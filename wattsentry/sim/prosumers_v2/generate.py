"""Generate data-v2 (see SCHEMA.md): homes with contracts, meters and battery strategies, their 15-min truth, and the
meter layer PPC sees. PV and household load come from the fleet simulator's physics (unchanged, called on new
homes); batteries run the K8 strategies kernel (wattsentry.kernels.battery_strategies)."""

from __future__ import annotations

import json
import os
import shutil
import time
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq

from wattsentry import kernels
from wattsentry.kernels.strategies import STRATEGIES, day_rank_median
from wattsentry.sim.fleet.generate import _ids, _ts_array, _write_weather
from wattsentry.sim.fleet.load import simulate_load
from wattsentry.sim.fleet.localities import BY_ID
from wattsentry.sim.fleet.pv import simulate_pv
from wattsentry.sim.fleet.ro_calendar import local_features, utc_index
from wattsentry.sim.fleet.weather import load_weather, weather_key

from . import metering
from .config import DT_H, PV2_VERSION, ProsumerConfig
from .homes import BATTERY_HIDDEN, PUBLIC_COLUMNS, build_homes, initial_soc_kwh, strategy_codes
from .prices import PRICES, prices_key, pzu_15min

PQ = dict(compression="zstd", compression_level=6)


def _pq(names: list[str]) -> dict:
    """Long tables: time columns delta-encoded (constant steps -> ~0 bytes), the rest dictionary-encoded (values are
    rounded to the meter resolution, so they repeat)."""
    ts = [c for c in names if c in ("ts", "available_at")]
    return dict(compression="zstd", compression_level=3, use_dictionary=[c for c in names if c not in ts],
                column_encoding={c: "DELTA_BINARY_PACKED" for c in ts})
TRAINING_FILES = ["homes_public.parquet", "meter_observed", "meter_monthly.parquet", "settlement",
                  "prices.parquet", "weather.parquet", "standard_profile.parquet"]
HIDDEN_FILES = ["homes.parquet", "truth/home_15min", "truth/portfolio"]
# Columns that must never appear in a training file (spec I step 3 + the battery parameters).
FORBIDDEN_COLUMNS = {"pv_kw", "load_kw", "batt_kw", "soc_kwh", "strategy", *BATTERY_HIDDEN, "pv_ac_w", "load_w",
                     "batt_w", "soc"}


def resolve_out(cfg: ProsumerConfig, out_dir: Path | None = None) -> Path:
    return Path(out_dir) if out_dir else cfg.out_dir(weather_key=weather_key(cfg.weather_source),
                                                     prices_key=prices_key())


def generate(cfg: ProsumerConfig, out_dir: Path | None = None, force: bool = False, log=print,
             workers: int | None = None) -> Path:
    out = resolve_out(cfg, out_dir)
    if (out / "manifest.json").exists() and not force:
        log(f"cached: {out}")
        return out
    if out.exists():
        shutil.rmtree(out)
    tmp = out.with_name(out.name + ".tmp")
    if tmp.exists():
        shutil.rmtree(tmp)
    for d in ("truth/portfolio", "settlement"):
        (tmp / d).mkdir(parents=True)
    for y in cfg.years:
        for d in ("truth/home_15min", "meter_observed"):
            (tmp / d / f"year={y}").mkdir(parents=True)
    t0 = time.perf_counter()
    timing: dict[str, float] = {}

    homes = build_homes(cfg)
    n = len(homes)
    locs = [BY_ID[i] for i in sorted(set(homes.locality_id))]
    all_ts = utc_index(cfg.years[0]).append([utc_index(y) for y in cfg.years[1:]])
    weather = load_weather(locs, all_ts, cfg.seed, cfg.weather_source)
    _write_weather(weather, tmp / "weather.parquet")
    prices = pzu_15min(all_ts)
    cust_all = np.clip(prices.pzu_lei_mwh.to_numpy(), cfg.maxcontrol_floor_lei_mwh, cfg.maxcontrol_cap_lei_mwh)
    prof = metering.standard_profile(all_ts, cfg.standard_profile)
    log(f"data-v2: {n} homes in {len(locs)} localities, weather={weather.source}, kernels={kernels.backend()} "
        f"({time.perf_counter() - t0:.1f}s)")

    years = cfg.years
    yts = {y: utc_index(y) for y in years}
    county_lab, county_idx = np.unique(homes.county.to_numpy(), return_inverse=True)
    gkey = homes.county + "|" + homes.meter + "|" + homes.strategy
    g_lab, g_idx = np.unique(gkey.to_numpy(), return_inverse=True)
    yinfo = {}
    for y in years:
        ts = yts[y]
        sl = slice(all_ts.get_loc(ts[0]), all_ts.get_loc(ts[0]) + len(ts))
        cal = local_features(ts)
        day = cal["day_index"].astype(np.int32)
        cust = cust_all[sl]
        rank, med = day_rank_median(cust, day)
        midx, mlab = metering.month_index(ts)
        yinfo[y] = dict(ts=ts, sl=sl, cal=cal, day=day, cust=cust, rank=rank, med=med, midx=midx, mlab=mlab,
                        avail=metering.available_at(ts))
    global _CTX
    _CTX = dict(cfg=cfg, homes=homes, weather=weather, yinfo=yinfo, g_idx=g_idx, n_groups=len(g_lab),
                county_idx=county_idx, n_counties=len(county_lab), tmp=tmp)
    chunks = list(range(0, n, cfg.chunk_homes))
    n_workers = max(1, min(workers or _default_workers(), len(chunks)))
    log(f"simulating {len(chunks)} chunks of <= {cfg.chunk_homes} homes x {len(years)} years on {n_workers} processes")
    if n_workers == 1:
        acc = _accumulate(map(_simulate_chunk, range(len(chunks))), years, log)
    else:
        # spawn, not fork: a parent that already ran a Rust kernel has a live rayon pool, and forked children
        # deadlock on it (seen in pytest). Each child gets the context once, through the initializer.
        import multiprocessing as mp
        with mp.get_context("spawn").Pool(n_workers, initializer=_init_worker, initargs=(_CTX,)) as pool:
            acc = _accumulate(pool.imap_unordered(_simulate_chunk, range(len(chunks))), years, log)
    _CTX = {}

    monthly_rows, summer = [], {s: [0.0, 0.0] for s in STRATEGIES}
    home_ids = homes.home_id.to_numpy()
    smart = (homes.meter == "smart").to_numpy()
    onehot_c = np.zeros((n, len(county_lab)))
    onehot_c[np.arange(n), county_idx] = 1
    parts = pd.Series(g_lab).str.split("|", expand=True)
    for y in years:
        a, yi = acc[y], yinfo[y]
        ts, midx, mlab, T = yi["ts"], yi["midx"], yi["mlab"], len(yi["ts"])
        # settlement per county: smart on the measured curve, profile on the standard profile
        prof_y = prof.iloc[yi["sl"]]
        pc_imp = onehot_c.T @ (a["mon_imp"] * (~smart)[:, None])  # (counties, months)
        pc_exp = onehot_c.T @ (a["mon_exp"] * (~smart)[:, None])
        s_imp = metering.settle_on_profile(pc_imp, prof_y.load_coef.to_numpy(), midx)
        s_exp = metering.settle_on_profile(pc_exp, prof_y.solar_coef.to_numpy(), midx)
        nc, ng = len(county_lab), len(g_lab)
        st = pa.table({
            "ts": _ts_array(ts, nc), "county": _ids(county_lab, T),
            "smart_import_kwh": pa.array(a["smart_import"].ravel()),
            "smart_export_kwh": pa.array(a["smart_export"].ravel()),
            "profile_import_kwh": pa.array(s_imp.ravel()), "profile_export_kwh": pa.array(s_exp.ravel()),
        })
        pq.write_table(st, tmp / "settlement" / f"year={y}.parquet", **_pq(st.column_names))
        pt = pa.table({
            "ts": _ts_array(ts, ng), "county": _ids(parts[0].to_numpy(), T), "meter": _ids(parts[1].to_numpy(), T),
            "strategy": _ids(parts[2].to_numpy(), T),
            "n_homes": pa.array(np.repeat(np.bincount(g_idx, minlength=ng).astype(np.int32), T)),
            **{f"{k}_kwh": pa.array(a[k].ravel()) for k in AGG},
        })
        pq.write_table(pt, tmp / "truth/portfolio" / f"year={y}.parquet", **_pq(pt.column_names))
        monthly_rows.append(pd.DataFrame({
            "home_id": np.repeat(home_ids, len(mlab)), "month": np.tile(mlab, n),
            "meter": np.repeat(homes.meter.to_numpy(), len(mlab)),
            "import_kwh": a["mon_imp"].ravel().round(3), "export_kwh": a["mon_exp"].ravel().round(3),
            "available_at": np.tile(metering.month_available_at(mlab), n)}))
        _summer_export(summer, a["export"], parts[2].to_numpy(), yi["cal"])
        for k in ("sim", "kernel", "write"):
            timing[f"t_{k}_{y}_s"] = a[f"t_{k}"]
        log(f"{y}: import {a['import'].sum() / 1e3:,.0f} MWh, export {a['export'].sum() / 1e3:,.0f} MWh, battery "
            f"throughput {np.abs(a['batt']).sum() / 1e3:,.0f} MWh (process-seconds: sim {a['t_sim']:.1f}, "
            f"kernel {a['t_kernel']:.1f}, write {a['t_write']:.1f})")
    timing["t_parallel_wall_s"] = acc["wall"]
    has_batt = homes.has_battery.to_numpy()

    pd.concat(monthly_rows, ignore_index=True).to_parquet(tmp / "meter_monthly.parquet", index=False, **PQ)
    pr = prices.assign(customer_price_lei_mwh=cust_all, day_rank=np.concatenate([yinfo[y]["rank"] for y in years]))
    pr.to_parquet(tmp / "prices.parquet", index=False, **PQ)
    prof.to_parquet(tmp / "standard_profile.parquet", index=False, **PQ)
    homes.to_parquet(tmp / "homes.parquet", index=False, **PQ)
    homes[PUBLIC_COLUMNS].to_parquet(tmp / "homes_public.parquet", index=False, **PQ)

    seconds = time.perf_counter() - t0
    share_18_21 = {s: (a / b if b > 0 else None) for s, (a, b) in summer.items()}
    stats = {
        "share_battery": float(homes.has_battery.mean()), "share_maxcontrol": float((homes.contract == "maxcontrol").mean()),
        "share_smart_meter": float(smart.mean()),
        "share_battery_declared_of_battery": float(homes.battery_declared[homes.has_battery].mean()) if has_batt.any() else None,
        "strategy_counts": homes.strategy.value_counts().to_dict(),
        "summer_export_share_18_21": share_18_21,
        "price_source_counts": prices.price_source.value_counts().to_dict(),
    }
    manifest = {
        "data_version": f"prosumers-v2/v{PV2_VERSION}", "config": cfg.to_json(), "digest": out.name,
        "weather_source": weather.source, "weather_key": weather_key(cfg.weather_source),
        "prices": str(PRICES.relative_to(PRICES.parents[2])), "prices_key": prices_key(),
        "kernel_backend": kernels.backend(), "n_homes": n, "years": cfg.years,
        "training_files": TRAINING_FILES, "hidden_files": HIDDEN_FILES,
        "forbidden_training_columns": sorted(FORBIDDEN_COLUMNS),
        "labels": {"homes, curves, contracts, strategies, meters": "synthetic RO",
                   "share_battery, share_smart_meter, MaxControl price rule": "real RO (cited in config.py)",
                   "PZU prices 2023-2025": "real RO (OPCOM)", "PZU prices 2022": "proxy (2023 shape, +371 d)",
                   "standard profile": prof.profile_source.iloc[0] if len(prof) else None,
                   "weather": "real RO (ERA5/Open-Meteo, #9)" if weather.source == "raw" else "synthetic"},
        "stats": stats, "timing": {k: round(v, 2) for k, v in timing.items()}, "seconds": round(seconds, 1),
        "files": sorted(str(p.relative_to(tmp)) for p in tmp.rglob("*.parquet")),
    }
    (tmp / "manifest.json").write_text(json.dumps(manifest, indent=2, ensure_ascii=False, default=str))
    tmp.rename(out)
    for s, v in share_18_21.items():
        if v is not None:
            log(f"LAB_METRIC pv2_summer_export_share_18_21_{s}={v:.4f}")
    log(f"LAB_METRIC pv2_share_battery={stats['share_battery']:.4f}")
    log(f"LAB_METRIC pv2_share_smart_meter={stats['share_smart_meter']:.4f}")
    log(f"LAB_METRIC pv2_n_price_driven={stats['strategy_counts'].get('price_driven', 0)}")
    log(f"LAB_METRIC pv2_gen_seconds={seconds:.1f}")
    for k, v in timing.items():
        log(f"LAB_METRIC pv2_{k}={v:.2f}")
    log(f"wrote {out}")
    return out


AGG = ("import", "export", "pv", "load", "batt")
_CTX: dict = {}


def _default_workers() -> int:
    env = os.environ.get("WATTSENTRY_PV2_WORKERS")
    return int(env) if env else max(1, min(4, (os.cpu_count() or 2) // 4))


WRITE_SLICE = 100  # homes per write call: keeps each process's Arrow buffers small (output bytes unchanged)


def _init_worker(ctx: dict) -> None:
    os.environ.setdefault("RAYON_NUM_THREADS", "2")  # before the first kernel call starts rayon's pool
    global _CTX
    _CTX = ctx


def _simulate_chunk(ci: int) -> tuple[int, dict]:
    """One chunk of homes through all years (SoC carries over): writes its truth and meter parts, returns the sums
    the parent needs (monthly totals per home, group and county curves)."""
    c = _CTX
    cfg, homes, weather = c["cfg"], c["homes"], c["weather"]
    n = len(homes)
    i0, i1 = ci * cfg.chunk_homes, min(n, (ci + 1) * cfg.chunk_homes)
    sub = homes.iloc[i0:i1]
    m = i1 - i0
    codes = strategy_codes(sub)
    cap, pmax = sub.cap_kwh.to_numpy(float), sub.p_kw.to_numpy(float)
    reserve, eta = sub.reserve_frac.to_numpy(float), np.where(sub.has_battery, sub.eta, 1.0)
    k_top = sub.k_top.to_numpy(np.int32)
    has_batt = sub.has_battery.to_numpy()
    sm = (sub.meter == "smart").to_numpy()
    ids = sub.home_id.to_numpy()
    soc_state = initial_soc_kwh(sub)
    G = np.zeros((c["n_groups"], m))
    G[c["g_idx"][i0:i1], np.arange(m)] = 1
    C = np.zeros((c["n_counties"], m))
    C[c["county_idx"][i0:i1], np.arange(m)] = 1
    C = C * sm[None]
    res = {}
    for year, yi in c["yinfo"].items():
        ts, T, day = yi["ts"], len(yi["ts"]), yi["day"]
        n_days = int(day[-1]) + 1 if T else 0
        rng = np.random.default_rng([cfg.seed, 172, year, 2])  # same stream in every chunk -> slice our rows
        comply = (rng.random((n, n_days)) >= cfg.noncompliance_prob)[i0:i1]
        missing = (rng.random((n, n_days)) < cfg.meter_missing_day_prob)[i0:i1]
        ta = time.perf_counter()
        w = weather.slice(ts)
        pv_w, _ = simulate_pv(sub, w)
        load_w = simulate_load(sub, w, yi["cal"], np.random.default_rng([cfg.seed, 172, year, 3, ci]))
        pv, load = pv_w.astype(np.float64) / 1000, load_w.astype(np.float64) / 1000
        del pv_w, load_w
        tb = time.perf_counter()
        batt, soc = kernels.battery_strategies(pv, load, codes, cap, pmax, reserve, eta, k_top, soc_state, yi["cust"],
                                               yi["rank"], yi["med"], day, comply, DT_H)
        tc = time.perf_counter()
        if T:
            soc_state = soc[:, -1].copy()
        net = load - pv + batt  # kW, > 0 import
        imp, exp = np.maximum(net, 0) * DT_H, np.maximum(-net, 0) * DT_H
        del net
        soc = np.where(has_batt[:, None], soc, np.nan)
        with pq.ParquetWriter(c["tmp"] / "truth/home_15min" / f"year={year}" / f"part-{ci:03d}.parquet",
                              _TRUTH_SCHEMA, **_pq(_TRUTH_SCHEMA.names)) as wr:
            for a in range(0, m, WRITE_SLICE):
                b = slice(a, min(m, a + WRITE_SLICE))
                k = b.stop - b.start
                wr.write_table(pa.table({
                    "ts": _ts_array(ts, k), "home_id": _ids(ids[b], T), "pv_kw": _r(pv[b]), "load_kw": _r(load[b]),
                    "batt_kw": _r(batt[b]), "soc_kwh": _r(soc[b]), "import_kwh": _r(imp[b]),
                    "export_kwh": _r(exp[b])}, schema=_TRUTH_SCHEMA), row_group_size=T * 32)
        if sm.any():
            rows = np.flatnonzero(sm)
            with pq.ParquetWriter(c["tmp"] / "meter_observed" / f"year={year}" / f"part-{ci:03d}.parquet",
                                  _METER_SCHEMA, **_pq(_METER_SCHEMA.names)) as wr:
                for a in range(0, len(rows), WRITE_SLICE):
                    rr = rows[a:a + WRITE_SLICE]
                    gone = missing[rr][:, day]
                    wr.write_table(pa.table({
                        "ts": _ts_array(ts, len(rr)), "home_id": _ids(ids[rr], T),
                        "import_kwh": _r(np.where(gone, np.nan, imp[rr])),
                        "export_kwh": _r(np.where(gone, np.nan, exp[rr])),
                        "available_at": _ts_array(yi["avail"], len(rr))}, schema=_METER_SCHEMA), row_group_size=T * 32)
        r = {"mon_imp": metering.monthly_sums(imp, yi["midx"], len(yi["mlab"])),
             "mon_exp": metering.monthly_sums(exp, yi["midx"], len(yi["mlab"])),
             "smart_import": C @ imp, "smart_export": C @ exp}
        for key, x in zip(AGG, (imp, exp, pv * DT_H, load * DT_H, batt * DT_H)):
            r[key] = G @ x
        r.update(t_sim=tb - ta, t_kernel=tc - tb, t_write=time.perf_counter() - tc)
        res[year] = r
    return ci, res


def _accumulate(results, years, log) -> dict:
    t0 = time.perf_counter()
    acc: dict = {}
    cs = _CTX["cfg"].chunk_homes
    for ci, res in results:
        i0 = ci * cs
        for y, r in res.items():
            a = acc.setdefault(y, {"mon_imp": {}, "mon_exp": {}})
            a["mon_imp"][i0], a["mon_exp"][i0] = r.pop("mon_imp"), r.pop("mon_exp")
            for k, v in r.items():
                a[k] = a[k] + v if k in a else v
        log(f"  chunk {ci} done ({time.perf_counter() - t0:.0f}s)")
    for y in years:
        for k in ("mon_imp", "mon_exp"):
            acc[y][k] = np.concatenate([acc[y][k][i] for i in sorted(acc[y][k])])
    acc["wall"] = time.perf_counter() - t0
    return acc


def _r(x: np.ndarray, step: float = 0.001) -> pa.Array:
    """Round to 1 Wh / 1 W (meter resolution), float32; NaN -> null."""
    return pa.array((np.round(x.ravel() / step) * step).astype(np.float32), from_pandas=True)


def _summer_export(acc: dict, export: np.ndarray, strat: np.ndarray, cal: dict) -> None:
    jja = np.isin(cal["month"], (6, 7, 8))
    eve = jja & (cal["hour"] >= 18) & (cal["hour"] < 21)
    for s in acc:
        rows = strat == s
        if rows.any():
            acc[s][0] += float(export[rows][:, eve].sum())
            acc[s][1] += float(export[rows][:, jja].sum())


_TS = pa.timestamp("ns", tz="UTC")
_TRUTH_SCHEMA = pa.schema([("ts", _TS), ("home_id", pa.string()), ("pv_kw", pa.float32()), ("load_kw", pa.float32()),
                           ("batt_kw", pa.float32()), ("soc_kwh", pa.float32()), ("import_kwh", pa.float32()),
                           ("export_kwh", pa.float32())])
_METER_SCHEMA = pa.schema([("ts", _TS), ("home_id", pa.string()), ("import_kwh", pa.float32()),
                           ("export_kwh", pa.float32()), ("available_at", _TS)])
