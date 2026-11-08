"""Orchestrates the fleet simulation and writes the dataset (see SCHEMA.md)."""

from __future__ import annotations

import json
import os
import shutil
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq

from wattsentry import kernels

from .battery import BatteryParams
from .config import DATA_VERSION, DT_H, FleetConfig
from .grid import feeder_flows, passive_load
from .load import simulate_load
from .localities import BY_ID
from .pv import simulate_pv
from .ro_calendar import local_features, utc_index
from .sites import build_world
from .weather import load_weather, weather_key

PQ = dict(compression="zstd", compression_level=6)


def _ts_array(ts: pd.DatetimeIndex, reps: int) -> pa.Array:
    return pa.array(np.tile(ts.tz_convert("UTC").tz_localize(None).to_numpy("datetime64[ns]"), reps),
                    pa.timestamp("ns", tz="UTC"))


def _ids(ids: np.ndarray, T: int) -> pa.Array:
    """Each id repeated T times, as a plain string column (the contract type); Parquet dictionary-encodes it."""
    return pa.DictionaryArray.from_arrays(pa.array(np.repeat(np.arange(len(ids), dtype=np.int32), T)),
                                          pa.array(list(ids))).cast(pa.string())


# simulate_pv is row-independent and works per locality, so running it on buckets of whole localities gives the
# same arrays (bit for bit) as one call; numpy releases the GIL, so the buckets run in parallel.
_PV_THREADS = max(1, min(8, (os.cpu_count() or 2) - 1))


def _locality_buckets(sites: pd.DataFrame) -> list[np.ndarray]:
    groups = sorted(sites.groupby("locality_id").indices.values(), key=len, reverse=True)
    buckets: list[list[np.ndarray]] = [[] for _ in range(min(len(groups), 2 * _PV_THREADS))]
    sizes = [0] * len(buckets)
    for g in groups:  # largest first into the emptiest bucket
        k = sizes.index(min(sizes))
        buckets[k].append(g)
        sizes[k] += len(g)
    return [np.sort(np.concatenate(b)) for b in buckets if b]


def _gather_pv(parts: dict, n: int, T: int) -> tuple[np.ndarray, np.ndarray]:
    pv = np.empty((n, T), np.float32)
    status = np.empty((n, T), np.int16)
    for f in as_completed(list(parts)):  # copy each bucket as it lands and drop it (bounds peak memory)
        pv[parts[f]], status[parts[f]] = f.result()
        del parts[f]
    return pv, status


def generate(cfg: FleetConfig, out_dir: Path | None = None, force: bool = False, plots: bool = True,
             log=print) -> Path:
    wkey = weather_key(cfg.weather_source)
    out = Path(out_dir) if out_dir else cfg.out_dir(weather_key=wkey)
    if (out / "manifest.json").exists() and not force:
        log(f"cached: {out}")
        return out
    if out.exists():
        shutil.rmtree(out)
    tmp = out.with_name(out.name + ".tmp")
    if tmp.exists():
        shutil.rmtree(tmp)
    (tmp / "telemetry").mkdir(parents=True)
    (tmp / "feeder_ts").mkdir()
    t0 = time.perf_counter()

    feeders, sites = build_world(cfg)
    locs = [BY_ID[i] for i in sorted(set(feeders.locality_id))]
    all_ts = utc_index(cfg.years[0]).append([utc_index(y) for y in cfg.years[1:]])
    weather = load_weather(locs, all_ts, cfg.seed, cfg.weather_source)
    log(f"world: {len(sites)} sites on {len(feeders)} feeders in {len(locs)} localities; weather={weather.source} "
        f"({time.perf_counter() - t0:.1f}s)")
    io = ThreadPoolExecutor(1)  # Parquet writes (pyarrow releases the GIL) overlap the simulation (#135)
    weather_job = io.submit(_write_weather, weather, tmp / "weather.parquet")

    bp = BatteryParams.from_sites(sites)
    e = None
    g_w = feeders.g_max_kw.to_numpy()[:, None] * 1000
    feeder_parts, stats = [], []
    for year in cfg.years:
        ty = time.perf_counter()
        ts = utc_index(year)
        w = weather.slice(ts)
        cal = local_features(ts)
        rng = np.random.default_rng([cfg.seed, year, 2])
        with ThreadPoolExecutor(_PV_THREADS) as ex:  # PV has no RNG: threads overlap it with simulate_load (#135)
            pv_parts = {ex.submit(simulate_pv, sites.iloc[rows], w): rows for rows in _locality_buckets(sites)}
            load = simulate_load(sites, w, cal, rng)
            pv, status = _gather_pv(pv_parts, len(sites), len(ts))
        batt, soc, e = kernels.self_consumption(bp, pv, load, DT_H, e)  # Rust K7 if built, else battery.py
        grid = pv - load - batt
        passive = passive_load(feeders, w, cal, rng)
        fdf, volt = feeder_flows(sites, feeders, grid, passive, w, rng)
        tel_job = io.submit(_write_telemetry, tmp / "telemetry" / f"year={year}.parquet", sites.site_id.to_numpy(),
                            ts, pv=pv, load=load, batt=batt, soc=soc, grid=grid, volt=volt, status=status)
        fdf.to_parquet(tmp / "feeder_ts" / f"year={year}.parquet", index=False, **PQ)
        net = fdf.net_export_w.to_numpy().reshape(len(feeders), -1)
        feeder_parts.append(net)
        stats.append(_year_stats(year, feeders, net, g_w, cal, sites, pv, load))
        tel_job.result()  # before the next year allocates its arrays (bounds peak memory)
        log(f"{year}: pv {pv.sum() * DT_H / 1e6:,.0f} MWh, load {load.sum() * DT_H / 1e6:,.0f} MWh, "
            f"feeders over limit: {int((stats[-1].exceed_days > 0).sum())}/{len(feeders)} "
            f"({time.perf_counter() - ty:.1f}s)")
        del pv, load, batt, soc, grid, volt, status
    weather_job.result()
    io.shutdown()

    sites.to_parquet(tmp / "sites.parquet", index=False, **PQ)
    feeders.to_parquet(tmp / "feeders.parquet", index=False, **PQ)
    st = pd.concat(stats).groupby("feeder_id", sort=False).agg(
        exceed_steps=("exceed_steps", "sum"), exceed_days=("exceed_days", "sum"), max_ratio=("max_ratio", "max"),
        sunny_summer_days=("sunny_summer_days", "sum"), sunny_summer_exceed_days=("sunny_summer_exceed_days", "sum"),
    ).reset_index()
    st["sunny_exceed_share"] = (st.sunny_summer_exceed_days / st.sunny_summer_days.clip(lower=1)).round(3)
    st.to_csv(tmp / "feeder_stats.csv", index=False)
    seconds = time.perf_counter() - t0
    congested = int((st.sunny_exceed_share >= 0.25).sum())  # over the limit on >= 1 in 4 sunny summer days
    manifest = {
        "data_version": DATA_VERSION, "config": cfg.to_json(), "digest": cfg.digest(wkey), "weather_source": weather.source, "weather_key": wkey,
        "n_sites": len(sites), "n_feeders": len(feeders), "steps_per_year": {y: len(utc_index(y)) for y in cfg.years},
        "feeders_congested_on_sunny_days": congested, "seconds": round(seconds, 1),
        "files": sorted(str(p.relative_to(tmp)) for p in tmp.rglob("*.parquet")),
    }
    (tmp / "manifest.json").write_text(json.dumps(manifest, indent=2, ensure_ascii=False))
    if plots:
        try:
            from .plots import sanity_plots
            sanity_plots(tmp, feeders, all_ts, np.concatenate(feeder_parts, 1))
        except ImportError:
            log("matplotlib not installed: skipping plots")
    tmp.rename(out)
    log(st.to_string(index=False))
    log(f"LAB_METRIC fleet_feeders_congested={congested}")
    log(f"LAB_METRIC fleet_gen_seconds={seconds:.1f}")
    log(f"wrote {out}")
    return out


def _write_weather(w, path: Path) -> None:
    T, n = len(w.ts), len(w.locality_ids)
    cols = {"ts": _ts_array(w.ts, n), "point_id": _ids(np.array(w.locality_ids), T)}
    for k, name in (("ghi", "ghi_wm2"), ("dni", "dni_wm2"), ("dhi", "dhi_wm2"), ("temp_air", "temp_c"),
                    ("wind_speed", "wind10_ms"), ("snow_depth_cm", "snow_cm")):
        cols[name] = pa.array(w.data[k].ravel())
    cols["source"] = pa.DictionaryArray.from_arrays(pa.array(np.zeros(T * n, np.int32)), pa.array([w.source]))
    pq.write_table(pa.table(cols), path, **PQ)


def _write_telemetry(path: Path, site_ids: np.ndarray, ts: pd.DatetimeIndex, *, pv, load, batt, soc, grid, volt,
                     status) -> None:
    T = len(ts)

    def r(x, step):  # round to the meter's resolution: smaller files, nothing a model could use is lost
        return pa.array((np.round(x.ravel() / step) * step).astype(np.float32), from_pandas=True)  # NaN -> null

    table = pa.table({
        "ts": _ts_array(ts, len(site_ids)),
        "site_id": _ids(site_ids, T),
        "pv_ac_w": r(pv, 1),
        "inv_voltage_v": r(volt, 0.1),
        "inv_status": pa.array(status.ravel()),
        "soc": r(soc, 0.001),
        "batt_w": r(batt, 1),
        "load_w": r(load, 1),
        "grid_w": r(grid, 1),
    })
    pq.write_table(table, path, row_group_size=T * 32, **PQ)


def _year_stats(year, feeders, net, g_w, cal, sites, pv, load) -> pd.DataFrame:
    day = cal["day_index"]
    n_days = int(day[-1]) + 1
    over = net > g_w
    over_day = np.zeros((len(feeders), n_days), bool)
    np.logical_or.at(over_day.T, day, over.T)
    # A sunny summer day for a feeder: Apr-Sep, its sites' PV >= 70 % of their best day that year.
    fidx = pd.Index(feeders.feeder_id).get_indexer(sites.feeder_id)
    onehot = np.zeros((len(feeders), len(sites)), np.float32)
    onehot[fidx, np.arange(len(sites))] = 1
    pv_f = onehot @ pv
    pv_day = np.zeros((len(feeders), n_days))
    np.add.at(pv_day.T, day, pv_f.T)
    month_day = np.zeros(n_days, int)
    month_day[day] = cal["month"]
    summer = (month_day >= 4) & (month_day <= 9)
    sunny = (pv_day >= 0.7 * pv_day.max(1, keepdims=True)) & summer[None]
    return pd.DataFrame({
        "feeder_id": feeders.feeder_id, "year": year,
        "exceed_steps": over.sum(1), "exceed_days": over_day.sum(1),
        "max_ratio": (net.max(1) / g_w[:, 0]).round(3),
        "sunny_summer_days": sunny.sum(1), "sunny_summer_exceed_days": (sunny & over_day).sum(1),
    })
