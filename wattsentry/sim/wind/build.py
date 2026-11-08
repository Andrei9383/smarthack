"""Run simulator + fault injector and write the contract tables (zstd Parquet) to the wind cache.

Files (wind and data contracts):
  wind_telemetry.parquet  long `ts, tag, value, is_code` (15 min; codes are integral floats with is_code = true)
  turbines.parquet        `turbine_id, park_id, model, rated_kw, hub_height_m, rotor_d_m, lat, lon, county,
                          power_unit, power_scale` (+ osm_id, park_name, era5_point)
  meteo_wind.parquet      5-min per turbine `ts, turbine_id, wind_ms, wdir_deg`
  weather.parquet         hourly ERA5 at each park (contract `weather`, point_id = park id, + era5_point)
  fault_labels.parquet    one row per episode (asset_kind = turbine, fault = WindAnomaly value)
  manifest.json           version, configs, digest, inputs, counts, shear fit, CF calibration, fault and quality stats
  plots/power_vs_vnorm.svg + power_curve_bins.csv   sanity plot (binned power vs density-normalised wind, per park)
"""

from __future__ import annotations

import json
import os
import shutil
import time
from pathlib import Path

import numpy as np

from wattsentry.model.wind.data import (
    fault_labels_table,
    labels_from_episodes,
    meteo_table,
    to_long,
    turbines_table,
)

from .config import RAW, WIND_DATA_VERSION, WindSimConfig, code_hash, default_out_dir, digest, weather_file
from .generate import CleanWind, generate_clean

PQ = dict(compression="zstd", compression_level=6)


def _inputs(cfg: WindSimConfig) -> dict:
    files = [weather_file(cfg), RAW / "osm_wind_turbines.parquet", RAW / "entsoe_ro.parquet",
             RAW / "ro_installed_capacity.parquet"]
    return {f.name: (f.stat().st_size if f.exists() else None) for f in files}


def simulate(cfg: WindSimConfig | None = None, fcfg=None, faults: bool = True, log=print):
    """In memory: (CleanWind, FaultResult | None, WindData with labels)."""
    from wattsentry.sim.wind_faults import FaultConfig, inject

    cfg = cfg or WindSimConfig()
    clean = generate_clean(cfg, log=log)
    if not faults:
        return clean, None, clean.to_wind_data(clean.render(), source=f"sim-wind-v{WIND_DATA_VERSION}:clean")
    res = inject(clean, fcfg or FaultConfig())
    labels = labels_from_episodes(res.episodes, clean.ts, [t["turbine_id"] for t in clean.turbines])
    data = clean.to_wind_data(res.sensors, res.err, labels, res.episodes, source=f"sim-wind-v{WIND_DATA_VERSION}")
    return clean, res, data


def _write_chunked(path: Path, frames) -> int:
    import pyarrow.parquet as pq

    writer, rows = None, 0
    for df in frames:
        t = df.to_arrow()
        if writer is None:
            writer = pq.ParquetWriter(path, t.schema, **PQ)
        writer.write_table(t, row_group_size=1 << 20)
        rows += t.num_rows
    if writer is not None:
        writer.close()
    return rows


def _chunks(n: int, size: int):
    return [np.arange(i, min(n, i + size)) for i in range(0, n, size)]


def power_curve_bins(clean: CleanWind, data) -> list[dict]:
    """Binned mean power (fraction of rated) vs v_norm per park: clean and with faults (normal + faulty steps)."""
    rows = []
    edges = np.arange(0, 26.5, 0.5)
    vn = clean.v_norm
    for p, pid in enumerate(clean.park_ids):
        cols = np.flatnonzero(clean.park_of == p)
        x = vn[:, cols].reshape(-1)
        b = np.digitize(x, edges) - 1
        for name, pw in (("clean", clean.avail), ("faulted", data.sensors["act_pwt"])):
            y = (pw[:, cols] / clean.rated[cols][None]).reshape(-1)
            ok = np.isfinite(y) & (b >= 0) & (b < len(edges) - 1)
            cnt = np.bincount(b[ok], minlength=len(edges) - 1)
            sm = np.bincount(b[ok], weights=y[ok], minlength=len(edges) - 1)
            for i in np.flatnonzero(cnt >= 20):
                rows.append({"park_id": pid, "series": name, "v_norm_ms": float(edges[i] + 0.25),
                             "p_frac": round(float(sm[i] / cnt[i]), 4), "n": int(cnt[i])})
    return rows


def _svg(bins: list[dict], clean: CleanWind, data, path: Path, seed: int = 0) -> None:
    """Dependency-free SVG: one panel per park, scatter sample of (v_norm, power/rated) + binned means."""
    rng = np.random.default_rng(seed)
    W, H, pad = 320, 240, 36
    parks = clean.park_ids
    out = [f'<svg xmlns="http://www.w3.org/2000/svg" width="{W * len(parks)}" height="{H + 30}" '
           f'font-family="sans-serif" font-size="11">', '<rect width="100%" height="100%" fill="white"/>']
    for k, pid in enumerate(parks):
        ox = k * W
        sx = lambda v: ox + pad + v / 26 * (W - pad - 10)  # noqa: E731
        sy = lambda f: H - pad + 10 - f / 1.1 * (H - pad - 10)  # noqa: E731
        out.append(f'<text x="{ox + pad}" y="14">{pid} {clean.layouts[k].park.name} '
                   f'({clean.layouts[k].n} x {clean.layouts[k].park.tclass.name})</text>')
        out.append(f'<line x1="{sx(0)}" y1="{sy(0)}" x2="{sx(26)}" y2="{sy(0)}" stroke="#888"/>')
        out.append(f'<line x1="{sx(0)}" y1="{sy(0)}" x2="{sx(0)}" y2="{sy(1.1)}" stroke="#888"/>')
        for v in (0, 5, 10, 15, 20, 25):
            out.append(f'<text x="{sx(v) - 4}" y="{sy(0) + 14}">{v}</text>')
        for f in (0, 0.5, 1.0):
            out.append(f'<text x="{ox + 4}" y="{sy(f) + 4}">{f:.1f}</text>')
        cols = np.flatnonzero(clean.park_of == k)
        tt = rng.integers(0, clean.T, 1500)
        jj = cols[rng.integers(0, len(cols), 1500)]
        y = data.sensors["act_pwt"][tt, jj] / clean.rated[jj]
        x = clean.v_norm[tt, jj]
        for xi, yi in zip(x, y):
            if np.isfinite(yi):
                out.append(f'<circle cx="{sx(min(xi, 26)):.1f}" cy="{sy(np.clip(yi, -0.05, 1.1)):.1f}" r="1.2" '
                           f'fill="#868e96" fill-opacity="0.5"/>')
        for series, color in (("clean", "#1c7ed6"), ("faulted", "#e8590c")):
            pts = [(r["v_norm_ms"], r["p_frac"]) for r in bins if r["park_id"] == pid and r["series"] == series]
            if pts:
                d = " ".join(f"{sx(v):.1f},{sy(f):.1f}" for v, f in pts)
                out.append(f'<polyline points="{d}" fill="none" stroke="{color}" stroke-width="2"/>')
    out.append(f'<text x="{pad}" y="{H + 22}">x: density-normalised hub wind v_norm (m/s); y: act_pwt / rated. '
               f'Blue: clean binned mean; orange: with faults; grey: sample of faulted steps. Synthetic RO, '
               f'ERA5-driven.</text>')
    out.append("</svg>")
    path.write_text("\n".join(out))


def build(cfg: WindSimConfig | None = None, fcfg=None, out_dir: Path | None = None, force: bool = False,
          faults: bool = True, plots: bool = True, quality: bool = True, log=print) -> Path:
    import polars as pl

    from wattsentry.sim.wind_faults import FaultConfig

    cfg = cfg or WindSimConfig()
    fcfg = fcfg or FaultConfig()
    out = Path(out_dir) if out_dir else default_out_dir()
    dg = digest(cfg.to_json(), fcfg.to_json() if faults else None, _inputs(cfg))
    man_path = out / "manifest.json"
    if man_path.exists() and not force:
        try:
            if json.loads(man_path.read_text()).get("digest") == dg:
                log(f"[sim.wind] cached: {out} (digest {dg}); --force to rebuild")
                return out
        except ValueError:
            pass
    t0 = time.perf_counter()
    clean, res, data = simulate(cfg, fcfg, faults=faults, log=log)
    t_sim = time.perf_counter() - t0

    tmp = out.with_name(f".{out.name}.tmp-{os.getpid()}")  # per process: parallel runs never share a tmp dir
    if tmp.exists():
        shutil.rmtree(tmp)
    tmp.mkdir(parents=True)
    N = len(data.turbines)
    n_tel = _write_chunked(tmp / "wind_telemetry.parquet",
                           (to_long(data.subset(idx), decimals=2) for idx in _chunks(N, 8)))
    n_met = _write_chunked(tmp / "meteo_wind.parquet",
                           (meteo_table(data.subset(idx), decimals=2) for idx in _chunks(N, 16)))
    turbines_table(data).write_parquet(tmp / "turbines.parquet")
    weather = clean.weather.table.sort(["point_id", "ts"])
    weather.write_parquet(tmp / "weather.parquet", **PQ)
    fault_labels_table(data).write_parquet(tmp / "fault_labels.parquet")
    t_write = time.perf_counter() - t0 - t_sim

    # stats
    rated = clean.rated
    years = clean.ts.astype("datetime64[Y]").astype(int) + 1970
    cf = {}
    for y in np.unique(years):
        m = years == y
        cf[str(y)] = {"clean": round(float(clean.avail[m].sum() / (rated.sum() * m.sum())), 4),
                      "faulted": round(float(np.nansum(np.clip(data.sensors["act_pwt"][m], 0, None))
                                             / (rated.sum() * m.sum())), 4)}
    parks = []
    for p, lay in enumerate(clean.layouts):
        cols = np.flatnonzero(clean.park_of == p)
        parks.append({
            "park_id": lay.park.park_id, "name": lay.park.name, "county": lay.park.county,
            "turbines": int(lay.n), "class": lay.park.tclass.name,
            "rated_mw": round(float(rated[cols].sum()) / 1000, 1),
            "lat": round(float(lay.lat.mean()), 4), "lon": round(float(lay.lon.mean()), 4),
            "era5_point": lay.era5_point, "era5_dist_km": round(lay.era5_dist_km, 1),
            "shear_alpha": clean.weather.alpha_table[lay.park.park_id],
            "clean_cf": round(float(clean.avail[:, cols].mean() / rated[cols].mean()), 4),
            "mean_v_free_hub_ms": round(float(clean.v_free[:, cols].mean()), 2),
            "wake_speed_loss": round(float(1 - clean.v_eff[:, cols].mean() / clean.v_free[:, cols].mean()), 4),
            "mean_rho": round(float(clean.rho[:, cols].mean()), 4),
            "source": "real OSM turbine positions (ODbL) + ERA5 (Open-Meteo, CC BY 4.0)", "note": lay.park.note,
        })
    qstats = {}
    if quality:
        from wattsentry.data.wind.quality import apply_rules, flag_counts, wide_from_arrays

        tt = turbines_table(data)
        for p, pid in enumerate(clean.park_ids):
            cols = np.flatnonzero(clean.park_of == p)
            qstats[pid] = flag_counts(apply_rules(wide_from_arrays(data, cols), tt))
    lab = data.labels
    from wattsentry.model.wind.data import CLASSES

    step_share = {c: round(float((lab == i).mean()), 5) for i, c in enumerate(CLASSES)} if lab is not None else {}
    manifest = {
        "dataset": "synthetic RO wind, ERA5-driven (cards #22 + #23)",
        "label": "synthetic RO",
        "wind_data_version": WIND_DATA_VERSION,
        "digest": dg,
        "code_hash": code_hash(),
        "config": cfg.to_json(),
        "fault_config": fcfg.to_json() if faults else None,
        "inputs": _inputs(cfg),
        "period": [str(clean.ts[0]), str(clean.ts[-1])],
        "steps": int(clean.T),
        "turbines": int(N),
        "parks": parks,
        "era5_speed_factor": clean.k_speed,
        "cf_target_ro_national_first_year": clean.cf_target,
        "capacity_factor": cf,
        "rows": {"wind_telemetry": n_tel, "meteo_wind": n_met, "fault_labels": len(data.episodes),
                 "weather": weather.height},
        "fault_stats": res.stats if res is not None else None,
        "step_label_share_effective": step_share,
        "quality": qstats,
        "splits": "M6 convention (wattsentry.model.wind.data.make_splits): test = 2025; 2024 day-of-month <= 19 train, "
                  ">= 22 val; park WP2 held out of training",
        "timings_s": {"simulate": round(t_sim, 1), "write": round(t_write, 1)},
    }
    if plots:
        (tmp / "plots").mkdir()
        bins = power_curve_bins(clean, data)
        pl.DataFrame(bins).write_csv(tmp / "plots" / "power_curve_bins.csv")
        _svg(bins, clean, data, tmp / "plots" / "power_vs_vnorm.svg")
    manifest["timings_s"]["total"] = round(time.perf_counter() - t0, 1)
    manifest["bytes"] = {f.name: f.stat().st_size for f in sorted(tmp.glob("*.parquet"))}
    (tmp / "manifest.json").write_text(json.dumps(manifest, indent=2, ensure_ascii=False, default=str))
    if out.exists():
        shutil.rmtree(out, ignore_errors=True)
    try:
        tmp.rename(out)
    except OSError:  # another process published the same dataset first
        shutil.rmtree(tmp, ignore_errors=True)
    mb = sum(manifest["bytes"].values()) / 1e6
    log(f"[sim.wind] wrote {out}: {N} turbines, {clean.T} steps, {n_tel:,} telemetry rows, "
        f"{len(data.episodes)} fault episodes, {mb:.0f} MB, {manifest['timings_s']['total']}s")
    return out
