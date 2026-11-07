"""Build the model window cache (card #10): fleet (#11) + faults (#12) -> the `sentinel-windows-v1` layout that
sentinel.model.data.ArraySource reads (#14), at ~/.cache/sentinel/data-v{N}/windows/.

    python -m sentinel.data.windows [--quick] [--force] [--out DIR]

Layout (memmap-able .npy + JSON, see #14 synth.py):
  site arrays [S, T] float16, kW / V / 0..1: pv_ac (REPORTED, with faults), pv_healthy (clean physics), load, batt,
              soc, inv_v, pvlib_rec, pvlib_rec_fc, pvclear_rec (pvlib from RECORDED metadata, actual / forecast
              weather / clear sky);  int8: inv_status (0 off, 1 producing, 2 limiting, 3 overvoltage trip),
              fault (PvFault index, 0 = healthy), valid (0 where the logger dropped out; values there are 0-filled)
  locality    [L, T, 7] float16 weather_act, weather_fc (ghi, dni, dhi, temp, cloud, wind, snow); [L, T, 3] geom
  time        [T, 8] float32 calendar
  sites.json  sites table rows + loc_idx + meta_wrong;  episodes.json  fault episodes with step onset / end
The time axis is whole UTC days. pvlib channels and geometry use sentinel.model.physics, the same code the ONNX
predictor runs at inference, so training and serving features match.
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import time
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow.parquet as pq

from sentinel.contracts.envelope import PvFault
from sentinel.sim.faults import FaultConfig, inject_dataset
from sentinel.sim.faults.kinds import INV_OVERVOLTAGE
from sentinel.sim.fleet import FleetConfig, generate
from sentinel.sim.fleet.config import DATA_VERSION, QUICK, default_cache_root
from sentinel.sim.fleet.localities import BY_ID

FORMAT = "sentinel-windows-v1"
FAULTS = [f.value for f in PvFault]  # index 0 = healthy
FAULT_IDX = {f: i for i, f in enumerate(FAULTS)}
DAY = 96
SITE_F16 = ("pv_ac", "pv_healthy", "load", "batt", "soc", "inv_v", "pvlib_rec", "pvlib_rec_fc", "pvclear_rec")
SITE_I8 = ("inv_status", "fault", "valid")


def physics():
    try:
        from sentinel.model import physics as ph
    except ImportError as e:  # pragma: no cover
        raise ImportError("sentinel.model.physics (#14) is needed for the pvlib channels and geometry") from e
    return ph


def default_out() -> Path:
    return default_cache_root() / f"data-v{DATA_VERSION}" / "windows"


def _axis(ts_all: pd.DatetimeIndex) -> tuple[int, int]:
    """[i0, i1) of the longest run of whole UTC days inside the fleet's local-year axis."""
    first_midnight = ts_all[0].ceil("D")
    i0 = int(np.searchsorted(ts_all.asi8, first_midnight.value))
    n_days = (len(ts_all) - i0) // DAY
    return i0, i0 + n_days * DAY


def _cloud_from_ghi(ghi: np.ndarray, cs: np.ndarray) -> np.ndarray:
    """Cloud % from the daily clear-sky index (synthetic weather has no cloud channel)."""
    T = len(ghi)
    k = np.divide(ghi.reshape(-1, DAY).sum(1), np.maximum(cs.reshape(-1, DAY).sum(1), 1e-6))
    return np.repeat(np.clip(100 * (1.05 - k) / 0.95, 0, 100), DAY)[:T]


def forecast_weather(act: np.ndarray, geom: np.ndarray, doy: np.ndarray, rng, ph) -> np.ndarray:
    """Day-ahead forecast from actuals, same error structure as #14's synth: a daily clear-sky-index regime error
    (15 % of days a big miss), smoothed intraday timing error, errors growing with lead over the target day."""
    T = act.shape[0]
    D = T // DAY
    lead = np.tile(np.linspace(0.8, 1.25, DAY), D)
    csg = geom[:, 2]
    csi = np.divide(act[:, 0], csg, out=np.zeros(T), where=csg > 20)

    def ar1(rho, n):
        from scipy.signal import lfilter

        return lfilter([1.0], [1.0, -rho], rng.standard_normal(n) * np.sqrt(1 - rho**2))

    miss = rng.random(D) < 0.15
    eps_d = rng.standard_normal(D) * np.where(miss, 0.22, 0.08)
    eps_t = np.repeat(eps_d, DAY) * lead + 0.04 * ar1(0.95, T) * lead
    smooth = np.convolve(csi, np.ones(9) / 9, mode="same")  # forecasts miss the fast cloud structure
    csi_fc = np.clip(smooth + eps_t, 0.04, 1.10)
    ghi = np.where(csg > 20, csi_fc * csg, 0.0)
    dni, dhi = ph.decompose(ghi, geom[:, 0], doy)
    cloud = np.clip(act[:, 4] - np.repeat(eps_d, DAY) * 120, 0, 100)
    temp = act[:, 3] + (np.repeat(rng.standard_normal(D) * 1.3, DAY) + 0.5 * ar1(0.97, T)) * lead
    wind = act[:, 5] * np.exp(np.repeat(rng.standard_normal(D) * 0.25, DAY))
    return np.stack([ghi, dni, dhi, temp, cloud, wind, act[:, 6]], 1)


def build_windows(fleet_dir: Path | None = None, fault_dir: Path | None = None, out: Path | None = None,
                  fleet_cfg: FleetConfig = FleetConfig(), fault_cfg: FaultConfig = FaultConfig(), seed: int = 5,
                  force: bool = False, batch: int = 64, log=print) -> Path:
    ph = physics()
    t_all = time.perf_counter()
    fleet_dir = Path(fleet_dir) if fleet_dir else generate(fleet_cfg, plots=False, log=log)
    fault_dir = Path(fault_dir) if fault_dir else inject_dataset(fleet_dir, fault_cfg, log=log)
    out = Path(out) if out else default_out()
    key = {"fleet": fleet_dir.name, "faults": fault_dir.name, "seed": seed, "format": FORMAT}
    if (out / "index.json").exists() and not force:
        idx = json.loads((out / "index.json").read_text())
        if idx.get("source_key") == key:
            log(f"cached: {out}")
            return out
    tmp = out.with_name(out.name + f".tmp{os.getpid()}")
    if tmp.exists():
        shutil.rmtree(tmp)
    tmp.mkdir(parents=True)

    sites = pd.read_parquet(fault_dir / "sites.parquet")
    labels = pd.read_parquet(fault_dir / "fault_labels.parquet")
    files = sorted((fault_dir / "telemetry").glob("year=*.parquet"))
    clean_files = sorted((fleet_dir / "telemetry").glob("year=*.parquet"))
    first = sites.site_id.iloc[0]
    ts_all = pd.DatetimeIndex(pd.concat([pd.read_parquet(f, columns=["ts"], filters=[("site_id", "==", first)]).ts
                                         for f in files]).sort_values())
    ts_all = ts_all.tz_localize("UTC") if ts_all.tz is None else ts_all.tz_convert("UTC")
    i0, i1 = _axis(ts_all)
    times = ts_all[i0:i1]
    T, S = len(times), len(sites)
    doy = times.dayofyear.to_numpy()
    log(f"[windows] axis {times[0]} .. {times[-1]} ({T // DAY} days), {S} sites")

    # --- localities: geometry, weather (actual + forecast), calendar --------------------------------------
    loc_ids = sorted(set(sites.locality_id))
    loc_index = {lid: i for i, lid in enumerate(loc_ids)}
    L = len(loc_ids)
    w = pd.read_parquet(fault_dir / "weather.parquet")
    w["ts"] = pd.to_datetime(w.ts, utc=True)
    geom = np.zeros((L, T, 3), np.float32)
    act = np.zeros((L, T, 7), np.float32)
    fc = np.zeros((L, T, 7), np.float32)
    rng = np.random.default_rng(seed)
    for li, lid in enumerate(loc_ids):
        loc = BY_ID[lid]
        g = ph.solar_geometry(times, loc.lat, loc.lon)
        geom[li] = g
        wl = w[w.point_id.astype(str) == lid].set_index("ts").reindex(times)
        ghi = wl.ghi_wm2.to_numpy(float)
        a = np.stack([ghi, wl.dni_wm2.to_numpy(float), wl.dhi_wm2.to_numpy(float), wl.temp_c.to_numpy(float),
                      _cloud_from_ghi(ghi, g[:, 2]), wl.wind10_ms.to_numpy(float), wl.snow_cm.to_numpy(float)], 1)
        act[li] = np.nan_to_num(a)
        fc[li] = forecast_weather(act[li].astype(float), g.astype(float), doy, rng, ph)
    np.save(tmp / "geom.npy", geom)
    np.save(tmp / "weather_act.npy", act.astype(np.float16))
    np.save(tmp / "weather_fc.npy", fc.astype(np.float16))
    np.save(tmp / "calendar.npy", ph.calendar_features(times).astype(np.float32))
    log(f"[windows] localities {time.perf_counter() - t_all:.0f}s")

    # --- site arrays, in batches ----------------------------------------------------------------------------
    def mm(name, dtype):
        return np.lib.format.open_memmap(tmp / f"{name}.npy", mode="w+", dtype=dtype, shape=(S, T))

    arr = {k: mm(k, np.float16) for k in SITE_F16}
    arr.update({k: mm(k, np.int8) for k in SITE_I8})
    site_order = list(sites.site_id)
    for b0 in range(0, S, batch):
        ids = site_order[b0:b0 + batch]

        def read(fs, cols):
            parts = []
            for f in fs:
                d = pq.read_table(f, columns=["ts", "site_id", *cols], filters=[("site_id", "in", ids)]).to_pandas()
                d["site_id"] = d.site_id.astype(str)
                parts.append(d)
            d = pd.concat(parts).sort_values(["site_id", "ts"], kind="stable").set_index("site_id").loc[ids]
            n = len(d) // len(ids)
            return {c: d[c].to_numpy().reshape(len(ids), n)[:, i0:i1] for c in cols}

        f = read(files, ["pv_ac_w", "load_w", "batt_w", "soc", "inv_voltage_v", "inv_status"])
        c = read(clean_files, ["pv_ac_w"])
        sl = slice(b0, b0 + len(ids))
        valid = np.isfinite(f["pv_ac_w"]) & np.isfinite(f["load_w"])
        arr["valid"][sl] = valid
        pv = np.nan_to_num(f["pv_ac_w"]) / 1000
        arr["pv_ac"][sl] = pv
        arr["pv_healthy"][sl] = c["pv_ac_w"] / 1000
        arr["load"][sl] = np.nan_to_num(f["load_w"]) / 1000
        arr["batt"][sl] = np.nan_to_num(f["batt_w"]) / 1000
        arr["soc"][sl] = np.nan_to_num(f["soc"])
        arr["inv_v"][sl] = np.nan_to_num(f["inv_voltage_v"], nan=230.0)
        sb = sites.iloc[b0:b0 + len(ids)]
        inv_kw = sb.inverter_kw.to_numpy()[:, None]
        st = np.where(pv > 0.005, 1, 0)
        st[(pv >= 0.995 * inv_kw) & (pv > 0.1)] = 2
        st[f["inv_status"] == INV_OVERVOLTAGE] = 3
        arr["inv_status"][sl] = st
        for r, s in enumerate(sb.itertuples()):
            li = loc_index[s.locality_id]
            meta = {"kwp_recorded": s.kwp_recorded, "tilt_recorded": s.tilt_recorded,
                    "azimuth_recorded": s.azimuth_recorded, "inverter_kw": s.inverter_kw}
            arr["pvlib_rec"][b0 + r] = ph.pvlib_expected_kw(meta, act[li].astype(np.float64), geom[li].astype(np.float64))
            arr["pvlib_rec_fc"][b0 + r] = ph.pvlib_expected_kw(meta, fc[li].astype(np.float64), geom[li].astype(np.float64))
            arr["pvclear_rec"][b0 + r] = ph.pvlib_clear_kw(meta, times, geom[li].astype(np.float64))
        log(f"[windows] sites {b0 + len(ids)}/{S} ({time.perf_counter() - t_all:.0f}s)")

    # --- labels ---------------------------------------------------------------------------------------------
    pos = {sid: i for i, sid in enumerate(site_order)}
    t_ns = times.asi8
    episodes = []
    fault = arr["fault"]
    for e in labels.itertuples():
        a = int(np.searchsorted(t_ns, pd.Timestamp(e.onset).value))
        z = T if pd.isna(e.end) else int(np.searchsorted(t_ns, pd.Timestamp(e.end).value))
        if z <= 0 or a >= T or a >= z:
            continue
        si = pos[e.asset_id]
        fault[si, a:z] = FAULT_IDX[e.fault]
        episodes.append(dict(site=si, asset_id=e.asset_id, fault=e.fault, onset=a, end=z,
                             severity=float(e.severity), params=json.loads(e.params) if e.params else {}))
    for a in arr.values():
        a.flush()
    del arr

    rows = []
    for s in sites.to_dict("records"):
        d = {k: (v.item() if hasattr(v, "item") else (None if pd.isna(v) else v)) for k, v in s.items()}
        d["loc_idx"] = loc_index[s["locality_id"]]
        d["meta_wrong"] = bool(s.get("true_metadata_wrong", False))
        rows.append(d)
    arrays = {}
    for fpath in sorted(tmp.glob("*.npy")):
        a = np.load(fpath, mmap_mode="r")
        arrays[fpath.stem] = {"file": fpath.name, "dtype": str(a.dtype), "shape": list(a.shape)}
    weather_source = json.loads((fleet_dir / "manifest.json").read_text()).get("weather_source", "?")
    index = {
        "format": FORMAT, "source": "sentinel.sim.fleet (#11) + sentinel.sim.faults (#12) via sentinel.data.windows (#10)",
        "label": f"synthetic RO ({weather_source} weather)", "source_key": key, "data_version": DATA_VERSION,
        "t_start": str(times[0]), "step_minutes": 15, "n_steps": T, "n_sites": S,
        "weather_channels": list(ph.WEATHER_CHANNELS), "geom_channels": list(ph.GEOM_CHANNELS),
        "calendar_channels": list(ph.CAL_CHANNELS), "faults": FAULTS,
        "localities": [dict(id=lid, name=BY_ID[lid].name, county=BY_ID[lid].county, lat=BY_ID[lid].lat,
                            lon=BY_ID[lid].lon) for lid in loc_ids],
        "arrays": arrays, "units": {"power": "kW", "inv_v": "V", "soc": "0..1"},
        "gen_seconds": round(time.perf_counter() - t_all, 1),
    }
    (tmp / "sites.json").write_text(json.dumps(rows, ensure_ascii=False))
    (tmp / "episodes.json").write_text(json.dumps(episodes))
    (tmp / "index.json").write_text(json.dumps(index, ensure_ascii=False, indent=1))
    if out.exists():
        shutil.rmtree(out)
    tmp.rename(out)
    log(f"LAB_METRIC windows_gen_seconds={index['gen_seconds']}")
    log(f"[windows] wrote {out}")
    return out


def main(argv=None) -> None:
    ap = argparse.ArgumentParser(prog="python -m sentinel.data.windows", description=__doc__.split("\n\n")[0])
    ap.add_argument("--quick", action="store_true", help="quick fleet (120 sites, 2024) into windows-quick/")
    ap.add_argument("--out", type=Path)
    ap.add_argument("--force", action="store_true")
    a = ap.parse_args(argv)
    if a.quick:
        cfg = FleetConfig(**QUICK)
        build_windows(fleet_cfg=cfg, out=a.out or default_out().with_name("windows-quick"), force=a.force)
    else:
        build_windows(out=a.out, force=a.force)


if __name__ == "__main__":
    main()
