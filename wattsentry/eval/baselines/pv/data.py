"""Load the synthetic RO fleet (#11) + fault injector output (#12) into wide day-aligned arrays for the baselines.

Everything is [n_sites, n_days, 96] with UTC days (00:00-24:00 UTC holds all Romanian daylight). Split by the UTC
day's year: train < val_year, val == val_year, test == test_year (plan §4). Site split (`train` / `unseen`) comes
from sites.parquet. Physical-model outputs (pvlib with recorded metadata) are computed here once per site.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd

from .physical import RECORDED, clearsky_power_kw, pv_power_kw, recorded_meta, solar_geometry

D = 96  # steps per day
PV_VISIBLE = ("soiling", "shading", "string_outage", "clipping", "overvoltage_trip", "sensor_fault")


@dataclass
class FleetArrays:
    sites: pd.DataFrame              # one row per site, recorded metadata + split (no truth used downstream)
    days: pd.DatetimeIndex           # UTC midnights, length n_days
    pv_kw: np.ndarray                # [n, d, 96] measured AC power (NaN = missing)
    load_kw: np.ndarray              # [n, d, 96]
    weather: dict[str, np.ndarray]   # ghi, dni, dhi, temp_c, wind10_ms, snow_cm: [n, d, 96]
    geom: dict[str, np.ndarray]      # zenith, azimuth, cs_ghi, doy: [n, d, 96]
    pvlib_kw: np.ndarray             # [n, d, 96] pvlib on actual weather, recorded metadata
    pvlib_cs_kw: np.ndarray          # [n, d, 96] pvlib under clear sky, recorded metadata
    fault: np.ndarray                # [n, d, 96] str-coded fault id (0 = healthy), see fault_names
    fault_names: list[str] = field(default_factory=list)
    labels: pd.DataFrame | None = None
    source: str = "synthetic RO"
    eval_truth: pd.DataFrame | None = None  # hidden site truth, ONLY for error analysis (never a model input)

    @property
    def n(self) -> int:
        return len(self.sites)

    def day_mask(self, years) -> np.ndarray:
        years = [years] if isinstance(years, int) else list(years)
        return np.isin(self.days.year, years)

    def ts(self) -> pd.DatetimeIndex:
        return day_grid(self.days)


def day_grid(days: pd.DatetimeIndex) -> pd.DatetimeIndex:
    """All 15-min step starts of the given UTC days, day-major."""
    off = np.arange(D) * np.int64(15 * 60 * 10**9)
    ns = (days.asi8[:, None] + off[None, :]).ravel()
    return pd.DatetimeIndex(pd.to_datetime(ns, utc=True))


def ensure_dataset(n_sites: int = 100, n_feeders: int = 6, start_year: int = 2023, end_year: int = 2024,
                   seed: int = 7, log=print) -> Path:
    """Generate (or reuse the cached) fleet and its fault-injected copy; returns the faults dir."""
    from wattsentry.sim.faults import inject_dataset
    from wattsentry.sim.fleet import FleetConfig, generate

    cfg = FleetConfig(seed=seed, n_sites=n_sites, n_feeders=n_feeders, n_congested=max(1, n_feeders // 2),
                      start_year=start_year, end_year=end_year)
    fleet = generate(cfg, plots=False)
    return inject_dataset(fleet, log=log)


def load_arrays(data_dir: Path, years: list[int] | None = None, max_sites: int | None = None,
                site_ids: list[str] | None = None) -> FleetArrays:
    data_dir = Path(data_dir)
    sites = pd.read_parquet(data_dir / "sites.parquet")
    sites["site_id"] = sites.site_id.astype(str)
    if site_ids is not None:
        sites = sites[sites.site_id.isin([str(s) for s in site_ids])]
    if max_sites:
        sites = sites.iloc[:max_sites]
    truth_cols = [c for c in ("true_metadata_wrong", "true_shaded") if c in sites.columns]
    eval_truth = sites[truth_cols].reset_index(drop=True) if truth_cols else None
    keep = [c for c in sites.columns if not (c.endswith("_true") or c.startswith("true_"))
            and c not in ("feeder_id", "phase")]
    sites = sites[keep].reset_index(drop=True)  # truth never reaches the baselines
    ids = sites.site_id.tolist()

    files = sorted((data_dir / "telemetry").glob("year=*.parquet"))
    if years:
        # local-year files overlap UTC years by 2-3 h: read one extra file on each side when present
        files = [p for p in files if int(p.stem.split("=")[1]) in range(min(years) - 1, max(years) + 2)]
    cols = ["ts", "site_id", "pv_ac_w", "load_w"]
    tel = pd.concat([pd.read_parquet(p, columns=cols, filters=[("site_id", "in", ids)]) for p in files],
                    ignore_index=True)
    tel["site_id"] = tel.site_id.astype(str)
    tel = tel.drop_duplicates(["site_id", "ts"]).sort_values(["site_id", "ts"], kind="stable")
    ts_all = pd.DatetimeIndex(np.sort(tel.ts.unique()))
    first = ts_all[ts_all.normalize() == ts_all][0]
    last_mid = ts_all[-1].normalize()
    if ts_all[-1] < last_mid + pd.Timedelta(minutes=15 * (D - 1)):
        last_mid -= pd.Timedelta(days=1)
    days = pd.date_range(first, last_mid, freq="D")
    if years:
        days = days[np.isin(days.year, years)]
    grid = day_grid(days)

    def wide(df: pd.DataFrame, value: str, key: str, keys: list[str]) -> np.ndarray:
        out = np.full((len(keys), len(grid)), np.nan)
        ki = pd.Index(keys).get_indexer(df[key].astype(str))
        ti = grid.get_indexer(pd.DatetimeIndex(df.ts))
        ok = (ki >= 0) & (ti >= 0)
        out[ki[ok], ti[ok]] = df[value].to_numpy(np.float64)[ok]
        return out.reshape(len(keys), len(days), D)

    pv = wide(tel, "pv_ac_w", "site_id", ids) / 1000.0
    load = wide(tel, "load_w", "site_id", ids) / 1000.0

    wx = pd.read_parquet(data_dir / "weather.parquet")
    wx["point_id"] = wx.point_id.astype(str)
    locs = sorted(sites.locality_id.astype(str).unique())
    loc_ix = {l: i for i, l in enumerate(locs)}
    site_loc = np.array([loc_ix[str(l)] for l in sites.locality_id])
    weather = {c.replace("_wm2", ""): wide(wx, c, "point_id", locs)[site_loc]
               for c in ("ghi_wm2", "dni_wm2", "dhi_wm2", "temp_c", "wind10_ms", "snow_cm")}

    n = len(ids)
    geom = {k: np.zeros((n, len(days), D)) for k in ("zenith", "azimuth", "cs_ghi", "doy")}
    pvlib_kw = np.zeros((n, len(days), D))
    pvlib_cs = np.zeros((n, len(days), D))
    for i, row in sites.iterrows():
        meta = recorded_meta(row)
        g = solar_geometry(grid, meta["lat"], meta["lon"])
        for k in geom:
            geom[k][i] = g[k].reshape(len(days), D)
        out = pv_power_kw(meta, g, weather["ghi"][i].ravel(), weather["dni"][i].ravel(), weather["dhi"][i].ravel(),
                          weather["temp_c"][i].ravel(), weather["wind10_ms"][i].ravel())
        pvlib_kw[i] = out["ac_kw"].reshape(len(days), D)
        pvlib_cs[i] = clearsky_power_kw(meta, g).reshape(len(days), D)

    labels = None
    fault = np.zeros((n, len(days), D), np.int8)
    names = ["healthy"]
    lp = data_dir / "fault_labels.parquet"
    if lp.exists():
        labels = pd.read_parquet(lp)
        labels["asset_id"] = labels.asset_id.astype(str)
        labels = labels[labels.asset_id.isin(ids)].reset_index(drop=True)
        names += sorted(labels.fault.unique())
        code = {f: i for i, f in enumerate(names)}
        site_ix = {s: i for i, s in enumerate(ids)}
        flat = fault.reshape(n, -1)
        g0, g_end = grid[0], grid[-1] + pd.Timedelta(minutes=15)
        for r in labels.itertuples():
            on = max(r.onset, g0)
            end = g_end if pd.isna(r.end) else min(r.end, g_end)
            if end <= on:
                continue
            i0 = int((on - g0) / pd.Timedelta(minutes=15))
            i1 = int(np.ceil((end - g0) / pd.Timedelta(minutes=15)))
            flat[site_ix[r.asset_id], i0:i1] = code[r.fault]
    assert all(c in sites.columns for c in RECORDED)
    return FleetArrays(sites=sites, days=days, pv_kw=pv, load_kw=load, weather=weather, geom=geom,
                       pvlib_kw=pvlib_kw, pvlib_cs_kw=pvlib_cs, fault=fault, fault_names=names, labels=labels,
                       eval_truth=eval_truth)
