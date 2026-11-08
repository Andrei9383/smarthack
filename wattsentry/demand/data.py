"""National RO load for M3 v2: ENTSO-E RO total load ("real RO") or a labelled synthetic fallback.

Looks under `data/raw/` for `*entsoe*load*.parquet` (`ts` + `load_mw`, or aliases) or any `*entsoe*.parquet` in long
format `ts, series, mw` (series == "load_actual"). Sub-hourly data is averaged to hourly MW. Temperature comes from
`*weather*.parquet` (`ts`, `temp_c`) when present, else a synthetic RO climatology.
"""

from __future__ import annotations

import glob
import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

import numpy as np

from .calendar import calendar_features

LABEL_REAL = "real RO"
LABEL_SYNTHETIC = "synthetic RO"
REPO_ROOT = Path(__file__).resolve().parents[2]
_TS_COLS = ("ts", "timestamp", "utc", "datetime", "time", "ts_utc", "timestamp_utc")
_LOAD_COLS = ("load_mw", "mw", "load", "value", "actual_load_mw")


@dataclass
class LoadSeries:
    """Hourly series on a regular UTC grid; NaN = missing."""

    ts: np.ndarray  # datetime64[s] UTC, hourly, regular
    load_mw: np.ndarray
    temp_c: np.ndarray
    label: str  # "real RO" | "synthetic RO"
    temp_label: str
    source: str
    zone_id: str = "RO"
    meta: dict = field(default_factory=dict)


def _ts_numpy(df, col: str) -> np.ndarray:
    import polars as pl

    s = df.get_column(col)
    if s.dtype == pl.Utf8 or s.dtype == pl.String:
        s = s.str.to_datetime(time_zone="UTC")
    if getattr(s.dtype, "time_zone", None):
        s = s.dt.convert_time_zone("UTC").dt.replace_time_zone(None)
    return s.cast(pl.Datetime("us")).to_numpy().astype("datetime64[s]")


def _hourly(ts: np.ndarray, vals: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Mean per UTC hour onto a regular grid (missing hours NaN)."""
    ok = np.isfinite(vals)
    ts, vals = ts[ok], vals[ok]
    hrs = ts.astype("datetime64[h]")
    grid = np.arange(hrs.min(), hrs.max() + np.timedelta64(1, "h"), np.timedelta64(1, "h"))
    idx = (hrs - grid[0]).astype(int)
    s = np.bincount(idx, weights=vals, minlength=len(grid))
    n = np.bincount(idx, minlength=len(grid))
    with np.errstate(invalid="ignore", divide="ignore"):
        out = np.where(n > 0, s / np.maximum(n, 1), np.nan)
    return grid.astype("datetime64[s]"), out


def _read_entsoe(path: str) -> Optional[tuple[np.ndarray, np.ndarray]]:
    import polars as pl

    df = pl.read_parquet(path)
    cols = {c.lower(): c for c in df.columns}
    tcol = next((cols[c] for c in _TS_COLS if c in cols), None)
    if tcol is None:
        return None
    if "series" in cols:  # long format
        df = df.filter(pl.col(cols["series"]) == "load_actual")
        vcol = cols.get("mw")
    else:
        vcol = next((cols[c] for c in _LOAD_COLS if c in cols), None)
    if vcol is None or df.height == 0:
        return None
    return _ts_numpy(df, tcol), df.get_column(vcol).cast(pl.Float64).to_numpy()


def find_entsoe_load(data_dir: Optional[str | Path] = None) -> Optional[str]:
    d = Path(data_dir) if data_dir else REPO_ROOT / "data" / "raw"
    cands = sorted(glob.glob(str(d / "*entsoe*load*.parquet"))) + sorted(glob.glob(str(d / "*entsoe*.parquet")))
    for p in dict.fromkeys(cands):
        try:
            if _read_entsoe(p) is not None:
                return p
        except Exception:  # unreadable / other schema: try the next file
            continue
    return None


def _read_temperature(data_dir: Path, grid: np.ndarray) -> Optional[np.ndarray]:
    import polars as pl

    for p in sorted(glob.glob(str(data_dir / "*weather*.parquet"))):
        try:
            df = pl.read_parquet(p, columns=None)
            cols = {c.lower(): c for c in df.columns}
            tcol = next((cols[c] for c in _TS_COLS if c in cols), None)
            if tcol is None or "temp_c" not in cols:
                continue
            if "source" in cols:  # train on reanalysis actuals, not forecasts
                df = df.filter(pl.col(cols["source"]).is_in(["era5", "era5_land"])) if df.height else df
            ts, v = _hourly(_ts_numpy(df, tcol), df.get_column(cols["temp_c"]).cast(pl.Float64).to_numpy())
            out = np.full(len(grid), np.nan)
            pos = (ts - grid[0]).astype("timedelta64[h]").astype(int)
            ok = (pos >= 0) & (pos < len(grid))
            out[pos[ok]] = v[ok]
            if np.isfinite(out).mean() > 0.5:
                return out
        except Exception:
            continue
    return None


def synthetic_temperature(ts: np.ndarray, seed: int = 1) -> np.ndarray:
    """RO-like hourly 2 m temperature (°C): annual cycle (−2 Jan .. 23 Jul mean), diurnal ±5, AR(1) weather."""
    ts = np.asarray(ts, dtype="datetime64[s]")
    rng = np.random.default_rng(seed)
    days = (ts.astype("datetime64[D]") - ts[0].astype("datetime64[D]")).astype(int)
    doy = (ts.astype("datetime64[D]") - ts.astype("datetime64[Y]")).astype(int)
    hour_utc = (ts - ts.astype("datetime64[D]")).astype("timedelta64[h]").astype(int)
    annual = 10.5 - 12.5 * np.cos(2 * np.pi * (doy - 15) / 365.25)
    diurnal = 5.0 * np.cos(2 * np.pi * (hour_utc - 12) / 24)  # max ~15 local
    nd = days.max() + 1
    w = np.zeros(nd)
    for d in range(1, nd):
        w[d] = 0.8 * w[d - 1] + 2.4 * rng.standard_normal()
    return annual + diurnal + w[days] + 0.6 * rng.standard_normal(len(ts))


def synthetic_ro_load(start: str = "2022-01-01", end: str = "2026-01-01", *, seed: int = 0,
                      level_mw: float = 6400.0, temp_c: Optional[np.ndarray] = None) -> LoadSeries:
    """Synthetic RO-shaped national load (MW, hourly): ENTSO-E-like daily shape (morning + evening peaks, evening
    later in summer), weekends and RO holidays lower, heating/cooling response to temperature, AR(1) noise.
    Always labelled "synthetic RO" — it only exercises the code path.
    """
    ts = np.arange(np.datetime64(start, "h"), np.datetime64(end, "h"), np.timedelta64(1, "h")).astype("datetime64[s]")
    temp = synthetic_temperature(ts, seed + 1) if temp_c is None else np.asarray(temp_c, dtype=float)
    cal = calendar_features(ts)
    h = cal["hour"].astype(float)
    month = cal["month"]
    summer = ((month >= 5) & (month <= 9)).astype(float)
    eve = 19.0 + 2.0 * summer  # evening peak ~19 h winter, ~21 h summer
    shape = (
        0.80
        + 0.16 * np.exp(-((h - 8.5) ** 2) / 6.0)
        + 0.22 * np.exp(-((h - eve) ** 2) / 4.0)
        + 0.10 * ((h >= 7) & (h <= 17))  # working hours plateau
        - 0.12 * np.exp(-((h - 3.5) ** 2) / 5.0)
    )
    daytype = cal["daytype"]
    day_f = np.where(daytype == 0, 1.0, np.where(daytype == 1, 0.92, 0.85))
    day_f = np.where(cal["bridge"] == 1, 0.93, day_f)
    temp_24 = np.convolve(np.nan_to_num(temp, nan=10.0), np.ones(24) / 24, mode="same")
    hdd = np.maximum(0, 15.0 - temp_24)
    cdd = np.maximum(0, temp_24 - 22.0)
    rng = np.random.default_rng(seed)
    nd = len(ts) // 24 + 2
    days = (ts.astype("datetime64[D]") - ts[0].astype("datetime64[D]")).astype(int)
    lvl = np.zeros(nd)
    for d in range(1, nd):
        lvl[d] = 0.9 * lvl[d - 1] + 0.012 * rng.standard_normal()
    hourly = np.zeros(len(ts))
    for i in range(1, len(ts)):
        hourly[i] = 0.7 * hourly[i - 1] + 0.008 * rng.standard_normal()
    years = (ts.astype("datetime64[Y]").astype(int) + 1970 - 2022).astype(float)
    load = level_mw * (1 - 0.01 * years) * shape * day_f * (1 + 0.012 * hdd + 0.02 * cdd) * (1 + lvl[days] + hourly)
    return LoadSeries(ts=ts, load_mw=load, temp_c=temp, label=LABEL_SYNTHETIC, temp_label=LABEL_SYNTHETIC,
                      source="synthetic_ro_load(seed=%d)" % seed)


def load_national(data_dir: Optional[str | Path] = None, *, allow_synthetic: bool = True, seed: int = 0) -> LoadSeries:
    """ENTSO-E RO total load if the parquet is present (label "real RO"), else synthetic RO."""
    d = Path(data_dir) if data_dir else REPO_ROOT / "data" / "raw"
    path = find_entsoe_load(d)
    if path is None:
        if not allow_synthetic:
            raise FileNotFoundError(f"no ENTSO-E load parquet under {d} (expected *entsoe*load*.parquet)")
        return synthetic_ro_load(seed=seed)
    ts, mw = _read_entsoe(path)
    grid, load = _hourly(ts, mw)
    temp = _read_temperature(d, grid)
    temp_label = LABEL_REAL
    if temp is None:
        temp, temp_label = synthetic_temperature(grid), LABEL_SYNTHETIC
    else:  # fill short gaps by climatology
        clim = synthetic_temperature(grid)
        temp = np.where(np.isfinite(temp), temp, clim)
    return LoadSeries(ts=grid, load_mw=load, temp_c=temp, label=LABEL_REAL, temp_label=temp_label,
                      source=os.path.relpath(path, REPO_ROOT) if path.startswith(str(REPO_ROOT)) else path)


def synthetic_county_split(national: LoadSeries, shares: dict[str, float], *, seed: int = 7) -> list[LoadSeries]:
    """County series from the national one: share × national × county temperature sensitivity × own noise.

    Always labelled "synthetic RO" (county load is not published).
    """
    rng = np.random.default_rng(seed)
    out = []
    for county, share in shares.items():
        t_off = rng.normal(0, 1.5)
        temp = national.temp_c + t_off
        sens = 1 + 0.006 * (np.maximum(0, 15 - temp) - np.maximum(0, 15 - national.temp_c))
        n = len(national.ts)
        noise = np.zeros(n)
        e = rng.standard_normal(n)
        for k in range(1, n):
            noise[k] = 0.85 * noise[k - 1] + 0.02 * e[k]
        out.append(LoadSeries(ts=national.ts, load_mw=national.load_mw * share * sens * (1 + noise), temp_c=temp,
                              label=LABEL_SYNTHETIC, temp_label=national.temp_label,
                              source=f"county split of {national.source}", zone_id=county))
    return out


__all__ = ["LoadSeries", "load_national", "synthetic_ro_load", "synthetic_temperature", "synthetic_county_split",
           "find_entsoe_load", "LABEL_REAL", "LABEL_SYNTHETIC"]
