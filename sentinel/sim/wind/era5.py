"""Real ERA5 (Open-Meteo, card #9) at each park: hourly table, 15-min interpolation, shear fit.

ERA5 values are hourly instantaneous fields on a 0.25° grid. The simulator interpolates them to 15 min (linear for
scalars, on the unit circle for direction) and adds the sub-hourly variability ERA5 cannot resolve as AR(1)
noise in generate.py. Shear (plan §3.5): alpha per park and meteorological season, fitted on the calendar year of
the first simulated step (2024 = the training year) from ERA5's own 10 m / 100 m wind (median of ln(v100/v10)/ln 10
over hours with v100 > 4 m/s, clipped to 0.08-0.30); "fixed" uses 0.14.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from sentinel.model.wind.physics import ALPHA_DEFAULT

from .physics import interp_circular

FIELDS = ["temp_c", "rh_pct", "precip_mm", "pressure_hpa", "wind10_ms", "wind100_ms", "wdir100_deg"]
SEASONS = {"DJF": (12, 1, 2), "MAM": (3, 4, 5), "JJA": (6, 7, 8), "SON": (9, 10, 11)}


@dataclass
class ParkWeather:
    ts_h: np.ndarray  # datetime64[m] hourly
    hourly: dict[str, np.ndarray]  # FIELDS -> [H, P]
    q15: dict[str, np.ndarray]  # FIELDS -> [T, P] at 15 min
    alpha: np.ndarray  # [T, P] shear exponent per step (season of the step)
    alpha_table: dict[str, dict[str, float]]  # park -> season -> alpha
    table: "object"  # polars frame: contract `weather` rows, point_id = park id (hourly)
    fit_ts_h: np.ndarray  # hourly grid of the fit year (calendar year of the first step)
    fit_hourly: dict[str, np.ndarray]  # FIELDS -> [H_fit, P]
    fit_alpha: np.ndarray  # [H_fit, P]


def weather_points(path):
    """Points in the ERA5 file that carry 100 m wind (localities + parks; the 2025-only grid file is not used)."""
    import polars as pl

    w = pl.scan_parquet(path)
    return (w.filter(pl.col("wind100_ms").is_not_null()).group_by("point_id")
            .agg(pl.col("lat").first(), pl.col("lon").first(), pl.col("ts").min().alias("t0"),
                 pl.col("ts").max().alias("t1"))
            .sort("point_id").collect())


def _read_hourly(path, layouts, h0, h1):
    """Hourly FIELDS [H, P] on the grid h0..h1 (exclusive) at each park's ERA5 point (+ the raw rows)."""
    import polars as pl

    ts_h = np.arange(h0, h1, np.timedelta64(1, "h")).astype("datetime64[m]")
    lo = ts_h[0].astype("datetime64[us]").item()
    hi = ts_h[-1].astype("datetime64[us]").item()
    pts = [lay.era5_point for lay in layouts]
    raw = (pl.scan_parquet(path).filter(pl.col("point_id").is_in(pts))
           .filter(pl.col("ts").dt.replace_time_zone(None).is_between(lo, hi)).collect())
    H, P = len(ts_h), len(layouts)
    hourly = {k: np.full((H, P), np.nan) for k in FIELDS}
    rows = []
    for p, lay in enumerate(layouts):
        r = raw.filter(pl.col("point_id") == lay.era5_point).sort("ts")
        if r.height == 0:
            raise FileNotFoundError(f"no ERA5 rows for {lay.era5_point} in {path}")
        rts = r["ts"].dt.replace_time_zone(None).to_numpy().astype("datetime64[m]")
        idx = ((rts - ts_h[0]) // np.timedelta64(60, "m")).astype(np.int64)
        ok = (idx >= 0) & (idx < H)
        for k in FIELDS:
            hourly[k][idx[ok], p] = r[k].cast(pl.Float64).to_numpy()[ok]
        for k in FIELDS:  # fill the odd missing hour (and the edges) by interpolation / nearest
            col = hourly[k][:, p]
            bad = ~np.isfinite(col)
            if bad.any():
                good = np.flatnonzero(~bad)
                col[bad] = np.interp(np.flatnonzero(bad), good, col[good])
        rows.append(r)
    return ts_h, hourly, rows


def fit_year_window(ts15: np.ndarray):
    """The calendar year of the first step (hourly grid): shear fit and CF calibration always use the WHOLE year,
    so a short test window gets exactly the same parameters as the full run."""
    y = ts15[0].astype("datetime64[Y]")
    return y.astype("datetime64[h]"), (y + np.timedelta64(1, "Y")).astype("datetime64[h]")


def load(path, layouts, ts15: np.ndarray, shear: str = "fit") -> ParkWeather:
    import polars as pl

    t0, t1 = ts15[0], ts15[-1]
    h0 = np.datetime64(t0, "h") - np.timedelta64(1, "h")
    h1 = np.datetime64(t1, "h") + np.timedelta64(2, "h")
    ts_h, hourly, rows = _read_hourly(path, layouts, h0, h1)
    P = len(layouts)
    tables = []
    for p, lay in enumerate(layouts):
        # contract `weather` rows for this park (hourly ERA5 at the park's point; point_id = park id)
        keep = rows[p].filter(pl.col("ts").dt.replace_time_zone(None).is_between(
            t0.astype("datetime64[us]").item(), (t1 + np.timedelta64(1, "h")).astype("datetime64[us]").item()))
        tables.append(keep.with_columns(
            pl.lit(lay.park.park_id).alias("point_id"),
            pl.lit(float(lay.lat.mean())).alias("lat"),
            pl.lit(float(lay.lon.mean())).alias("lon"),
            pl.lit(lay.era5_point).alias("era5_point"),
        ))
    tq = ts15.astype(np.int64)
    th = ts_h.astype(np.int64)
    q15 = {}
    for k in FIELDS:
        if k == "wdir100_deg":
            q15[k] = np.stack([interp_circular(tq, th, hourly[k][:, p]) for p in range(P)], 1)
        else:  # precip stays mm per hour (a rate), like the hourly table
            q15[k] = np.stack([np.interp(tq, th, hourly[k][:, p]) for p in range(P)], 1)

    # shear exponent per park and season, fitted on the calendar year of the first step (the training year)
    fy_ts, fy, _ = _read_hourly(path, layouts, *fit_year_window(ts15))
    months_h = fy_ts.astype("datetime64[M]").astype(int) % 12 + 1
    months_q = ts15.astype("datetime64[M]").astype(int) % 12 + 1
    alpha = np.full((len(ts15), P), ALPHA_DEFAULT)
    alpha_h = np.full((len(fy_ts), P), ALPHA_DEFAULT)
    table: dict[str, dict[str, float]] = {}
    for p, lay in enumerate(layouts):
        table[lay.park.park_id] = {}
        for sname, months in SEASONS.items():
            a = ALPHA_DEFAULT
            if shear == "fit":
                m = np.isin(months_h, months) & (fy["wind100_ms"][:, p] > 4) & (fy["wind10_ms"][:, p] > 1)
                if m.sum() >= 50:
                    ratio = fy["wind100_ms"][m, p] / fy["wind10_ms"][m, p]
                    a = float(np.clip(np.median(np.log(ratio) / np.log(10.0)), 0.08, 0.30))
            table[lay.park.park_id][sname] = round(a, 4)
            alpha[np.isin(months_q, months), p] = round(a, 4)
            alpha_h[np.isin(months_h, months), p] = round(a, 4)
    out_table = pl.concat(tables, how="diagonal_relaxed")
    return ParkWeather(ts_h, hourly, q15, alpha, table, out_table, fy_ts, fy, alpha_h)
