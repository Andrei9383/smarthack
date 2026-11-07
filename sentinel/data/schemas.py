"""Data checks for telemetry (data.md): Pandera schemas built from sentinel.contracts.schemas, plus quality checks.

    from sentinel.data.schemas import validate_telemetry, quality_report
    validate_telemetry(df)              # raises pandera SchemaErrors: missing columns, types, physical ranges
    issues = quality_report(df, sites)  # one row per (site, issue): gaps, frozen, night_power, soc_over_100, units

The schema is about shape and hard physics (power >= 0, SoC in [0, 1], voltage plausible). quality_report finds what
a fault injector or a broken logger produces and what models must not train on blindly; drills (#28) use it too.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from sentinel.contracts import schemas as contract

try:  # pandera >= 0.20 exposes the pandas API under pandera.pandas
    import pandera.pandas as pa
except ImportError:  # pragma: no cover
    import pandera as pa

_ARROW_TO_PANDAS = {"float32": "float", "float64": "float", "int16": "int", "int32": "int", "string": "str",
                    "bool": "bool", "timestamp[UTC]": "datetime64[ns, UTC]"}

# Hard physical ranges (W, V, 0..1). Wide on purpose: they catch unit errors, not faults.
RANGES = {
    "pv_ac_w": (0, 30_000),
    "load_w": (0, 50_000),
    "batt_w": (-20_000, 20_000),
    "soc": (0, 1.0 + 1e-6),
    "inv_voltage_v": (150, 300),
}


def _column(name: str, arrow_type: str) -> pa.Column:
    nullable = arrow_type.endswith("?")
    t = arrow_type.rstrip("?")
    checks = []
    if name in RANGES:
        lo, hi = RANGES[name]
        checks.append(pa.Check.in_range(lo, hi, error=f"{name} outside [{lo}, {hi}]"))
    dtype = _ARROW_TO_PANDAS.get(t)
    if dtype in ("float", "int"):
        return pa.Column(None, checks=checks, nullable=nullable, coerce=False, required=True,
                         description=f"{arrow_type} (numeric)")
    if dtype == "datetime64[ns, UTC]":
        return pa.Column(checks=[pa.Check(lambda s: s.dt.tz is not None, element_wise=False,
                                          error="timestamps must be tz-aware UTC")], nullable=nullable)
    return pa.Column(None, nullable=nullable)


def schema_for(table: str) -> pa.DataFrameSchema:
    cols = {name: _column(name, t) for name, t in contract.TABLES[table].items()}
    return pa.DataFrameSchema(cols, strict=False, name=table)  # extra columns are allowed (data.md)


TELEMETRY = schema_for("telemetry")


def validate_telemetry(df: pd.DataFrame, lazy: bool = True) -> pd.DataFrame:
    numeric = [c for c in ("pv_ac_w", "load_w", "batt_w", "soc", "inv_voltage_v", "inv_status") if c in df]
    for c in numeric:
        if not pd.api.types.is_numeric_dtype(df[c]):
            raise TypeError(f"telemetry column {c} is {df[c].dtype}, expected numeric")
    return TELEMETRY.validate(df, lazy=lazy)


# --- quality checks ----------------------------------------------------------------------------------------
def _sun_up(ts: pd.DatetimeIndex, lat: float, lon: float) -> np.ndarray:
    from sentinel.sim.fleet.solar import solar_position

    zen, _ = solar_position(ts, lat, lon)
    return zen < 93  # twilight: diffuse light still gives some PV until the sun is ~3 deg below the horizon


def _runs(mask: np.ndarray) -> list[tuple[int, int]]:
    """[start, end) of each run of True."""
    if not mask.any():
        return []
    d = np.diff(np.r_[0, mask.astype(np.int8), 0])
    return list(zip(np.flatnonzero(d == 1), np.flatnonzero(d == -1)))


def quality_report(df: pd.DataFrame, sites: pd.DataFrame | None = None, step: str = "15min",
                   frozen_steps: int = 8, night_w: float = 50.0) -> pd.DataFrame:
    """One row per (site_id, issue, start, end, n_steps, detail).

    gaps          missing timestamps or null pv_ac_w
    frozen        pv_ac_w > 0 and exactly constant for >= frozen_steps consecutive daylight steps
    night_power   pv_ac_w > night_w while the sun is down (needs sites lat/lon; skipped otherwise)
    soc_over_100  soc > 1
    units         pv peak far above the inverter rating (W vs kW x1000) or far below it (kW stored as W)
    """
    sites_i = sites.set_index("site_id") if sites is not None else None
    rows = []
    for sid, g in df.groupby("site_id", sort=False, observed=True):
        g = g.sort_values("ts")
        ts = pd.DatetimeIndex(g.ts)
        full = pd.date_range(ts[0], ts[-1], freq=step)
        missing = len(full) - len(ts)
        pv = g.pv_ac_w.to_numpy(float)
        if missing > 0:
            rows.append((sid, "gaps", ts[0], ts[-1], missing, "missing timestamps"))
        for a, b in _runs(np.isnan(pv)):
            rows.append((sid, "gaps", ts[a], ts[b - 1], b - a, "null pv_ac_w"))
        same = np.r_[False, (pv[1:] == pv[:-1]) & (pv[1:] > 0)]
        for a, b in _runs(same):
            if b - a + 1 >= frozen_steps:
                rows.append((sid, "frozen", ts[a - 1], ts[b - 1], b - a + 1, f"pv stuck at {pv[a]:.0f} W"))
        if "soc" in g:
            for a, b in _runs(g.soc.to_numpy(float) > 1 + 1e-6):
                rows.append((sid, "soc_over_100", ts[a], ts[b - 1], b - a, "soc > 1"))
        if sites_i is not None and sid in sites_i.index:
            s = sites_i.loc[sid]
            if {"lat", "lon"} <= set(s.index):
                night = ~_sun_up(ts, float(s.lat), float(s.lon))
                for a, b in _runs(night & (np.nan_to_num(pv) > night_w)):
                    rows.append((sid, "night_power", ts[a], ts[b - 1], b - a, f"max {np.nanmax(pv[a:b]):.0f} W"))
            if "inverter_kw" in s.index and np.isfinite(pv).any():
                peak, rating = np.nanquantile(pv, 0.999), float(s.inverter_kw) * 1000
                if peak > 3 * rating:
                    rows.append((sid, "units", ts[0], ts[-1], len(ts), f"p99.9 {peak:.0f} W vs inverter {rating:.0f} W (x1000?)"))
                elif 0 < peak < rating / 100:
                    rows.append((sid, "units", ts[0], ts[-1], len(ts), f"p99.9 {peak:.2f} vs inverter {rating:.0f} W (kW as W?)"))
    return pd.DataFrame(rows, columns=["site_id", "issue", "start", "end", "n_steps", "detail"])
