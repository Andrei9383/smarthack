"""Metering layer: what PPC actually sees.

- `smart` homes: the 15-min import/export curve, available the next day (``available_at`` = local D+1 00:00).
- `profile` homes: only monthly import/export totals.
- Settlement: smart homes on their measured curve; profile homes on the standard profile (their monthly totals
  spread over the month's quarter-hours by the profile), as the DSO settles unmetered intervals. So gains on
  profile homes never show up in lei, and reporting stays honest.

The standard profile here is a built-in stand-in (``proxy``): a household load shape (occupancy x darkness x
season, the fleet's household model without weather) and a clear-sky solar shape for export, both at a central
RO point. To use the distributor's published profile pass ``standard_profile="file:<parquet>"`` with columns ts,
load_coef, solar_coef.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from wattsentry.sim.fleet import solar
from wattsentry.sim.fleet.load import _OCC_OFFDAY, _OCC_WEEKDAY, national_shape
from wattsentry.sim.fleet.ro_calendar import local_features

TZ = "Europe/Bucharest"
CENTER = (45.9, 24.9)  # geographic centre of Romania (lat, lon)


def standard_profile(ts_utc: pd.DatetimeIndex, spec: str = "builtin") -> pd.DataFrame:
    """ts -> DataFrame(ts, load_coef, solar_coef, profile_source); coefficients are relative (normalised per month
    at settlement)."""
    if spec.startswith("file:"):
        df = pd.read_parquet(spec[5:], columns=["ts", "load_coef", "solar_coef"])
        df["ts"] = pd.to_datetime(df.ts, utc=True).dt.as_unit("ns")
        out = df.set_index("ts").reindex(pd.DatetimeIndex(ts_utc).as_unit("ns"))
        if out.isna().any().any():
            raise ValueError(f"{spec}: standard profile does not cover every quarter-hour")
        return out.reset_index().assign(profile_source=spec[5:])
    if spec != "builtin":
        raise ValueError(f"standard_profile must be 'builtin' or 'file:<path>', got {spec!r}")
    cal = local_features(ts_utc)
    zen, _ = solar.solar_position(ts_utc, *CENTER)
    occ = np.where(cal["off_day"], np.repeat(_OCC_OFFDAY, 4)[cal["qh"]], np.repeat(_OCC_WEEKDAY, 4)[cal["qh"]])
    dark = (zen > 85).astype(float)
    load = (100 + 320 * occ + 150 * occ * dark) * national_shape(cal["doy"])
    sol = np.clip(np.cos(np.radians(zen)), 0, None)
    return pd.DataFrame({"ts": ts_utc, "load_coef": load.astype(np.float64), "solar_coef": sol.astype(np.float64),
                         "profile_source": "builtin_proxy"})


def month_index(ts_utc: pd.DatetimeIndex) -> tuple[np.ndarray, list[str]]:
    """Per step: index of its local month in the returned labels ('YYYY-MM')."""
    loc = pd.DatetimeIndex(ts_utc).tz_convert(TZ)
    lab = (loc.year * 100 + loc.month).to_numpy()
    uniq, idx = np.unique(lab, return_inverse=True)
    return idx, [f"{u // 100}-{u % 100:02d}" for u in uniq]


def available_at(ts_utc: pd.DatetimeIndex) -> pd.DatetimeIndex:
    """Smart-meter D+1 rule: a reading for local day D arrives at local D+1 00:00 (returned in UTC)."""
    loc = pd.DatetimeIndex(ts_utc).tz_convert(TZ).tz_localize(None).normalize() + pd.Timedelta(days=1)
    return loc.tz_localize(TZ).tz_convert("UTC")


def month_available_at(labels: list[str]) -> pd.DatetimeIndex:
    """Monthly totals arrive at the start of the next local month."""
    start = pd.DatetimeIndex([pd.Timestamp(f"{m}-01") for m in labels]) + pd.offsets.MonthBegin(1)
    return start.tz_localize(TZ).tz_convert("UTC")


def monthly_sums(x: np.ndarray, midx: np.ndarray, n_months: int) -> np.ndarray:
    """x (n, T) per-step energy -> (n, n_months) totals."""
    out = np.zeros((x.shape[0], n_months))
    for k in range(n_months):
        out[:, k] = x[:, midx == k].sum(1)
    return out


def settle_on_profile(monthly_kwh: np.ndarray, coef: np.ndarray, midx: np.ndarray) -> np.ndarray:
    """Spread monthly totals over the month's steps by a profile. monthly_kwh (..., n_months), coef (T,),
    midx (T,) -> (..., T). Each month's settled curve sums back to its total."""
    coef = np.asarray(coef, dtype=np.float64)
    norm = np.bincount(midx, weights=coef, minlength=monthly_kwh.shape[-1])
    w = np.divide(coef, norm[midx], out=np.zeros_like(coef), where=norm[midx] > 0)
    return monthly_kwh[..., midx] * w
