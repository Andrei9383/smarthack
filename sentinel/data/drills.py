"""Robustness drills (card #28): corrupted variants of the test inputs. The agent must keep working or say "not
sure" (human review), never crash and never commit flexibility it cannot deliver.

Two forms of the same drills:
  DRILLS[name](asset, window, rng) -> (asset, window)   per prediction window; eval.py --drills scores them
  telemetry_drill(df, name, rng) -> df                  on long telemetry (ts, site_id, pv_ac_w, ...), for replay

Drills
  gaps_10 / gaps_40     10 % / 40 % of past readings missing (isolated + short runs)
  missing_day           the last full day before t0 is missing
  frozen_sensor         past PV stuck at one daytime value for the last 3 days
  units_kw / units_x1000  past power sent in kW (or x1000) while the metadata still says W
  unseen_site_type      east-west flat commercial roof, kWp far outside the training range
  telemetry_dropout     nothing received in the 6 h before delivery; meta['telemetry_age_steps'] says so
  forecast_noise        day-ahead weather error: irradiance x a daily factor, timing shift, temperature bias
Drills never modify their inputs (copies), and are deterministic for a given rng.
"""

from __future__ import annotations

import copy
from typing import Callable

import numpy as np
import pandas as pd

DAY = 96
POWER_KEYS = ("reference", "history")  # past pv_ac_w and load_w in the Window


def _copy(asset: dict, window):
    return copy.deepcopy(dict(asset)), copy.deepcopy(window)


def _arr(x) -> np.ndarray | None:
    return None if x is None else np.array(x, dtype=np.float64, copy=True)


def _gap_mask(n: int, share: float, rng) -> np.ndarray:
    """~share of n steps missing: half isolated readings, half runs of 1-8 h."""
    m = np.zeros(n, bool)
    m[rng.random(n) < share / 2] = True
    while m.mean() < share:
        a = int(rng.integers(0, n))
        m[a:a + int(rng.integers(4, 33))] = True
    return m


def _past(window, fn) -> None:
    for k in POWER_KEYS:
        x = _arr(getattr(window, k))
        if x is not None and len(x):
            setattr(window, k, fn(x, k))


def _blank(x: np.ndarray, idx) -> np.ndarray:
    x[idx] = np.nan
    return x


def _tag(window, drill: str, **extra) -> None:
    window.meta = {**(window.meta or {}), "drill": drill, **extra}


def gaps(share: float) -> Callable:
    def drill(asset, window, rng):
        asset, window = _copy(asset, window)
        n = len(window.reference) if window.reference is not None else 0
        mask = _gap_mask(n, share, rng)
        _past(window, lambda x, _: _blank(x, mask[: len(x)]))
        _tag(window, f"gaps_{int(share * 100)}")
        return asset, window
    return drill


def missing_day(asset, window, rng):
    asset, window = _copy(asset, window)
    _past(window, lambda x, _: _blank(x, slice(-DAY, None)))
    _tag(window, "missing_day")
    return asset, window


def frozen_sensor(asset, window, rng):
    asset, window = _copy(asset, window)
    x = _arr(window.reference)
    if x is not None and len(x) > 3 * DAY:
        start = len(x) - 3 * DAY
        day = np.flatnonzero(np.nan_to_num(x[start - DAY:start]) > 0)
        val = float(x[start - DAY + day[len(day) // 2]]) if len(day) else 0.0
        x[start:] = val
        window.reference = x
    _tag(window, "frozen_sensor")
    return asset, window


def units(factor: float, name: str) -> Callable:
    def drill(asset, window, rng):
        asset, window = _copy(asset, window)
        _past(window, lambda x, _: x * factor)
        _tag(window, name)  # power_unit still claims "W"
        return asset, window
    return drill


def unseen_site_type(asset, window, rng):
    asset, window = _copy(asset, window)
    scale = float(rng.uniform(2.5, 4.0))
    for k in ("kwp_recorded", "kwp", "kwp_true"):
        if k in asset and asset[k] is not None:
            asset[k] = float(asset[k]) * scale
    if "inverter_kw" in asset:
        asset["inverter_kw"] = float(asset["inverter_kw"]) * scale
    for k in ("tilt_recorded", "tilt_deg"):
        if k in asset:
            asset[k] = 8.0
    for k in ("azimuth_recorded", "azimuth_deg"):
        if k in asset:
            asset[k] = float(rng.choice([90.0, 270.0]))  # east-west
    asset["archetype"] = "commercial_flat_roof"
    _tag(window, "unseen_site_type")
    return asset, window


def telemetry_dropout(asset, window, rng, hours: float = 6.0):
    asset, window = _copy(asset, window)
    k = int(hours * 4)
    _past(window, lambda x, _: _blank(x, slice(-k, None)))
    _tag(window, "telemetry_dropout", telemetry_age_steps=k)
    return asset, window


def forecast_noise(asset, window, rng):
    asset, window = _copy(asset, window)
    w = _arr(window.weather)
    if w is not None and w.ndim == 2:
        chans = list((window.meta or {}).get("weather_channels", []))
        irr = [i for i, c in enumerate(chans) if any(s in str(c).lower() for s in ("ghi", "dni", "dhi", "irr"))]
        temp = [i for i, c in enumerate(chans) if "temp" in str(c).lower()]
        if not chans:  # unknown layout: assume data.md order ghi, dni, dhi, temp first
            irr, temp = [0, 1, 2][: w.shape[1]], [3] if w.shape[1] > 3 else []
        factor = float(np.exp(rng.normal(0, 0.3)))
        shift = int(rng.integers(-3, 4))
        for i in irr:
            w[:, i] = np.clip(np.roll(w[:, i], shift) * factor, 0, None)
        for i in temp:
            w[:, i] = w[:, i] + rng.normal(0, 2.0)
        window.weather = w
        _tag(window, "forecast_noise", drill_ghi_factor=round(factor, 3), drill_shift_steps=shift)
    return asset, window


DRILLS: dict[str, Callable] = {
    "gaps_10": gaps(0.10),
    "gaps_40": gaps(0.40),
    "missing_day": missing_day,
    "frozen_sensor": frozen_sensor,
    "units_kw": units(1e-3, "units_kw"),
    "units_x1000": units(1e3, "units_x1000"),
    "unseen_site_type": unseen_site_type,
    "telemetry_dropout": telemetry_dropout,
    "forecast_noise": forecast_noise,
}


# --- telemetry form (long tables), for replay and the data checks ---------------------------------------------
POWER_COLS = ("pv_ac_w", "load_w", "batt_w", "grid_w")


def telemetry_drill(df: pd.DataFrame, name: str, rng, sites: list[str] | None = None,
                    delivery_hour_utc: int = 8) -> pd.DataFrame:
    """Apply a drill to long telemetry (copy). `sites` limits it to some sites (default: all).
    telemetry_dropout removes the readings in the 6 h before each day's delivery hour (UTC)."""
    out = df.copy()
    pick = out.site_id.astype(str).isin(sites) if sites is not None else pd.Series(True, index=out.index)
    cols = [c for c in POWER_COLS if c in out]
    if name in ("gaps_10", "gaps_40"):
        m = pick.to_numpy() & (rng.random(len(out)) < (0.1 if name == "gaps_10" else 0.4))
        out.loc[m, cols + [c for c in ("soc", "inv_voltage_v") if c in out]] = np.nan
    elif name == "missing_day":
        day = out.ts.dt.floor("D")
        target = day.iloc[int(rng.integers(0, len(day)))]
        out.loc[pick & (day == target), cols] = np.nan
    elif name == "frozen_sensor":
        for sid, g in out[pick].groupby("site_id", observed=True):
            g = g.sort_values("ts")
            d = g[g.pv_ac_w > 0]
            if d.empty:
                continue
            i0 = d.index[len(d) // 2]
            sel = g.index[g.index.get_loc(i0): g.index.get_loc(i0) + 3 * DAY]
            out.loc[sel, "pv_ac_w"] = out.at[i0, "pv_ac_w"]
    elif name in ("units_kw", "units_x1000"):
        out.loc[pick, cols] = out.loc[pick, cols] * (1e-3 if name == "units_kw" else 1e3)
    elif name == "telemetry_dropout":
        h = out.ts.dt.hour
        before = (h >= (delivery_hour_utc - 6) % 24) & (h < delivery_hour_utc) if delivery_hour_utc >= 6 else \
            (h >= (delivery_hour_utc - 6) % 24) | (h < delivery_hour_utc)
        drop = pick & before
        out = out[~drop]  # not delivered at all: the rows are absent, not null
    elif name in ("unseen_site_type", "forecast_noise"):
        pass  # metadata / weather drills: nothing to change in telemetry
    else:
        raise KeyError(f"unknown drill {name!r}; one of {list(DRILLS)}")
    return out
