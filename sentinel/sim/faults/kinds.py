"""One function per PvFault class: apply a fault episode to one site's series, in place.

Each `apply_<kind>(ep)` gets an Episode with the site's arrays (`ep.cols`, 15-min, length T) and the step range
[i0, i1) the fault is active, and returns the params / severity to put in the label. Physical faults change
`pv_ac_w` (and battery capacity); the battery and grid are re-simulated afterwards by inject.py. Sensor faults
only corrupt what is reported. Signatures (what M4 has to learn, what the tests check):

  soiling           PV loss ramps up linearly from 0 at onset, gone after the cleaning rain at the end
  shading           PV loss only while the sun is inside an azimuth sector and below an elevation
  string_outage     PV steps down by 1/N strings for the whole episode (above the clipping level too)
  clipping          inverter derated: PV capped at a fraction of its AC rating
  overvoltage_trip  PV drops to 0 (inv_status = INV_OVERVOLTAGE) when the voltage passes a low trip setting
  battery_fade      usable capacity shrinks (fade) or the battery stops cycling most of its power (underuse)
  sensor_fault      stuck PV value, logger dropout (all columns null) or PV spikes; physics unchanged
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from ..fleet import solar

INV_OVERVOLTAGE = 37  # inverter code "grid voltage too high" (0 = ok, data.md)
STEPS_PER_DAY = 96


@dataclass
class Episode:
    ts: pd.DatetimeIndex
    cols: dict[str, np.ndarray]       # pv_ac_w, inv_voltage_v, inv_status, soc, batt_w, load_w, grid_w (if present)
    site: pd.Series                   # sites row (lat, lon, inverter_kw, battery_kwh, battery_kw, ...)
    i0: int
    i1: int
    rng: np.random.Generator
    capacity_factor: np.ndarray | None = None  # set by battery_fade: usable capacity per step (1 = nominal)
    power_factor: np.ndarray | None = None     # set by battery_fade (underuse): usable power per step
    extra: dict = field(default_factory=dict)

    @property
    def sl(self) -> slice:
        return slice(self.i0, self.i1)


def _days_since_onset(ep: Episode) -> np.ndarray:
    return np.arange(ep.i1 - ep.i0) / STEPS_PER_DAY


def _lost(before: np.ndarray, after: np.ndarray) -> float:
    e = np.nansum(before)
    return float(np.clip(1 - np.nansum(after) / e, 0, 1)) if e > 0 else 0.0


def apply_soiling(ep: Episode) -> tuple[dict, float]:
    max_loss = ep.rng.uniform(0.06, 0.30)
    days = (ep.i1 - ep.i0) / STEPS_PER_DAY
    rate = max_loss / max(days, 1)  # reaches max_loss just before the cleaning rain
    pv = ep.cols["pv_ac_w"]
    before = pv[ep.sl].copy()
    pv[ep.sl] = before * (1 - np.minimum(rate * _days_since_onset(ep), max_loss))
    return {"loss_per_day": round(rate, 5), "max_loss": round(max_loss, 3), "cleaned_by": "rain"}, \
        _lost(before, pv[ep.sl])


def apply_shading(ep: Episode) -> tuple[dict, float]:
    zen, az = solar.solar_position(ep.ts[ep.sl], float(ep.site.lat), float(ep.site.lon))
    elev = 90 - zen
    center = ep.rng.uniform(110, 250)
    width = ep.rng.uniform(25, 70)
    max_elev = ep.rng.uniform(20, 45)
    depth = ep.rng.uniform(0.3, 0.8)
    d_az = np.abs((az - center + 180) % 360 - 180)
    shaded = (d_az < width / 2) & (elev > 0) & (elev < max_elev)
    pv = ep.cols["pv_ac_w"]
    before = pv[ep.sl].copy()
    pv[ep.sl] = np.where(shaded, before * (1 - depth), before)
    return {"az_center_deg": round(center, 1), "az_width_deg": round(width, 1), "max_elev_deg": round(max_elev, 1),
            "depth": round(depth, 3), "cause": ep.rng.choice(["tree", "new_building", "chimney"])}, \
        _lost(before, pv[ep.sl])


def apply_string_outage(ep: Episode) -> tuple[dict, float]:
    n_strings = int(ep.rng.choice([2, 3, 4]))
    pv = ep.cols["pv_ac_w"]
    before = pv[ep.sl].copy()
    # Losing a string removes DC power before the inverter: a clipped plateau drops too, but less than 1/N.
    ac_w = float(ep.site.inverter_kw) * 1000
    dc_proxy = np.where(before >= 0.99 * ac_w, before * 1.15, before)
    pv[ep.sl] = np.minimum(dc_proxy * (1 - 1 / n_strings), ac_w)
    pv[ep.sl] = np.where(before > 0, pv[ep.sl], 0)
    return {"n_strings": n_strings, "strings_lost": 1}, _lost(before, pv[ep.sl])


def apply_clipping(ep: Episode) -> tuple[dict, float]:
    factor = ep.rng.uniform(0.35, 0.75)
    limit = float(ep.site.inverter_kw) * 1000 * factor
    pv = ep.cols["pv_ac_w"]
    before = pv[ep.sl].copy()
    pv[ep.sl] = np.minimum(before, limit)
    return {"derate_factor": round(factor, 3), "limit_w": round(limit, 1),
            "cause": ep.rng.choice(["thermal_derating", "misconfigured_export_limit", "firmware"])}, \
        _lost(before, pv[ep.sl])


def overvoltage_threshold(v: np.ndarray, pv: np.ndarray, rng) -> float | None:
    """A trip setting this site would actually hit: between its 97th and 99.7th daytime voltage percentile."""
    day = np.isfinite(v) & (pv > 0)
    if day.sum() < STEPS_PER_DAY:
        return None
    q = np.quantile(v[day], [0.97, 0.997])
    return float(rng.uniform(q[0], q[1]))


def apply_overvoltage_trip(ep: Episode) -> tuple[dict, float]:
    pv, v, st = ep.cols["pv_ac_w"], ep.cols["inv_voltage_v"], ep.cols["inv_status"]
    thr = ep.extra.get("threshold_v") or overvoltage_threshold(v[ep.sl], pv[ep.sl], ep.rng) or 248.0
    reconnect = int(ep.rng.integers(1, 4))  # steps the inverter stays off after a trip
    before = pv[ep.sl].copy()
    off_until = -1
    trips = 0
    for i in range(ep.i0, ep.i1):
        if i <= off_until:
            pv[i], st[i] = 0.0, INV_OVERVOLTAGE
        elif pv[i] > 0 and v[i] > thr:
            pv[i] *= ep.rng.uniform(0.0, 0.5)  # tripped part-way through the 15 min
            st[i] = INV_OVERVOLTAGE
            off_until = i + reconnect
            trips += 1
    return {"trip_setting_v": round(thr, 1), "reconnect_steps": reconnect, "trips": trips,
            "cause": "trip setting below 253 V (EN 50549)"}, _lost(before, pv[ep.sl])


def apply_battery_fade(ep: Episode) -> tuple[dict, float]:
    T = len(ep.ts)
    mode = ep.rng.choice(["fade", "underuse"], p=[0.65, 0.35])
    n = ep.i1 - ep.i0
    if mode == "fade":
        final = ep.rng.uniform(0.45, 0.8)
        ramp = np.minimum(np.arange(n) / max(int(n * ep.rng.uniform(0.2, 0.6)), 1), 1)
        cf = np.ones(T)
        cf[ep.sl] = 1 - (1 - final) * ramp
        ep.capacity_factor = cf if ep.capacity_factor is None else np.minimum(ep.capacity_factor, cf)
        return {"mode": "fade", "final_capacity": round(final, 3)}, round(1 - final, 3)
    final = ep.rng.uniform(0.05, 0.3)
    pf = np.ones(T)
    pf[ep.sl] = final
    ep.power_factor = pf if ep.power_factor is None else np.minimum(ep.power_factor, pf)
    return {"mode": "underuse", "power_factor": round(final, 3),
            "cause": ep.rng.choice(["bms_limit", "wrong_mode_backup_only", "firmware"])}, round(1 - final, 3)


def apply_sensor_fault(ep: Episode) -> tuple[dict, float]:
    """Corrupts the REPORTED series; call after the physics has been re-simulated."""
    mode = ep.rng.choice(["stuck", "dropout", "spikes"], p=[0.35, 0.4, 0.25])
    pv = ep.cols["pv_ac_w"]
    if mode == "stuck":
        pv[ep.sl] = pv[ep.i0]
        frac = 1.0
    elif mode == "dropout":
        for k in ep.cols:
            if k != "inv_status" and ep.cols[k].dtype.kind == "f":
                ep.cols[k][ep.sl] = np.nan
        frac = 1.0
    else:
        day = np.flatnonzero(pv[ep.sl] > 0) + ep.i0
        k = max(1, int(len(day) * ep.rng.uniform(0.01, 0.05)))
        idx = ep.rng.choice(day, min(k, len(day)), replace=False) if len(day) else np.array([], int)
        pv[idx] = pv[idx] * ep.rng.uniform(3, 10, len(idx))
        frac = len(idx) / max(ep.i1 - ep.i0, 1)
    return {"mode": mode}, round(frac, 4)


APPLY = {
    "soiling": apply_soiling,
    "shading": apply_shading,
    "string_outage": apply_string_outage,
    "clipping": apply_clipping,
    "overvoltage_trip": apply_overvoltage_trip,
    "battery_fade": apply_battery_fade,
    "sensor_fault": apply_sensor_fault,
}
