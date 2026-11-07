"""Home battery model and the baseline self-consumption rule. Shared by the simulator (#11), the LP (#18) and the
P10 rollout (#27): import from here, don't re-implement.

Units: power in W, energy in kWh, dt in hours. Sign conventions (also used in the telemetry):
  batt_w > 0  battery charging (power into the battery, measured at the AC side)
  batt_w < 0  battery discharging
  grid_w > 0  export to the feeder;  grid_w < 0  import
  grid_w = pv_ac_w - load_w - batt_w

The energy dynamics match the LP in plan §3.5:
  e[t+1] = e[t] + eta_c * c[t] * dt - d[t] * dt / eta_d,   R <= e <= E,   c, d <= P
with the lower bound max(soc_min * E, comfort_reserve) — the reserve is never used for self-consumption.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd


@dataclass
class BatteryParams:
    """Vectorised over sites; every field is an array of shape (n,). Sites without a battery have capacity 0."""
    capacity_kwh: np.ndarray
    p_max_kw: np.ndarray
    eta_charge: np.ndarray
    eta_discharge: np.ndarray
    soc_min: np.ndarray            # fraction of capacity
    comfort_reserve_kwh: np.ndarray

    @classmethod
    def from_sites(cls, sites: pd.DataFrame) -> "BatteryParams":
        return cls(
            capacity_kwh=sites.battery_kwh.to_numpy(float),
            p_max_kw=sites.battery_kw.to_numpy(float),
            eta_charge=sites.battery_eta_charge.to_numpy(float),
            eta_discharge=sites.battery_eta_discharge.to_numpy(float),
            soc_min=sites.battery_soc_min.to_numpy(float),
            comfort_reserve_kwh=sites.comfort_reserve_kwh.to_numpy(float),
        )

    @property
    def e_floor_kwh(self) -> np.ndarray:
        return np.maximum(self.soc_min * self.capacity_kwh, self.comfort_reserve_kwh)


def apply_power(p: BatteryParams, e_kwh: np.ndarray, batt_w: np.ndarray, dt_h: float) -> tuple[np.ndarray, np.ndarray]:
    """Clip a requested battery power to what is feasible and return (actual batt_w, new energy kWh).
    Use this to replay any schedule (LP output, P10 scenarios) through the physical battery."""
    cap = p.capacity_kwh
    p_max_w = p.p_max_kw * 1000
    room_w = np.where(p.eta_charge > 0, (cap - e_kwh) / np.maximum(p.eta_charge, 1e-9) / dt_h * 1000, 0)
    avail_w = np.maximum(e_kwh - p.e_floor_kwh, 0) * p.eta_discharge / dt_h * 1000
    charge = np.clip(batt_w, 0, np.minimum(p_max_w, room_w))
    discharge = np.clip(-batt_w, 0, np.minimum(p_max_w, avail_w))
    e_new = e_kwh + (p.eta_charge * charge - discharge / np.maximum(p.eta_discharge, 1e-9)) * dt_h / 1000
    e_new = np.clip(e_new, 0, cap)
    return charge - discharge, e_new


def self_consumption_step(p: BatteryParams, e_kwh: np.ndarray, pv_w: np.ndarray, load_w: np.ndarray,
                          dt_h: float) -> tuple[np.ndarray, np.ndarray]:
    """Baseline rule: charge from PV surplus, discharge to cover the deficit (down to the floor). One step."""
    return apply_power(p, e_kwh, pv_w - load_w, dt_h)


def simulate_self_consumption(p: BatteryParams, pv_w: np.ndarray, load_w: np.ndarray, dt_h: float,
                              e0_kwh: np.ndarray | None = None) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Run the baseline rule over time. pv_w, load_w: (n, T). Returns (batt_w, soc, e_end_kwh);
    soc is the state of charge (0..1) at the END of each step, 0 for sites without a battery."""
    n, T = pv_w.shape
    e = (p.capacity_kwh * 0.5) if e0_kwh is None else e0_kwh.copy()
    batt = np.empty((n, T), np.float32)
    soc = np.empty((n, T), np.float32)
    inv_cap = np.where(p.capacity_kwh > 0, 1 / np.maximum(p.capacity_kwh, 1e-9), 0)
    surplus = pv_w - load_w
    for t in range(T):
        b, e = apply_power(p, e, surplus[:, t], dt_h)
        batt[:, t] = b
        soc[:, t] = e * inv_cap
    return batt, soc, e
