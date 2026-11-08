"""Vectorised home battery + self-consumption rule: the same semantics as `BatterySpec.step` and
`self_consumption_step` in wattsentry/contracts/opt.py (team/contracts/opt.md), over arrays of sites.
tests/test_fleet_sim.py checks the two agree. Use the contract version for one site, this one for fleets
(the simulator runs 600 sites x 140k steps; #27's P10 rollout runs ~200 scenarios per feeder).

Units here: power in W, energy in kWh, dt in hours (the contract uses kW). Signs, as in the telemetry:
  batt_w > 0 charging, < 0 discharging;  grid_w > 0 export, < 0 import;  grid_w = pv_ac_w - load_w - batt_w

  e[t+1] = e[t] + eta_c * c[t] * dt - d[t] * dt / eta_d,   c, d <= P,   e in [soc_min * E, soc_max * E]

The physical battery (and the self-consumption rule) only respects the soc_min floor. The comfort reserve binds
the planner: `e_min_planner_kwh` = max(soc_min * E, comfort_reserve) is the floor the LP (#18) must use.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd


@dataclass
class BatteryParams:
    """Every field is an array of shape (n,). Sites without a battery have capacity 0."""
    capacity_kwh: np.ndarray
    p_max_kw: np.ndarray
    eta_charge: np.ndarray
    eta_discharge: np.ndarray
    soc_min: np.ndarray
    soc_max: np.ndarray
    comfort_reserve_kwh: np.ndarray

    @classmethod
    def from_sites(cls, sites: pd.DataFrame) -> "BatteryParams":
        """From a `sites` table (data.md column names, the same fields as BatterySpec)."""
        return cls(
            capacity_kwh=sites.battery_kwh.to_numpy(float),
            p_max_kw=sites.battery_kw.to_numpy(float),
            eta_charge=sites.eta_charge.to_numpy(float),
            eta_discharge=sites.eta_discharge.to_numpy(float),
            soc_min=sites.soc_min.to_numpy(float),
            soc_max=sites.soc_max.to_numpy(float),
            comfort_reserve_kwh=sites.comfort_reserve_kwh.to_numpy(float),
        )

    @classmethod
    def from_specs(cls, specs) -> "BatteryParams":
        """From a list of contracts.opt.BatterySpec."""
        f = lambda k: np.array([getattr(s, k) for s in specs], float)  # noqa: E731
        return cls(*(f(k) for k in ("capacity_kwh", "p_max_kw", "eta_charge", "eta_discharge", "soc_min",
                                    "soc_max", "comfort_reserve_kwh")))

    @property
    def e_min_kwh(self) -> np.ndarray:
        return self.soc_min * self.capacity_kwh

    @property
    def e_max_kwh(self) -> np.ndarray:
        return self.soc_max * self.capacity_kwh

    @property
    def e_min_planner_kwh(self) -> np.ndarray:
        return np.maximum(self.e_min_kwh, self.comfort_reserve_kwh)


def apply_power(p: BatteryParams, e_kwh: np.ndarray, batt_w: np.ndarray, dt_h: float) -> tuple[np.ndarray, np.ndarray]:
    """Clip a requested battery power to power and energy limits (BatterySpec.step, vectorised).
    Returns (applied batt_w, next energy kWh). Use it to replay any schedule (LP output, P10 scenarios)."""
    has = p.capacity_kwh > 0
    lo, hi = p.e_min_kwh, p.e_max_kwh
    p_max_w = p.p_max_kw * 1000
    req = np.clip(batt_w, -p_max_w, p_max_w)
    room_w = np.maximum(0, (hi - e_kwh) / (p.eta_charge * dt_h)) * 1000
    avail_w = np.maximum(0, (e_kwh - lo) * p.eta_discharge / dt_h) * 1000
    applied = np.where(req >= 0, np.minimum(req, room_w), np.maximum(req, -avail_w))
    applied = np.where(has, applied, 0.0)
    e_next = np.where(applied >= 0, e_kwh + p.eta_charge * applied / 1000 * dt_h,
                      e_kwh + applied / 1000 * dt_h / p.eta_discharge)
    e_next = np.where(has, np.minimum(hi, np.maximum(lo, e_next)), e_kwh)
    return applied, e_next


def self_consumption_step(p: BatteryParams, e_kwh: np.ndarray, pv_w: np.ndarray, load_w: np.ndarray,
                          dt_h: float) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Baseline rule, one step: charge from PV surplus, discharge to cover load. Returns (batt_w, e_next, grid_w)."""
    surplus = pv_w - load_w
    b, e = apply_power(p, e_kwh, surplus, dt_h)
    return b, e, surplus - b


def simulate_self_consumption(p: BatteryParams, pv_w: np.ndarray, load_w: np.ndarray, dt_h: float,
                              e0_kwh: np.ndarray | None = None) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Run the baseline rule over time. pv_w, load_w: (n, T). Returns (batt_w, soc, e_end_kwh);
    soc is the state of charge (0..1) at the END of each step, NaN for sites without a battery."""
    n, T = pv_w.shape
    e = (p.capacity_kwh * 0.5) if e0_kwh is None else e0_kwh.copy()
    batt = np.empty((n, T), np.float32)
    soc = np.empty((n, T), np.float32)
    inv_cap = np.where(p.capacity_kwh > 0, 1 / np.maximum(p.capacity_kwh, 1e-9), np.nan)
    surplus = pv_w - load_w
    for t in range(T):
        b, e = apply_power(p, e, surplus[:, t], dt_h)
        batt[:, t] = b
        soc[:, t] = e * inv_cap
    return batt, soc, e
