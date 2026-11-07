"""K7 battery re-simulation for the PV fault injector (#12, #134): re-run the self-consumption rule for faulty rows
from their fault onset, with per-step capacity / power derating (battery_fade).

Source of truth: ``battery_resim_py`` below, the time loop moved verbatim out of
``sentinel.sim.faults.inject._resimulate_battery`` (#134). The Rust copy ``sentinel._core.battery_resim`` matches it
bit for bit (same float operations in numpy's order, numpy NaN semantics for maximum/minimum/clip).

Inputs, all float64 C-contiguous, m rows x T steps unless noted:
  surplus    pv_ac_w - load_w (W)
  soc0, batt0  the original (pre-fault) soc and batt_w series
  cf, pf     capacity / power factor per step (1 = nominal)
  start      (m,) int64: each row's first fault onset; before it a row keeps its original series
  e0         (m,) energy (kWh) at the start of step t0
  cap, pmax, eta_c, eta_d, soc_min, soc_max  (m,) nominal battery parameters (BatteryParams fields)
  t0         first step to simulate (min of start); columns before t0 are copied from batt0 / soc0
Returns (batt, soc), each (m, T) float64.
"""

from __future__ import annotations

from typing import Any

import numpy as np


def _prep(surplus, soc0, batt0, cf, pf, start, e0, cap, pmax, eta_c, eta_d, soc_min, soc_max):
    f2 = [np.ascontiguousarray(a, dtype=np.float64) for a in (surplus, soc0, batt0, cf, pf)]
    m, T = f2[0].shape
    if any(a.shape != (m, T) for a in f2):
        raise ValueError("surplus, soc0, batt0, cf, pf must all have shape (m, T)")
    f1 = [np.ascontiguousarray(np.broadcast_to(np.asarray(a, dtype=np.float64), (m,)))
          for a in (e0, cap, pmax, eta_c, eta_d, soc_min, soc_max)]
    st = np.ascontiguousarray(np.broadcast_to(np.asarray(start, dtype=np.int64), (m,)))
    return f2, st, f1


def battery_resim_py(surplus: Any, soc0: Any, batt0: Any, cf: Any, pf: Any, start: Any, e0: Any, cap: Any,
                     pmax: Any, eta_c: Any, eta_d: Any, soc_min: Any, soc_max: Any, t0: int,
                     dt_h: float = 0.25) -> tuple[np.ndarray, np.ndarray]:
    from sentinel.sim.fleet.battery import BatteryParams, apply_power

    (surplus, soc0, batt0, cf, pf), start, (e, cap, pmax, eta_c, eta_d, soc_min, soc_max) = _prep(
        surplus, soc0, batt0, cf, pf, start, e0, cap, pmax, eta_c, eta_d, soc_min, soc_max)
    T = surplus.shape[1]
    batt = batt0.copy()
    soc = soc0.copy()
    sub = BatteryParams(cap.copy(), pmax.copy(), eta_c, eta_d, soc_min, soc_max, np.zeros_like(cap))
    nominal_cap, nominal_p = cap, pmax
    if T == 0 or len(cap) == 0:
        return batt, soc
    prev_cap = nominal_cap * cf[:, min(t0, T - 1)]
    for t in range(t0, T):
        before = t < start
        sub.capacity_kwh = nominal_cap * cf[:, t]
        sub.p_max_kw = nominal_p * pf[:, t]
        # Fading cells lose stored energy in proportion (SoC is preserved), never gain usable energy.
        e = e * np.divide(sub.capacity_kwh, prev_cap, out=np.ones_like(e), where=prev_cap > 0)
        prev_cap = sub.capacity_kwh
        b, e_new = apply_power(sub, e, surplus[:, t], dt_h)
        # Before a row's own onset, keep the original series exactly (and follow its energy).
        e_orig = np.nan_to_num(soc0[:, t]) * nominal_cap
        e = np.where(before, e_orig, e_new)
        batt[:, t] = np.where(before, batt0[:, t], b)
        soc[:, t] = np.where(before, soc0[:, t], e / np.maximum(sub.capacity_kwh, 1e-9))
    return batt, soc


def battery_resim_rust(surplus: Any, soc0: Any, batt0: Any, cf: Any, pf: Any, start: Any, e0: Any, cap: Any,
                       pmax: Any, eta_c: Any, eta_d: Any, soc_min: Any, soc_max: Any, t0: int,
                       dt_h: float = 0.25) -> tuple[np.ndarray, np.ndarray]:
    from sentinel import kernels

    f2, st, f1 = _prep(surplus, soc0, batt0, cf, pf, start, e0, cap, pmax, eta_c, eta_d, soc_min, soc_max)
    return kernels._load_core().battery_resim(*f2, st, *f1, int(t0), float(dt_h))
