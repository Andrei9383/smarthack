"""K7 self-consumption: the fleet simulator's home-battery rule over a year of 15-min steps (#135).

Source of truth: ``wattsentry.sim.fleet.battery.simulate_self_consumption`` (Andrei, #11); the Rust copy
``wattsentry._core.self_consumption`` reproduces it bit for bit (same IEEE operations in numpy's order, numpy's
min/max/clip semantics for NaN and signed zeros). Same signature as the reference, so the call site is a drop-in:

    batt_w, soc, e_end_kwh = self_consumption(p: BatteryParams, pv_w (n, T), load_w (n, T), dt_h, e0_kwh=None)

The loop is stateful over time (each step's energy depends on the last), so numpy runs it as T Python-level steps
over all sites; Rust runs each site's year as one tight loop, sites in parallel. The Rust path needs float32
pv/load (what the simulator produces); other dtypes use the reference.
"""

from __future__ import annotations

from typing import Any

import numpy as np


def self_consumption_py(p: Any, pv_w: np.ndarray, load_w: np.ndarray, dt_h: float,
                        e0_kwh: np.ndarray | None = None) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Reference: ``wattsentry.sim.fleet.battery.simulate_self_consumption``."""
    from wattsentry.sim.fleet.battery import simulate_self_consumption

    return simulate_self_consumption(p, pv_w, load_w, dt_h, e0_kwh)


def self_consumption_rust(p: Any, pv_w: np.ndarray, load_w: np.ndarray, dt_h: float,
                          e0_kwh: np.ndarray | None = None) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    from wattsentry import kernels

    pv_w, load_w = np.asarray(pv_w), np.asarray(load_w)
    if pv_w.dtype != np.float32 or load_w.dtype != np.float32 or pv_w.ndim != 2 or pv_w.shape != load_w.shape:
        return self_consumption_py(p, pv_w, load_w, dt_h, e0_kwh)
    n, T = pv_w.shape
    f64 = lambda a: np.ascontiguousarray(np.broadcast_to(np.asarray(a, dtype=np.float64), (n,)))  # noqa: E731
    cap = f64(p.capacity_kwh)
    # The same expressions as the reference, evaluated once in numpy (bounds, start energy, 1 / capacity).
    e0 = f64((cap * 0.5) if e0_kwh is None else e0_kwh)
    inv_cap = np.where(cap > 0, 1 / np.maximum(cap, 1e-9), np.nan)
    batt = np.empty((n, T), np.float32)
    soc = np.empty((n, T), np.float32)
    e_end = kernels._load_core().self_consumption(
        np.ascontiguousarray(pv_w), np.ascontiguousarray(load_w), cap, f64(p.p_max_kw), f64(p.eta_charge),
        f64(p.eta_discharge), f64(p.e_min_kwh), f64(p.e_max_kwh), f64(inv_cap), e0, float(dt_h), batt, soc)
    return batt, soc, e_end
