"""K1 battery rollout: replay requested battery power through the battery model with clipping.

Source of truth: ``wattsentry.opt.commit.rollout`` (#27); the Rust copy ``wattsentry._core.battery_rollout`` matches it
bit for bit. req (S, T) or (N, S, T) kW (+ charge, - discharge); soc0 (S,) or (N, S) fraction of capacity; battery
arrays (S,) as in ``BatteryParams``. Returns soc (fraction at the END of each step) and applied power, req's shape.
"""

from __future__ import annotations

from typing import Any

import numpy as np


def _prep(req, soc0, cap, pmax, eta_c, eta_d, e_lo, e_hi):
    req = np.asarray(req, dtype=np.float64)
    if req.ndim not in (2, 3):
        raise ValueError(f"req must be (S, T) or (N, S, T), got shape {req.shape}")
    r3 = req[None] if req.ndim == 2 else req
    n, s, _ = r3.shape
    params = [np.ascontiguousarray(np.broadcast_to(np.asarray(a, dtype=np.float64), (s,)))
              for a in (cap, pmax, eta_c, eta_d, e_lo, e_hi)]
    s0 = np.ascontiguousarray(np.broadcast_to(np.asarray(soc0, dtype=np.float64), (n, s)))
    return np.ascontiguousarray(r3), s0, params, req.ndim == 2


def battery_rollout_py(req: Any, soc0: Any, cap: Any, pmax: Any, eta_c: Any, eta_d: Any, e_lo: Any, e_hi: Any,
                       dt_h: float = 0.25) -> tuple[np.ndarray, np.ndarray]:
    """Reference: ``wattsentry.opt.commit.rollout`` with pv = load = 0."""
    from wattsentry.opt.commit import BatteryParams, rollout

    r3, s0, params, two_d = _prep(req, soc0, cap, pmax, eta_c, eta_d, e_lo, e_hi)
    soc, batt, _ = rollout(r3, s0, BatteryParams(*params), np.zeros(1), np.zeros(1), float(dt_h))
    return (soc[0], batt[0]) if two_d else (soc, batt)


def battery_rollout_rust(req: Any, soc0: Any, cap: Any, pmax: Any, eta_c: Any, eta_d: Any, e_lo: Any, e_hi: Any,
                         dt_h: float = 0.25) -> tuple[np.ndarray, np.ndarray]:
    from wattsentry import kernels

    r3, s0, params, two_d = _prep(req, soc0, cap, pmax, eta_c, eta_d, e_lo, e_hi)
    soc, batt = kernels._load_core().battery_rollout(r3, s0, *params, float(dt_h))
    return (soc[0], batt[0]) if two_d else (soc, batt)
