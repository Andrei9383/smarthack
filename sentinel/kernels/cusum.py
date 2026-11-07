"""K3 two-sided CUSUM: Python reference (source of truth) + wrapper for the Rust copy (team/infra.md §2).

    from sentinel.kernels import cusum_two_sided
    s_pos, s_neg, alarm = cusum_two_sided(x, k=0.5, h=5.0)            # uncapped sums
    s_pos, s_neg, alarm = cusum_two_sided(x, k=0.1, h=0.1, cap=0.1)   # sums saturate at cap (decision-interval clamp)

The Rust copy is ``sentinel._core.cusum_two_sided`` (rust/sentinel_core/src/cusum.rs) and must match this file bit
for bit. The PV fault baseline (``sentinel.eval.baselines.pv.cusum``) is a thin wrapper over the dispatched kernel
with ``cap=h``.
"""

from __future__ import annotations

from typing import Any

import numpy as np


def _cusum_prep(x: Any, k: float, h: float, cap: float | None) -> tuple[np.ndarray, bool, float, float, float | None]:
    x = np.asarray(x, dtype=np.float64)
    if x.ndim not in (1, 2):
        raise ValueError(f"x must be 1-D (n,) or 2-D (series, n), got shape {x.shape}")
    k, h = float(k), float(h)
    if not (np.isfinite(k) and np.isfinite(h)):
        raise ValueError("k and h must be finite")
    if cap is not None:
        cap = float(cap)
        if not (np.isfinite(cap) and cap >= 0.0):
            raise ValueError("cap must be finite and >= 0 (or None)")
    one_d = x.ndim == 1
    x2 = np.ascontiguousarray(x.reshape(1, -1) if one_d else x)
    return x2, one_d, k, h, cap


def cusum_two_sided_py(x: Any, k: float, h: float,
                       cap: float | None = None) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Two-sided CUSUM over the last axis of ``x`` (shape (n,) or (series, n)).

    For each step, with sums starting at 0::

        x[t] not finite (NaN/inf = gap): s_pos = s_neg = 0, alarm = False
        otherwise: s_pos = max(s_pos + x[t] - k, 0)      then, if cap is not None, s_pos = min(s_pos, cap)
                   s_neg = max(s_neg - x[t] - k, 0)      (same cap)
                   alarm = s_pos > h or s_neg > h

    ``x`` should already be standardised (target mean removed, scaled); ``k`` is the allowance (slack) and
    ``h`` the decision threshold, in the same units. Sums are not reset after an alarm. With ``cap=h`` the sums
    saturate at h, so ``alarm`` (strict >) never fires; callers that cap use ``s >= h`` as their alarm.
    Returns ``(s_pos, s_neg, alarm)`` with the shape of ``x`` (float64, float64, bool).
    """
    x2, one_d, k, h, cap = _cusum_prep(x, k, h, cap)
    m, n = x2.shape
    xt = np.ascontiguousarray(x2.T)  # (n, m): one contiguous row per time step, vectorised over series
    sp = np.empty((n, m))
    sn = np.empty((n, m))
    cp = np.zeros(m)
    cn = np.zeros(m)
    finite = np.isfinite(xt)
    with np.errstate(invalid="ignore"):
        for t in range(n):
            v = xt[t]
            ok = finite[t]
            up = np.maximum(cp + v - k, 0.0)
            dn = np.maximum(cn - v - k, 0.0)
            if cap is not None:
                up = np.minimum(up, cap)
                dn = np.minimum(dn, cap)
            cp = np.where(ok, up, 0.0)
            cn = np.where(ok, dn, 0.0)
            sp[t] = cp
            sn[t] = cn
    sp = np.ascontiguousarray(sp.T)
    sn = np.ascontiguousarray(sn.T)
    alarm = (sp > h) | (sn > h)
    if one_d:
        return sp[0], sn[0], alarm[0]
    return sp, sn, alarm


def cusum_two_sided_rust(x: Any, k: float, h: float,
                         cap: float | None = None) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    from sentinel import kernels

    x2, one_d, k, h, cap = _cusum_prep(x, k, h, cap)
    sp, sn, alarm = kernels._load_core().cusum_two_sided(x2, k, h, cap)
    if one_d:
        return sp[0], sn[0], alarm[0]
    return sp, sn, alarm
