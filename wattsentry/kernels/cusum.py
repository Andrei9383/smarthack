"""K3 two-sided CUSUM: Python reference (source of truth); the Rust copy ``wattsentry._core.cusum_two_sided``
(rust/wattsentry_core/src/cusum.rs) must match it bit for bit. The PV fault baseline uses it with ``cap=h``."""

from __future__ import annotations

from typing import Any

import numpy as np


def _prep(x: Any, k: float, h: float, cap: float | None) -> tuple[np.ndarray, bool, float, float, float | None]:
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
    return np.ascontiguousarray(x.reshape(1, -1) if x.ndim == 1 else x), x.ndim == 1, k, h, cap


def cusum_two_sided_py(x: Any, k: float, h: float,
                       cap: float | None = None) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Two-sided CUSUM over the last axis of ``x`` ((n,) or (series, n)), sums starting at 0::

        x[t] not finite (gap): s_pos = s_neg = 0, alarm = False
        else: s_pos = max(s_pos + x[t] - k, 0), s_neg = max(s_neg - x[t] - k, 0)  (each then min(., cap) if cap)
              alarm = s_pos > h or s_neg > h

    ``x`` standardised; ``k`` allowance, ``h`` threshold, same units. Sums are not reset after an alarm; with
    ``cap=h`` the strict alarm never fires, so capping callers use ``s >= h``. Returns float64, float64, bool.
    """
    x2, one_d, k, h, cap = _prep(x, k, h, cap)
    m, n = x2.shape
    xt = np.ascontiguousarray(x2.T)  # (n, m): one contiguous row per time step, vectorised over series
    finite = np.isfinite(xt)
    sp, sn = np.empty((n, m)), np.empty((n, m))
    cp, cn = np.zeros(m), np.zeros(m)
    with np.errstate(invalid="ignore"):
        for t in range(n):
            up, dn = np.maximum(cp + xt[t] - k, 0.0), np.maximum(cn - xt[t] - k, 0.0)
            if cap is not None:
                up, dn = np.minimum(up, cap), np.minimum(dn, cap)
            sp[t] = cp = np.where(finite[t], up, 0.0)
            sn[t] = cn = np.where(finite[t], dn, 0.0)
    sp, sn = np.ascontiguousarray(sp.T), np.ascontiguousarray(sn.T)
    alarm = (sp > h) | (sn > h)
    return (sp[0], sn[0], alarm[0]) if one_d else (sp, sn, alarm)


def cusum_two_sided_rust(x: Any, k: float, h: float,
                         cap: float | None = None) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    from wattsentry import kernels

    x2, one_d, k, h, cap = _prep(x, k, h, cap)
    sp, sn, alarm = kernels._load_core().cusum_two_sided(x2, k, h, cap)
    return (sp[0], sn[0], alarm[0]) if one_d else (sp, sn, alarm)
