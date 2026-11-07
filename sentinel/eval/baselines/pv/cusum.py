"""Two-sided CUSUM, the numpy reference for kernel K3 (team/infra.md §2; a Rust twin mirrors this signature).

    s_pos[t] = min(h, max(0, s_pos[t-1] + x[t] - k))     upward shift   (e.g. PV under-production residual)
    s_neg[t] = min(h, max(0, s_neg[t-1] - x[t] - k))     downward shift
    alarm[t] = +1 if s_pos[t] >= h, -1 if s_neg[t] >= h (and s_pos < h), else 0

- A NaN in x resets both statistics to 0 at that step (a data gap restarts the evidence) and gives alarm 0.
- Statistics saturate at h ("decision-interval clamp"): an alarm stays on while the shift lasts and clears about
  h / k steps after it ends, instead of a long tail after a long fault.
- x is 1-D [T] or 2-D [n_series, T]; time is the last axis. Outputs are new float64 / int8 arrays, same shape.
"""

from __future__ import annotations

import numpy as np


def cusum_two_sided(x: np.ndarray, k: float, h: float) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    if h <= 0 or k < 0:
        raise ValueError("need h > 0 and k >= 0")
    x = np.asarray(x, np.float64)
    one_d = x.ndim == 1
    x2 = np.atleast_2d(x)
    if x2.ndim != 2:
        raise ValueError("x must be 1-D or 2-D [n_series, T]")
    n, T = x2.shape
    s_pos = np.zeros((n, T))
    s_neg = np.zeros((n, T))
    sp = np.zeros(n)
    sn = np.zeros(n)
    for t in range(T):
        xt = x2[:, t]
        gap = np.isnan(xt)
        xt = np.where(gap, 0.0, xt)
        sp = np.where(gap, 0.0, np.minimum(h, np.maximum(0.0, sp + xt - k)))
        sn = np.where(gap, 0.0, np.minimum(h, np.maximum(0.0, sn - xt - k)))
        s_pos[:, t] = sp
        s_neg[:, t] = sn
    alarm = np.where(s_pos >= h, 1, np.where(s_neg >= h, -1, 0)).astype(np.int8)
    if one_d:
        return s_pos[0], s_neg[0], alarm[0]
    return s_pos, s_neg, alarm
