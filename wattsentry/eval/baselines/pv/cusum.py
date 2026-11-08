"""Two-sided CUSUM for the PV fault baselines: a thin wrapper over kernel K3 (``wattsentry.kernels.cusum_two_sided``).

    s_pos[t] = min(h, max(0, s_pos[t-1] + x[t] - k))     upward shift   (e.g. PV under-production residual)
    s_neg[t] = min(h, max(0, s_neg[t-1] - x[t] - k))     downward shift
    alarm[t] = +1 if s_pos[t] >= h, -1 if s_neg[t] >= h (and s_pos < h), else 0

- A gap (NaN, or +-inf) in x resets both statistics to 0 at that step (a data gap restarts the evidence), alarm 0.
- Statistics saturate at h (kernel ``cap=h``, "decision-interval clamp"): an alarm stays on while the shift lasts
  and clears about h / k steps after it ends, instead of a long tail after a long fault.
- x is 1-D [T] or 2-D [n_series, T]; time is the last axis. Outputs are new float64 / int8 arrays, same shape.
"""

from __future__ import annotations

import numpy as np

from wattsentry import kernels


def cusum_two_sided(x: np.ndarray, k: float, h: float) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    if h <= 0 or k < 0:
        raise ValueError("need h > 0 and k >= 0")
    s_pos, s_neg, _ = kernels.cusum_two_sided(x, k, h, cap=h)
    alarm = np.where(s_pos >= h, 1, np.where(s_neg >= h, -1, 0)).astype(np.int8)
    return s_pos, s_neg, alarm
