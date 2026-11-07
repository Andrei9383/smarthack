"""Forecast metrics used by the baselines (same definitions #14's train.py should use; #16 owns the shared module).

nMAE     = sum |q50 - y| / sum y over the scored steps (energy-weighted; scale-free across site sizes).
pinball  = mean over q in (0.1, 0.5, 0.9) of the mean pinball loss, divided by mean y (so it is also scale-free).
coverage = share of scored steps with q10 <= y <= q90 (target 78-82 %, decisions.md).
"""

from __future__ import annotations

import numpy as np

QS = (0.1, 0.5, 0.9)


def nmae(q50: np.ndarray, y: np.ndarray, mask: np.ndarray) -> float:
    m = mask & np.isfinite(y) & np.isfinite(q50)
    den = np.abs(y[m]).sum()
    return float(np.abs(q50[m] - y[m]).sum() / den) if den > 0 else float("nan")


def pinball(q: np.ndarray, y: np.ndarray, mask: np.ndarray) -> float:
    """q [3, ...] for QS."""
    m = mask & np.isfinite(y) & np.isfinite(q).all(0)
    yy = y[m]
    if not len(yy) or yy.mean() <= 0:
        return float("nan")
    losses = []
    for i, a in enumerate(QS):
        e = yy - q[i][m]
        losses.append(np.mean(np.maximum(a * e, (a - 1) * e)))
    return float(np.mean(losses) / yy.mean())


def coverage(q: np.ndarray, y: np.ndarray, mask: np.ndarray) -> float:
    m = mask & np.isfinite(y) & np.isfinite(q).all(0)
    if not m.any():
        return float("nan")
    eps = 1e-9
    return float(((q[0][m] <= y[m] + eps) & (y[m] <= q[2][m] + eps)).mean())


def forecast_metrics(q: np.ndarray, y: np.ndarray, mask: np.ndarray) -> dict[str, float]:
    return {"nmae": nmae(q[1], y, mask), "pinball": pinball(q, y, mask), "coverage": coverage(q, y, mask)}
