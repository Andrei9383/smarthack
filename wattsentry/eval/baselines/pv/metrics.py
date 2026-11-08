"""Baseline forecast metrics on stacked quantiles q [3, ...] (q10, q50, q90): adapters over wattsentry.eval.metrics."""

from __future__ import annotations

import numpy as np

from wattsentry.eval import metrics as M


def nmae(q50: np.ndarray, y: np.ndarray, mask: np.ndarray) -> float:
    """sum |q50 - y| / sum y (same value as M.nmae up to rounding; kept for bit-stable baseline numbers)."""
    m = mask & np.isfinite(y) & np.isfinite(q50)
    den = np.abs(y[m]).sum()
    return float(np.abs(q50[m] - y[m]).sum() / den) if den > 0 else float("nan")


def pinball(q: np.ndarray, y: np.ndarray, mask: np.ndarray) -> float:
    return M.npinball(q[0], q[1], q[2], y, mask)


def coverage(q: np.ndarray, y: np.ndarray, mask: np.ndarray) -> float:
    return M.coverage(q[0], q[2], y, mask, eps=1e-9)


def forecast_metrics(q: np.ndarray, y: np.ndarray, mask: np.ndarray) -> dict[str, float]:
    return {"nmae": nmae(q[1], y, mask), "pinball": pinball(q, y, mask), "coverage": coverage(q, y, mask)}
