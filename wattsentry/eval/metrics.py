"""Test-set metrics shared by eval.py and the baselines; numpy only, NaN in the truth = missing (masked out).

Forecasts: arrays [N, H] of truth y and quantiles q10/q50/q90 in the same unit. Detection: one score per case.
"""

from __future__ import annotations

import numpy as np

NAN = float("nan")


def _mask(y: np.ndarray, *qs: np.ndarray, mask: np.ndarray | None = None) -> np.ndarray:
    m = np.isfinite(y)
    for q in qs:
        m &= np.isfinite(q)
    return m if mask is None else m & mask


def nmae(q50: np.ndarray, y: np.ndarray, mask: np.ndarray | None = None) -> float:
    """Mean absolute error / mean truth (0.25 = 25 % of the average level)."""
    m = _mask(y, q50, mask=mask)
    denom = np.abs(y[m]).mean() if m.any() else np.nan
    return float(np.abs(q50[m] - y[m]).mean() / denom) if m.any() and denom > 0 else NAN


def pinball(q: np.ndarray, y: np.ndarray, tau: float, mask: np.ndarray | None = None) -> float:
    m = _mask(y, q, mask=mask)
    d = y[m] - q[m]
    return float(np.maximum(tau * d, (tau - 1) * d).mean()) if m.any() else NAN


def npinball(q10, q50, q90, y, mask=None) -> float:
    """Mean pinball over the three quantiles / mean truth (comparable across sites and units)."""
    m = _mask(y, q10, q50, q90, mask=mask)
    if not m.any() or np.abs(y[m]).mean() <= 0:
        return NAN
    return float(np.mean([pinball(q, y, t, m) for q, t in ((q10, 0.1), (q50, 0.5), (q90, 0.9))])
                 / np.abs(y[m]).mean())


def coverage(q10, q90, y, mask=None, eps: float = 0.0) -> float:
    """Share of truth inside [q10 - eps, q90 + eps]; target 78-82 %."""
    m = _mask(y, q10, q90, mask=mask)
    return float(((q10[m] <= y[m] + eps) & (y[m] <= q90[m] + eps)).mean()) if m.any() else NAN


def nwidth(q10, q90, y, mask=None) -> float:
    """Mean interval width / mean truth: coverage is only good if the interval is also narrow."""
    m = _mask(y, q10, q90, mask=mask)
    return float((q90[m] - q10[m]).mean() / np.abs(y[m]).mean()) if m.any() and np.abs(y[m]).mean() > 0 else NAN


def forecast_metrics(q10, q50, q90, y, mask=None) -> dict[str, float]:
    return {"nmae": nmae(q50, y, mask), "npinball": npinball(q10, q50, q90, y, mask),
            "coverage": coverage(q10, q90, y, mask), "nwidth": nwidth(q10, q90, y, mask)}


def average_precision(y_true: np.ndarray, score: np.ndarray) -> float:
    """Area under the precision-recall curve, step-wise (stable sort; ties are not grouped)."""
    y_true = np.asarray(y_true, bool)
    if y_true.sum() == 0:
        return NAN
    y = y_true[np.argsort(-np.asarray(score, float), kind="stable")]
    return float((np.cumsum(y) / np.arange(1, len(y) + 1) * y).sum() / y.sum())


def prf(y_true: np.ndarray, y_pred: np.ndarray) -> dict[str, float]:
    y_true, y_pred = np.asarray(y_true, bool), np.asarray(y_pred, bool)
    tp = float((y_true & y_pred).sum())
    p = tp / y_pred.sum() if y_pred.sum() else NAN
    r = tp / y_true.sum() if y_true.sum() else NAN
    return {"precision": p, "recall": r, "f1": 2 * p * r / (p + r) if p == p and r == r and (p + r) > 0 else NAN}


def confusion(labels: list[str], preds: list[str], classes: list[str]) -> np.ndarray:
    """Rows = truth, columns = prediction, in `classes` order."""
    idx = {c: i for i, c in enumerate(classes)}
    cm = np.zeros((len(classes), len(classes)), int)
    for a, b in zip(labels, preds):
        if a in idx and b in idx:
            cm[idx[a], idx[b]] += 1
    return cm


def per_class_ap(labels: list[str], probs: list[dict[str, float]], classes: list[str]) -> dict[str, float]:
    """One-vs-rest average precision per fault class (classes absent from the test set are NaN)."""
    return {c: average_precision(np.array([lab == c for lab in labels]), np.array([p.get(c, 0.0) for p in probs]))
            for c in classes}


def detection_delays(hours: list[float | None]) -> dict[str, float]:
    """Hours from onset to the first alarm per episode; None = never detected inside the search window."""
    found = np.array([h for h in hours if h is not None], float)
    return {"episodes": float(len(hours)), "detected_share": float(len(found) / len(hours)) if hours else NAN,
            "hours_to_detect_median": float(np.median(found)) if len(found) else NAN,
            "hours_to_detect_p90": float(np.percentile(found, 90)) if len(found) else NAN}
