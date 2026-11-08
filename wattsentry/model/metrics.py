"""Metrics (numpy / sklearn only). Definitions used in every M1 / M2 / M4 LAB_METRIC and report:

- nMAE = sum|q50 - y| / sum(y) over the evaluated steps (energy-normalised; stricter than per-kWp nMAE).
- pinball = mean over (q10, q50, q90) and steps of the pinball loss, divided by mean(y) (same scale as nMAE).
- coverage_10_90 = share of steps with q10 <= y <= q90 (PV: daylight steps only; target 78-82 %).
- PR-AUC = average precision; macro = mean over the fault classes present (one-vs-rest).
- hours_to_detect = hours from the exact onset to the end of the first 24 h window flagged (threshold frozen
  on validation); episodes not flagged within the search horizon count as missed.
"""

from __future__ import annotations

import numpy as np

TAUS = np.array([0.1, 0.5, 0.9])


def nmae(pred, y, mask=None) -> float:
    pred, y = np.asarray(pred, np.float64), np.asarray(y, np.float64)
    if mask is not None:
        pred, y = pred[mask], y[mask]
    return float(np.abs(pred - y).sum() / max(y.sum(), 1e-9))


def pinball(q, y, mask=None) -> float:
    q, y = np.asarray(q, np.float64), np.asarray(y, np.float64)
    e = y[..., None] - q
    l = np.maximum(TAUS * e, (TAUS - 1) * e).mean(-1)
    if mask is not None:
        l, y = l[mask], y[mask]
    return float(l.mean() / max(y.mean(), 1e-9))


def coverage(q, y, mask=None) -> float:
    q, y = np.asarray(q), np.asarray(y)
    inside = (y >= q[..., 0] - 1e-6) & (y <= q[..., 2] + 1e-6)
    if mask is not None:
        inside = inside[mask]
    return float(inside.mean()) if inside.size else float("nan")


def ap(y_true, score) -> float:
    from sklearn.metrics import average_precision_score

    y_true = np.asarray(y_true).astype(int)
    if y_true.sum() == 0 or y_true.sum() == len(y_true):
        return float("nan")
    return float(average_precision_score(y_true, score))


def prauc_macro(probs, labels, n_cls=8) -> tuple[float, dict]:
    per = {}
    for c in range(1, n_cls):
        if (labels == c).any():
            per[c] = ap(labels == c, probs[:, c])
    vals = [v for v in per.values() if not np.isnan(v)]
    return (float(np.mean(vals)) if vals else float("nan")), per


def confusion(pred, labels, n_cls=8) -> np.ndarray:
    m = np.zeros((n_cls, n_cls), np.int64)
    np.add.at(m, (labels, pred), 1)
    return m


def best_f1_threshold(y_true, score) -> tuple[float, float]:
    from sklearn.metrics import precision_recall_curve

    p, r, t = precision_recall_curve(np.asarray(y_true).astype(int), score)
    f1 = 2 * p * r / np.maximum(p + r, 1e-9)
    i = int(np.nanargmax(f1[:-1])) if len(t) else 0
    return (float(t[i]) if len(t) else 0.5), float(f1[i])


def summarize_detection(hours: list[float], horizon_h: float) -> dict:
    h = np.array(hours, np.float64)
    hit = ~np.isnan(h)
    return {
        "n_episodes": int(len(h)),
        "detected_frac": float(hit.mean()) if len(h) else float("nan"),
        "median_hours": float(np.median(h[hit])) if hit.any() else float("nan"),
        "mean_hours_capped": float(np.where(hit, h, horizon_h).mean()) if len(h) else float("nan"),
    }
