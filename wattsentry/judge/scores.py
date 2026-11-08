"""Scores that cannot be gamed (team/judge-design.md §5.3). All in 0..1, higher is better; NaN = ungraded.

Vectorised: every function takes [P, T] arrays (P asset-days, T = 96 steps) plus a validity mask and returns [P].
"""

from __future__ import annotations

import numpy as np

TARGET_COVERAGE = 0.80  # team decision 13:52: P10-P90 should hold 80 %
ALPHA = 1.0 - TARGET_COVERAGE  # central interval -> Winkler alpha = 0.2
MIN_STEPS = 8  # fewer valid steps in a day -> ungraded


def hit_rate(q10: np.ndarray, q90: np.ndarray, y: np.ndarray, valid: np.ndarray) -> np.ndarray:
    """Share of valid steps with q10 <= y <= q90; NaN when fewer than MIN_STEPS valid steps."""
    inside = (y >= q10) & (y <= q90) & valid
    n = valid.sum(-1)
    with np.errstate(invalid="ignore", divide="ignore"):
        return np.where(n >= MIN_STEPS, inside.sum(-1) / np.maximum(n, 1), np.nan)


def calibration(hit: np.ndarray, target: float = TARGET_COVERAGE) -> np.ndarray:
    """1 - |hit - 0.80| / 0.80: both too narrow and too wide lose. NaN stays NaN."""
    return np.clip(1.0 - np.abs(hit - target) / target, 0.0, 1.0)


def interval_score(q10: np.ndarray, q90: np.ndarray, y: np.ndarray, valid: np.ndarray,
                   alpha: float = ALPHA) -> np.ndarray:
    """Mean Winkler interval score over valid steps: width + (2/alpha) x distance outside the band."""
    lo = np.minimum(q10, q90)
    hi = np.maximum(q10, q90)
    s = (hi - lo) + (2.0 / alpha) * (np.maximum(lo - y, 0.0) + np.maximum(y - hi, 0.0))
    s = np.where(valid, s, 0.0)
    n = valid.sum(-1)
    with np.errstate(invalid="ignore", divide="ignore"):
        return np.where(n >= MIN_STEPS, s.sum(-1) / np.maximum(n, 1), np.nan)


def skill(is_model: np.ndarray, is_ref: np.ndarray) -> np.ndarray:
    """clip(1 - IS_model / IS_ref, 0, 1): 'valid' = better than what we would otherwise use. NaN if no reference."""
    with np.errstate(invalid="ignore", divide="ignore"):
        r = np.where(is_ref > 1e-12, 1.0 - is_model / is_ref, np.where(is_model <= 1e-12, 1.0, 0.0))
    r = np.where(np.isfinite(is_ref) & np.isfinite(is_model), r, np.nan)
    return np.clip(r, 0.0, 1.0)


def forecast_scores(q10: np.ndarray, q90: np.ndarray, y: np.ndarray, valid: np.ndarray,
                    ref_q10: np.ndarray, ref_q90: np.ndarray, ref_valid: np.ndarray) -> dict[str, np.ndarray]:
    """Both forecast scores per asset-day. The reference is scored on exactly the same steps as the model."""
    both = valid & ref_valid
    hit = hit_rate(q10, q90, y, valid)
    is_m = interval_score(q10, q90, y, both)
    is_r = interval_score(ref_q10, ref_q90, y, both)
    return {"calibration": calibration(hit), "skill": skill(is_m, is_r), "hit": hit, "is_model": is_m,
            "is_ref": is_r}


# ---------------------------------------------------------------------------------- detectors
HEALTHY = ("healthy", "normal")


def alarm_precision(claimed: np.ndarray, found: np.ndarray, fault_existed: np.ndarray) -> np.ndarray:
    """1 if the found class is the claimed one, 0.5 if a fault existed but of another class, 0 if nothing.

    claimed / found: object arrays of class labels; fault_existed: the found fault had started by the claim day.
    """
    found_fault = ~np.isin(found, HEALTHY) & fault_existed
    return np.where(found_fault & (found == claimed), 1.0, np.where(found_fault, 0.5, 0.0))


def m7_hit(alarm: np.ndarray, failed: np.ndarray) -> np.ndarray:
    """M7: hit if (alarm and failure within 14 d) or (no alarm and no failure)."""
    return (np.asarray(alarm, bool) == np.asarray(failed, bool)).astype(np.float64)


def brier_score(p: np.ndarray, y: np.ndarray) -> np.ndarray:
    """1 - Brier (so higher is better, 0..1); kept as a detail next to the M7 hit."""
    return 1.0 - (np.clip(np.asarray(p, np.float64), 0, 1) - np.asarray(y, np.float64)) ** 2


def nanquantile(x: np.ndarray, qs, axis: int = -1) -> np.ndarray:
    """NaN-aware linear quantiles along `axis`, fully vectorised; result shape = x without `axis` + [len(qs)].

    Same values as np.nanquantile(method='linear'), which falls back to a per-row Python loop when NaNs are present
    (that loop was ~95 % of grading time). Rows with no finite value give NaN.
    """
    a = np.moveaxis(np.asarray(x, np.float64), axis, -1)
    s = np.sort(a, axis=-1)  # NaN sorts last
    n = np.isfinite(a).sum(-1)
    q = np.atleast_1d(np.asarray(qs, np.float64))
    pos = (np.maximum(n, 1) - 1)[..., None] * q
    lo = np.floor(pos).astype(np.intp)
    hi = np.minimum(lo + 1, np.maximum(n, 1)[..., None] - 1)
    vlo = np.take_along_axis(s, lo, -1)
    vhi = np.take_along_axis(s, hi, -1)
    out = vlo + (vhi - vlo) * (pos - lo)
    return np.where((n > 0)[..., None], out, np.nan)
