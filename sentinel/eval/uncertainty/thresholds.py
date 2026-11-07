"""'Not sure' thresholds for human review, tuned on validation and frozen in models/thresholds.json.

Schema (shared with #26 qualify, which reads it — do not change without `lab announce`):

    {"version": 1, "source_run": str,
     "site":    {"stale_minutes": 60, "pfault_confident": 0.6, "pfault_std_max": 0.15,
                 "rel_width_max": 0.8, "ood_max": 3.0},
     "turbine": {"stale_minutes": 60, "anomaly_conf_min": 0.6, "ood_max": 3.0}}

Signals: `pfault_std` = ensemble std of P_fault (M4 members), `rel_width` = (q90 − q10)/max(q50, eps) averaged over
daylight steps (`rel_width`), `ood` = Mahalanobis score (`ood.MahalanobisOOD`, same scale as
`Prediction.ood_score`). A signal above its *_max → human review; P_fault ≥ pfault_confident → confident fault.
"""

from __future__ import annotations

import copy
import json
from pathlib import Path
from typing import Any, Mapping, Optional, Sequence, Union

import numpy as np

from sentinel.contracts import Prediction

REPO_ROOT = Path(__file__).resolve().parents[3]
THRESHOLDS_PATH = REPO_ROOT / "models" / "thresholds.json"

DEFAULT_THRESHOLDS: dict[str, Any] = {
    "version": 1,
    "source_run": "defaults",
    "site": {"stale_minutes": 60, "pfault_confident": 0.6, "pfault_std_max": 0.15, "rel_width_max": 0.8,
             "ood_max": 3.0},
    "turbine": {"stale_minutes": 60, "anomaly_conf_min": 0.6, "ood_max": 3.0},
}
_INT_KEYS = {"stale_minutes"}

EPS_KW = 1e-3
DAYLIGHT_FRAC = 0.05  # daylight = q50 above 5% of the day's q50 peak (drops dawn/dusk blow-ups)


# --- schema --------------------------------------------------------------------------------------------------
def validate_thresholds(d: Mapping[str, Any]) -> dict[str, Any]:
    """Check the exact schema (same keys as DEFAULT_THRESHOLDS, numeric values); return a normalised copy."""
    if set(d) != set(DEFAULT_THRESHOLDS):
        raise ValueError(f"thresholds keys {sorted(d)} != {sorted(DEFAULT_THRESHOLDS)}")
    if int(d["version"]) != 1:
        raise ValueError("thresholds version must be 1")
    if not isinstance(d["source_run"], str):
        raise ValueError("source_run must be a string")
    out: dict[str, Any] = {"version": 1, "source_run": d["source_run"]}
    for group in ("site", "turbine"):
        g = d[group]
        if set(g) != set(DEFAULT_THRESHOLDS[group]):
            raise ValueError(f"{group} keys {sorted(g)} != {sorted(DEFAULT_THRESHOLDS[group])}")
        out[group] = {}
        for k in DEFAULT_THRESHOLDS[group]:
            v = g[k]
            if isinstance(v, bool) or not isinstance(v, (int, float)) or not np.isfinite(v) or v < 0:
                raise ValueError(f"{group}.{k} must be a finite number >= 0, got {v!r}")
            out[group][k] = int(v) if k in _INT_KEYS else float(v)
    return out


def load_thresholds(path: Union[str, Path, None] = None) -> dict[str, Any]:
    """Read models/thresholds.json (or `path`); defaults if the file does not exist."""
    p = Path(path) if path is not None else THRESHOLDS_PATH
    if not p.exists():
        return copy.deepcopy(DEFAULT_THRESHOLDS)
    return validate_thresholds(json.loads(p.read_text()))


def save_thresholds(d: Mapping[str, Any], path: Union[str, Path, None] = None) -> Path:
    p = Path(path) if path is not None else THRESHOLDS_PATH
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(validate_thresholds(d), indent=2) + "\n")
    return p


# --- signals -------------------------------------------------------------------------------------------------
def daylight_mask(q50: np.ndarray, frac: float = DAYLIGHT_FRAC, eps: float = EPS_KW) -> np.ndarray:
    """q50 > max(eps, frac · peak) per row (last axis = horizon)."""
    q50 = np.asarray(q50, dtype=float)
    peak = q50.max(axis=-1, keepdims=True) if q50.size else 0.0
    return q50 > np.maximum(eps, frac * peak)


def rel_width(q10: Any, q50: Any = None, q90: Any = None, mask: Optional[np.ndarray] = None,
              eps: float = EPS_KW) -> Union[float, np.ndarray]:
    """(q90 − q10) / max(q50, eps) averaged over daylight steps. Accepts a Prediction or three arrays
    ((H,) → float, (n, H) → (n,)). No daylight step → 0 (nothing to be unsure about)."""
    if isinstance(q10, Prediction):
        q10, q50, q90 = q10.q10, q10.q50, q10.q90
    q10, q50, q90 = (np.asarray(a, dtype=float) for a in (q10, q50, q90))
    m = daylight_mask(q50, eps=eps) if mask is None else np.asarray(mask, dtype=bool)
    r = (q90 - q10) / np.maximum(q50, eps)
    num = np.where(m, r, 0.0).sum(axis=-1)
    den = m.sum(axis=-1)
    out = np.where(den > 0, num / np.maximum(den, 1), 0.0)
    return float(out) if out.ndim == 0 else out


def pfault_std(member_fault_probs: Sequence[Mapping[str, float]]) -> float:
    """Ensemble std of P_fault = 1 − P(healthy) across M4 members (population std, 0 for one member)."""
    p = [1.0 - float(m.get("healthy", 0.0)) for m in member_fault_probs]
    return float(np.std(p)) if p else 0.0


# --- tuning --------------------------------------------------------------------------------------------------
def tune_upper_threshold(signal: Any, wrong: Optional[Any] = None, max_review_rate: float = 0.1,
                         default: Optional[float] = None, n_candidates: int = 200) -> float:
    """Threshold t for 'signal > t → human review'.

    With labels (`wrong[i]` = the model was wrong on sample i: outside the calibrated band, misclassified, …):
    the t that maximises F1 of flagging wrong samples, subject to review rate P(signal > t) ≤ max_review_rate.
    Without labels: the (1 − max_review_rate) quantile, i.e. a fixed human-review budget.
    """
    x = np.asarray(signal, dtype=float)
    x = x[np.isfinite(x)]
    if x.size == 0:
        if default is None:
            raise ValueError("empty signal and no default")
        return float(default)
    floor_t = float(np.quantile(x, 1 - max_review_rate))  # smallest t that respects the budget
    if wrong is None:
        return floor_t
    w = np.asarray(wrong, dtype=bool)[np.isfinite(np.asarray(signal, dtype=float))]
    if not w.any():
        return floor_t
    cands = np.unique(np.quantile(x, np.linspace(0, 1, n_candidates)))
    cands = cands[cands >= floor_t] if (cands >= floor_t).any() else np.array([floor_t])
    best_t, best_f1 = floor_t, -1.0
    for t in cands:
        flag = x > t
        tp = float((flag & w).sum())
        if flag.sum() == 0:
            f1 = 0.0
        else:
            prec, rec = tp / flag.sum(), tp / w.sum()
            f1 = 2 * prec * rec / (prec + rec) if prec + rec > 0 else 0.0
        if f1 > best_f1 + 1e-12:
            best_t, best_f1 = float(t), f1
    return best_t


def tune_confidence_threshold(score: Any, positive: Any, min_precision: float = 0.8,
                              default: float = 0.6) -> float:
    """Smallest t with precision(score ≥ t → positive) ≥ min_precision (e.g. pfault_confident, anomaly_conf_min)."""
    s = np.asarray(score, dtype=float)
    y = np.asarray(positive, dtype=bool)
    if s.size == 0 or not y.any():
        return float(default)
    for t in np.unique(s):
        flag = s >= t
        if flag.sum() and (flag & y).sum() / flag.sum() >= min_precision:
            return float(t)
    return float(default)


def tune_thresholds(site: Optional[Mapping[str, Any]] = None, turbine: Optional[Mapping[str, Any]] = None, *,
                    source_run: str, max_review_rate: float = 0.1, min_precision: float = 0.8,
                    base: Optional[Mapping[str, Any]] = None) -> dict[str, Any]:
    """Tune on validation arrays and return a schema-valid dict (missing signals keep `base`/defaults).

    site keys (all optional, aligned arrays): pfault_std, rel_width, ood, wrong (model wrong on the sample),
    p_fault + is_fault (for pfault_confident). turbine keys: ood, wrong, anomaly_score + is_anomaly.
    """
    th = copy.deepcopy(dict(base) if base is not None else DEFAULT_THRESHOLDS)
    th["source_run"] = source_run
    if site:
        wrong = site.get("wrong")
        for sig, key in (("pfault_std", "pfault_std_max"), ("rel_width", "rel_width_max"), ("ood", "ood_max")):
            if site.get(sig) is not None:
                th["site"][key] = round(tune_upper_threshold(site[sig], wrong, max_review_rate,
                                                             th["site"][key]), 4)
        if site.get("p_fault") is not None and site.get("is_fault") is not None:
            th["site"]["pfault_confident"] = round(
                tune_confidence_threshold(site["p_fault"], site["is_fault"], min_precision,
                                          th["site"]["pfault_confident"]), 4)
    if turbine:
        if turbine.get("ood") is not None:
            th["turbine"]["ood_max"] = round(tune_upper_threshold(turbine["ood"], turbine.get("wrong"),
                                                                  max_review_rate, th["turbine"]["ood_max"]), 4)
        if turbine.get("anomaly_score") is not None and turbine.get("is_anomaly") is not None:
            th["turbine"]["anomaly_conf_min"] = round(
                tune_confidence_threshold(turbine["anomaly_score"], turbine["is_anomaly"], min_precision,
                                          th["turbine"]["anomaly_conf_min"]), 4)
    return validate_thresholds(th)
