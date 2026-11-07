"""Event-level wind anomaly metrics, identical for M6 and the WindMon rules (numpy).

Definitions (also written into the JSON report):
- truth episode: contiguous run of one non-normal label on one turbine; data_gap runs closer than `gap_merge`
  steps are one offline window.
- alert: contiguous run of steps whose predicted class is ACTIONABLE (everything except normal and curtailment,
  which is logged without a work order) on one turbine; runs closer than `merge_tol` steps merge (generous to the
  rules: a step firing 3 rules is one alert). Alert class = majority predicted class in the run.
- wind_precision = alerts overlapping (± overlap_tol) an actionable truth episode / alerts.
- wind_recall = actionable truth episodes overlapped by >= 1 alert / actionable truth episodes (class-agnostic).
- alerts_per_turbine_day = alerts / (turbines × days).
- curtail_vs_fault_f1: over truth episodes of class curtailment or fault_stop, episode prediction = majority
  non-normal predicted class inside it ("normal" if none); macro F1 of the two classes.
"""

from __future__ import annotations

import numpy as np

from .data import ACTIONABLE, CLASS_INDEX, CLASSES

A_IDX = np.array([CLASS_INDEX[c] for c in ACTIONABLE])
GAP = CLASS_INDEX["data_gap"]
CURT = CLASS_INDEX["curtailment"]
STOP = CLASS_INDEX["fault_stop"]
K = len(CLASSES)


def _runs(mask: np.ndarray, merge: int = 0) -> tuple[np.ndarray, np.ndarray]:
    d = np.diff(np.concatenate([[0], mask.astype(np.int8), [0]]))
    s, e = np.flatnonzero(d == 1), np.flatnonzero(d == -1)
    if merge > 0 and len(s) > 1:
        keep = np.concatenate([[True], (s[1:] - e[:-1]) > merge])
        grp = np.cumsum(keep) - 1
        s2 = s[keep]
        e2 = np.zeros_like(s2)
        np.maximum.at(e2, grp, e)
        s, e = s2, e2
    return s, e


def truth_episodes(lab: np.ndarray, gap_merge: int = 32) -> list[tuple[int, int, int]]:
    """(class, start, end) for one turbine's label column."""
    out = []
    for c in range(1, K):
        s, e = _runs(lab == c, merge=gap_merge if c == GAP else 0)
        out += [(c, int(a), int(b)) for a, b in zip(s, e)]
    return sorted(out, key=lambda x: x[1])


def _majority(pred: np.ndarray, s: int, e: int, skip_normal: bool = True) -> int:
    seg = pred[s:e]
    if skip_normal:
        seg = seg[seg != 0]
    if len(seg) == 0:
        return 0
    return int(np.bincount(seg, minlength=K).argmax())


def event_metrics(
    truth: np.ndarray,
    pred: np.ndarray,
    steps_per_day: int = 96,
    merge_tol: int = 4,
    gap_merge: int = 32,
    overlap_tol: int = 4,
    full: bool = True,
) -> dict:
    T, N = truth.shape
    n_alerts = tp_alerts = 0
    n_true = det_true = 0
    per_cls_n = np.zeros(K, int)
    per_cls_det = np.zeros(K, int)
    per_cls_correct = np.zeros(K, int)
    ep_conf = np.zeros((K, K), int)  # truth episode class x predicted episode class; row 0 = false alerts
    delays = {c: [] for c in range(K)}
    cvf = []  # (truth, pred) for curtailment / fault_stop episodes
    is_act = np.isin(pred, A_IDX)
    for j in range(N):
        a_s, a_e = _runs(is_act[:, j], merge=merge_tol)
        n_alerts += len(a_s)
        eps = truth_episodes(truth[:, j], gap_merge=gap_merge)
        act_eps = [(c, s, e) for c, s, e in eps if c in A_IDX]
        # alerts: TP if overlapping any actionable truth episode
        if len(a_s):
            if act_eps:
                es = np.array([x[1] for x in act_eps])
                ee = np.array([x[2] for x in act_eps])
                order = np.argsort(es)
                es, ee = es[order], np.maximum.accumulate(ee[order])
                lo = np.searchsorted(ee, a_s - overlap_tol, side="right")
                hi = np.searchsorted(es, a_e + overlap_tol, side="left")
                hit = hi > lo
            else:
                hit = np.zeros(len(a_s), bool)
            tp_alerts += int(hit.sum())
            if full:
                for k in np.flatnonzero(~hit):
                    ep_conf[0, _majority(pred[:, j], a_s[k], a_e[k])] += 1
        for c, s, e in eps:
            if len(a_s):
                lo = np.searchsorted(a_e, s - overlap_tol, side="right")
                hi = np.searchsorted(a_s, e + overlap_tol, side="left")
                detected = hi > lo
            else:
                detected = False
            pc = _majority(pred[:, j], max(0, s - overlap_tol), min(T, e + overlap_tol))
            per_cls_n[c] += 1
            per_cls_det[c] += int(detected)
            per_cls_correct[c] += int(pc == c)
            ep_conf[c, pc] += 1
            if c in A_IDX:
                n_true += 1
                det_true += int(detected)
                if detected and full:
                    first = max(int(a_s[lo]), s - overlap_tol)
                    delays[c].append(max(0, first - s) / 4.0)
            if c in (CURT, STOP):
                cvf.append((c, pc))
    days = T / steps_per_day
    prec = tp_alerts / n_alerts if n_alerts else 0.0
    rec = det_true / n_true if n_true else 0.0
    f1s = []
    for c in (CURT, STOP):
        tp = sum(1 for t, p in cvf if t == c and p == c)
        fp = sum(1 for t, p in cvf if t != c and p == c)
        fn = sum(1 for t, p in cvf if t == c and p != c)
        f1s.append(2 * tp / (2 * tp + fp + fn) if tp else 0.0)
    out = {
        "wind_precision": prec,
        "wind_recall": rec,
        "wind_f1": 2 * prec * rec / (prec + rec) if prec + rec else 0.0,
        "alerts_per_turbine_day": n_alerts / (N * days) if N and days else 0.0,
        "curtail_vs_fault_f1": float(np.mean(f1s)),
        "n_alerts": n_alerts,
        "n_true_actionable": n_true,
        "true_actionable_per_turbine_day": n_true / (N * days) if N and days else 0.0,
        "class_correct_recall_macro": float(
            np.mean([per_cls_correct[c] / per_cls_n[c] for c in range(1, K) if per_cls_n[c]] or [0.0])
        ),
    }
    if full:
        out["per_class"] = {
            CLASSES[c]: {
                "episodes": int(per_cls_n[c]),
                "detected_any_alert": int(per_cls_det[c]),
                "class_correct": int(per_cls_correct[c]),
                "median_delay_h": float(np.median(delays[c])) if delays[c] else None,
            }
            for c in range(1, K)
        }
        out["episode_confusion"] = {
            "rows_truth_cols_pred": CLASSES,
            "note": "row 'normal' = alerts that overlap no actionable truth episode, by predicted class",
            "matrix": ep_conf.tolist(),
        }
        out["curtail_vs_fault_f1_per_class"] = {"curtailment": f1s[0], "fault_stop": f1s[1]}
    return out


def step_confusion(truth: np.ndarray, pred: np.ndarray) -> list[list[int]]:
    m = np.zeros((K, K), dtype=np.int64)
    np.add.at(m, (truth.ravel().astype(np.int64), pred.ravel().astype(np.int64)), 1)
    return m.tolist()
