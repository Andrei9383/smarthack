"""Event-level metrics for daily failure alarms (shared by M7 and the fixed-threshold baseline).

Definitions (horizon H = 14 days):
- an alarm is raised for (turbine j, component c) on day d when its score >= threshold; alarm time = end of day d;
- alarm days of one (j, c) form EPISODES (runs, gaps <= 2 days merged): one episode = one work order / truck roll;
- an alarm day is TRUE if a failure of (j, c) starts within (end_d, end_d + H]; an episode is TRUE if it contains
  a true day, otherwise it is a FALSE ALARM (so an episode that starts > H days before the failure, on the same
  drift, is not penalised for being early);
- a failure event is DETECTED if it has a true alarm day; lead time = failure onset − end of the first day of the
  episode that contains its first true day (hours, > 0);
- false alarms per turbine-year = false episodes / (eligible turbine-days / 365.25). False episodes that overlap a
  degradation drift which did NOT end in a failure (the simulator fails 60 % of drifts at random) are also reported
  separately: no detector can tell those apart from failing drifts before the drift ends.
"""

from __future__ import annotations

import numpy as np

from . import COMPONENTS

DAY_M = np.timedelta64(1440, "m")


def _runs(x: np.ndarray, merge_gap: int) -> list[tuple[int, int]]:
    """[(start, end_inclusive)] of True runs in a 1-D bool array, runs separated by <= merge_gap False merged."""
    idx = np.flatnonzero(x)
    if idx.size == 0:
        return []
    cut = np.flatnonzero(np.diff(idx) > merge_gap + 1)
    starts = np.concatenate([[idx[0]], idx[cut + 1]])
    ends = np.concatenate([idx[cut], [idx[-1]]])
    return list(zip(starts.tolist(), ends.tolist()))


def evaluate_alarms(alarm: np.ndarray, eligible: np.ndarray, failures: list, days: np.ndarray, horizon_d: int = 14,
                    merge_gap: int = 2, drifts: list | None = None) -> dict:
    """alarm bool [D, N, C]; eligible bool [D, N] (evaluated turbine-days); failures: labels.Failure in scope."""
    ends = days + DAY_M
    H = np.timedelta64(horizon_d * 1440, "m")
    alarm = alarm & eligible[..., None]
    true_day = np.zeros_like(alarm)
    for f in failures:
        c = COMPONENTS.index(f.component)
        window = (f.t_fail > ends) & (f.t_fail <= ends + H)
        true_day[:, f.turbine, c] |= alarm[:, f.turbine, c] & window
    in_drift = np.zeros(alarm.shape[:2], bool)  # turbine-days inside a drift that did not end in a failure
    for d in drifts or []:
        if d.get("ends_in_failure") or d["onset"] is None:
            continue
        end = d["end"] if d["end"] is not None else ends[-1]
        in_drift[(days < end) & (ends > d["onset"]), d["turbine"]] = True
    run_start = {}  # (j, c, day) of a true day -> start day of its episode
    n_false, n_false_drift, false_days = 0, 0, 0
    per_comp_false = {c: 0 for c in COMPONENTS}
    jj, cc = np.nonzero(alarm.any(0))
    for j, c in zip(jj.tolist(), cc.tolist()):
        for s, e in _runs(alarm[:, j, c], merge_gap):
            td = np.flatnonzero(true_day[s : e + 1, j, c])
            if td.size:
                for k in td:
                    run_start[(j, c, s + int(k))] = s
            else:
                n_false += 1
                per_comp_false[COMPONENTS[c]] += 1
                false_days += int(alarm[s : e + 1, j, c].sum())
                n_false_drift += int(in_drift[s : e + 1, j].any())
    events = []
    for f in failures:
        c = COMPONENTS.index(f.component)
        hits = np.flatnonzero(true_day[:, f.turbine, c] & (f.t_fail > ends) & (f.t_fail <= ends + H))
        ev = {**f.to_json(), "detected": bool(hits.size), "lead_h": None}
        if hits.size:
            s = run_start[(f.turbine, c, int(hits[0]))]
            ev["lead_h"] = float((f.t_fail - ends[s]) / np.timedelta64(1, "m") / 60.0)
            ev["first_alarm_day"] = str(days[s].astype("datetime64[D]"))
        events.append(ev)
    ty = float(eligible.sum()) / 365.25
    det = [e for e in events if e["detected"]]
    leads = [e["lead_h"] for e in det]
    out = {
        "n_events": len(events),
        "n_detected": len(det),
        "event_recall": len(det) / len(events) if events else float("nan"),
        "lead_time_h": float(np.median(leads)) if leads else 0.0,
        "lead_time_h_mean": float(np.mean(leads)) if leads else 0.0,
        "false_alarm_episodes": n_false,
        "false_alarm_episodes_in_nonfailing_drift": n_false_drift,
        "false_alarm_days": false_days,
        "turbine_years": ty,
        "false_alarms_per_turbine_year": n_false / ty if ty > 0 else float("nan"),
        "false_alarms_outside_drift_per_turbine_year": (n_false - n_false_drift) / ty if ty > 0 else float("nan"),
        "alarm_turbine_days": int(alarm.any(2).sum()),
        "events": events,
    }
    per_comp = {}
    for c in COMPONENTS:
        evc = [e for e in events if e["component"] == c]
        dc = [e for e in evc if e["detected"]]
        per_comp[c] = {
            "n_events": len(evc), "n_detected": len(dc),
            "event_recall": len(dc) / len(evc) if evc else None,
            "lead_time_h": float(np.median([e["lead_h"] for e in dc])) if dc else None,
            "false_alarm_episodes": per_comp_false[c],
        }
    out["per_component"] = per_comp
    return out


def tune_threshold(score: np.ndarray, eligible: np.ndarray, failures: list, days: np.ndarray, fa_budget: float,
                   horizon_d: int = 14, n_cand: int = 160) -> tuple[float, dict, list]:
    """Highest event recall with false alarms/turbine-year <= fa_budget (ties → highest threshold).
    Falls back to the lowest-FA threshold when no candidate meets the budget."""
    vals = score[eligible[..., None].repeat(score.shape[2], 2) & np.isfinite(score)]
    if vals.size == 0:
        return float("inf"), {}, []
    qs = np.unique(np.quantile(vals, np.concatenate([np.linspace(0.90, 0.999, n_cand // 2),
                                                     np.linspace(0.999, 0.99999, n_cand // 2)])))
    curve = []
    for t in qs:
        m = evaluate_alarms(np.nan_to_num(score, nan=-np.inf) >= t, eligible, failures, days, horizon_d)
        curve.append((float(t), m["event_recall"], m["false_alarms_per_turbine_year"], m["lead_time_h"]))
    ok = [c for c in curve if c[2] <= fa_budget]
    if ok:
        best = max(ok, key=lambda c: (c[1], c[0]))
    else:
        best = min(curve, key=lambda c: (c[2], -c[0]))
    m = evaluate_alarms(np.nan_to_num(score, nan=-np.inf) >= best[0], eligible, failures, days, horizon_d)
    return best[0], m, curve


def recall_at_fa(score: np.ndarray, eligible: np.ndarray, failures: list, days: np.ndarray, fa_max: float,
                 horizon_d: int = 14) -> dict:
    """Diagnostic only (never used to pick a shipped threshold): best recall on THIS data at FA <= fa_max."""
    _, m, _ = tune_threshold(score, eligible, failures, days, fa_max, horizon_d)
    return {k: m.get(k) for k in ("event_recall", "lead_time_h", "false_alarms_per_turbine_year")}


def pr_auc(y: np.ndarray, p: np.ndarray) -> float:
    from sklearn.metrics import average_precision_score

    ok = np.isfinite(p)
    y, p = y[ok], p[ok]
    if y.sum() == 0 or y.sum() == len(y):
        return float("nan")
    return float(average_precision_score(y, p))
