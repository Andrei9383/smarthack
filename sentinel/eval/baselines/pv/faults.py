"""Fault baselines on the pvlib residual (M4 SiteHealth fallback): a threshold rule and a CUSUM.

Daily performance ratio r[s, d] = measured / pvlib(recorded metadata) energy over daylight steps. A day is
- uninformative (skipped, CUSUM state carried) when pvlib expects < 0.3 kWh/kWp or there is snow (> 1 cm),
- missing (NaN, CUSUM reset) when more than half of the daylight readings are null.
Each site's ratio is divided by its own reference level (median r over the calibration = training days), so wrong
recorded metadata does not look like a fault: q = r / ref, x = 1 - q (> 0 = under-producing).

threshold rule (card): alarm on day d when q < theta on d-2, d-1 and d (theta = 0.8, i.e. "< 80 % for 3 days").
CUSUM: two-sided CUSUM on x (cusum.py); alarm while s_pos >= h. k, h tuned for day-level F1 on training days.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping

import numpy as np

from sentinel.contracts.envelope import PvFault, Prediction, Window

from .cusum import cusum_two_sided
from .data import PV_VISIBLE, FleetArrays

DAYLIGHT_ZENITH = 85.0
MIN_KWH_PER_KWP = 0.3
SNOW_CM = 1.0
THETA = 0.8
RUN_DAYS = 3
K_GRID = (0.02, 0.05, 0.1, 0.15, 0.2, 0.3)
H_GRID = (0.03, 0.05, 0.1, 0.2, 0.3, 0.5, 0.8, 1.2)


def daily_ratio(a: FleetArrays) -> tuple[np.ndarray, np.ndarray]:
    """(r [n, d] with NaN = missing data, informative [n, d] bool)."""
    day = a.geom["zenith"] < DAYLIGHT_ZENITH
    meas = np.isfinite(a.pv_kw) & day
    e_act = np.where(meas, a.pv_kw, 0.0).sum(-1)
    e_exp = np.where(meas, a.pvlib_kw, 0.0).sum(-1)
    e_exp_all = np.where(day, a.pvlib_kw, 0.0).sum(-1) * 0.25  # kWh
    frac = meas.sum(-1) / np.maximum(day.sum(-1), 1)
    kwp = a.sites.kwp_recorded.to_numpy()[:, None]
    informative = (e_exp_all > MIN_KWH_PER_KWP * kwp) & (np.nanmean(a.weather["snow_cm"], -1) <= SNOW_CM)
    with np.errstate(invalid="ignore", divide="ignore"):
        r = np.where(frac >= 0.5, e_act / np.maximum(e_exp, 1e-9), np.nan)
    return r, informative


def site_reference(r: np.ndarray, informative: np.ndarray, calib_days: np.ndarray) -> np.ndarray:
    v = np.where(informative[:, calib_days], r[:, calib_days], np.nan)
    ref = np.nanmedian(v, axis=1)
    return np.where(np.isfinite(ref) & (ref > 0.2), ref, 1.0)


def threshold_rule(q: np.ndarray, informative: np.ndarray, theta: float = THETA, run: int = RUN_DAYS):
    """(alarm [n, d] bool, score [n, d]): score = 1 - max q over the last `run` informative-or-missing days."""
    n, d = q.shape
    alarm = np.zeros((n, d), bool)
    score = np.zeros((n, d))
    for s in range(n):
        idx = np.flatnonzero(informative[s] | np.isnan(q[s]))  # skipped days don't break a run, missing ones do
        qs = q[s, idx]
        for j in range(run - 1, len(idx)):
            w = qs[j - run + 1:j + 1]
            if np.isnan(w).any():
                continue
            m = w.max()
            score[s, idx[j]] = 1 - m
            alarm[s, idx[j]] = m < theta
        # carry the state through skipped days
        last_a, last_s = False, 0.0
        for t in range(d):
            if informative[s, t] or np.isnan(q[s, t]):
                last_a, last_s = alarm[s, t], score[s, t]
            else:
                alarm[s, t], score[s, t] = last_a, last_s
    return alarm, score


def cusum_detector(q: np.ndarray, informative: np.ndarray, k: float, h: float):
    """(alarm [n, d] bool, s_pos [n, d]); skipped days carry the state, missing days (NaN) reset it."""
    n, d = q.shape
    s_out = np.zeros((n, d))
    for s in range(n):
        idx = np.flatnonzero(informative[s] | np.isnan(q[s]))
        sp, _, _ = cusum_two_sided(1.0 - q[s, idx], k, h)
        full = np.full(d, np.nan)
        full[idx] = sp
        # forward fill skipped days
        last = 0.0
        for t in range(d):
            if np.isnan(full[t]):
                full[t] = last
            last = full[t]
        s_out[s] = full
    return s_out >= h, s_out


# ------------------------------------------------------------------------------------------------ evaluation
def day_labels(a: FleetArrays) -> tuple[np.ndarray, np.ndarray]:
    """(faulty [n, d] = a PV-visible fault active in daylight, class code [n, d] = most common daylight code)."""
    visible = np.array([name in PV_VISIBLE for name in a.fault_names])
    day = a.geom["zenith"] < 90
    f = np.where(day, a.fault, 0)
    vis = visible[f] & day
    faulty = vis.any(-1)
    counts = np.stack([(f == c).sum(-1) for c in range(len(a.fault_names))], -1)
    counts[..., 0] = 0
    code = np.where(counts.max(-1) > 0, counts.argmax(-1), 0).astype(np.int16)
    return faulty, code


def prf(alarm: np.ndarray, truth: np.ndarray) -> dict[str, float]:
    tp = float((alarm & truth).sum())
    fp = float((alarm & ~truth).sum())
    fn = float((~alarm & truth).sum())
    p = tp / (tp + fp) if tp + fp else 0.0
    r = tp / (tp + fn) if tp + fn else 0.0
    return {"precision": p, "recall": r, "f1": 2 * p * r / (p + r) if p + r else 0.0}


def pr_auc(score: np.ndarray, truth: np.ndarray) -> float:
    from sklearn.metrics import average_precision_score

    s = np.nan_to_num(score.ravel(), nan=-1.0)
    t = truth.ravel()
    return float(average_precision_score(t, s)) if t.any() else float("nan")


def events(a: FleetArrays, alarm: np.ndarray, day_mask: np.ndarray) -> dict[str, Any]:
    """Episode-level recall and hours from onset to the end of the first alarm day inside the episode."""
    if a.labels is None or not len(a.labels):
        return {"event_recall": float("nan"), "hours_to_detect": float("nan"), "by_class": {}}
    ids = {s: i for i, s in enumerate(a.sites.site_id)}
    days = a.days
    lo, hi = days[day_mask].min(), days[day_mask].max() + np.timedelta64(1, "D")
    rows = []
    for r in a.labels.itertuples():
        if r.fault not in PV_VISIBLE or not (lo <= r.onset < hi):
            continue
        end = hi if r.end is None or str(r.end) == "NaT" else min(r.end, hi)
        d0 = int((r.onset.normalize() - days[0]).days)
        d1 = int(np.ceil((end - days[0]) / np.timedelta64(1, "D")))
        hit = np.flatnonzero(alarm[ids[r.asset_id], d0:max(d1, d0 + 1)])
        delay = ((days[0] + np.timedelta64(d0 + int(hit[0]) + 1, "D")) - r.onset).total_seconds() / 3600 \
            if len(hit) else np.nan
        rows.append((r.fault, len(hit) > 0, delay))
    if not rows:
        return {"event_recall": float("nan"), "hours_to_detect": float("nan"), "by_class": {}}
    det = np.array([x[1] for x in rows])
    delays = np.array([x[2] for x in rows], float)
    by = {}
    for c in sorted({x[0] for x in rows}):
        m = np.array([x[0] == c for x in rows])
        by[c] = {"n": int(m.sum()), "recall": float(det[m].mean()),
                 "hours_to_detect": float(np.nanmedian(delays[m])) if det[m].any() else float("nan")}
    return {"n_events": len(rows), "event_recall": float(det.mean()),
            "hours_to_detect": float(np.nanmedian(delays)) if det.any() else float("nan"), "by_class": by}


@dataclass
class FaultBaselineResult:
    theta: float
    k: float
    h: float
    metrics: dict[str, Any]


def run_fault_baselines(a: FleetArrays, train_days: np.ndarray, eval_days: np.ndarray, log=print
                        ) -> FaultBaselineResult:
    r, inf = daily_ratio(a)
    ref = site_reference(r, inf, train_days)
    q = r / ref[:, None]
    truth, _ = day_labels(a)

    best = (-1.0, None)
    for k in K_GRID:
        for h in H_GRID:
            al, _ = cusum_detector(q, inf, k, h)
            f1 = prf(al[:, train_days], truth[:, train_days])["f1"]
            if f1 > best[0]:
                best = (f1, (k, h))
    k, h = best[1]
    log(f"cusum tuned on training days: k={k} h={h} (train F1 {best[0]:.3f})")

    out: dict[str, Any] = {}
    th_alarm, th_score = threshold_rule(q, inf)
    cu_alarm, cu_score = cusum_detector(q, inf, k, h)
    for name, al, sc in (("threshold", th_alarm, th_score), ("cusum", cu_alarm, cu_score)):
        sl = (slice(None), eval_days)
        m = prf(al[sl], truth[sl])
        m["prauc"] = pr_auc(sc[sl], truth[sl])
        m.update(events(a, al, eval_days))
        healthy_days = (~truth[sl]).sum()
        m["false_alarms_per_site_year"] = float((al[sl] & ~truth[sl]).sum() / max(healthy_days, 1) * 365)
        out[name] = m
    out["faulty_day_share"] = float(truth[:, eval_days].mean())
    return FaultBaselineResult(THETA, k, h, out)


# ------------------------------------------------------------------------------------------------ predictor
class ThresholdCusumPredictor:
    """M4 fallback as a contracts.Predictor.

    window.residual_summary [14, C]: column 0 = daily measured / pvlib ratio (NaN = no data, most recent last).
    asset may carry `ratio_ref` (the site's healthy ratio); otherwise the median of the 14 days is used.
    q10/q50/q90 = pvlib expectation on window.weather (zeros if no weather). fault_probs is a heuristic split:
    near-zero output -> sensor_fault, threshold alarm -> string_outage, CUSUM-only alarm -> soiling.
    """

    model = "threshold_cusum"

    def __init__(self, k: float = 0.05, h: float = 0.3, theta: float = THETA, model_version: str = "threshold_cusum-1"):
        self.k, self.h, self.theta = k, h, theta
        self.model_version = model_version

    def predict(self, asset: Mapping[str, Any], window: Window) -> Prediction:
        from .persistence import PvlibPredictor

        H = int(window.horizon)
        q50 = PvlibPredictor().expected_kw(asset, window) if window.weather is not None else np.zeros(H)
        rs = np.asarray(window.residual_summary if window.residual_summary is not None else np.full((14, 1), np.nan),
                        np.float64)
        r = rs[:, 0] if rs.ndim == 2 else rs
        ref = asset.get("ratio_ref")
        ref = float(ref) if ref is not None else (float(np.nanmedian(r)) if np.isfinite(r).any() else 1.0)
        q = r / max(ref, 1e-6)
        valid = q[np.isfinite(q)]
        th = len(valid) >= RUN_DAYS and bool(np.all(valid[-RUN_DAYS:] < self.theta))
        sp, _, _ = cusum_two_sided(1.0 - q, self.k, self.h)
        cu = bool(sp[-1] >= self.h) if len(sp) else False
        p = 0.9 if th else min(0.85, 0.85 * float(sp[-1]) / self.h if len(sp) else 0.0)
        if len(r) == 0 or not np.isfinite(r[-1]):
            cls = PvFault.SENSOR_FAULT.value  # no data at all: logger dropout looks like this
            p = max(p, 0.5)
        elif len(valid) and valid[-1] < 0.1:
            cls = PvFault.SENSOR_FAULT.value
        elif th:
            cls = PvFault.STRING_OUTAGE.value
        else:
            cls = PvFault.SOILING.value
        probs = {f.value: 0.0 for f in PvFault}
        probs[PvFault.HEALTHY.value] = 1 - p
        probs[cls] += p
        return Prediction(asset_id=str(asset.get("site_id", "?")), model=self.model, model_version=self.model_version,
                          t0=window.t0, horizon=H, q10=q50, q50=q50, q90=q50, fault_probs=probs,
                          ood_score=float(np.isnan(r).mean()) if len(r) else 1.0,
                          extra={"threshold_alarm": th, "cusum_alarm": cu, "cusum_s": float(sp[-1]) if len(sp) else 0.0,
                                 "ratio_ref": ref, "rule": "heuristic class split"})
