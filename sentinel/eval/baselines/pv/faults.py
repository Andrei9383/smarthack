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
import pandas as pd

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
        s_out[s] = pd.Series(full).ffill().fillna(0.0).to_numpy()  # skipped days carry the state
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
INV_OVERVOLTAGE = 37  # inverter code "grid voltage too high" (sim/faults/kinds.py)


class ThresholdCusumPredictor:
    """M4 fallback as a contracts.Predictor (same Window conventions as #14's SiteHealth).

    Today's ratio = window.meta["telemetry"]["pv_ac_w"] / pvlib(window.weather, recorded metadata) over daylight.
    Past ratios come from window.residual_summary[:, 0] (oldest first) when given, otherwise from the ratios this
    predictor computed for the same asset on earlier days (it remembers them, so calling it day by day per site
    reproduces the bulk rule). q = ratio / ratio_ref (asset["ratio_ref"], else the median of the past ratios, else 1).
    Rules: threshold (q < 0.8 on the last 3 informative days), CUSUM on 1 - q, inverter overvoltage code, no data.
    fault_probs: the alarm probability goes to one heuristic class (no data / ~0 output -> sensor_fault, overvoltage
    code -> overvoltage_trip, threshold -> string_outage, CUSUM only -> soiling); extra["p_fault_threshold"] = 0.5.
    """

    model = "threshold_cusum"
    P_THRESHOLD = 0.5

    def __init__(self, k: float = 0.1, h: float = 0.1, theta: float = THETA, model_version: str = "threshold_cusum-1"):
        self.k, self.h, self.theta = k, h, theta
        self.model_version = model_version
        self._memory: dict[str, dict[str, tuple[float, bool]]] = {}

    def _today(self, asset, window: Window, H: int, expected: np.ndarray):
        """(ratio or NaN if missing, informative, inverter overvoltage code seen)."""
        from .lgbm import _to_steps, parse_weather, power_scale, weather_channels
        from .physical import recorded_meta, solar_geometry

        tel = (window.meta or {}).get("telemetry") or {}
        if "pv_ac_w" not in tel or window.weather is None:
            return np.nan, False, False
        meas = _to_steps(np.asarray(tel["pv_ac_w"], np.float64), H) / power_scale(window)
        status = np.nan_to_num(np.asarray(tel.get("inv_status", []), np.float64))
        ov = bool((status == INV_OVERVOLTAGE).any())
        meta = recorded_meta(asset)
        g = solar_geometry(pd.date_range(pd.Timestamp(window.t0), periods=H, freq="15min"), meta["lat"], meta["lon"])
        day = g["zenith"] < DAYLIGHT_ZENITH
        ok = day & np.isfinite(meas)
        snow = parse_weather(window.weather, H, weather_channels(window))["snow_cm"]
        informative = expected[day].sum() * 0.25 > MIN_KWH_PER_KWP * meta["kwp"] and np.nanmean(snow) <= SNOW_CM
        if ok.sum() < 0.5 * max(day.sum(), 1):
            return np.nan, True, ov
        return float(meas[ok].sum() / max(expected[ok].sum(), 1e-9)), bool(informative), ov

    def predict(self, asset: Mapping[str, Any], window: Window) -> Prediction:
        from .persistence import PvlibPredictor

        H = int(window.horizon)
        aid = str(asset.get("site_id", asset.get("asset_id", "?")))
        has_tel = bool((window.meta or {}).get("telemetry"))
        q50 = PvlibPredictor().expected_kw(asset, window) if window.weather is not None else np.zeros(H)
        r0, inf0, ov = self._today(asset, window, H, q50)
        day0 = pd.Timestamp(window.t0).normalize()
        if window.residual_summary is not None:
            rs = np.asarray(window.residual_summary, np.float64)
            past = list(rs[:, 0] if rs.ndim == 2 else rs)
            past_inf = [True] * len(past)
        else:
            mem = self._memory.get(aid, {})
            keys = [(day0 - pd.Timedelta(days=i)).strftime("%Y-%m-%d") for i in range(14, 0, -1)]
            past = [mem[k][0] for k in keys if k in mem]
            past_inf = [mem[k][1] for k in keys if k in mem]
        if has_tel:
            self._memory.setdefault(aid, {})[day0.strftime("%Y-%m-%d")] = (r0, inf0)
        if has_tel or not past:
            r = np.array(past + [r0], np.float64)
            inf = np.array(past_inf + [inf0], bool)
        else:  # only a residual summary: its last row is the assessed day
            r, inf = np.array(past, np.float64), np.array(past_inf, bool)
        hist = r[:-1][inf[:-1]]
        hist = hist[np.isfinite(hist)]
        ref = asset.get("ratio_ref")
        ref = float(ref) if ref is not None else (float(np.median(hist)) if len(hist) >= 3 else 1.0)
        q = r / max(ref, 1e-6)
        qs = q[inf | np.isnan(q)]  # skipped (uninformative) days drop out, missing ones reset the CUSUM
        last = qs[-RUN_DAYS:]
        th = len(last) == RUN_DAYS and bool(np.isfinite(last).all()) and bool(last.max() < self.theta)
        s_last = float(cusum_two_sided(1.0 - qs, self.k, self.h)[0][-1]) if len(qs) else 0.0
        cu = s_last >= self.h
        q0 = q[-1]
        if ov:
            p, cls = 0.8, PvFault.OVERVOLTAGE_TRIP.value
        elif not np.isfinite(q0):
            p, cls = (0.6 if has_tel or window.residual_summary is not None else 0.0), PvFault.SENSOR_FAULT.value
        elif th or cu:
            p = 0.9 if th else 0.5 + 0.25 * min(1.0, s_last / self.h)
            cls = (PvFault.SENSOR_FAULT.value if q0 < 0.1 else
                   PvFault.STRING_OUTAGE.value if th else PvFault.SOILING.value)
        else:
            day_score = float(np.clip((1 - q0) / (1 - self.theta), 0, 1))
            p, cls = 0.45 * max(day_score, s_last / self.h), PvFault.SOILING.value
        probs = {f.value: 0.0 for f in PvFault}
        probs[PvFault.HEALTHY.value] = 1 - p
        probs[cls] += p
        return Prediction(asset_id=aid, model=self.model, model_version=self.model_version, t0=window.t0, horizon=H,
                          q10=q50, q50=q50, q90=q50, fault_probs=probs, ood_score=float(np.isnan(r).mean()),
                          extra={"p_fault_threshold": self.P_THRESHOLD, "ratio_today": None if np.isnan(r0) else r0,
                                 "ratio_ref": ref, "threshold_alarm": bool(th), "cusum_alarm": bool(cu),
                                 "cusum_s": s_last, "overvoltage_code": ov, "rule": "heuristic class split"})
