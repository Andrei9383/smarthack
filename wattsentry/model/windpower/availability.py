"""W2 (card #165, spec W step 3): tomorrow's availability of every turbine, from the Judge-validated M6 / M7 reads.

    from wattsentry.model.windpower.availability import M6Batch, M7Batch, judge_reads, availability
    reads = judge_reads(ids, day, M6Batch(cls, score, loss_frac, stopped_h), M7Batch(p_fail_7d, alarm))
    av = availability(reads, ts_tomorrow, weather=w, work_orders={"WP1.TID07": np.datetime64("2025-03-02T14:00")})
    av.a            # [T, N] in 0..1 = p_up x derate (what M5 is multiplied by)
    av.table()      # one row per turbine: rule, mean availability, uncertain, Judge verdict / source

Every model read goes through wattsentry/judge/adapter.py (C1 #157): M6 ("windnbm") and M7 ("windfailure") outputs are
passed as raw batches; a "dont_use" verdict swaps in the named substitute (e.g. "windmon_rules") when the caller
supplies it, else the turbine is unknown and keeps availability 1, marked uncertain (= windpower_only for that
turbine). The measured extras (underperformance loss, stop age) are taken from whichever source the Judge accepted.

The table (spec W step 3):

| M6 / M7 says                         | availability tomorrow                                                       |
|--------------------------------------|-----------------------------------------------------------------------------|
| normal                               | 1                                                                           |
| fault_stop + open work order (ETA)   | 0 until the repair ETA, then 1                                              |
| fault_stop, no ETA                   | P(repaired by t | stopped for `stopped_h`) from the stop-duration survival  |
| underperformance / yaw               | 1 - measured mean loss (no measurement: 1 - 0.15)                           |
| icing risk (weather forecast)        | x icing_factor only on risk steps: T < 1 C and (RH >= 90 % or precip)       |
| M7 high risk (alarm)                 | x (1 - P(failure in the next 24 h)), P24 = 1 - (1 - P7d)^(1/7)               |
| curtailment                          | 1 (dispatcher limit, handled separately)                                    |
| anemometer fault / data gap          | 1, uncertain (M5 already runs on the weather wind)                          |
| overheating                          | 1 (the risk is M7's)                                                        |
| Judge dont_use without substitute    | 1, uncertain                                                                |

Two factors are kept apart so park scenarios can treat them correctly: `p_up` is a probability that the turbine runs
(stop / repair, M7 failure; a scenario may draw it as on/off), `derate` is a deterministic multiplier (loss, ice).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Mapping, Optional, Sequence, Union

import numpy as np

from wattsentry.judge.adapter import RawBatch, VerdictSource, validated_many

M6, M7 = "windnbm", "windfailure"
STEP_H = 0.25

RULES = {
    "normal": "1",
    "fault_stop_eta": "0 until the open work order's repair ETA, then 1",
    "fault_stop": "P(repaired by t | stopped so far) from the stop-duration survival curve",
    "underperformance": "1 - measured mean loss",
    "icing": "x icing_factor on forecast icing-risk steps only",
    "m7_high": "x (1 - P(failure within 24 h))",
    "curtailment": "1 (dispatcher limit, handled separately)",
    "anemometer_fault": "1, uncertain: M5 uses the weather wind",
    "data_gap": "1, uncertain",
    "overheating": "1 (M7 carries the risk)",
    "unknown": "1, uncertain: Judge dont_use and no substitute",
}
UNCERTAIN = {"anemometer_fault", "data_gap", "unknown"}


# ============================================================================================ stop survival
@dataclass
class StopSurvival:
    """S(h) = P(a fault stop lasts more than h hours), on a 15-min grid. `p_repaired(age, dt)` = P(running again
    dt hours from now | stopped for `age` hours already) = 1 - S(age + dt) / S(age)."""

    surv: np.ndarray  # [K] S at h = k * STEP_H, S[0] = 1, non-increasing
    n: int = 0
    source: str = "default"

    @classmethod
    def exponential(cls, mean_h: float = 30.0, max_h: float = 480.0) -> "StopSurvival":
        h = np.arange(0, max_h + STEP_H, STEP_H)
        return cls(np.exp(-h / mean_h), 0, f"exponential(mean {mean_h:g} h)")

    @classmethod
    def fit(cls, durations_h: Sequence[float], max_h: float = 480.0, min_n: int = 20) -> "StopSurvival":
        """Empirical survival of observed stop durations (label-free when the runs come from M6 fault_stop runs).
        Beyond the longest observation the tail decays exponentially with the empirical mean."""
        d = np.asarray([x for x in durations_h if np.isfinite(x) and x > 0], np.float64)
        if len(d) < min_n:
            return cls.exponential(float(d.mean()) if len(d) else 30.0, max_h)
        h = np.arange(0, max_h + STEP_H, STEP_H)
        s = 1.0 - np.searchsorted(np.sort(d), h, side="right") / len(d)
        tail = h > d.max()
        s[tail] = s[~tail][-1] * np.exp(-(h[tail] - d.max()) / max(d.mean(), 1.0)) if (~tail).any() else 0
        s = np.maximum(s, np.exp(-h / max(d.mean(), 1.0)) * 1e-6)  # never exactly 0: conditional stays defined
        return cls(np.minimum.accumulate(s), len(d), f"empirical ({len(d)} stops)")

    def _at(self, h: np.ndarray) -> np.ndarray:
        k = np.clip(np.asarray(h, np.float64) / STEP_H, 0, len(self.surv) - 1)
        lo = np.floor(k).astype(int)
        hi = np.minimum(lo + 1, len(self.surv) - 1)
        return self.surv[lo] + (self.surv[hi] - self.surv[lo]) * (k - lo)

    def p_repaired(self, age_h: Union[float, np.ndarray], dt_h: np.ndarray) -> np.ndarray:
        s0 = np.maximum(self._at(age_h), 1e-12)
        return np.clip(1.0 - self._at(np.asarray(age_h) + np.asarray(dt_h)) / s0, 0.0, 1.0)

    def to_json(self) -> dict:
        return {"surv": np.round(self.surv[:: 4], 5).tolist(), "step_h": 1.0, "n": self.n, "source": self.source}

    @classmethod
    def from_json(cls, d: dict) -> "StopSurvival":
        s = np.asarray(d["surv"], np.float64)
        h = np.arange(len(s)) * float(d.get("step_h", 1.0))
        return cls(np.interp(np.arange(0, h[-1] + STEP_H, STEP_H), h, s), int(d.get("n", 0)), str(d.get("source", "")))


def stop_runs_h(cls_steps: np.ndarray, stop_code: int, step_h: float = STEP_H, min_steps: int = 2) -> list[float]:
    """Durations (h) of the runs of `stop_code` in a [T, N] class array (e.g. M6 fault_stop runs on training days).
    Runs touching the array edges are dropped (censored)."""
    out = []
    T = cls_steps.shape[0]
    for j in range(cls_steps.shape[1]):
        x = np.concatenate([[False], cls_steps[:, j] == stop_code, [False]]).astype(np.int8)
        d = np.diff(x)
        for s, e in zip(np.flatnonzero(d == 1), np.flatnonzero(d == -1)):
            if s > 0 and e < T and e - s >= min_steps:
                out.append((e - s) * step_h)
    return out


# ============================================================================================ config
@dataclass(frozen=True)
class AvailConfig:
    conf_min: float = 0.6  # = qualify's anomaly_conf_min: weaker M6 classes count as normal
    default_underperf_loss: float = 0.15  # = qualify.WIND_DEFAULT_LOSS["underperformance"]
    icing_temp_c: float = 1.0
    icing_rh: float = 90.0
    icing_precip_mm: float = 0.05
    icing_min_h: float = 3.0  # risk spells shorter than this are ignored (the sim needs >= 3 h to ice)
    icing_persist_h: float = 6.0  # ice stays after the spell ends
    icing_factor: float = 0.75  # availability on risk steps; fit_icing_factor() on training days
    m7_horizon_days: float = 7.0
    survival: StopSurvival = field(default_factory=StopSurvival.exponential)


# ============================================================================================ Judge reads
@dataclass
class M6Batch:
    """Raw M6 output per turbine at the forecast origin. cls: (N,) class names; score: (N,) anomaly score;
    loss_frac: (N,) measured mean loss of the current episode (NaN = not measured); stopped_h: (N,) age of the
    current fault_stop (h). The same shape is used for an M6 substitute (e.g. windmon_rules)."""

    cls: Sequence[Any]
    score: Sequence[float]
    loss_frac: Optional[Sequence[float]] = None
    stopped_h: Optional[Sequence[float]] = None


@dataclass
class M7Batch:
    """Raw M7 output per turbine: p_fail_7d (N,) of the top component; alarm (N,) bool (risk >= frozen threshold)."""

    p_fail_7d: Sequence[float]
    alarm: Optional[Sequence[bool]] = None


@dataclass
class Reads:
    """Judge-validated reads, one row per turbine (arrays (N,))."""

    turbine_ids: list[str]
    cls: np.ndarray  # object: class name ("unknown" when the Judge rejected M6 and no substitute was given)
    score: np.ndarray
    loss_frac: np.ndarray
    stopped_h: np.ndarray
    p_fail_7d: np.ndarray  # NaN = no usable M7
    m7_alarm: np.ndarray
    verdict: np.ndarray  # M6 verdict
    source: np.ndarray  # whose numbers cls/score are (windnbm, the substitute, or None)
    m7_verdict: np.ndarray


def _col(x, n, fill=np.nan) -> np.ndarray:
    return np.full(n, fill, np.float64) if x is None else np.asarray(x, np.float64).reshape(n)


def judge_reads(turbine_ids: Sequence[Any], days: Union[str, Sequence[Any]], m6: M6Batch, m7: Optional[M7Batch] = None,
                substitutes: Optional[Mapping[str, M6Batch]] = None, judge: Optional[VerdictSource] = None) -> Reads:
    """M6 class / score and M7 risk through the Judge adapter (`validated_many`). days: the M6 / M7 verdict day
    (the day the observation ends, i.e. the forecast origin's day), one for all or one per turbine."""
    ids = [str(t) for t in turbine_ids]
    n = len(ids)
    subs = dict(substitutes or {})
    raw6 = RawBatch(cls=np.asarray(m6.cls, dtype=object), score=_col(m6.score, n, 0.0))
    v6 = validated_many(M6, ids, days, raw=raw6, judge=judge,
                        substitutes={k: RawBatch(cls=np.asarray(b.cls, dtype=object), score=_col(b.score, n, 0.0))
                                     for k, b in subs.items()})
    loss, age = np.full(n, np.nan), np.zeros(n)
    for name, b in [(M6, m6), *subs.items()]:  # extras from the source the Judge accepted
        rows = np.asarray([v6.known[i] and v6.source_model[i] == name for i in range(n)], bool)
        loss[rows] = _col(b.loss_frac, n)[rows]
        age[rows] = np.nan_to_num(_col(b.stopped_h, n, 0.0))[rows]
    cls = np.asarray([c if k and c is not None else ("unknown" if not k else "normal")
                      for c, k in zip(v6.cls, v6.known)], dtype=object)
    p7, alarm, v7 = np.full(n, np.nan), np.zeros(n, bool), np.full(n, "use", dtype=object)
    if m7 is not None:
        b7 = validated_many(M7, ids, days, raw=RawBatch(score=_col(m7.p_fail_7d, n)), judge=judge)
        p7 = np.where(b7.known, b7.score, np.nan)
        alarm = b7.known & (np.zeros(n, bool) if m7.alarm is None else np.asarray(m7.alarm, bool).reshape(n))
        v7 = b7.verdict.astype(object)
    return Reads(ids, cls, np.nan_to_num(v6.score if v6.score is not None else np.zeros(n)), loss, age, p7, alarm,
                 v6.verdict.astype(object), v6.source_model, v7)


# ============================================================================================ availability
@dataclass
class Availability:
    turbine_ids: list[str]
    ts: np.ndarray  # [T] datetime64[m] UTC
    p_up: np.ndarray  # [T, N] probability the turbine runs (stop / repair, M7)
    derate: np.ndarray  # [T, N] deterministic multiplier (underperformance loss, icing)
    uncertain: np.ndarray  # [N] bool
    rule: list[str]  # [N] main rule applied (RULES key)
    reads: Optional[Reads] = None

    @property
    def a(self) -> np.ndarray:
        """[T, N] availability in 0..1 (what M5's turbine forecast is multiplied by)."""
        return self.p_up * self.derate

    def table(self) -> list[dict]:
        a = self.a
        rows = []
        for j, tid in enumerate(self.turbine_ids):
            r = {"turbine_id": tid, "rule": self.rule[j], "availability_mean": round(float(a[:, j].mean()), 4),
                 "availability_min": round(float(a[:, j].min()), 4), "uncertain": bool(self.uncertain[j])}
            if self.reads is not None:
                r |= {"m6_class": str(self.reads.cls[j]), "m6_verdict": str(self.reads.verdict[j]),
                      "m6_source": self.reads.source[j], "m7_p_fail_7d": None if np.isnan(self.reads.p_fail_7d[j])
                      else float(self.reads.p_fail_7d[j])}
            rows.append(r)
        return rows


def _runs_filter(m: np.ndarray, min_steps: int, persist_steps: int) -> np.ndarray:
    """[T, N] bool: keep runs >= min_steps along time, extend each kept run by persist_steps."""
    out = np.zeros_like(m)
    T = m.shape[0]
    for j in range(m.shape[1]):
        x = np.concatenate([[0], m[:, j].astype(np.int8), [0]])
        d = np.diff(x)
        for s, e in zip(np.flatnonzero(d == 1), np.flatnonzero(d == -1)):
            if e - s >= min_steps:
                out[s:min(T, e + persist_steps), j] = True
    return out


def icing_risk(weather: Mapping[str, Any], T: int, N: int, cfg: AvailConfig = AvailConfig(),
               iced_now: Optional[np.ndarray] = None) -> np.ndarray:
    """[T, N] bool icing-risk steps from the weather forecast: T < icing_temp_c and (RH >= icing_rh or precip >
    icing_precip_mm), spells >= icing_min_h, extended by icing_persist_h. Turbines M6 sees iced now (`iced_now` (N,))
    need only the temperature condition. weather fields [T, N] or [T, 1]; missing temp -> no risk."""
    if weather is None or weather.get("temp_c") is None:
        return np.zeros((T, N), bool)
    f = lambda k, d: np.broadcast_to(np.nan_to_num(np.asarray(weather[k], np.float64).reshape(T, -1), nan=d), (T, N)) \
        if weather.get(k) is not None else np.full((T, N), d)  # noqa: E731
    cold = f("temp_c", 10.0) < cfg.icing_temp_c
    wet = (f("rh_pct", 0.0) >= cfg.icing_rh) | (f("precip_mm", 0.0) > cfg.icing_precip_mm)
    risk = cold & wet
    if iced_now is not None:
        risk = risk | (cold & np.asarray(iced_now, bool)[None])
    return _runs_filter(risk, int(round(cfg.icing_min_h / STEP_H)), int(round(cfg.icing_persist_h / STEP_H)))


def availability(reads: Reads, ts, weather: Optional[Mapping[str, Any]] = None,
                 work_orders: Optional[Mapping[str, Any]] = None, origin=None,
                 cfg: AvailConfig = AvailConfig()) -> Availability:
    """Availability [T, N] over the forecast steps `ts` (UTC, 15 min) from the Judge reads at the origin.

    work_orders: {turbine_id: repair ETA (datetime64 / ISO str) or None} = open work orders; a fault_stop turbine
    with an ETA is 0 until it, without one the survival curve applies. origin: forecast origin (default: ts[0]);
    the stop age at step t is stopped_h + (t - origin). weather: forecast fields for the icing rule."""
    ts = np.asarray(ts).astype("datetime64[m]")
    T, N = len(ts), len(reads.turbine_ids)
    origin = ts[0] if origin is None else np.datetime64(str(origin).replace("Z", ""), "m")
    dt_h = (ts - origin).astype(np.int64) / 60.0  # hours since origin, [T]
    wo = {str(k): v for k, v in (work_orders or {}).items()}
    p_up, derate = np.ones((T, N)), np.ones((T, N))
    uncertain = np.zeros(N, bool)
    rule = ["normal"] * N
    for j, tid in enumerate(reads.turbine_ids):
        c = str(reads.cls[j])
        conf = c == "unknown" or float(reads.score[j]) >= cfg.conf_min
        if not conf or c in ("normal", "None"):
            continue
        rule[j] = c
        if c == "fault_stop":
            eta = wo.get(tid)
            if eta is not None:
                p_up[:, j] = (ts >= np.datetime64(str(eta).replace("Z", ""), "m")).astype(np.float64)
                rule[j] = "fault_stop_eta"
            else:
                p_up[:, j] = cfg.survival.p_repaired(float(reads.stopped_h[j]), np.maximum(dt_h, 0.0))
        elif c == "underperformance":
            loss = reads.loss_frac[j]
            derate[:, j] = 1.0 - float(np.clip(cfg.default_underperf_loss if not np.isfinite(loss) else loss, 0, 1))
        elif c in UNCERTAIN:
            uncertain[j] = True
    iced_now = np.array([str(c) == "icing" and float(s) >= cfg.conf_min for c, s in zip(reads.cls, reads.score)])
    risk = icing_risk(weather, T, N, cfg, iced_now)
    derate = np.where(risk, derate * cfg.icing_factor, derate)
    for j in np.flatnonzero(risk.any(0)):
        if rule[j] == "normal":
            rule[j] = "icing"
    p24 = 1.0 - (1.0 - np.clip(np.nan_to_num(reads.p_fail_7d), 0, 1)) ** (1.0 / cfg.m7_horizon_days)
    hi = reads.m7_alarm & np.isfinite(reads.p_fail_7d)
    p_up[:, hi] *= (1.0 - p24[hi])[None]
    for j in np.flatnonzero(hi):
        if rule[j] == "normal":
            rule[j] = "m7_high"
    return Availability(list(reads.turbine_ids), ts, p_up, derate, uncertain, rule, reads)


def fit_icing_factor(actual: np.ndarray, expected: np.ndarray, risk: np.ndarray, lo: float = 0.2,
                     default: float = 0.75, min_steps: int = 96) -> tuple[float, int]:
    """Label-free icing factor = sum(actual) / sum(expected) over training steps the weather rule flags (actual /
    expected [T, N] kW, e.g. measured power and M5 P50). Returns (factor, n_steps)."""
    m = risk & np.isfinite(actual) & np.isfinite(expected) & (expected > 0)
    if m.sum() < min_steps:
        return default, int(m.sum())
    return float(np.clip(actual[m].sum() / expected[m].sum(), lo, 1.0)), int(m.sum())


__all__ = ["AvailConfig", "Availability", "M6Batch", "M7Batch", "Reads", "RULES", "StopSurvival", "availability",
           "fit_icing_factor", "icing_risk", "judge_reads", "stop_runs_h"]
