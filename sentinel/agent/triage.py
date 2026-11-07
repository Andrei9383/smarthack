"""O&M triage: assessments -> work_order / alert / human_review actions, top-K per day per crew type. Card #26.

Plan §3.5:

    score_i = P_fault,i × ( lost_kWh_day,i × π  +  lost_flex_MWh_day,i × v_flex ) × D_i
    lost_kWh_day,i = Σ (P50_i,t − actual_i,t)+ · Δt          D_i = days until the next possible visit
    v_flex = value per committed MWh of flexibility (0 for turbines)

π in lei/MWh (OPCOM PZU), so the score is lei expected to be lost before a crew can get there. Crews:
residential installers, wind technicians, remote check (sensor / anemometer faults: never a truck). Each crew type has
its own daily budget K. Icing -> safety alert, no truck. Curtailment -> curtailed-energy log, no work order. Stale /
data gap -> one offline window per asset (one alert per turbine, no storm). Every action carries an audit record
(model_version, input_hash, compact prediction, threshold, status).

`triage()` is pure: it takes the previous `TriageState` and returns a new one, so the loop (#17) can call it every
tick (open work orders, alerts, reviews and offline windows are not re-issued) or once per day.

Turbines (Wind D2, card #167, spec W step 5, team/judge-design.md §10.4) are ranked by lei per day, not by the
site formula above:

    lei_per_day   = lost_MWh × mean PZU over the next 24 h
    lost_MWh      = fault_stop: M5 P50 summed over the next 24 h; else M6 Σ (expected − actual)+ over the last 24 h
    triage_score  = P(fault) × lei_per_day × days_to_visit

P(fault) and the expected power are read through the C1 adapter (`judge_validated`, one function), never raw.
Judge "no reliable source" -> remote check, ranked after crew orders. Sensor faults -> remote check; data gaps ->
the offline-window alert, routed to remote check (never a crew); curtailment -> log only. Site scoring is unchanged.
"""

from __future__ import annotations

import copy
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from functools import lru_cache
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Callable, Iterable, Mapping, Optional, Sequence

import numpy as np

from sentinel.contracts import Action, ActionType, AuditRecord, CrewType, Severity, Status, WindAnomaly, input_hash
from sentinel.contracts.envelope import parse_utc

from .qualify import Assessment, rel_width


@dataclass
class TriageConfig:
    price_lei_mwh: float = 450.0  # π: energy value (OPCOM PZU order of magnitude); pass the day's mean price
    v_flex_lei_mwh: dict[str, float] = field(default_factory=lambda: {"site": 400.0, "turbine": 0.0})
    k_per_crew: dict[str, Optional[int]] = field(default_factory=lambda: {
        CrewType.RESIDENTIAL.value: 5, CrewType.WIND.value: 3, CrewType.REMOTE.value: 20})  # None = unlimited
    visit_days: dict[str, float] = field(default_factory=lambda: {
        CrewType.RESIDENTIAL.value: 1.0, CrewType.WIND.value: 2.0, CrewType.REMOTE.value: 1.0})
    policy: str = "sentinel"
    run_id: str = "local"
    # Turbines: "pzu" = mean OPCOM PZU over the next 24 h (price_lei_mwh where the file has no prices);
    # "config" = price_lei_mwh.
    wind_price: str = "pzu"


@dataclass
class TriageState:
    """What triage remembers between ticks. Keys are asset ids (alerts: 'asset_id|code', issued: 'date|crew')."""

    open_work_orders: dict[str, str] = field(default_factory=dict)  # asset_id -> action_id
    issued: dict[str, int] = field(default_factory=dict)  # "YYYY-MM-DD|crew" -> work orders issued that day
    active_alerts: dict[str, str] = field(default_factory=dict)  # "asset_id|code" -> action_id
    open_reviews: dict[str, str] = field(default_factory=dict)  # asset_id -> action_id
    offline_since: dict[str, str] = field(default_factory=dict)  # asset_id -> window start (UTC ISO)


@dataclass
class TriageResult:
    work_orders: list[Action]
    alerts: list[Action]
    human_reviews: list[Action]
    backlog: list[dict[str, Any]]  # warranted work orders that did not fit today's K, ranked
    curtailment_log: list[dict[str, Any]]  # curtailed-energy entries (no work order)
    offline_opened: list[dict[str, Any]]  # windows opened this tick
    offline_closed: list[dict[str, Any]]  # windows closed this tick (asset reported again)
    state: TriageState

    @property
    def actions(self) -> list[Action]:
        return [*self.work_orders, *self.alerts, *self.human_reviews]

    def om_queue(self, asset_kind: Optional[str] = None) -> list[dict[str, Any]]:
        """The O&M queue: issued work orders and backlog, ordered by triage score (lei), highest first.

        Judge-unknown turbines (remote check, no score) come last. asset_kind: optional "turbine" / "site" filter.
        """
        rows = [w.payload for w in self.work_orders] + list(self.backlog)
        if asset_kind is not None:
            rows = [r for r in rows if r.get("asset_kind") == asset_kind]
        return sorted(rows, key=lambda r: (bool(r.get("judge_unknown")), -float(r["score"]), r["asset_id"]))


def triage_score(p_fault: float, lost_kwh_day: float, lost_flex_mwh_day: float, price_lei_mwh: float,
                 v_flex_lei_mwh: float, visit_days: float) -> float:
    """Plan §3.5 score, in lei: P_fault × (lost kWh × π + lost flex MWh × v_flex) × D."""
    value = max(0.0, lost_kwh_day) / 1000.0 * price_lei_mwh + max(0.0, lost_flex_mwh_day) * v_flex_lei_mwh
    return float(p_fault * value * visit_days)


def lost_kwh(q50: Sequence[float], actual: Sequence[float], step_minutes: int = 15) -> float:
    """Σ (P50 − actual)+ · Δt over observed steps (NaN = not observed)."""
    e = np.asarray(q50, dtype=float)
    a = np.asarray(actual, dtype=float)
    n = min(len(e), len(a))
    m = np.isfinite(a[:n])
    return float(np.clip(e[:n][m] - a[:n][m], 0.0, None).sum() * step_minutes / 60.0)


# --- Wind D2 (card #167): turbine work orders by lei per day ------------------------------------------------
WIND_P_MODEL, WIND_P_SUBSTITUTE = "windnbm", "windmon_rules"  # M6 and its §3 substitute (ModelId values)
WIND_POWER_MODEL, WIND_POWER_SUBSTITUTE = "windpower", "binned_curve"  # M5 and its substitute
LOCAL_TZ = "Europe/Bucharest"
PRICES_PATH = Path(__file__).resolve().parents[2] / "data" / "raw" / "prices.parquet"
_NO_JUDGE_REASON = "no Judge verdict: model used at its validated strength"


@dataclass
class WindInputs:
    """Optional turbine inputs for triage, keyed by turbine id. Missing entries use the fallbacks below."""

    m5: Mapping[str, Any] = field(default_factory=dict)  # M5 WindPower Prediction (card #164), next 1-24 h
    m5_binned: Mapping[str, Any] = field(default_factory=dict)  # binned power curve Prediction (M5's substitute)
    m6_rules: Mapping[str, Any] = field(default_factory=dict)  # WindMon rules output (M6's substitute)
    judge: Any = None  # adapter VerdictSource; None = the adapter's default (NoJudge unless configured)
    pzu_next24: Optional[float] = None  # lei/MWh override of the mean PZU over the next 24 h


def wind_triage_score(p_fault: float, lei_day: float, days_to_visit: float) -> float:
    """Spec W step 5: P(fault) × lei_per_day × days until a crew can get there (lei; no flexibility term)."""
    return float(p_fault * max(0.0, lei_day) * days_to_visit)


def lei_per_day(lost_mwh: float, pzu_mean_lei_mwh: float) -> float:
    """Lost MWh (per 24 h) × mean PZU over the next 24 h."""
    return float(max(0.0, lost_mwh) * pzu_mean_lei_mwh)


# The one door to the Judge: the C1 adapter (card #157). Once #157 is on main the import succeeds by itself; to drop
# the stand-in, make the body of `_adapter_validated` just `from sentinel.judge.adapter import validated`.
@lru_cache(maxsize=1)
def _adapter_validated() -> Callable[..., Any]:
    try:
        from sentinel.judge.adapter import validated  # card #157
    except ImportError:
        return _no_judge_validated
    return validated


def _no_judge_validated(asset_id: Any, model: str, day: str, *, raw: Any, substitutes: Any = None,
                        judge: Any = None) -> SimpleNamespace:
    """Stand-in for the adapter's no-Judge mode until #157 merges: raw passes as "use" / "no_judge" (§10.1)."""
    base = dict(asset_id=str(asset_id), model=model, day=day, verdict="use", p_valid=None, trust=None, n_eff=0.0,
                band_scale=1.0, substitute=None, rule="no_judge", judge_version=None, raw=raw)
    if raw is None:
        return SimpleNamespace(**base, source_model=None, known=False, q10=None, q50=None, q90=None, cls=None,
                               score=None, fault_probs=None, reasons=())
    arr = (lambda k: None if getattr(raw, k, None) is None else np.asarray(getattr(raw, k), dtype=float))
    score = getattr(raw, "anomaly_score", None)
    return SimpleNamespace(**base, source_model=model, known=True, q10=arr("q10"), q50=arr("q50"), q90=arr("q90"),
                           cls=getattr(raw, "anomaly_class", None), score=None if score is None else float(score),
                           fault_probs=getattr(raw, "fault_probs", None), reasons=(_NO_JUDGE_REASON,))


def judge_validated(asset_id: Any, model: str, day: str, *, raw: Any,
                    substitutes: Optional[Mapping[str, Any]] = None, judge: Any = None) -> Any:
    """`sentinel.judge.adapter.validated` (§10.1): verdict + the numbers to act on (band scaled, substitute swapped
    in on dont_use; known=False = no reliable source). Every turbine number W4 acts on comes through here."""
    subs = {k: v for k, v in (substitutes or {}).items() if v is not None} or None
    return _adapter_validated()(asset_id, model, day, raw=raw, substitutes=subs, judge=judge)


def local_day(t: str) -> str:
    """Local (Europe/Bucharest) date of a UTC tick: the `judge_verdicts.day` key."""
    from zoneinfo import ZoneInfo
    return parse_utc(t).astimezone(ZoneInfo(LOCAL_TZ)).date().isoformat()


def _step_index(pred: Any, t: str) -> int:
    """Index of the step of `pred` that starts at / contains tick t (may be < 0 or ≥ horizon)."""
    dt = (parse_utc(t) - parse_utc(pred.t0)).total_seconds() / 60.0
    return int(np.floor(dt / pred.step_minutes))


def _next24_mwh(q50: Any, pred: Any, t: str) -> Optional[float]:
    """Σ P50·Δt over the 24 h from tick t, in MWh (kW in). A shorter remaining horizon is scaled to 24 h."""
    if q50 is None:
        return None
    q = np.asarray(q50, dtype=float)
    n24 = int(round(24 * 60 / pred.step_minutes))
    i = min(max(0, _step_index(pred, t)), len(q))
    fut = q[i:i + n24]
    fut = fut[np.isfinite(fut)]
    if not fut.size:  # tick past the horizon: the whole window is the best estimate of the next day
        fut = q[np.isfinite(q)]
    if not fut.size:
        return None
    return float(fut.mean() * 24.0 / 1000.0)


def m5_p50_next24_mwh(a: Assessment, t: str, wind: WindInputs, day: str, m6: Any) -> tuple[float, str]:
    """Lost MWh of a stopped turbine = what it would have produced: M5 P50 over the next 24 h, and its source.

    M5 WindPower (card #164) is not on main yet. Order: M5 (through the adapter, binned curve as its substitute)
    -> binned curve -> M6 expected power (the NBM's q50, already validated) -> the class-default loss.
    """
    aid = a.asset_id
    m5, binned = wind.m5.get(aid), wind.m5_binned.get(aid)
    if m5 is not None:
        v = judge_validated(aid, WIND_POWER_MODEL, day, raw=m5, substitutes={WIND_POWER_SUBSTITUTE: binned},
                            judge=wind.judge)
        mwh = _next24_mwh(v.q50, m5, t) if v.known else None
        if mwh is not None:
            return mwh, str(v.source_model)
    if binned is not None:
        mwh = _next24_mwh(binned.q50, binned, t)
        if mwh is not None:
            return mwh, WIND_POWER_SUBSTITUTE
    if m6 is not None and m6.known and a.prediction is not None:
        mwh = _next24_mwh(m6.q50, a.prediction, t)
        if mwh is not None:
            return mwh, f"{m6.source_model}_expected"  # stand-in for M5 until #164 lands
    return a.lost_kwh_day / 1000.0, "class_default"


def m6_lost_mwh_last24h(a: Assessment, t: str, m6: Any) -> tuple[float, str, float]:
    """M6 Σ (expected − actual)+·Δt over the observed steps of the last 24 h, as MWh per 24 h.

    Returns (MWh/day, source, observed hours). Fewer than 24 observed hours are scaled to 24 h; no observed step
    (or no actual) -> the assessment's class-default loss.
    """
    pred = a.prediction
    if pred is None or a.actual is None or m6 is None or not m6.known or m6.q50 is None:
        return a.lost_kwh_day / 1000.0, "class_default", 0.0
    e = np.asarray(m6.q50, dtype=float)
    act = np.asarray(a.actual, dtype=float)
    n = min(len(e), len(act))
    n24 = int(round(24 * 60 / pred.step_minutes))
    i = _step_index(pred, t)
    hi = n if i <= 0 else min(i, n)
    lo = max(0, hi - n24)
    seg_e, seg_a = e[lo:hi], act[lo:hi]
    m = np.isfinite(seg_a) & np.isfinite(seg_e)
    if not m.any():
        return a.lost_kwh_day / 1000.0, "class_default", 0.0
    dt_h = pred.step_minutes / 60.0
    obs_h = float(m.sum() * dt_h)
    lost = float(np.clip(seg_e[m] - seg_a[m], 0.0, None).sum() * dt_h / 1000.0)
    return lost * 24.0 / obs_h, f"{m6.source_model}_measured", obs_h


@lru_cache(maxsize=2)
def _pzu_rows(path: str) -> Optional[tuple[np.ndarray, np.ndarray, np.ndarray]]:
    """(start epoch s, length s, lei/MWh) of every OPCOM PZU row in a prices parquet, sorted by time."""
    try:
        import polars as pl
        p = pl.read_parquet(path).filter(pl.col("market") == "opcom_pzu").sort("ts")
    except Exception:
        return None
    start = p["ts"].dt.epoch("s").to_numpy().astype(np.int64)
    length = np.where(p["resolution"].to_numpy() == "PT60M", 3600, 900).astype(np.int64)
    return start, length, p["price_lei_mwh"].to_numpy().astype(float)


def _pzu_mean_direct(lo: datetime, hi: datetime, path: str) -> Optional[float]:
    """Time-weighted mean PZU over [lo, hi) straight from the parquet (an hourly row counts for its whole hour)."""
    rows = _pzu_rows(path)
    if rows is None:
        return None
    start, length, price = rows
    a, b = int(lo.timestamp()), int(hi.timestamp())
    w = np.clip(np.minimum(start + length, b) - np.maximum(start, a), 0, None).astype(float)
    m = (w > 0) & np.isfinite(price)
    if w[m].sum() < 0.5 * (b - a):  # less than half the window priced: not covered
        return None
    return float((price[m] * w[m]).sum() / w[m].sum())


def _pzu_mean_module(lo: datetime, hi: datetime) -> Optional[float]:
    """Mean PZU over [lo, hi) from sentinel.data.prices (card #158); None while that module is not on main."""
    try:
        from sentinel.data.prices import load_prices  # card #158
    except ImportError:
        return None
    df = load_prices(lo, hi)
    if df.height < 48:  # less than half the window priced
        return None
    return float(df["pzu_lei_mwh"].mean())


@lru_cache(maxsize=4096)
def pzu_next24_mean(t: str, default: float = 450.0, path: str = str(PRICES_PATH)) -> tuple[float, str]:
    """Mean OPCOM PZU (lei/MWh) over the 24 h from tick t, and its source: 'opcom_pzu' (real RO) or 'config'.

    Reads sentinel/data/prices.py (#158) when it is on main, else data/raw/prices.parquet directly. Dates the file
    does not cover (synthetic years, the future) fall back to `default` (TriageConfig.price_lei_mwh).
    """
    lo = parse_utc(t)
    lo = lo.replace(minute=lo.minute - lo.minute % 15, second=0, microsecond=0)  # the tick's quarter-hour
    hi = lo + timedelta(hours=24)
    v = None
    if path == str(PRICES_PATH):
        try:
            v = _pzu_mean_module(lo, hi)
        except Exception:
            v = None
    if v is None:
        v = _pzu_mean_direct(lo, hi, path)
    return (float(v), "opcom_pzu") if v is not None and np.isfinite(v) else (float(default), "config")


def wind_economics(a: Assessment, t: str, d: float, wind: WindInputs, pzu: tuple[float, str]) -> dict[str, Any]:
    """Score and work-order fields of one turbine candidate (spec W step 5, judge-design §10.4)."""
    day = local_day(t)
    m6 = judge_validated(a.asset_id, WIND_P_MODEL, day, raw=a.prediction,
                         substitutes={WIND_P_SUBSTITUTE: wind.m6_rules.get(a.asset_id)}, judge=wind.judge)
    unknown = not m6.known or m6.score is None
    obs_h = None
    if a.fault == WindAnomaly.FAULT_STOP.value:
        lost_mwh, lost_src = m5_p50_next24_mwh(a, t, wind, day, m6)
    else:
        lost_mwh, lost_src, obs_h = m6_lost_mwh_last24h(a, t, m6)
    price, price_src = pzu
    lei = lei_per_day(lost_mwh, price)
    p = None if unknown else float(m6.score)
    score = 0.0 if p is None else wind_triage_score(p, lei, d)
    out: dict[str, Any] = {
        "score": score, "triage_score": round(score, 2), "lei_per_day": round(lei, 2),
        "lost_mwh_24h": round(lost_mwh, 4), "lost_src": lost_src, "pzu_next24_lei_mwh": round(price, 2),
        "price_src": price_src, "p_fault": p, "p_fault_src": m6.source_model,
        "judge_verdict": m6.verdict, "judge_rule": m6.rule, "judge_reasons": list(m6.reasons),
        "judge_unknown": unknown,
    }
    if obs_h is not None:
        out["lost_obs_h"] = round(obs_h, 2)
    if unknown:  # §10.4: no reliable source -> no crew; a remote check, ranked after the scored orders
        out["crew_type"] = CrewType.REMOTE.value
    return out


# --- audit --------------------------------------------------------------------------------------------------
def compact_prediction(a: Assessment) -> dict[str, Any]:
    p = a.prediction
    if p is None:
        return {"model": None}
    dt_h = p.step_minutes / 60.0
    out: dict[str, Any] = {
        "model": p.model,
        "t0": p.t0,
        "q50_sum_kwh": round(sum(p.q50) * dt_h, 3),
        "q10_sum_kwh": round(sum(p.q10) * dt_h, 3),
        "q90_sum_kwh": round(sum(p.q90) * dt_h, 3),
        "ood_score": p.ood_score,
        "member_std": p.member_std,
        "rel_width": round(rel_width(p), 4),
        "loss_frac": round(a.loss_frac, 4),
        "loss_measured": a.measured,
    }
    if p.fault_probs:
        out["p_fault"] = round(p.p_fault, 4)
        out["top_fault"] = p.top_fault
    if p.anomaly_class is not None:
        out["anomaly_class"] = p.anomaly_class
        out["anomaly_score"] = p.anomaly_score
    return out


def audit_for(a: Assessment, note: str = "") -> AuditRecord:
    p = a.prediction
    seen = {
        "prediction": p.to_dict() if p is not None else None,
        "actual": None if a.actual is None else [None if not np.isfinite(v) else float(v) for v in a.actual],
        "telemetry_age_min": a.telemetry_age_min,
    }
    return AuditRecord(
        model_version=p.model_version if p is not None else "none",
        input_hash=input_hash(seen),
        prediction=compact_prediction(a),
        threshold=a.threshold,
        status=a.status.status,
        note=note or a.status.reason,
    )


def _action(kind: ActionType, a: Assessment, t: str, payload: dict[str, Any], cfg: TriageConfig,
            code: str) -> Action:
    aid = f"{kind.value[:2]}-" + input_hash([kind.value, a.asset_id, t, code, cfg.run_id, cfg.policy])
    return Action(action_id=aid, type=kind.value, t=t, policy=cfg.policy, run_id=cfg.run_id, payload=payload,
                  audit=audit_for(a))


# --- triage -------------------------------------------------------------------------------------------------
def triage(
    assessments: Iterable[Assessment],
    t: str,
    state: Optional[TriageState] = None,
    config: Optional[TriageConfig] = None,
    *,
    visit_days: Optional[Mapping[str, float]] = None,
    wind: Optional[WindInputs] = None,
) -> TriageResult:
    """Turn one tick's assessments into actions. `t` = the tick (UTC ISO); the K budget is per UTC day of `t`.

    visit_days: optional per-asset override of D_i (days until the next possible visit).
    wind: optional turbine inputs (M5, substitutes, Judge source, PZU override) for the lei-per-day ranking.
    """
    cfg = config or TriageConfig()
    st = copy.deepcopy(state) if state is not None else TriageState()
    day = t[:10]
    visit_days = visit_days or {}
    alerts: list[Action] = []
    reviews: list[Action] = []
    curtail: list[dict[str, Any]] = []
    opened: list[dict[str, Any]] = []
    closed: list[dict[str, Any]] = []
    candidates: dict[str, list[tuple[float, Assessment, float, dict[str, Any]]]] = {}
    wind = wind or WindInputs()
    pzu: Optional[tuple[float, str]] = None

    for a in assessments:
        s = a.status
        kind = a.asset_kind

        # Offline windows: open once, close when the asset reports again. One alert per turbine window.
        if a.offline:
            if a.asset_id not in st.offline_since:
                st.offline_since[a.asset_id] = t
                opened.append({"asset_id": a.asset_id, "asset_kind": kind, "start": t, "reason": s.reason})
                if kind == "turbine":
                    alerts.append(_action(ActionType.ALERT, a, t, {
                        "asset_id": a.asset_id, "asset_kind": kind, "severity": Severity.WARNING.value,
                        "code": "offline", "message": f"{a.asset_id} offline since {t} ({s.reason}); one window "
                                                      f"until it reports again. Remote check, no crew.",
                        "park_id": a.group_id, "route": CrewType.REMOTE.value}, cfg, "offline"))
        elif a.asset_id in st.offline_since:
            closed.append({"asset_id": a.asset_id, "asset_kind": kind, "start": st.offline_since.pop(a.asset_id),
                           "end": t})

        # Alerts from qualify (icing = safety, overheating = warning), once while the condition lasts.
        code_now = a.alert[1] if a.alert else None
        for key in [k for k in st.active_alerts if k.split("|", 1)[0] == a.asset_id]:
            if key.split("|", 1)[1] != code_now:
                del st.active_alerts[key]
        if a.alert:
            sev, code, msg = a.alert
            key = f"{a.asset_id}|{code}"
            if key not in st.active_alerts:
                act = _action(ActionType.ALERT, a, t, {"asset_id": a.asset_id, "asset_kind": kind, "severity": sev,
                                                       "code": code, "message": msg, "group_id": a.group_id},
                              cfg, code)
                st.active_alerts[key] = act.action_id
                alerts.append(act)

        # Human review: once per episode; no work order.
        if s.status == Status.HUMAN_REVIEW.value:
            if a.asset_id not in st.open_reviews:
                act = _action(ActionType.HUMAN_REVIEW, a, t, {
                    "asset_id": a.asset_id, "asset_kind": kind, "reason": s.reason, "detail": s.detail,
                    "group_id": a.group_id}, cfg, s.reason)
                st.open_reviews[a.asset_id] = act.action_id
                reviews.append(act)
        else:
            st.open_reviews.pop(a.asset_id, None)

        # Curtailment: log the energy, never a work order.
        if a.curtailed_kwh is not None or s.reason.startswith("curtailed"):
            curtail.append({"t": t, "asset_id": a.asset_id, "park_id": a.group_id, "reason": s.reason,
                            "curtailed_kwh": a.curtailed_kwh,
                            "window_start": a.prediction.t0 if a.prediction is not None else None})

        # Work orders: repaired assets (eligible, nothing to do) close theirs and rejoin the pool.
        if a.crew_type is None:
            if s.status == Status.ELIGIBLE.value:
                st.open_work_orders.pop(a.asset_id, None)
            continue
        if a.asset_id in st.open_work_orders:
            continue
        d = float(visit_days.get(a.asset_id, cfg.visit_days.get(a.crew_type, 1.0)))
        if kind == "turbine":
            if pzu is None:  # one price per tick
                pzu = ((float(wind.pzu_next24), "override") if wind.pzu_next24 is not None
                       else pzu_next24_mean(t, cfg.price_lei_mwh) if cfg.wind_price == "pzu"
                       else (cfg.price_lei_mwh, "config"))
            extra = wind_economics(a, t, d, wind, pzu)
            candidates.setdefault(extra.pop("crew_type", a.crew_type), []).append((extra.pop("score"), a, d, extra))
            continue
        v_flex = cfg.v_flex_lei_mwh.get(kind, 0.0)
        score = triage_score(a.p_fault, a.lost_kwh_day, a.lost_flex_mwh_day, cfg.price_lei_mwh, v_flex, d)
        candidates.setdefault(a.crew_type, []).append((score, a, d, {}))

    work_orders: list[Action] = []
    backlog: list[dict[str, Any]] = []
    for crew in sorted(candidates):
        # Judge-unknown turbines (remote check, no score) rank after everything that has a score.
        ranked = sorted(candidates[crew], key=lambda x: (bool(x[3].get("judge_unknown")), -x[0],
                                                         -x[1].lost_kwh_day, x[1].asset_id))
        k = cfg.k_per_crew.get(crew)
        used = st.issued.get(f"{day}|{crew}", 0)
        budget = len(ranked) if k is None else max(0, k - used)
        for i, (score, a, d, extra) in enumerate(ranked):
            if extra:  # turbine: lei per day from M5/M6 × PZU, P(fault) from the adapter
                lost_kwh_day, lost_lei, p_fault = extra["lost_mwh_24h"] * 1000.0, extra["lei_per_day"], extra["p_fault"]
            else:
                lost_kwh_day, lost_lei, p_fault = a.lost_kwh_day, a.lost_kwh_day / 1000.0 * cfg.price_lei_mwh, a.p_fault
            row = {
                "asset_id": a.asset_id, "asset_kind": a.asset_kind, "fault": a.fault, "crew_type": crew,
                "score": round(score, 2), "lost_kwh_day": round(lost_kwh_day, 2),
                "lost_lei_day": round(lost_lei, 2), "lost_flex_mwh_day": round(a.lost_flex_mwh_day, 6),
                "mw_at_risk": round(a.mw_at_risk, 6), "p_fault": None if p_fault is None else round(p_fault, 4),
                "visit_days": d, "status": a.status.status, "reason": a.status.reason,
                "headroom_factor": a.status.headroom_factor,
                "detail": a.status.detail, ("park_id" if a.asset_kind == "turbine" else "feeder_id"): a.group_id,
                **{k_: v for k_, v in extra.items() if k_ != "p_fault"},
            }
            if i < budget:
                row["rank"] = used + i + 1
                act = _action(ActionType.WORK_ORDER, a, t, row, cfg, crew)
                st.open_work_orders[a.asset_id] = act.action_id
                work_orders.append(act)
            else:
                row["backlog_rank"] = i - budget + 1
                backlog.append(row)
        issued_now = min(budget, len(ranked))
        if issued_now:
            st.issued[f"{day}|{crew}"] = used + issued_now

    return TriageResult(work_orders=work_orders, alerts=alerts, human_reviews=reviews, backlog=backlog,
                        curtailment_log=curtail, offline_opened=opened, offline_closed=closed, state=st)


def offline_windows(records: Iterable[tuple[str, str, bool]]) -> list[dict[str, Any]]:
    """Merge (asset_id, t, offline) ticks, in time order, into windows {asset_id, start, end}; end None = still open.

    Offline-ness is consecutive in tick order: a run of offline ticks is one window, however long.
    """
    open_: dict[str, str] = {}
    out: list[dict[str, Any]] = []
    for aid, t, off in records:
        if off and aid not in open_:
            open_[aid] = t
        elif not off and aid in open_:
            out.append({"asset_id": aid, "start": open_.pop(aid), "end": t})
    out.extend({"asset_id": aid, "start": s, "end": None} for aid, s in open_.items())
    return out
