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
"""

from __future__ import annotations

import copy
from dataclasses import dataclass, field
from typing import Any, Iterable, Mapping, Optional, Sequence

import numpy as np

from sentinel.contracts import Action, ActionType, AuditRecord, CrewType, Severity, Status, input_hash

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
) -> TriageResult:
    """Turn one tick's assessments into actions. `t` = the tick (UTC ISO); the K budget is per UTC day of `t`.

    visit_days: optional per-asset override of D_i (days until the next possible visit).
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
    candidates: dict[str, list[tuple[float, Assessment, float]]] = {}

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
                                                      f"until it reports again.",
                        "park_id": a.group_id}, cfg, "offline"))
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
        v_flex = cfg.v_flex_lei_mwh.get(kind, 0.0)
        score = triage_score(a.p_fault, a.lost_kwh_day, a.lost_flex_mwh_day, cfg.price_lei_mwh, v_flex, d)
        candidates.setdefault(a.crew_type, []).append((score, a, d))

    work_orders: list[Action] = []
    backlog: list[dict[str, Any]] = []
    for crew in sorted(candidates):
        ranked = sorted(candidates[crew], key=lambda x: (-x[0], -x[1].lost_kwh_day, x[1].asset_id))
        k = cfg.k_per_crew.get(crew)
        used = st.issued.get(f"{day}|{crew}", 0)
        budget = len(ranked) if k is None else max(0, k - used)
        for i, (score, a, d) in enumerate(ranked):
            lost_lei = a.lost_kwh_day / 1000.0 * cfg.price_lei_mwh
            row = {
                "asset_id": a.asset_id, "asset_kind": a.asset_kind, "fault": a.fault, "crew_type": crew,
                "score": round(score, 2), "lost_kwh_day": round(a.lost_kwh_day, 2),
                "lost_lei_day": round(lost_lei, 2), "lost_flex_mwh_day": round(a.lost_flex_mwh_day, 6),
                "mw_at_risk": round(a.mw_at_risk, 6), "p_fault": round(a.p_fault, 4), "visit_days": d,
                "status": a.status.status, "reason": a.status.reason, "headroom_factor": a.status.headroom_factor,
                "detail": a.status.detail, ("park_id" if a.asset_kind == "turbine" else "feeder_id"): a.group_id,
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
