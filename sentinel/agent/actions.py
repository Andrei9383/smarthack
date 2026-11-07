"""Agent actions -> contract `Action` (+ audit record) -> SQLite rows (card #17; team/contracts/actions.md, replay.md).

Every decision the loop takes is an `Action` with an `AuditRecord`. `rows_for(action, ts)` maps an action's
payload onto the #38 tables (work_orders, alerts, review_queue, gis_flags, reports); battery schedules, feeder
commitments and zone demand are per-step series the loop writes itself, keyed by the same run_id. Column names
here follow #38; `audit.AuditDB` maps them onto older drafts via aliases and drops what the schema lacks.
"""

from __future__ import annotations

from typing import Any, Mapping, Optional, Sequence

from sentinel.contracts import Action, ActionType, AuditRecord, CrewType, Prediction, input_hash

from .audit import Ts


def action_id(kind: str, *parts: Any) -> str:
    """Deterministic, unique per (kind, parts): e.g. bs-1a2b3c4d5e6f7a8b."""
    return f"{kind[:2]}-" + input_hash([kind, *parts])


def compact(pred: Optional[Prediction]) -> dict[str, Any]:
    """Compact prediction for the audit record (sums, p_fault, top class), never the full curves."""
    if pred is None:
        return {}
    dt = pred.step_minutes / 60.0
    out: dict[str, Any] = {"model": pred.model, "model_version": pred.model_version, "t0": pred.t0,
                           "q10_sum": round(sum(pred.q10) * dt, 3), "q50_sum": round(sum(pred.q50) * dt, 3),
                           "q90_sum": round(sum(pred.q90) * dt, 3), "ood_score": pred.ood_score,
                           "member_std": pred.member_std, "unit": pred.unit}
    if pred.fault_probs:
        out["p_fault"] = round(pred.p_fault, 4)
        out["top_fault"] = pred.top_fault
    if pred.anomaly_class is not None:
        out["anomaly_class"] = pred.anomaly_class
        out["anomaly_score"] = pred.anomaly_score
    return out


def audit(model_version: str, inputs: Any, prediction: Optional[dict[str, Any]] = None,
          threshold: Optional[float] = None, status: str = "", schedule: Optional[Sequence[float]] = None,
          note: str = "") -> AuditRecord:
    return AuditRecord(model_version=model_version, input_hash=input_hash(inputs), prediction=prediction or {},
                       threshold=threshold, status=status,
                       schedule=None if schedule is None else [round(float(x), 4) for x in schedule], note=note)


def make(kind: ActionType | str, t: str, policy: str, run_id: str, payload: dict[str, Any], rec: AuditRecord,
         *id_parts: Any, meta: Optional[dict[str, Any]] = None) -> Action:
    kind = ActionType(kind).value
    return Action(action_id=action_id(kind, run_id, policy, t, *id_parts), type=kind, t=t, policy=policy,
                  run_id=run_id, payload=payload, audit=rec, meta=dict(meta or {}))


def battery_schedule(t: str, policy: str, run_id: str, site_id: str, feeder_id: str, batt_kw: Sequence[float],
                     rec: AuditRecord, **extra: Any) -> Action:
    payload = {"site_id": site_id, "feeder_id": feeder_id, "batt_kw": [round(float(x), 4) for x in batt_kw],
               **extra}
    return make(ActionType.BATTERY_SCHEDULE, t, policy, run_id, payload, rec, site_id, extra.get("revision", 0))


def feeder_commitment(t: str, policy: str, run_id: str, feeder_id: str, nominal: Sequence[float],
                      available: Sequence[float], committed: Sequence[float], rec: AuditRecord,
                      **extra: Any) -> Action:
    r4 = lambda xs: [round(float(x), 4) for x in xs]  # noqa: E731
    payload = {"feeder_id": feeder_id, "nominal": r4(nominal), "available": r4(available),
               "committed": r4(committed), **extra}
    return make(ActionType.FEEDER_COMMITMENT, t, policy, run_id, payload, rec, feeder_id)


def _crew_action(crew: Optional[str], kind: str, fault: Optional[str]) -> str:
    """#38 work_orders.action: truck | remote_check | inspection. Sensor / anemometer -> remote, never a truck."""
    if crew == CrewType.REMOTE.value:
        return "remote_check"
    if kind == "turbine" and fault in ("underperformance", "overheating"):
        return "inspection"
    return "truck"


def rows_for(action: Action, ts: int) -> dict[str, list[dict[str, Any]]]:
    """Payload -> table rows for the action types that have their own table (replay.md)."""
    p = action.payload
    base = {"action_id": action.action_id, "id": action.action_id, "run_id": action.run_id}
    t = Ts(ts)
    if action.type == ActionType.WORK_ORDER.value:
        kind = p.get("asset_kind", "site")
        fault = p.get("fault")
        return {"work_orders": [{
            **base, "asset_id": p["asset_id"], "asset_kind": kind, "crew_type": p.get("crew_type"),
            "action": _crew_action(p.get("crew_type"), kind, fault), "ts": t, "fault": fault,
            "confidence": float(p.get("p_fault", 0.0) or 0.0), "lost_kwh_day": p.get("lost_kwh_day"),
            "lost_flex_mwh_day": p.get("lost_flex_mwh_day"),
            "lei_at_risk": p.get("lost_lei_day"), "mw_at_risk": p.get("mw_at_risk"),
            "days_to_visit": int(round(float(p.get("visit_days", 1) or 1))), "score": float(p.get("score", 0.0)),
            "rank_k": p.get("rank"), "status": "open", "evidence_json": p}]}
    if action.type == ActionType.ALERT.value:
        kind = p.get("asset_kind")
        park = p.get("park_id") or (p.get("group_id") if kind == "turbine" else None)
        feeder = p.get("feeder_id") or (p.get("group_id") if kind == "site" else None)
        return {"alerts": [{
            **base, "asset_id": None if kind in ("park", "feeder") else p["asset_id"], "park_id": park,
            "feeder_id": feeder, "ts": t, "kind": p["code"], "severity": p["severity"],
            "creates_work_order": int(bool(p.get("creates_work_order", False))),
            "curtailed_mwh": p.get("curtailed_mwh"), "message": p["message"], "evidence_json": p}]}
    if action.type == ActionType.HUMAN_REVIEW.value:
        pred = action.audit.prediction or {}
        return {"review_queue": [{
            **base, "asset_id": p["asset_id"], "ts": t, "reason": p["reason"], "resolved": 0,
            "ood_score": pred.get("ood_score"), "member_std": pred.get("member_std"),
            "interval_width": pred.get("rel_width")}]}
    if action.type == ActionType.GIS_FLAG.value:
        return {"gis_flags": [{
            **base, "section_id": p["section_id"], "recorded_feeder": p["recorded_feeder"], "suggested_feeder": p["suggested_feeder"],
            "recorded_phase": p["recorded_phase"], "suggested_phase": p["suggested_phase"], "tau": p["tau"],
            "confidence": p.get("confidence"), "geometry_json": p.get("geometry")}]}
    if action.type == ActionType.PARK_REPORT.value:
        return {"reports": [{
            **base, "park_id": p["park_id"], "ts": t, "production_mwh": p["energy_mwh"],
            "expected_mwh": p["expected_mwh"], "availability_pct": round(100 * p["availability"], 2),
            "curtailed_mwh": p["curtailed_mwh"], "data_quality_pct": p.get("data_quality_pct"),
            "issues_json": p.get("issues", []), "body_json": p}]}
    return {}


def site_inputs(site_row: Mapping[str, Any], preds: Mapping[str, Optional[Prediction]], actual_digest: str,
                prices_digest: str, g_kw: float) -> dict[str, Any]:
    """What a site-day decision depends on (both policies hash exactly this, so identical inputs = equal hashes)."""
    return {"site_id": site_row["site_id"], "feeder_id": site_row.get("feeder_id"),
            "preds": {k: (None if p is None else p.to_dict()) for k, p in sorted(preds.items())},
            "actual": actual_digest, "prices": prices_digest, "g_kw": round(float(g_kw), 6)}
