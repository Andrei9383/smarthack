"""The agent loop: perceive -> estimate -> qualify -> decide -> act -> log, every 15 min, for one policy.

    comps = default_components(world)          # real modules where they import, built-in fallbacks otherwise
    res = run_policy(world, comps, LoopConfig(policy="sentinel", run_id="r1"), sink=AuditDB(...))

Per day: predictions (estimate) -> qualify at t0 -> per-feeder day-ahead plan + P10 commitment -> every tick:
re-qualify, triage, execute the plan through the battery model with a feasibility clip that never goes below the
comfort reserve, optional hourly re-solve. Turbines get statuses, work orders, curtailment alerts and a park report
but never enter the LP. Policies: `baseline` (self-consumption rule), `threshold` (per-home export threshold) and
`sentinel`; all see identical inputs (`PolicyResult.input_hashes`), only decide/act differ.

Observation vs physics: every telemetry read goes through `world.observed_site_day`, while `world.site_day` drives
the physics. A pool site dropped only for missing telemetry keeps running today's schedule on its device; it rejoins
the pool once its telemetry is fresh and it re-qualifies; at commit time the planned reduction from sites expected to
be blind (same steps of the trailing day, +-1 h) is not committed.
"""

from __future__ import annotations

import copy
import hashlib
import sys
import traceback
from dataclasses import dataclass, field
from typing import Any, Callable, Mapping, Optional, Sequence

import numpy as np

from wattsentry.contracts import (
    DAY_STEPS,
    Action,
    ActionType,
    AssetStatus,
    BatterySpec,
    Commitment,
    CrewType,
    FeederPlan,
    FeederSpec,
    Prediction,
    PvFault,
    Severity,
    Status,
    WindAnomaly,
    Window,
    input_hash,
    is_error_code,
    self_consumption_step,
)

from . import actions as A
from .audit import Ts, iso, local_dt, parse_iso
from .world import SiteDay, World

DT = 0.25
EPS = 1e-6
BLIND_REASONS = ("stale_telemetry", "data_gap")  # qualify reasons that mean "not seen", not "not fit"
STALE_AGE_S = 3600.0  # qualify DEFAULT_THRESHOLDS["site"]["stale_minutes"] = 60: older telemetry = out of the pool
OBSERVER_SLOTS = ("judge",)  # slots that record but never change a decision (not listed as fallbacks)
BASELINE_ALIASES = {"baseline": "baseline", "self_consumption": "baseline", "threshold": "threshold",
                    "sentinel": "sentinel"}


def log(msg: str) -> None:
    print(f"[agent] {msg}", file=sys.stderr, flush=True)


# =============================================================================================== fallbacks
def _pv_factor(status: Optional[AssetStatus]) -> float:
    if status is None or not status.in_pool:
        return 1.0
    return float(status.headroom_factor)


def _scaled(pred: Prediction, f: float) -> Prediction:
    if abs(f - 1.0) < 1e-9:
        return pred
    p = copy.copy(pred)
    p.q10, p.q50, p.q90 = [v * f for v in pred.q10], [v * f for v in pred.q50], [v * f for v in pred.q90]
    return p


def _clip_step(b: BatterySpec, e: float, req: float, floor: Optional[float] = None) -> tuple[float, float]:
    """BatterySpec.step plus a discharge floor (comfort reserve for planned schedules)."""
    p, e_next = b.step(e, req)
    fl = b.e_min if floor is None else floor
    fl = min(fl, e)  # already below the reserve: no further discharge, not forced up
    if p < 0 and e_next < fl - 1e-12:
        p = -max(0.0, (e - fl)) * b.eta_discharge / DT
        e_next = e + p * DT / b.eta_discharge
    return p, e_next


def _rest_arrays(rest: Any, horizon: int) -> tuple[np.ndarray, np.ndarray]:
    if rest is None:
        z = np.zeros(horizon)
        return z, z
    if hasattr(rest, "net_kw"):
        return np.asarray(rest.net_kw, float), np.asarray(getattr(rest, "worst_kw", rest.net_kw), float)
    if isinstance(rest, tuple) and len(rest) == 2:
        return np.asarray(rest[0], float), np.asarray(rest[1], float)
    a = np.asarray(rest, float)
    return a, a


def _battery_of(row: Mapping[str, Any]) -> Optional[BatterySpec]:
    b = row.get("battery")
    if isinstance(b, BatterySpec):
        return b
    cap = float(row.get("battery_kwh") or 0.0)
    if cap <= 0:
        return None
    return BatterySpec(cap, float(row.get("battery_kw") or 0.0),
                       comfort_reserve_kwh=float(row.get("comfort_reserve_kwh") or 0.0))


def builtin_plan_feeder(feeder: FeederSpec, eligible_sites: Sequence[Mapping[str, Any]], rest_of_feeder: Any,
                        forecasts: Mapping[str, Mapping[str, Prediction]], G: Optional[float] = None,
                        prices: Any = None, *, statuses: Optional[Mapping[str, AssetStatus]] = None,
                        t0: Optional[str] = None, **_: Any) -> FeederPlan:
    """FALLBACK planner (not an LP): clip the worst-case export peak with the pool's battery headroom.

    Finds the lowest level L >= G that the pool's energy headroom can hold the worst plausible export
    (P90 PV - P10 load) under, splits the required absorption by headroom, keeps that much room free until the
    window, charges own surplus otherwise and discharges to cover load. Same signature as `opt.lp.plan_feeder`.
    """
    statuses = statuses or {}
    g = float(feeder.g_kw if G is None else G)
    sites = []
    for row in eligible_sites:
        b = _battery_of(row)
        sid = row["site_id"]
        if b is None or sid not in forecasts:
            continue
        f = _pv_factor(statuses.get(sid))
        pv, ld = forecasts[sid]["pv"], forecasts[sid]["load"]
        e0 = float(row.get("e0_kwh") if row.get("e0_kwh") is not None else b.e_min)
        sites.append((sid, b, e0, np.asarray(pv.q50) * f, np.asarray(pv.q90) * f, np.asarray(ld.q10),
                      np.asarray(ld.q50)))
    H = len(sites[0][3]) if sites else len(_rest_arrays(rest_of_feeder, DAY_STEPS)[0])
    rest_net, rest_worst = _rest_arrays(rest_of_feeder, H)
    worst = rest_worst.copy()
    for s in sites:
        worst += s[4] - s[5]
    caps = np.array([max(0.0, (b.e_max - max(e0, b.e_min)) / b.eta_charge) for _, b, e0, *_ in sites])
    pmax = np.array([b.p_max_kw for _, b, *_ in sites])
    sched: dict[str, list[float]] = {}
    if sites:
        tot_p, tot_e = pmax.sum(), max(caps.sum(), EPS)

        def need(level: float) -> np.ndarray:
            return np.minimum(np.maximum(0.0, worst - level), tot_p)

        lo, hi = g, max(g, float(worst.max()))
        if need(lo).sum() * DT > tot_e:
            for _ in range(40):
                mid = 0.5 * (lo + hi)
                lo, hi = (mid, hi) if need(mid).sum() * DT > tot_e else (lo, mid)
            level = hi
        else:
            level = lo
        req_total = need(level)
        w = caps / tot_e
        for i, (sid, b, e0, pv50, pv90, l10, l50) in enumerate(sites):
            absorb = np.minimum(req_total * w[i], b.p_max_kw)
            future = np.concatenate([np.cumsum(absorb[::-1])[::-1][1:], [0.0]]) * b.eta_charge * DT
            e, out = e0, []
            for t in range(H):
                room = max(0.0, (b.e_max - future[t] - e) / (b.eta_charge * DT))
                surplus = pv50[t] - l50[t]
                if absorb[t] > 0:
                    req = max(absorb[t], min(max(surplus, 0.0), room))
                elif surplus > 0:
                    req = min(surplus, room)
                else:
                    req = surplus
                p, e = _clip_step(b, e, req)
                out.append(p)
            sched[sid] = out
    export = rest_net.copy()
    wst = rest_worst.copy()
    for sid, b, e0, pv50, pv90, l10, l50 in sites:
        p = np.asarray(sched.get(sid, np.zeros(H)))
        export += pv50 - l50 - p
        wst += pv90 - l10 - p
    t0 = t0 or (next(iter(forecasts.values()))["pv"].t0 if forecasts else "1970-01-01T00:00:00Z")
    return FeederPlan(feeder_id=feeder.feeder_id, t0=t0, policy="sentinel", g_kw=g, schedule=sched,
                      export_curve=list(export), slack=list(np.maximum(0.0, wst - g)), horizon=H,
                      solver_status="builtin_greedy", objective=None)


@dataclass
class FallbackAssessment:
    """Minimal stand-in for `qualify.Assessment` (the attributes the loop and fallback triage read)."""

    asset_id: str
    asset_kind: str
    status: AssetStatus
    group_id: Optional[str] = None
    fault: Optional[str] = None
    p_fault: float = 0.0
    crew_type: Optional[str] = None
    alert: Optional[tuple[str, str, str]] = None
    threshold: Optional[float] = None
    loss_frac: float = 0.0
    lost_kwh_day: float = 0.0
    lost_flex_mwh_day: float = 0.0
    mw_at_risk: float = 0.0
    curtailed_kwh: Optional[float] = None
    offline: bool = False
    prediction: Optional[Prediction] = None
    actual: Optional[np.ndarray] = None


_PV_LOSS = {"soiling": 0.08, "shading": 0.12, "string_outage": 0.30, "clipping": 0.05, "overvoltage_trip": 0.20}


def fallback_qualify_many(assets: Sequence[Mapping[str, Any]], predictions: Mapping[str, Prediction],
                          telemetry_ages: Mapping[str, Optional[float]], thresholds: Any = None, *,
                          actuals: Optional[Mapping[str, Any]] = None,
                          flags: Optional[Mapping[str, Mapping[str, Any]]] = None,
                          park_consensus: bool = True) -> list[FallbackAssessment]:
    """FALLBACK qualify: everything eligible unless stale / data gap / OOD > 3 / confident (p >= 0.6) fault."""
    flags = flags or {}
    out: list[FallbackAssessment] = []
    for a in assets:
        aid = a.get("site_id") or a.get("turbine_id")
        kind = "turbine" if a.get("turbine_id") else "site"
        grp = a.get("park_id") if kind == "turbine" else a.get("feeder_id")
        p = predictions.get(aid)
        age = telemetry_ages.get(aid)
        fl = flags.get(aid, {})
        mk = lambda st, reason, h=None, ok=True: AssetStatus(  # noqa: E731
            aid, st, 0.0 if st in ("excluded", "human_review") else (1.0 if st == "eligible" else h), reason, ok)
        x = FallbackAssessment(aid, kind, mk("eligible", "ok"), grp, prediction=p)
        exp_kwh = float(np.mean(p.q50) * 24.0) if p is not None else 0.0
        if (age is not None and age > 60) or fl.get("data_gap"):
            x.status, x.offline = mk("excluded", "stale_telemetry" if not fl.get("data_gap") else "data_gap"), True
            if kind == "turbine":
                x.fault = WindAnomaly.DATA_GAP.value
        elif p is None:
            x.status = mk("excluded", "no_prediction")
        elif kind == "site":
            top, pf = p.top_fault, p.p_fault
            if p.ood_score > 3.0:
                x.status, x.threshold = mk("human_review", "ood"), 3.0
            elif top and top != "healthy" and pf >= 0.6:
                x.fault, x.p_fault, x.threshold = top, pf, 0.6
                if top == PvFault.SENSOR_FAULT.value:
                    x.status, x.crew_type = mk("excluded", "sensor_fault"), CrewType.REMOTE.value
                elif top == PvFault.BATTERY_FADE.value:
                    x.status, x.crew_type = mk("excluded", "battery_fault", ok=False), CrewType.RESIDENTIAL.value
                else:
                    loss = _PV_LOSS.get(top, 0.1)
                    x.status = mk("derated", f"pv_fault:{top}", 1 - loss)
                    x.crew_type, x.loss_frac, x.lost_kwh_day = CrewType.RESIDENTIAL.value, loss, loss * exp_kwh
                    x.mw_at_risk = max(p.q50) * loss / 1000
        else:
            cls, score = p.anomaly_class, float(p.anomaly_score or 0.0)
            if cls and cls != "normal" and score >= 0.6:
                x.fault, x.p_fault, x.threshold = cls, score, 0.6
                rated = float(a.get("rated_kw") or 0.0)
                if cls == WindAnomaly.FAULT_STOP.value:
                    x.status, x.crew_type, x.loss_frac = mk("excluded", "fault_stop"), CrewType.WIND.value, 1.0
                    x.lost_kwh_day, x.mw_at_risk = exp_kwh, rated / 1000
                elif cls == WindAnomaly.UNDERPERFORMANCE.value:
                    x.status, x.crew_type = mk("derated", "underperformance", 0.85), CrewType.WIND.value
                    x.loss_frac, x.lost_kwh_day = 0.15, 0.15 * exp_kwh
                elif cls == WindAnomaly.CURTAILMENT.value:
                    x.status = mk("eligible", "curtailed")
                elif cls == WindAnomaly.ICING.value:
                    x.status = mk("excluded", "icing")
                    x.alert = (Severity.SAFETY.value, "icing", f"Icing risk on {aid}: no crew until it clears.")
                elif cls == WindAnomaly.ANEMOMETER_FAULT.value:
                    x.status, x.crew_type = mk("eligible", "anemometer_fault"), CrewType.REMOTE.value
        out.append(x)
    if park_consensus:
        parks: dict[str, list[FallbackAssessment]] = {}
        for x in out:
            if x.asset_kind == "turbine" and not x.offline:
                parks.setdefault(x.group_id or "", []).append(x)
        for xs in parks.values():
            hit = [x for x in xs if x.fault in ("fault_stop", "underperformance") and x.crew_type]
            if len(hit) >= 3 and len(hit) >= 0.6 * len(xs):
                for x in hit:
                    x.status = AssetStatus(x.asset_id, "eligible", 1.0, "curtailed_park_wide")
                    x.fault, x.crew_type, x.lost_kwh_day, x.mw_at_risk = "curtailment", None, 0.0, 0.0
    return out


@dataclass
class FallbackTriageConfig:
    price_lei_mwh: float = 450.0
    policy: str = "sentinel"
    run_id: str = "local"
    k_per_crew: dict[str, int] = field(default_factory=lambda: {
        CrewType.RESIDENTIAL.value: 5, CrewType.WIND.value: 3, CrewType.REMOTE.value: 20})
    v_flex_lei_mwh: float = 400.0


@dataclass
class FallbackTriageResult:
    work_orders: list[Action]
    alerts: list[Action]
    human_reviews: list[Action]
    state: dict[str, Any]


def fallback_triage(assessments: Sequence[Any], t: str, state: Optional[dict[str, Any]] = None,
                    config: Optional[FallbackTriageConfig] = None, **_: Any) -> FallbackTriageResult:
    """FALLBACK triage: one action per episode; work orders top-K per UTC day per crew type by triage score."""
    cfg = config or FallbackTriageConfig()
    st = copy.deepcopy(state) if state else {"wo": set(), "issued": {}, "offline": set(), "rev": set(), "al": set()}
    wos, als, revs, cands = [], [], [], {}

    def rec(x, note=""):
        return A.audit(x.prediction.model_version if x.prediction else "none",
                       {"p": x.prediction.to_dict() if x.prediction else None, "t": t, "id": x.asset_id},
                       A.compact(x.prediction), x.threshold, x.status.status, note=note or x.status.reason)

    for x in assessments:
        aid, kind, s = x.asset_id, x.asset_kind, x.status
        if x.offline and aid not in st["offline"]:
            st["offline"].add(aid)
            if kind == "turbine":
                als.append(A.make(ActionType.ALERT, t, cfg.policy, cfg.run_id, {
                    "asset_id": aid, "asset_kind": kind, "severity": Severity.WARNING.value, "code": "offline",
                    "message": f"{aid} offline since {t}", "park_id": x.group_id}, rec(x), aid, "offline"))
        elif not x.offline:
            st["offline"].discard(aid)
        if x.alert and f"{aid}|{x.alert[1]}" not in st["al"]:
            st["al"].add(f"{aid}|{x.alert[1]}")
            als.append(A.make(ActionType.ALERT, t, cfg.policy, cfg.run_id, {
                "asset_id": aid, "asset_kind": kind, "severity": x.alert[0], "code": x.alert[1],
                "message": x.alert[2], "group_id": x.group_id}, rec(x), aid, x.alert[1]))
        if s.status == Status.HUMAN_REVIEW.value:
            if aid not in st["rev"]:
                st["rev"].add(aid)
                revs.append(A.make(ActionType.HUMAN_REVIEW, t, cfg.policy, cfg.run_id, {
                    "asset_id": aid, "asset_kind": kind, "reason": s.reason, "group_id": x.group_id}, rec(x), aid))
        else:
            st["rev"].discard(aid)
        if x.crew_type is None:
            if s.status == Status.ELIGIBLE.value:
                st["wo"].discard(aid)
            continue
        if aid in st["wo"]:
            continue
        v_flex = cfg.v_flex_lei_mwh if kind == "site" else 0.0
        score = x.p_fault * (x.lost_kwh_day / 1000 * cfg.price_lei_mwh + x.lost_flex_mwh_day * v_flex)
        cands.setdefault(x.crew_type, []).append((score, x))
    day = t[:10]
    for crew, xs in sorted(cands.items()):
        xs.sort(key=lambda sx: (-sx[0], sx[1].asset_id))
        used = st["issued"].get(f"{day}|{crew}", 0)
        for i, (score, x) in enumerate(xs[:max(0, cfg.k_per_crew.get(crew, 5) - used)]):
            st["wo"].add(x.asset_id)
            st["issued"][f"{day}|{crew}"] = used + i + 1
            wos.append(A.make(ActionType.WORK_ORDER, t, cfg.policy, cfg.run_id, {
                "asset_id": x.asset_id, "asset_kind": x.asset_kind, "fault": x.fault, "crew_type": crew,
                "score": round(score, 2), "lost_kwh_day": round(x.lost_kwh_day, 2),
                "lost_lei_day": round(x.lost_kwh_day / 1000 * cfg.price_lei_mwh, 2),
                "lost_flex_mwh_day": x.lost_flex_mwh_day, "mw_at_risk": x.mw_at_risk, "p_fault": x.p_fault,
                "visit_days": 1.0, "rank": used + i + 1, "status": x.status.status, "reason": x.status.reason,
                ("park_id" if x.asset_kind == "turbine" else "feeder_id"): x.group_id}, rec(x), x.asset_id, crew))
    return FallbackTriageResult(wos, als, revs, st)


def fallback_commit(feeder: FeederSpec, plan: FeederPlan, ctx: Mapping[str, Any]) -> Commitment:
    """FALLBACK commitment: nominal = available = planned export reduction vs the rule on P50, committed = 0.8x."""
    red = np.maximum(0.0, np.asarray(ctx["planned_reduction"], float))
    h = float(ctx.get("haircut", 0.0))
    return Commitment(feeder_id=feeder.feeder_id, t0=plan.t0, nominal=list(red), available=list(red),
                      committed=list(0.8 * red * (1 - h)), haircut=h, horizon=plan.horizon,
                      extra={"method": "fallback_0.8x_planned"})


def fallback_zone_demand(world: World, d: int, load_preds: Mapping[str, Prediction], run_id: str, policy: str
                         ) -> tuple[list[dict[str, Any]], list[Action]]:
    """FALLBACK demand: per feeder / locality / county, q50 summed, spread by sqrt-sum of squares (MW)."""
    t0 = world.day_t0(d)
    groups: dict[tuple[str, str, str, Optional[str]], list[str]] = {}
    for sid, site in world.sites.items():
        r = site.row
        loc_id = f"{r['county']}/{r['locality']}"
        groups.setdefault(("feeder", site.feeder_id, site.feeder_id, loc_id), []).append(sid)
        groups.setdefault(("locality", loc_id, r["locality"], r["county"]), []).append(sid)
        groups.setdefault(("county", r["county"], r["county"], None), []).append(sid)
    rows, acts = [], []
    for (level, zid, name, parent), sids in sorted(groups.items(), key=lambda kv: kv[0][:2]):
        q50 = np.sum([load_preds[s].q50 for s in sids], axis=0) / 1000
        lo = np.sqrt(np.sum([(np.asarray(load_preds[s].q50) - load_preds[s].q10) ** 2 for s in sids], axis=0)) / 1000
        hi = np.sqrt(np.sum([(np.asarray(load_preds[s].q90) - load_preds[s].q50) ** 2 for s in sids], axis=0)) / 1000
        k = int(np.argmax(q50))
        cap = sum(world.feeders[f].spec.g_max_kw for f in {world.sites[s].feeder_id for s in sids}) / 1000
        stress = float(min(1.0, (q50[k] + hi[k]) / cap)) if cap > 0 else 0.0
        for j in range(len(q50)):
            ts = world.tick_ts(d, j)
            rows.append({"run_id": run_id, "zone_type": level, "zone_id": zid, "zone_name": name,
                         "parent_zone_id": parent, "ts": Ts(ts), "p10_mw": max(0.0, q50[j] - lo[j]),
                         "p50_mw": q50[j], "p90_mw": q50[j] + hi[j], "peak_stress": stress,
                         "is_peak_hour": int(j // 4 == k // 4), "label": world.label})
        rec = A.audit("demand-fallback-sum", {"zone": zid, "t0": t0, "sites": sorted(sids)},
                      {"peak_mw": float(q50[k]), "peak_stress": stress}, None, "ok")
        acts.append(A.make(ActionType.ZONE_DEMAND, t0, policy, run_id, {
            "zone_id": zid, "level": level, "q50_mw": [round(v, 5) for v in q50],
            "q90_mw": [round(v, 5) for v in q50 + hi], "peak_t": iso(world.tick_ts(d, k)),
            "peak_mw": float(q50[k])}, rec, level, zid))
    return rows, acts


# =============================================================================================== components
@dataclass
class Components:
    predictors: dict[str, Any]
    plan_feeder: Callable[..., FeederPlan]
    plan_feeder_backup: Optional[Callable[..., FeederPlan]]
    qualify_many: Callable[..., list[Any]]
    triage: Callable[..., Any]
    triage_config: Callable[[float, str, str], Any]
    commit: Callable[[FeederSpec, FeederPlan, Mapping[str, Any]], Commitment]
    zone_demand: Callable[..., tuple[list[dict[str, Any]], list[Action]]]
    rest_of_feeder: Callable[..., Any]
    threshold_step: Callable[..., tuple[float, float, float]]
    sources: dict[str, str] = field(default_factory=dict)
    errors: dict[str, int] = field(default_factory=dict)
    windmon_flags: Optional[Callable[..., dict[str, np.ndarray]]] = None  # WindMon rules (baseline policy); None = builtin
    judge: Optional[Any] = None  # wattsentry.judge.slot.JudgeSlot; None = no verdicts

    @property
    def fallbacks(self) -> list[str]:
        return sorted(k for k, v in self.sources.items() if not v.startswith("real") and k not in OBSERVER_SLOTS)

    def note_error(self, slot: str, exc: BaseException) -> None:
        n = self.errors.get(slot, 0)
        self.errors[slot] = n + 1
        if n == 0:
            log(f"WARNING {slot} ({self.sources.get(slot)}) raised {type(exc).__name__}: {exc}; using its fallback "
                f"for this call (further errors counted, not printed)")
            traceback.print_exc(limit=3, file=sys.stderr)


def _builtin_rest(sites: Sequence[Mapping[str, Any]], forecasts: Mapping[str, Mapping[str, Prediction]],
                  statuses: Optional[Mapping[str, AssetStatus]] = None, horizon: Optional[int] = None):
    statuses = statuses or {}
    H = horizon or DAY_STEPS
    net, worst = np.zeros(H), np.zeros(H)
    for row in sites:
        sid = row["site_id"]
        b = _battery_of(row)
        f = _pv_factor(statuses.get(sid))
        pv, ld = forecasts[sid]["pv"], forecasts[sid]["load"]
        for arr, pvq, ldq in ((net, pv.q50, ld.q50), (worst, pv.q90, ld.q10)):
            e = float(row.get("e0_kwh") if row.get("e0_kwh") is not None else (b.e_min if b else 0.0))
            for t in range(H):
                _, e, g = self_consumption_step(b, e, pvq[t] * f, ldq[t])
                arr[t] += g
    return net, worst


def _builtin_threshold_step(battery, e, pv, load, threshold_kw, dt_h=DT):
    surplus = pv - load
    if battery is None or battery.capacity_kwh == 0:
        return 0.0, e, surplus
    req = surplus - threshold_kw if surplus > threshold_kw else (surplus if surplus < 0 else 0.0)
    p, e2 = battery.step(e, req, dt_h)
    return p, e2, surplus - p


def default_components(world: World, force_fallback: bool | Sequence[str] = False,
                       predictors: Optional[dict[str, Any]] = None, judge: bool = True) -> Components:
    """Real modules where they import (lazy), built-in fallbacks otherwise. `force_fallback=True` or a list of
    slots (planner, qualify, triage, committer, demand) forces the built-ins (tests, A/B checks)."""
    forced = set(["planner", "qualify", "triage", "committer", "demand", "wind_baseline"] if force_fallback is True
                 else (force_fallback or []))
    src: dict[str, str] = {}
    preds = predictors or world.default_predictors()
    versions = sorted({getattr(p, "model_version", "?") for k, p in preds.items() if k != "zone"})  # zone: unused
    src["predictors"] = ("mock:" if all(str(v).startswith("mock") for v in versions) else "real:") + ",".join(versions)
    for k, p in sorted((k, p) for k, p in preds.items() if k != "zone"):  # zone: unused by the loop
        v = str(getattr(p, "model_version", "?"))
        src[f"predictor_{k}"] = ("fallback:mock " if v.startswith("mock") else "real:") + \
            f"{getattr(p, 'model', k)}:{v}"
    wind_real = not str(getattr(preds.get("wind"), "model_version", "mock")).startswith("mock")
    src["wind_classifier"] = ("real:windnbm per-step classes" if wind_real
                              else "fallback:builtin_rules (M6 stand-in)")
    src["wind_baseline"] = "fallback:builtin_windmon_rules (#24 stand-in)"

    def _try(slot: str, fn: Callable[[], Any]) -> Any:
        if slot in forced:
            return None
        try:
            return fn()
        except ImportError as e:
            log(f"{slot}: real module not importable ({e}); FALLBACK in use")
        except Exception as e:  # module exists but is broken: loud
            log(f"WARNING {slot}: real module failed to load ({type(e).__name__}: {e}); FALLBACK in use")
        return None

    # planner: LP, aggregate LP as backup, greedy built-in
    plan, backup, rest_fn, thr_step = builtin_plan_feeder, None, None, _builtin_threshold_step
    src["planner"] = "fallback:builtin_greedy"
    src["rest_of_feeder"] = "fallback:builtin"
    src["threshold_rule"] = "fallback:builtin"
    lp = _try("planner", lambda: __import__("wattsentry.opt.lp", fromlist=["plan_feeder"]))
    if lp is not None:
        plan, src["planner"] = lp.plan_feeder, "real:wattsentry.opt.lp.plan_feeder"
        agg = _try("planner", lambda: __import__("wattsentry.opt.aggregate", fromlist=["plan_feeder_aggregate"]))
        if agg is not None:
            backup = agg.plan_feeder_aggregate
    pol = _try("planner", lambda: __import__("wattsentry.opt.policies", fromlist=["threshold_step"]))
    if pol is not None:
        src["rest_of_feeder"] = "real:wattsentry.opt.policies.rest_of_feeder_from_forecasts"
        src["threshold_rule"] = "real:wattsentry.opt.policies.threshold_step"
        thr_step = pol.threshold_step

        def rest_fn(sites, forecasts, statuses=None, horizon=None, _pol=pol):
            return _pol.rest_of_feeder_from_forecasts([_pol.site_from_row(s) for s in sites], forecasts,
                                                      statuses, horizon)
    if rest_fn is None:
        rest_fn = _builtin_rest

    # qualify + triage
    qmany, tri = fallback_qualify_many, fallback_triage
    tcfg = lambda price, run_id, policy: FallbackTriageConfig(price, policy, run_id)  # noqa: E731
    src["qualify"], src["triage"] = "fallback:builtin_qualify", "fallback:builtin_triage"
    q = _try("qualify", lambda: __import__("wattsentry.agent.qualify", fromlist=["qualify_many"]))
    tr = _try("triage", lambda: __import__("wattsentry.agent.triage", fromlist=["triage"])) if q is not None else None
    if tr is not None:  # q is not None too
        qmany, src["qualify"] = q.qualify_many, "real:wattsentry.agent.qualify.qualify_many"
        tri, src["triage"] = tr.triage, "real:wattsentry.agent.triage.triage"
        tcfg = lambda price, run_id, policy, _tr=tr: _tr.TriageConfig(price_lei_mwh=price, run_id=run_id,  # noqa
                                                                        policy=policy)
    elif q is not None:
        log("WARNING qualify and triage must come from the same module set; using both fallbacks")

    # committer
    commit_fn: Callable = fallback_commit
    src["committer"] = "fallback:0.8x_planned_reduction"
    cm = _try("committer", lambda: __import__("wattsentry.opt.commit", fromlist=["commit", "make_scenarios"]))
    if cm is not None:
        src["committer"] = "real:wattsentry.opt.commit.make_scenarios+commit"

        def commit_fn(feeder, plan, ctx, _cm=cm):
            pool = list(plan.schedule)
            if not pool:
                return fallback_commit(feeder, plan, ctx)
            scen = _cm.make_scenarios({s: ctx["pv"][s] for s in pool}, {s: ctx["load"][s] for s in pool},
                                      n=ctx.get("n_scenarios", 200), seed=ctx.get("seed", 0), site_ids=pool,
                                      pv_cap=ctx.get("pv_cap"), batteries={s: ctx["batteries"][s] for s in pool},
                                      soc0={s: ctx["soc0"][s] for s in pool}, feeder_id=feeder.feeder_id,
                                      t0=plan.t0)
            return _cm.commit(feeder, plan, scen, haircut=ctx.get("haircut", 0.05), t0=plan.t0)

    # zone demand
    zd: Callable = fallback_zone_demand
    src["demand"] = "fallback:summed_quantiles"
    dm = _try("demand", lambda: __import__("wattsentry.demand.aggregate", fromlist=["aggregate_scenarios"]))
    if dm is not None:
        src["demand"] = "real:wattsentry.demand.aggregate_scenarios+zone_demand_rows"

        def zd(world, d, load_preds, run_id, policy, _dm=dm):
            sids = sorted(load_preds)
            q10 = np.array([load_preds[s].q10 for s in sids])
            q50 = np.array([load_preds[s].q50 for s in sids])
            q90 = np.array([load_preds[s].q90 for s in sids])
            scen = _dm.scenarios_from_quantiles(q10, q50, q90, 200, seed=world.seed + d)
            agg = _dm.aggregate_scenarios(scen, [world.sites[s].row for s in sids],
                                          [f.row() for f in world.feeders.values()], t0=world.day_t0(d),
                                          label=world.label)
            return _dm.zone_demand_rows(agg, run_id), _dm.zone_demand_actions(agg, run_id=run_id,
                                                                                t=world.day_t0(d), policy=policy)

    comps = Components(preds, plan, backup, qmany, tri, tcfg, commit_fn, zd, rest_fn, thr_step, src)
    wm = _try("wind_baseline", lambda: __import__("wattsentry.eval.baselines.wind.windmon_rules",
                                                  fromlist=["windmon_flags"]))
    if wm is not None and hasattr(wm, "windmon_flags"):
        comps.windmon_flags = wm.windmon_flags
        src["wind_baseline"] = "real:wattsentry.eval.baselines.wind.windmon_rules (#24)"
    # judge: observer slot, never fails the loop
    src["judge"] = "off"
    if judge:
        try:
            from wattsentry.judge.slot import JudgeSlot

            comps.judge = JudgeSlot.load()
            src["judge"] = comps.judge.source
        except Exception as e:  # e.g. pandas missing: the agent runs as before, without verdicts
            log(f"WARNING judge slot unavailable ({type(e).__name__}: {e}); no verdicts this run")
            src["judge"] = f"unavailable ({type(e).__name__})"
    banner = ", ".join(f"{k}={v}" for k, v in sorted(src.items()))
    log(f"components: {banner}")
    if comps.fallbacks:
        log(f"FALLBACK components in use: {', '.join(comps.fallbacks)} -- numbers from this run are NOT results "
            f"of the real modules (recorded in runs.notes)")
    return comps


# =============================================================================================== loop
@dataclass
class LoopConfig:
    policy: str = "sentinel"
    run_id: str = "local"
    resolve_hourly: bool = False
    write_inputs: bool = True  # predictions, measurements, zone demand, gis flags: once per DB (shared inputs)
    step_predictions: bool = True  # per-step pv/load rows in predictions (False = lite DB; asset_steps keeps PV q's)
    detail_sites: Optional[frozenset[str]] = None  # large worlds: per-site rows only for these; None = every site
    write_schedules: bool = True  # per-step schedules rows (replay default: sentinel only; the UI shows policy 'sentinel')
    beta: float = 0.5  # export remuneration ratio (same default as the LP)
    c_deg_lei_kwh: float = 0.12  # battery degradation, ~EUR 0.025/kWh per direction, both policies
    threshold_alpha: float = 0.5
    haircut: float = 0.05
    conformal_buckets: int = 24  # hour-of-day buckets (t0 = local midnight): dawn load misses differ from midday PV
    n_scenarios: int = 200
    seed: int = 0
    judge: bool = True  # run comps.judge (if any) on the sentinel policy: verdicts only, decisions unchanged
    # live driver: called once per tick after the feeder curves with that tick's decisions; it may block until the
    # next reading arrives. None = batch replay.
    on_tick: Optional[Callable[[dict], None]] = None


@dataclass
class PolicyResult:
    policy: str
    run_id: str
    days: int
    feeder_days: dict[tuple[str, int], dict[str, float]] = field(default_factory=dict)
    input_hashes: dict[tuple[int, str], str] = field(default_factory=dict)
    day_hashes: dict[int, str] = field(default_factory=dict)
    comfort_breaches: int = 0
    agent_steps: int = 0
    clipped_steps: int = 0
    cost_lei: float = 0.0
    n_homes: int = 0
    hits: int = 0
    commit_steps: int = 0
    turbine_alerts: int = 0
    turbine_days: int = 0
    n_actions: dict[str, int] = field(default_factory=dict)
    action_ids: list[str] = field(default_factory=list)
    plan_status: dict[str, int] = field(default_factory=dict)
    conformal: dict[int, dict[str, Any]] = field(default_factory=dict)  # day -> commit-time calibration (runs.notes)
    judge: dict[str, Any] = field(default_factory=dict)  # JudgeSlot.summary(): mode, verdict counts, t_judge_s per day

    @property
    def minutes_over_limit(self) -> float:
        return sum(v["minutes_over"] for v in self.feeder_days.values())

    @property
    def energy_over_limit_kwh(self) -> float:
        return sum(v.get("kwh_over", 0.0) for v in self.feeder_days.values())

    @property
    def peak_overshoot_kw(self) -> float:
        return max((v.get("peak_overshoot", 0.0) for v in self.feeder_days.values()), default=0.0)

    @property
    def p10_hit_rate(self) -> Optional[float]:
        return self.hits / self.commit_steps if self.commit_steps else None


class _MemSink:
    """Sink that keeps rows in memory (tests / dry runs)."""

    def __init__(self):
        self.rows: dict[str, list[dict[str, Any]]] = {}
        self.actions: list[Action] = []

    def add(self, table, row):
        self.rows.setdefault(table, []).append(dict(row))

    def add_many(self, table, rows):
        for r in rows:
            self.add(table, r)

    def has_table(self, table):
        return table != "asset_steps"

    def write_action(self, action, rows=None, *, ts=None, extra=None):
        self.actions.append(action)
        self.add("audit_log", {"id": action.action_id, "action_type": action.type, "input_hash":
                               action.audit.input_hash, "asset_id": action.payload.get("asset_id") or
                               action.payload.get("site_id"), "ts": ts, "run_id": action.run_id})
        for t, rs in (rows or {}).items():
            self.add_many(t, rs)


def _digest(*arrays: Sequence[float]) -> str:
    h = hashlib.sha256()
    for a in arrays:
        h.update(np.asarray(a, dtype=np.float64).tobytes())
    return h.hexdigest()[:16]


def _is_fallback(p: Prediction, comps: "Components") -> int:
    """predictions.is_fallback: the envelope's own flag (real adapters), else whether its version is a mock."""
    fb = (p.extra or {}).get("is_fallback")
    if fb is not None:
        return int(fb)
    return int(str(p.model_version).startswith("mock") or "predictors" in comps.fallbacks)


def _health_row(run_id: str, s: str, h: Prediction, ts: Any, k: int, comps: "Components") -> dict[str, Any]:
    """predictions row of a SiteHealth assessment made at `ts` (= its t0), horizon index k."""
    return {"run_id": run_id, "asset_id": s, "model": h.model, "model_version": h.model_version, "ts": ts,
            "q10_kw": h.q10[k], "q50_kw": h.q50[k], "q90_kw": h.q90[k], "fault_probs_json": h.fault_probs,
            "ood_score": h.ood_score, "member_std": h.member_std, "t0": ts, "horizon_idx": k,
            "is_fallback": _is_fallback(h, comps)}


def _predict(pred: Any, asset: Mapping[str, Any], t0: str) -> Prediction:
    return pred.predict(asset, Window(t0=t0, horizon=DAY_STEPS))


def _wind_rules(rated: float, act: Optional[float], exp: float, run: int) -> tuple[str, float, int]:
    """M6 stand-in. Returns (anomaly_class, score, underperformance run length)."""
    if act is None:
        return WindAnomaly.DATA_GAP.value, 0.95, 0
    if act < 0.05 * rated and exp > 0.15 * rated:
        return WindAnomaly.FAULT_STOP.value, 0.9, 0
    if exp > 0.2 * rated and act < 0.7 * exp:
        run += 1
        return (WindAnomaly.UNDERPERFORMANCE.value, 0.7, run) if run >= 8 else ("normal", 0.3, run)
    return WindAnomaly.NORMAL.value, 0.05, 0


_WINDMON = (("HIGH_WIND_ZERO_POWER", 15.0), ("DATA_GAP", None), ("TEMPERATURE_SPIKE", 85.0), ("SUSPECT_DATA", None))


def run_policy(world: World, comps: Components, cfg: LoopConfig, sink: Any = None) -> PolicyResult:
    """Replay the world's days under one policy. Rows go to `sink` (AuditDB or in-memory)."""
    policy = BASELINE_ALIASES.get(cfg.policy)
    if policy is None:
        raise ValueError(f"unknown policy {cfg.policy!r} (baseline | self_consumption | threshold | sentinel)")
    sink = sink if sink is not None else _MemSink()
    run_id = cfg.run_id
    res = PolicyResult(policy, run_id, world.days)
    sentinel = policy == "sentinel"
    write_inputs = cfg.write_inputs
    asset_steps = sink.has_table("asset_steps") if hasattr(sink, "has_table") else False
    preds = comps.predictors
    sites = world.sites
    batt_sites = [s for s in sites if sites[s].battery is not None]
    res.n_homes = len(sites)
    # state carried across days
    energy = {s: sites[s].battery.e_min for s in batt_sites}
    triage_state: Any = None
    last_status: dict[str, tuple[str, str]] = {}
    under_run: dict[str, int] = {}
    curt_ep: dict[str, dict[str, Any]] = {}
    windmon_active: dict[tuple[str, str], bool] = {}
    windmon = _windmon_precompute(world, comps) if (not sentinel and world.turbines) else None
    res.turbine_days = len(world.turbines) * world.days

    def act(action: Action, ts: int, rows: Optional[dict[str, list[dict[str, Any]]]] = None,
            extra: Optional[dict[str, Any]] = None) -> None:
        r = A.rows_for(action, ts)
        for k, v in (rows or {}).items():
            r.setdefault(k, []).extend(v)
        sink.write_action(action, r, ts=ts, extra=extra)
        res.n_actions[action.type] = res.n_actions.get(action.type, 0) + 1
        res.action_ids.append(action.action_id)

    judge = comps.judge if (sentinel and cfg.judge) else None
    gis: list[dict[str, Any]] = []
    if write_inputs:  # grid-twin flags (M8 stand-in in the mock world), once per run
        t0 = world.day_t0(0)
        gis = list(world.gis_flags())
        for fl in gis:
            twin_v = getattr(world, "gridtwin_version", "gridtwin:mock-1")
            rec = A.audit(twin_v, fl, {"tau": fl["tau"], "score": fl.get("score")}, fl.get("threshold", 0.5),
                          "flagged", note=fl.get("message") or "mock grid-twin (M8 stand-in)")
            a = A.make(ActionType.GIS_FLAG, t0, policy, run_id, {
                "section_id": fl["section_id"], "recorded_feeder": fl["recorded_feeder"],
                "recorded_phase": fl["recorded_phase"], "suggested_feeder": fl["suggested_feeder"],
                "suggested_phase": fl["suggested_phase"], "tau": fl["tau"], "confidence": 1 - fl["tau"],
                "geometry": {"type": "Point", "coordinates": [fl["lon"], fl["lat"]]}}, rec, fl["section_id"])
            act(a, world.day_ts(0))

    if judge is not None:
        judge.start_run(world, run_id, sink, gis_flags=gis)
    prev_oday: dict[str, SiteDay] = {}  # yesterday's observation: the blind-haircut profile at commit time
    cal_hist: dict[str, list[tuple[np.ndarray, ...]]] = {"pv": [], "load": []}  # (y, q10, q50, q90, observed)
    for d in range(world.days):
        t0 = world.day_t0(d)
        day_ts = world.day_ts(d)
        prices = world.prices(d)
        prices_kwh = [p / 1000 for p in prices]
        pdig = _digest(prices)
        mean_price = float(np.mean(prices))
        # ---------------------------------------------------------------- perceive
        sday = {s: world.site_day(s, d) for s in sites}  # physics (truth): battery step, grid, realised
        oday = {s: world.observed_site_day(s, d) for s in sites}  # observation: everything the agent reads
        tday = {t: world.turbine_day(t, d) for t in world.turbines}
        # ---------------------------------------------------------------- estimate
        pv_p = {s: _predict(preds["pv"], sites[s].row, t0) for s in sites}
        ld_p = {s: _predict(preds["load"], sites[s].row, t0) for s in sites}
        if hasattr(preds["sitehealth"], "prefetch"):  # batch M4 over all sites (real models)
            preds["sitehealth"].prefetch(list(sites), t0, 0)
        hl_p = {s: _predict(preds["sitehealth"], sites[s].row, t0) for s in sites}
        reassess = set(getattr(preds["sitehealth"], "reassess_ticks", ()) or ())
        wd_p = {t: _predict(preds["wind"], world.turbines[t].row, t0) for t in world.turbines}
        day_inputs = {}
        for s in sites:
            g = world.feeders[sites[s].feeder_id].spec.g_kw
            inp = A.site_inputs(sites[s].row, {"pv": pv_p[s], "load": ld_p[s], "sitehealth": hl_p[s]},
                                _digest(oday[s].pv_kw, oday[s].load_kw, oday[s].telemetry_age_s), pdig, g)
            res.input_hashes[(d, s)] = day_inputs[s] = input_hash(inp)
        res.day_hashes[d] = input_hash([day_inputs, {t: _digest([x if x is not None else -1 for x in tday[t].act_pwt_kw])
                                                     for t in tday}, {t: wd_p[t].to_dict() for t in wd_p}])
        if write_inputs:
            created = Ts(day_ts)
            for s in sites:
                for p in ((pv_p[s], ld_p[s]) if cfg.step_predictions else ()):
                    for k in range(DAY_STEPS):
                        sink.add("predictions", {"run_id": run_id, "asset_id": s, "model": p.model,
                                                 "model_version": p.model_version, "ts": Ts(world.tick_ts(d, k)),
                                                 "q10_kw": p.q10[k], "q50_kw": p.q50[k], "q90_kw": p.q90[k],
                                                 "ood_score": p.ood_score, "member_std": p.member_std,
                                                 "t0": created, "horizon_idx": k,
                                                 "is_fallback": _is_fallback(p, comps)})
                if cfg.detail_sites is not None and s not in cfg.detail_sites:
                    continue
                sink.add("predictions", _health_row(run_id, s, hl_p[s], created, 0, comps))
            rows, acts = _guarded(comps, "demand", comps.zone_demand, fallback_zone_demand, world, d, ld_p, run_id,
                                  policy)
            sink.add_many("zone_demand", rows)
            for a in acts:
                act(a, day_ts)
        # ---------------------------------------------------------------- qualify @ t0 + decide (day-ahead)
        site_rows = {s: sites[s].row for s in sites}
        plans: dict[str, FeederPlan] = {}
        pool: dict[str, list[str]] = {f: [] for f in world.feeders}
        statuses: dict[str, AssetStatus] = {}
        thresholds: dict[str, float] = {}
        commitments: dict[str, Commitment] = {}
        e_day0 = dict(energy)
        if judge is not None:  # verdicts at t0: predictions + telemetry age now + past observed days only
            judge.at_t0(d, t0, pv_p, ld_p, hl_p, {s: oday[s].telemetry_age_s[0] for s in sites})
        if sentinel:
            ass = _safe_qualify(comps, [site_rows[s] for s in sites], hl_p,
                                {s: oday[s].telemetry_age_s[0] / 60 for s in sites}, None, {})
            for x in ass:
                statuses[x.asset_id] = x.status
            blind = _blind_profile(prev_oday)
            cal = _fit_conformal(cal_hist, cfg)
            res.conformal[d] = {k: c.to_dict() for k, c in cal.items()}
            for fid, feeder in world.feeders.items():
                members = feeder.site_ids
                pool[fid] = [s for s in members if s in energy and statuses[s].in_pool and statuses[s].battery_ok]
                pl, comm = _plan_and_commit(comps, cfg, world, feeder.spec, members, pool[fid], pv_p, ld_p,
                                            statuses, energy, prices_kwh, t0, res, blind, cal)
                plans[fid], commitments[fid] = pl, comm
                gi = feeder.spec.g_kw
                for s in pool[fid]:
                    rec = A.audit(f"{pv_p[s].model}:{pv_p[s].model_version}|{ld_p[s].model}:{ld_p[s].model_version}",
                                  {"site_inputs": day_inputs[s]},
                                  {**A.compact(pv_p[s]), "load_q50_sum": round(sum(ld_p[s].q50) * DT, 3),
                                   "p_fault": round(hl_p[s].p_fault, 4), "e0_kwh": round(energy[s], 3),
                                   "headroom_factor": statuses[s].headroom_factor},
                                  gi, statuses[s].status, pl.schedule.get(s, [0.0] * DAY_STEPS),
                                  note=f"day-ahead plan ({pl.solver_status}); threshold = feeder limit kW")
                    rec.input_hash = day_inputs[s]  # the hash pins the exogenous inputs both policies share
                    act(A.battery_schedule(t0, policy, run_id, s, fid, pl.schedule.get(s, [0.0] * DAY_STEPS), rec),
                        day_ts)
        elif policy == "threshold":
            for s in batt_sites:
                thresholds[s] = cfg.threshold_alpha * max(0.0, max(a - b for a, b in zip(pv_p[s].q50, ld_p[s].q50)))
        # ---------------------------------------------------------------- tick loop: act + log
        live_pool = {f: set(pool[f]) for f in pool}
        held = {f: set() for f in pool}  # blind pool sites still running today's schedule on the device (no re-plan)
        cf_e = {s: e_day0[s] for f in pool for s in pool[f]}  # rule counterfactual for realised
        realised = {f: [0.0] * DAY_STEPS for f in world.feeders}
        executed = {s: [0.0] * DAY_STEPS for s in batt_sites}
        pv_obs = {s: np.full(DAY_STEPS, np.nan) for s in sites}
        tb_obs = {t: np.full(DAY_STEPS, np.nan) for t in world.turbines}
        fd = {f: {"peak_export": 0.0, "minutes_over": 0.0, "peak_import_evening": 0.0, "kwh_over": 0.0,
                  "peak_overshoot": 0.0} for f in world.feeders}  # overshoot kW: max(0, export - limit)
        park_acc = {p: {"prod": 0.0, "exp": 0.0, "avail": 0.0, "curt": 0.0, "dq": 0.0, "n": 0, "issues": []}
                    for p in world.parks}
        revision = 0
        for k in range(DAY_STEPS):
            ts = world.tick_ts(d, k)
            tsx = Ts(ts)
            t_iso = iso(ts)
            local_h = local_dt(ts).hour
            # -- hourly re-solve on the remaining horizon with current energies
            if sentinel and cfg.resolve_hourly and k > 0 and k % 4 == 0:
                revision += 1
                for fid, feeder in world.feeders.items():
                    lp = sorted(live_pool[fid])
                    if not lp:
                        continue
                    pvk = {s: _slice(pv_p[s], k) for s in feeder.site_ids}
                    ldk = {s: _slice(ld_p[s], k) for s in feeder.site_ids}
                    pl = _plan(comps, world, feeder.spec, feeder.site_ids, lp, pvk, ldk, statuses, energy,
                               prices_kwh[k:], iso(ts), res)
                    old = plans[fid]
                    plans[fid] = FeederPlan(fid, old.t0, old.policy, old.g_kw,
                                            {s: old.schedule[s] if s in held[fid] else  # the device keeps its copy
                                             old.schedule.get(s, [0.0] * DAY_STEPS)[:k] + pl.schedule.get(s, [0.0] * (DAY_STEPS - k))
                                             for s in set(old.schedule) | set(pl.schedule)},
                                            old.export_curve[:k] + pl.export_curve, old.slack[:k] + pl.slack,
                                            solver_status=pl.solver_status)
                    for s in lp:
                        rec = A.audit("replan", {}, {"e_kwh": round(energy[s], 3), "from_step": k},
                                      feeder.spec.g_kw, statuses[s].status, plans[fid].schedule.get(s),
                                      note=f"hourly re-solve r{revision} from step {k} ({pl.solver_status})")
                        rec.input_hash = day_inputs[s]
                        act(A.battery_schedule(t0, policy, run_id, s, fid,
                                               plans[fid].schedule.get(s, [0.0] * DAY_STEPS), rec,
                                               revision=revision), ts)
            # -- perceive this tick
            for s in sites:
                if oday[s].telemetry_age_s[k] == 0:
                    pv_obs[s][k] = oday[s].pv_kw[k]
            # -- intraday SiteHealth re-check (real M4 on the 24 h ending now): new fault evidence mid-day
            if k in reassess:
                if hasattr(preds["sitehealth"], "prefetch"):
                    preds["sitehealth"].prefetch(list(sites), t0, k)
                for s in sites:
                    hl_p[s] = preds["sitehealth"].predict(sites[s].row, Window(t0=t0, horizon=DAY_STEPS,
                                                                               meta={"tick": k}))
                    if write_inputs:
                        sink.add("predictions", _health_row(run_id, s, hl_p[s], tsx, k, comps))
            # -- qualify every tick (sentinel): telemetry age + measured loss; triage
            if sentinel:
                ages_min = {**{s: oday[s].telemetry_age_s[k] / 60 for s in sites},
                            **{t: _gap_age(tday[t].act_pwt_kw, k) for t in world.turbines}}
                wind_k = _wind_preds(world, wd_p, tday, k, tb_obs, under_run)
                if judge is not None:
                    judge.observe_wind_tick(d, k, wind_k)
                ass = _safe_qualify(comps, [site_rows[s] for s in sites] + [world.turbines[t].row for t in world.turbines],
                                    {**hl_p, **wind_k}, ages_min,
                                    {**pv_obs, **tb_obs},
                                    {t: {"data_gap": True} for t in world.turbines if tday[t].act_pwt_kw[k] is None})
                for x in ass:
                    st = x.status
                    key = (st.status, st.reason)
                    if k == 0 or last_status.get(x.asset_id) != key:
                        last_status[x.asset_id] = key
                        sink.add("statuses", {"run_id": run_id, "asset_id": x.asset_id, "ts": tsx,
                                              "status": st.status, "headroom_factor": st.headroom_factor,
                                              "reason": st.reason,
                                              "telemetry_age_s": round(60 * ages_min.get(x.asset_id, 0.0), 1)})
                    if x.asset_kind == "site":
                        fid = sites[x.asset_id].feeder_id
                        if x.asset_id in live_pool[fid] and not st.in_pool:
                            live_pool[fid].discard(x.asset_id)  # out of re-planning
                            if st.reason in BLIND_REASONS:
                                held[fid].add(x.asset_id)  # blind only: the device keeps today's schedule
                        elif x.asset_id in held[fid] and not st.in_pool and st.reason not in BLIND_REASONS:
                            held[fid].discard(x.asset_id)  # a real problem now: its own local rule
                        elif (x.asset_id in cf_e and x.asset_id not in live_pool[fid]
                              and st.in_pool and st.battery_ok):
                            live_pool[fid].add(x.asset_id)  # telemetry fresh again and re-qualified: back on plan
                            held[fid].discard(x.asset_id)
                            statuses[x.asset_id] = st  # (the next hourly re-solve starts from its current energy)
                tri = _safe_triage(comps, ass, t_iso, triage_state, comps.triage_config(mean_price, run_id, policy))
                triage_state = tri.state
                for a in [*tri.work_orders, *tri.alerts, *tri.human_reviews]:
                    act(a, ts)
                    if a.payload.get("asset_kind") == "turbine":
                        res.turbine_alerts += 1
                _wind_tick(k, ts, ass, wd_p, tday, park_acc, curt_ep, act, run_id, policy, sink,
                           write_inputs, res)
            elif world.turbines:
                _windmon_tick(world, k, ts, tday, windmon_active, act, run_id, policy, res, windmon, d)
            # -- act: batteries
            fexp = {f: 0.0 for f in world.feeders}
            for s in sites:
                site = sites[s]
                b = site.battery
                sd = sday[s]
                pv, ld = sd.pv_kw[k], sd.load_kw[k]
                fid = site.feeder_id
                clipped = breached = False
                if b is None:
                    p, grid = 0.0, pv - ld
                else:
                    e = energy[s]
                    if sentinel and (s in live_pool[fid] or s in held[fid]):
                        req = plans[fid].schedule.get(s, [0.0] * DAY_STEPS)[k]
                        p, e2 = _clip_step(b, e, req)
                        clipped = abs(p - req) > 1e-3
                        breached = p < 0 and e2 < b.e_min - EPS
                        res.agent_steps += 1
                        res.clipped_steps += clipped
                    elif policy == "threshold":
                        p, e2, _ = comps.threshold_step(b, e, pv, ld, thresholds.get(s, 0.0))
                        breached = p < 0 and e2 < b.e_min - EPS
                    else:
                        p, e2, _ = self_consumption_step(b, e, pv, ld)
                        breached = p < 0 and e2 < b.e_min - EPS and not sentinel  # non-pool sites: their own rule
                    res.comfort_breaches += int(breached)
                    energy[s] = e2
                    executed[s][k] = p
                    grid = pv - ld - p
                    if cfg.write_schedules and (cfg.detail_sites is None or s in cfg.detail_sites):
                        sink.add("schedules", {"run_id": run_id, "asset_id": s, "policy": policy, "ts": tsx,
                                               "batt_kw": round(p, 4), "soc": round(e2 / b.capacity_kwh, 4),
                                               "comfort_reserve_kwh": round(b.comfort_reserve_kwh, 4),
                                               "comfort_breached": int(breached), "clipped": int(clipped)})
                fexp[fid] += grid
                res.cost_lei += prices_kwh[k] * (max(-grid, 0.0) - cfg.beta * max(grid, 0.0)) * DT \
                    + cfg.c_deg_lei_kwh * abs(executed[s][k] if b else 0.0) * DT
                od = oday[s]
                stale = od.telemetry_age_s[k] > 0
                detail = cfg.detail_sites is None or s in cfg.detail_sites
                if write_inputs and detail:  # telemetry once per DB (per-policy battery rows are in schedules)
                    sink.add("measurements", {"run_id": run_id, "asset_id": s, "ts": tsx,
                                              "pv_ac_kw": None if stale else od.pv_kw[k],
                                              "load_kw": None if stale else od.load_kw[k],
                                              "batt_kw": None if (stale or b is None) else round(p, 4),
                                              "soc": None if (stale or b is None) else round(energy[s] / b.capacity_kwh, 4),
                                              "inv_voltage_v": None if stale else od.inv_voltage_v[k],
                                              "inv_status": str(od.inv_status[k]), "data_quality": "gap" if stale else "ok"})
                if asset_steps and write_inputs and detail and (cfg.step_predictions or not _night_interior(sd.pv_kw, pv_p[s], k)):
                    sink.add("asset_steps", {"run_id": run_id, "asset_id": s, "ts": tsx, "actual_kw": pv,
                                             "q10_kw": pv_p[s].q10[k], "q50_kw": pv_p[s].q50[k],
                                             "q90_kw": pv_p[s].q90[k]})
                # realised flexibility vs the rule counterfactual (pool sites at t0)
                if sentinel and s in cf_e:
                    _, cf_e[s], cf_grid = self_consumption_step(b, cf_e[s], pv, ld)
                    realised[fid][k] += cf_grid - grid
            if write_inputs:
                for t in world.turbines:
                    td = tday[t]
                    sink.add("measurements", {"run_id": run_id, "asset_id": t, "ts": tsx,
                                              "act_pwt_kw": td.act_pwt_kw[k], "wd_spd_kmh": td.wd_spd_kmh[k],
                                              "ambient_temp_c": td.turbine_t_c[k], "gearbox_temp_c": td.gearbox_t_c[k],
                                              "error_code": None if td.error_code[k] is None else int(td.error_code[k]),
                                              "data_quality": "gap" if td.act_pwt_kw[k] is None else
                                              ("suspect" if td.error_code[k] else "ok")})
            for fid, feeder in world.feeders.items():
                g = feeder.spec.g_kw
                x = fexp[fid]
                over = x > g + 1e-6
                acc = fd[fid]
                acc["peak_export"] = max(acc["peak_export"], x)
                acc["minutes_over"] += 15.0 * over
                acc["kwh_over"] += max(0.0, x - g) * DT
                acc["peak_overshoot"] = max(acc["peak_overshoot"], x - g)
                if 17 <= local_h < 22:
                    acc["peak_import_evening"] = max(acc["peak_import_evening"], -x)
                sink.add("feeder_curves", {"run_id": run_id, "feeder_id": fid, "policy": policy, "ts": tsx,
                                           "export_kw": round(max(x, 0.0), 4), "import_kw": round(max(-x, 0.0), 4),
                                           "limit_kw": g, "slack_kw": round(max(0.0, x - g), 4),
                                           "is_violation": int(over)})
            if cfg.on_tick is not None:  # live driver: hand over this tick's decisions, wait for the next
                cfg.on_tick({"d": d, "k": k, "ts": ts, "feeder_net_kw": dict(fexp),
                             "batt_kw": {s: executed[s][k] for s in batt_sites},
                             "soc": {s: energy[s] / sites[s].battery.capacity_kwh for s in batt_sites},
                             "status": {s: last_status.get(s) for s in sites},
                             "pool": {f: sorted(live_pool[f] | held[f]) for f in live_pool},
                             "planned_kw": {s: plans[f].schedule.get(s, [0.0] * DAY_STEPS)[k]
                                            for f in plans for s in plans[f].schedule},
                             "committed_kw": {f: c.committed[k] for f, c in commitments.items()}})
        # ---------------------------------------------------------------- end of day: log
        prev_oday = oday
        for s in sites:  # calibration slice for the next days: what was forecast vs what was observed
            ok = np.asarray(oday[s].telemetry_age_s, float) == 0
            for key, p, y in (("pv", pv_p[s], oday[s].pv_kw), ("load", ld_p[s], oday[s].load_kw)):
                y, q50 = np.asarray(y, float), np.asarray(p.q50, float)
                e_obs, e_fc = float(y[ok].sum()), float(q50[ok].sum())
                if ok.sum() < DAY_STEPS // 2 or e_fc < 1.0 or not 1 / 3 <= e_obs / e_fc <= 3:
                    continue  # data-quality gate: a site-day off by > 3x in energy is a unit / sensor problem
                cal_hist[key].append((y, np.asarray(p.q10, float), q50, np.asarray(p.q90, float), ok))
        for fid in world.feeders:
            res.feeder_days[(fid, d)] = fd[fid]
        if judge is not None:  # M6 day verdicts, grading hook, ledger update (the day's telemetry is now observed)
            judge.at_end_of_day(d, t0, oday, tday, wd_p, pv_p, ld_p, hl_p)
        if sentinel:
            for fid, c in commitments.items():
                tot_c = tot_r = 0.0
                for k in range(DAY_STEPS):
                    cm, r = c.committed[k], realised[fid][k]
                    hit = r + 1e-6 >= cm if cm > 0 else None
                    if cm > 0:
                        res.commit_steps += 1
                        res.hits += int(hit)
                    tot_c += cm * DT
                    tot_r += r * DT
                    sink.add("commitments", {"run_id": run_id, "feeder_id": fid, "ts": Ts(world.tick_ts(d, k)),
                                             "nominal_kw": c.nominal[k], "available_kw": c.available[k],
                                             "committed_kw": cm, "realised_kw": round(r, 4),
                                             "haircut_h": c.extra.get("haircut_k", [c.haircut] * DAY_STEPS)[k],
                                             "hit": None if hit is None else int(hit)})
                c.realised = [round(v, 4) for v in realised[fid]]
                rec = A.audit(comps.sources["committer"], {"feeder": fid, "t0": t0, "pool": sorted(pool[fid]),
                                                           "inputs": [day_inputs[s] for s in sorted(pool[fid])]},
                              {"committed_kwh": round(tot_c, 3), "realised_kwh": round(tot_r, 3),
                               "nominal_kwh": round(sum(c.nominal) * DT, 3), "hit_rate": c.hit_rate(),
                               "n_pool": len(pool[fid])}, 0.10, "committed",
                              note=f"P10 commitment, haircut {c.haircut}; realised logged after delivery")
                act(A.feeder_commitment(t0, policy, run_id, fid, c.nominal, c.available, c.committed, rec,
                                        realised=c.realised, haircut=c.haircut),
                    day_ts, extra={"committed_kw": round(max(c.committed), 4), "realised_kw": None})
            for pid, acc in park_acc.items():
                if not acc["n"]:
                    continue
                rec = A.audit("park-report", {"park": pid, "t0": t0}, {"production_mwh": acc["prod"]}, None, "ok")
                a = A.make(ActionType.PARK_REPORT, t0, policy, run_id, {
                    "park_id": pid, "date": t0[:10], "energy_mwh": round(acc["prod"], 4),
                    "expected_mwh": round(acc["exp"], 4), "availability": round(acc["avail"] / acc["n"], 4),
                    "curtailed_mwh": round(acc["curt"], 4), "issues": acc["issues"],
                    "data_quality_pct": round(100 * acc["dq"] / acc["n"], 2)}, rec, pid)
                act(a, day_ts + DAY_STEPS * 900 - 900)
        else:  # baseline / threshold: one battery_schedule action per site-day (what the rule did), audited
            for s in batt_sites:
                rec = A.audit("rule:" + policy, {"site_inputs": day_inputs[s]}, {"e0_kwh": round(e_day0[s], 3)},
                              thresholds.get(s), "rule", executed[s], note=f"{policy} rule, executed schedule")
                rec.input_hash = day_inputs[s]
                act(A.battery_schedule(t0, policy, run_id, s, sites[s].feeder_id, executed[s], rec), day_ts)
    # close open curtailment episodes
    for pid in list(curt_ep):
        _close_curtailment(pid, curt_ep, act, run_id, policy, world.tick_ts(world.days - 1, DAY_STEPS - 1), res)
    if judge is not None:
        res.judge = judge.summary()
    if hasattr(sink, "flush"):
        sink.flush()
    return res


# ----------------------------------------------------------------------------------------------- helpers
def _slice(p: Prediction, k: int) -> Prediction:
    q = copy.copy(p)
    q.q10, q.q50, q.q90 = p.q10[k:], p.q50[k:], p.q90[k:]
    q.horizon = len(q.q50)
    q.t0 = iso(parse_iso(p.t0) + k * 900)
    return q


def _night_interior(actual: Sequence[float], pred: Prediction, k: int) -> bool:
    """Lite DB: an asset_steps row is skipped when it and both neighbours are all-zero (night). The rows that
    bound the night stay, so a chart interpolating between rows still draws 0 overnight."""
    def zero(j: int) -> bool:
        return 0 <= j < len(actual) and not actual[j] and not pred.q90[j] and not pred.q50[j] and not pred.q10[j]
    return zero(k - 1) and zero(k) and zero(k + 1)


def _gap_age(series: Sequence[Optional[float]], k: int) -> float:
    n = 0
    while k - n >= 0 and series[k - n] is None:
        n += 1
    return n * 15.0


def _wind_preds(world, wd_p, tday, k, tb_obs, under_run) -> dict[str, Prediction]:
    out = {}
    for t, base in wd_p.items():
        rated = world.turbines[t].rated_kw
        a = tday[t].act_pwt_kw[k]
        if a is not None:
            tb_obs[t][k] = a
        p = copy.copy(base)
        steps = (base.extra or {}).get("step_classes")
        if steps is not None and a is not None:  # real M6: its class at this step
            cls, score = steps[k], float(base.extra["step_scores"][k])
            if cls == "normal":
                score = 1.0 - score
        else:
            cls, score, under_run[t] = _wind_rules(rated, a, base.q50[k], under_run.get(t, 0))
            p.model_version = f"{base.model_version}+rules"
        p.anomaly_class, p.anomaly_score = cls, score
        out[t] = p
    return out


def _guarded(comps: Components, slot: str, fn: Callable, fallback: Callable, *args: Any, **kw: Any) -> Any:
    """fn(*args, **kw); if a real component raises, count the error and use its built-in fallback instead."""
    try:
        return fn(*args, **kw)
    except Exception as e:
        if fn is fallback:
            raise
        comps.note_error(slot, e)
        return fallback(*args, **kw)


def _safe_qualify(comps, assets, preds, ages, actuals, flags):
    return _guarded(comps, "qualify", comps.qualify_many, fallback_qualify_many, assets, preds, ages, None,
                    actuals=actuals, flags=flags)


def _safe_triage(comps, ass, t, state, config):
    try:
        return comps.triage(ass, t, state, config)
    except Exception as e:
        if comps.triage is fallback_triage:
            raise
        comps.note_error("triage", e)
        return fallback_triage(ass, t, None, FallbackTriageConfig(config.price_lei_mwh, config.policy,
                                                                  config.run_id))


def _plan(comps, world, spec, members, pool_ids, pv_p, ld_p, statuses, energy, prices_kwh, t0, res):
    """Call the planner for one feeder: pool = controlled sites, the others are the rest of the feeder."""
    sites = world.sites
    rows = {s: {**sites[s].row, "e0_kwh": energy.get(s)} for s in members}
    fc = {s: {"pv": pv_p[s], "load": ld_p[s]} for s in members}
    rest_ids = [s for s in members if s not in set(pool_ids)]
    H = len(next(iter(pv_p.values())).q50) if pv_p else DAY_STEPS
    try:
        rest = comps.rest_of_feeder([rows[s] for s in rest_ids], fc, statuses, H) if rest_ids else None
    except Exception as e:
        comps.note_error("rest_of_feeder", e)
        rest = _builtin_rest([rows[s] for s in rest_ids], fc, statuses, H)
    elig = [rows[s] for s in pool_ids]
    st = {s: statuses[s] for s in members if s in statuses}
    for fn, slot in ((comps.plan_feeder, "planner"), (comps.plan_feeder_backup, "planner_backup")):
        if fn is None or fn is builtin_plan_feeder:
            continue
        try:
            pl = fn(spec, elig, rest, fc, spec.g_kw, prices_kwh, statuses=st, t0=t0)
            break
        except Exception as e:
            comps.note_error(slot, e)
    else:
        pl = builtin_plan_feeder(spec, elig, rest, fc, spec.g_kw, prices_kwh, statuses=st, t0=t0)
    res.plan_status[pl.solver_status] = res.plan_status.get(pl.solver_status, 0) + 1
    return pl


def _blind_profile(prev_oday: Mapping[str, SiteDay]) -> dict[str, np.ndarray]:
    """Per site, the steps of the day it is expected to be blind (telemetry age over the qualify stale limit), from
    the same steps of the trailing day's observation, widened by +-1 h. Only what is known at t0; {} on day 0."""
    out = {}
    w = 4
    for s, od in prev_oday.items():
        blind = np.asarray(od.telemetry_age_s, float) > STALE_AGE_S
        if blind.any():
            pad = np.concatenate([np.zeros(w, bool), blind, np.zeros(w, bool)])
            out[s] = np.array([pad[k:k + 2 * w + 1].any() for k in range(len(blind))])
    return out


def _blind_haircut(c: Commitment, contrib: Mapping[str, np.ndarray], blind: Mapping[str, np.ndarray]) -> Commitment:
    """Commit-time telemetry haircut by step: committed_k *= 1 - loss_k, loss_k = share of the planned reduction at
    step k that comes from pool sites expected to be blind then (their flex cannot be verified or steered, so it is
    not sold). Effective h_k = 1 - (1 - h)(1 - loss_k) goes to extra['haircut_k'] and to commitments.haircut_h."""
    T = len(c.committed)
    num, den = np.zeros(T), np.zeros(T)
    for s, r in contrib.items():
        r = np.maximum(0.0, np.asarray(r, float)[:T])
        den += r
        if s in blind:
            num += r * blind[s][:T]
    loss = np.divide(num, den, out=np.zeros(T), where=den > 1e-9)
    if loss.any():
        c.committed = [float(x) * (1.0 - l) for x, l in zip(c.committed, loss)]
        c.extra["haircut_k"] = [round(1.0 - (1.0 - c.haircut) * (1.0 - l), 4) for l in loss]
        c.extra["blind_sites"] = sorted(s for s in contrib if s in blind)
    return c


def _fit_conformal(hist: Mapping[str, list[tuple[np.ndarray, ...]]], cfg: LoopConfig) -> dict[str, Any]:
    """Split-conformal offsets (wattsentry.eval.uncertainty.conformal, normalised asymmetric CQR) per quantity, fitted on
    every site-day already replayed (observed steps only): no information from the day being committed. Site-days
    whose observed energy is off the forecast by > 3x are dropped (unit / sensor faults would poison the offsets) and
    offsets are clipped to +-2 band widths. {} on day 0 or when the module is missing."""
    try:
        from wattsentry.eval.uncertainty.conformal import ConformalCalibration, default_mask
    except ImportError:
        return {}
    out = {}
    for key, rows in hist.items():
        if not rows:
            continue
        y, q10, q50, q90, ok = (np.stack([r[i] for r in rows]) for i in range(5))
        m = ok & default_mask(y, q10, q90)
        if m.sum() < 200:
            continue
        c = ConformalCalibration.fit(y, (q10, q50, q90), mask=m, n_buckets=cfg.conformal_buckets,
                                     model=key, model_version="online-past-days")
        for mc in c.members.values():  # bounded: at most +-2 band widths per quantile
            mc.offsets = {k: [float(np.clip(v, -2.0, 2.0)) for v in vs] for k, vs in mc.offsets.items()}
        out[key] = c
    return out


def _plan_and_commit(comps, cfg, world, spec, members, pool_ids, pv_p, ld_p, statuses, energy, prices_kwh, t0,
                     res, blind: Optional[Mapping[str, np.ndarray]] = None, cal: Optional[Mapping[str, Any]] = None):
    pl = _plan(comps, world, spec, members, pool_ids, pv_p, ld_p, statuses, energy, prices_kwh, t0, res)
    sites = world.sites
    pv_d = {s: _scaled(pv_p[s], _pv_factor(statuses.get(s))) for s in pool_ids}
    red = np.zeros(pl.horizon)
    contrib = {}
    for s in pool_ids:  # planned export reduction vs the rule on P50 (fallback committer, audit, blind haircut)
        b = sites[s].battery
        e = energy[s]
        sched = pl.schedule.get(s, [0.0] * pl.horizon)
        r = contrib[s] = np.zeros(pl.horizon)
        for t in range(pl.horizon):
            p_rule, e, _ = self_consumption_step(b, e, pv_d[s].q50[t], ld_p[s].q50[t])
            r[t] = sched[t] - p_rule
        red += r
    ld_c = {s: ld_p[s] for s in pool_ids}
    if cal:  # the commitment samples calibrated bands; the plan itself stays on the models' quantiles
        if "pv" in cal:
            pv_d = {s: cal["pv"].apply_prediction(pv_d[s]) for s in pool_ids}
        if "load" in cal:
            ld_c = {s: cal["load"].apply_prediction(ld_c[s]) for s in pool_ids}
    ctx = {"planned_reduction": red, "haircut": cfg.haircut, "pv": pv_d, "load": ld_c,
           "batteries": {s: sites[s].battery for s in pool_ids},
           "soc0": {s: energy[s] / sites[s].battery.capacity_kwh for s in pool_ids},
           "pv_cap": {s: float(sites[s].row.get("kwp_recorded") or 10.0) for s in pool_ids},
           "n_scenarios": cfg.n_scenarios, "seed": cfg.seed}
    c = _guarded(comps, "committer", comps.commit, fallback_commit, spec, pl, ctx)
    if blind:
        c = _blind_haircut(c, contrib, blind)
    return pl, c


def _wind_tick(k, ts, ass, wd_p, tday, park_acc, curt_ep, act, run_id, policy, sink, write_inputs, res):
    """Park forecasts, curtailment episodes (one alert per park episode, no work order), park report sums."""
    by_park: dict[str, list[Any]] = {}
    for x in ass:
        if x.asset_kind == "turbine":
            by_park.setdefault(x.group_id, []).append(x)
    for pid, xs in by_park.items():
        acc = park_acc[pid]
        q10 = q50 = q90 = prod = curt = 0.0
        avail = dq = 0
        curtailed = []
        for x in xs:
            t = x.asset_id
            p = wd_p[t]
            a = tday[t].act_pwt_kw[k]
            q10, q50, q90 = q10 + p.q10[k], q50 + p.q50[k], q90 + p.q90[k]
            prod += a or 0.0
            avail += x.status.in_pool
            dq += a is not None
            if x.status.reason.startswith("curtailed"):
                curtailed.append(t)
                curt += max(0.0, p.q50[k] - (a or 0.0))
        n = len(xs)
        acc["prod"] += prod * DT / 1000
        acc["exp"] += q50 * DT / 1000
        acc["curt"] += curt * DT / 1000
        acc["avail"] += avail / n
        acc["dq"] += dq / n
        acc["n"] += 1
        if write_inputs:
            sink.add("park_forecasts", {"run_id": run_id, "park_id": pid, "ts": Ts(ts), "p10_mw": q10 / 1000,
                                        "p50_mw": q50 / 1000, "p90_mw": q90 / 1000, "expected_mwh": q50 * DT / 1000,
                                        "availability_pct": round(100 * avail / n, 2),
                                        "curtailed_mwh": curt * DT / 1000, "data_quality_pct": round(100 * dq / n, 2)})
        ep = curt_ep.get(pid)
        if curtailed:
            if ep is None:
                ep = curt_ep[pid] = {"start": ts, "mwh": 0.0, "turbines": set(), "x": xs[0]}
                acc["issues"].append({"t": iso(ts), "code": "curtailment", "turbines": len(curtailed)})
            ep["mwh"] += curt * DT / 1000
            ep["turbines"].update(curtailed)
            ep["end"] = ts
        elif ep is not None:
            _close_curtailment(pid, curt_ep, act, run_id, policy, ts, res)
    for x in ass:  # issues for the park report
        if x.asset_kind == "turbine" and x.crew_type and k % 16 == 0:
            park_acc[x.group_id]["issues"].append({"t": iso(ts), "asset_id": x.asset_id, "code": x.fault})


def _close_curtailment(pid, curt_ep, act, run_id, policy, ts, res):
    ep = curt_ep.pop(pid)
    start = ep["start"]
    rec = A.audit("rules+qualify", {"park": pid, "start": start, "turbines": sorted(ep["turbines"])},
                  {"curtailed_mwh": round(ep["mwh"], 4), "turbines": len(ep["turbines"])}, 0.6, "eligible",
                  note="park-wide cap: curtailment, logged, no work order")
    a = A.make(ActionType.ALERT, iso(start), policy, run_id, {
        "asset_id": pid, "asset_kind": "park", "park_id": pid, "severity": Severity.INFO.value,
        "code": "curtailment", "creates_work_order": False, "curtailed_mwh": round(ep["mwh"], 4),
        "message": f"Park {pid}: {len(ep['turbines'])} turbines capped together from {iso(start)} to "
                   f"{iso(ep.get('end', ts))}: curtailment, {ep['mwh']:.2f} MWh, no work order."}, rec, pid, start)
    act(a, start)
    res.turbine_alerts += 1


def _windmon_precompute(world, comps):
    """WindMon rules over the whole replay (causal: each step uses only past steps); None -> builtin rules."""
    fn = comps.windmon_flags
    if fn is None:
        return None
    try:
        got = world.turbine_sensors()
        if got is None:
            return None
        ids, sens, err, rated = got
        flags = fn(sens, err, rated)
        return {"col": {t: j for j, t in enumerate(ids)}, "flags": {r: np.asarray(f, bool) for r, f in flags.items()}}
    except Exception as e:
        comps.note_error("wind_baseline", e)
        comps.sources["wind_baseline"] = "fallback:builtin_windmon_rules (#24 import ok, run failed)"
        return None


def _windmon_tick(world, k, ts, tday, active, act, run_id, policy, res, windmon=None, d=0):
    """Baseline wind O&M: WindMon's four rules on raw SCADA, one alert per rule episode per turbine (the eval
    baseline's rules when importable, else the builtin stand-in)."""
    for t, tb in world.turbines.items():
        td = tday[t]
        a, w, gb = td.act_pwt_kw[k], td.wd_spd_kmh[k], td.gearbox_t_c[k]
        if windmon is not None:
            row, j = d * DAY_STEPS + k, windmon["col"][t]
            fired = {r: bool(f[row, j]) for r, f in windmon["flags"].items()}
            model, note = "windmon_rules:#24", "WindMon rule (#24, wattsentry.eval.baselines.wind)"
        else:
            fired = {
                "HIGH_WIND_ZERO_POWER": a is not None and w is not None and w > 15.0 and a < 10.0,
                "DATA_GAP": a is None,
                "TEMPERATURE_SPIKE": gb is not None and gb > 85.0,
                "SUSPECT_DATA": td.error_code[k] != 0 and is_error_code("gearbox_t", float(td.error_code[k])),
            }
            model, note = "windmon_rules:builtin", "WindMon rule (builtin #24 stand-in)"
        for rule, thr in _WINDMON:
            on = fired.get(rule, False)
            if on and not active.get((t, rule)):
                rec = A.audit(model, {"t": t, "ts": ts, "act": a, "wd_spd": w, "gearbox_t": gb},
                              {"act_pwt_kw": a, "wd_spd_kmh": w}, thr, "alert", note=note)
                act(A.make(ActionType.ALERT, iso(ts), policy, run_id, {
                    "asset_id": t, "asset_kind": "turbine", "park_id": tb.park_id,
                    "severity": Severity.WARNING.value, "code": rule,
                    "message": f"{rule} on {t}"}, rec, t, rule), ts)
                res.turbine_alerts += 1
            active[(t, rule)] = on
