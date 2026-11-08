"""GridTwin (M8) in the loop (card G3 #161, team/features/grid.md step 4): does checking the map change the decision?

Three measurements, all on the replay world (synthetic RO fleet; label every number "synthetic RO" + run id):

1. `recovery(...)`: gis_recovery_rate, the share of the corrupted GIS records (~10 % wrong phase + ~3 % wrong feeder,
   injected by the simulator) that M8 flags *and* puts back on the right feeder (feeder errors) or phase (phase
   errors). Same definitions as `run.score`, pooled over both kinds.
2. `tau_effect(...)`: the same replay week with the LP limit G = tau*G_max + (1-tau)*G_cons (M8's trust) and with
   tau = 1 (G = G_max, map taken as-is). Minutes / kWh over the limit are counted against both the physical cable
   limit G_max and the tau limit, so the comparison does not move the goalposts.
3. `moved_home_case(...)`: one home the GIS puts on the wrong feeder, which M8 moves. The hero feeder's (F7)
   day-ahead plan is solved twice, on the GIS membership and on the M8-corrected one, and both plans are played
   against the physical feeder (truth membership, actual PV / load). Output: the plan diff and the over-limit result.

M8 outputs are read through the C1 adapter (wattsentry/judge/adapter.py, #157): a flag the Judge says "dont_use" is
not applied, and a feeder tau it says "dont_use" becomes 0 (the conservative limit). Until #157 merges a pass-through
shim with the same signature stands in, so the swap is the import below.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from types import SimpleNamespace
from typing import Any, Mapping, Optional, Sequence

import numpy as np

from wattsentry.contracts.opt import FeederSpec, self_consumption_step

DT = 0.25
STEPS = 96
MODEL = "gridtwin"

try:  # C1 #157: every model read goes through the adapter
    from wattsentry.judge.adapter import validated as _validated
    ADAPTER = "wattsentry.judge.adapter"
except ImportError:  # pragma: no cover - until #157 merges: pass-through with the adapter's signature
    ADAPTER = "pass-through shim (C1 #157 not on main yet)"

    def _validated(asset_id: Any, model: str, day: Any, *, raw: Any, substitutes: Any = None, judge: Any = None):
        get = raw.get if isinstance(raw, Mapping) else (lambda k, d=None: getattr(raw, k, d))
        return SimpleNamespace(known=raw is not None, verdict="use", rule="no_judge", reasons=(),
                               cls=None if raw is None else get("cls"), score=None if raw is None else get("score"))


# ============================================================================================ 1. recovery
def recovery(sites_res, truth) -> dict[str, Any]:
    """gis_recovery_rate over the corrupted records. `sites_res`: TwinResult.sites (gis_*, likely_*, *_flag);
    `truth`: the simulator's sites (site_id, feeder_id, phase), never seen by the twin."""
    s = sites_res.merge(truth[["site_id", "feeder_id", "phase"]], on="site_id")
    wrong_feeder = s.gis_feeder_id != s.feeder_id
    wrong_phase = (s.gis_phase != s.phase) & ~wrong_feeder
    flagged = s.feeder_flag | s.phase_flag
    ok_feeder = s.feeder_flag & (s.likely_feeder_id == s.feeder_id)
    ok_phase = s.phase_flag & (s.likely_phase == s.phase)
    rec_f, rec_p = int((ok_feeder & wrong_feeder).sum()), int((ok_phase & wrong_phase).sum())
    n_f, n_p = int(wrong_feeder.sum()), int(wrong_phase.sum())
    n_bad = n_f + n_p
    correct_flags = int((ok_feeder & wrong_feeder).sum() + (ok_phase & wrong_phase).sum())
    return {
        "gis_recovery_rate": round((rec_f + rec_p) / n_bad, 4) if n_bad else 1.0,
        "gis_flag_precision": round(correct_flags / int(flagged.sum()), 4) if flagged.any() else 1.0,
        "n_records": int(len(s)), "n_corrupted": n_bad, "corrupted_share": round(n_bad / max(1, len(s)), 4),
        "n_wrong_feeder": n_f, "n_wrong_phase": n_p, "recovered_feeder": rec_f, "recovered_phase": rec_p,
        "n_flags": int(flagged.sum()), "false_flags_on_clean": int((flagged & ~(wrong_feeder | (s.gis_phase != s.phase))).sum()),
    }


# ============================================================================================ adapter reads
def validated_tau(world, day: str) -> dict[str, float]:
    """Feeder tau through the adapter; unknown (Judge: dont_use, no substitute) -> 0, i.e. G = G_conservative."""
    out = {}
    for fid, f in world.feeders.items():
        v = _validated(fid, MODEL, day, raw={"cls": "feeder_trust", "score": float(f.spec.tau)})
        out[fid] = float(v.score) if v.known and v.score is not None else 0.0
    return out


def validated_moves(flags: Sequence[Mapping[str, Any]], day: str) -> list[dict[str, Any]]:
    """Feeder moves M8 proposes (world.gis_flags() rows) that the adapter lets through."""
    out = []
    for fl in flags:
        if fl.get("suggested_feeder") in (None, fl.get("recorded_feeder")):
            continue  # phase-only flag: the feeder LP does not change
        v = _validated(fl["site_id"], MODEL, day, raw={"cls": "feeder", "score": fl.get("score")})
        if v.known and v.verdict != "dont_use":
            out.append({**fl, "verdict": v.verdict, "rule": v.rule})
    return out


# ============================================================================================ 2. tau effect
@dataclass
class OverLimit:
    minutes: float = 0.0
    kwh: float = 0.0
    peak_overshoot_kw: float = 0.0
    feeder_days_over: int = 0
    per_feeder: dict[str, float] = field(default_factory=dict)  # minutes

    def to_dict(self) -> dict[str, Any]:
        return {"minutes": round(self.minutes, 1), "kwh": round(self.kwh, 2),
                "peak_overshoot_kw": round(self.peak_overshoot_kw, 2), "feeder_days_over": self.feeder_days_over,
                "per_feeder_minutes": {k: round(v, 1) for k, v in sorted(self.per_feeder.items())}}


def over_limit(curves: Sequence[Mapping[str, Any]], limits: Mapping[str, float]) -> OverLimit:
    """Minutes / kWh above `limits[feeder]` from feeder_curves rows (export_kw, import_kw per 15-min step)."""
    o = OverLimit(per_feeder={f: 0.0 for f in limits})
    days: set[tuple[str, int]] = set()
    for r in curves:
        f = r["feeder_id"]
        if f not in limits:
            continue
        x = float(r["export_kw"]) - float(r.get("import_kw") or 0.0)
        g = float(limits[f])
        if x > g + 1e-6:
            o.minutes += 15.0
            o.per_feeder[f] += 15.0
            o.kwh += (x - g) * DT
            o.peak_overshoot_kw = max(o.peak_overshoot_kw, x - g)
            days.add((f, int(r["ts"]) // 86400))
    o.feeder_days_over = len(days)
    return o


def _set_tau(world, tau: Mapping[str, float]) -> dict[str, FeederSpec]:
    old = {fid: f.spec for fid, f in world.feeders.items()}
    for fid, f in world.feeders.items():
        sp = f.spec
        f.spec = FeederSpec(sp.feeder_id, sp.g_max_kw, sp.g_conservative_kw, float(tau.get(fid, sp.tau)))
    return old


def _restore(world, old: Mapping[str, FeederSpec]) -> None:
    for fid, sp in old.items():
        world.feeders[fid].spec = sp


def tau_effect(world, comps=None, run_id: str = "local", policies: Sequence[str] = ("baseline", "sentinel")
               ) -> dict[str, Any]:
    """Replay the world's week with M8's tau limit and with tau = 1. Baseline batteries ignore the limit, so its
    curves are the same in both arms (one run, both yardsticks)."""
    from wattsentry.agent.loop import LoopConfig, _MemSink, default_components, run_policy

    comps = comps or default_components(world)
    day0 = world.week_start
    tau_m8 = validated_tau(world, day0)
    g_max = {fid: f.spec.g_max_kw for fid, f in world.feeders.items()}
    old = _set_tau(world, tau_m8)
    g_tau = {fid: f.spec.g_kw for fid, f in world.feeders.items()}
    arms: dict[str, Any] = {}
    try:
        for arm, tau in (("tau", tau_m8), ("no_tau", {f: 1.0 for f in world.feeders})):
            _set_tau(world, tau)
            for pol in policies:
                if pol != "sentinel" and arm == "no_tau":
                    continue
                sink = _MemSink()
                cfg = LoopConfig(policy=pol, run_id=f"{run_id}:{pol}:{arm}", write_inputs=False,
                                 step_predictions=False, write_schedules=False)
                res = run_policy(world, comps, cfg, sink=sink)
                curves = sink.rows.get("feeder_curves", [])
                arms[f"{pol}_{arm}" if pol == "sentinel" else pol] = {
                    "vs_g_max": over_limit(curves, g_max).to_dict(),
                    "vs_g_tau": over_limit(curves, g_tau).to_dict(),
                    "lei_cost": round(res.cost_lei, 2), "comfort_breaches": res.comfort_breaches,
                    "plan_status": dict(res.plan_status)}
    finally:
        _restore(world, old)
    s_t, s_n = arms.get("sentinel_tau"), arms.get("sentinel_no_tau")
    out = {"tau": {f: round(t, 3) for f, t in tau_m8.items()},
           "g_max_kw": {f: round(g, 2) for f, g in g_max.items()},
           "g_tau_kw": {f: round(g, 2) for f, g in g_tau.items()}, "arms": arms, "adapter": ADAPTER}
    if s_t and s_n:
        out["minutes_over_g_max_tau"] = s_t["vs_g_max"]["minutes"]
        out["minutes_over_g_max_no_tau"] = s_n["vs_g_max"]["minutes"]
        out["minutes_over_g_tau_tau"] = s_t["vs_g_tau"]["minutes"]
        out["minutes_over_g_tau_no_tau"] = s_n["vs_g_tau"]["minutes"]
        out["kwh_over_g_max_tau"] = s_t["vs_g_max"]["kwh"]
        out["kwh_over_g_max_no_tau"] = s_n["vs_g_max"]["kwh"]
        out["lei_cost_delta_tau"] = round(s_t["lei_cost"] - s_n["lei_cost"], 2)
    return out


# ============================================================================================ 3. one moved home
def gis_members(world) -> dict[str, list[str]]:
    """Feeder -> sites as the GIS records them (what Sentinel would plan on without M8)."""
    out: dict[str, list[str]] = {f: [] for f in world.feeders}
    for sid, s in world.sites.items():
        g = s.row.get("gis_feeder_id") or s.row["feeder_id"]
        out.setdefault(g, []).append(sid)
    return out


def apply_moves(members: Mapping[str, Sequence[str]], moves: Sequence[Mapping[str, Any]]) -> dict[str, list[str]]:
    out = {f: list(v) for f, v in members.items()}
    for m in moves:
        sid, a, b = m["site_id"], m["recorded_feeder"], m["suggested_feeder"]
        if sid in out.get(a, []):
            out[a].remove(sid)
        out.setdefault(b, [])
        if sid not in out[b]:
            out[b].append(sid)
    return out


def pick_case(flags: Sequence[Mapping[str, Any]], truth_feeder: Mapping[str, str], hero: str = "F7"
              ) -> Optional[dict[str, Any]]:
    """The demo case: a correct M8 feeder move involving the hero feeder. Prefer a home the GIS wrongly puts ON the
    hero feeder (grid.md: "placed wrongly on F7, M8 moves it to F3"), else one it wrongly puts elsewhere; highest
    score first. Only moves that match the truth qualify (we show a recovery, not a false alarm)."""
    ok = [f for f in flags if f.get("suggested_feeder") != f.get("recorded_feeder")
          and truth_feeder.get(f["site_id"]) == f["suggested_feeder"]]
    for want in ("recorded_feeder", "suggested_feeder"):
        c = [f for f in ok if f[want] == hero]
        if c:
            return dict(max(c, key=lambda f: (f.get("score") or 0, f["site_id"])), case=f"{want}={hero}")
    return None


def _day_inputs(world, comps, d: int, sids: Sequence[str]):
    from wattsentry.agent.loop import _predict, _safe_qualify

    t0 = world.day_t0(d)
    preds = comps.predictors
    rows = {s: world.sites[s].row for s in sids}
    pv = {s: _predict(preds["pv"], rows[s], t0) for s in sids}
    ld = {s: _predict(preds["load"], rows[s], t0) for s in sids}
    if hasattr(preds["sitehealth"], "prefetch"):
        preds["sitehealth"].prefetch(list(sids), t0, 0)
    hl = {s: _predict(preds["sitehealth"], rows[s], t0) for s in sids}
    od = {s: world.observed_site_day(s, d) for s in sids}
    ass = _safe_qualify(comps, [rows[s] for s in sids], hl, {s: od[s].telemetry_age_s[0] / 60 for s in sids}, None, {})
    return t0, pv, ld, {x.asset_id: x.status for x in ass}


def plan_on(world, comps, spec: FeederSpec, members: Sequence[str], d: int, inputs=None):
    """Day-ahead plan for one feeder on a given membership (loop._plan, batteries start at e_min like day 0)."""
    from wattsentry.agent.loop import PolicyResult, _plan

    members = [s for s in members if s in world.sites]
    t0, pv, ld, st = inputs or _day_inputs(world, comps, d, members)
    energy = {s: world.sites[s].battery.e_min for s in members if world.sites[s].battery is not None}
    pool = [s for s in members if s in energy and st[s].in_pool and st[s].battery_ok]
    prices_kwh = [p / 1000 for p in world.prices(d)]
    res = PolicyResult("sentinel", "case", 1)
    return _plan(comps, world, spec, members, pool, pv, ld, st, energy, prices_kwh, t0, res), pool


def play_feeder(world, fid: str, d: int, schedule: Mapping[str, Sequence[float]], limit_kw: float) -> dict[str, Any]:
    """Physical feeder `fid` on day d: its true member sites with actual PV / load; a site with a schedule follows it
    (clipped like the loop), every other site runs its own self-consumption rule."""
    from wattsentry.agent.loop import _clip_step

    x = np.zeros(STEPS)
    for s in world.feeders[fid].site_ids:
        site, sd = world.sites[s], world.site_day(s, d)
        b = site.battery
        e = b.e_min if b is not None else 0.0
        req = schedule.get(s)
        for k in range(STEPS):
            pv, ldk = sd.pv_kw[k], sd.load_kw[k]
            if b is None:
                x[k] += pv - ldk
            elif req is not None:
                p, e = _clip_step(b, e, req[k])
                x[k] += pv - ldk - p
            else:
                _, e, g = self_consumption_step(b, e, pv, ldk)
                x[k] += g
    over = np.maximum(x - limit_kw, 0.0)
    return {"minutes_over": float(15 * (over > 1e-6).sum()), "kwh_over": round(float(over.sum() * DT), 2),
            "peak_kw": round(float(x.max()), 2), "limit_kw": round(limit_kw, 2), "export_kw": np.round(x, 2).tolist()}


def _midday_kwh(sched: Sequence[float], world, d: int) -> float:
    from wattsentry.agent.audit import local_dt

    return round(sum(max(0.0, p) * DT for k, p in enumerate(sched) if 10 <= local_dt(world.tick_ts(d, k)).hour < 15), 2)


def moved_home_case(world, comps=None, d: int = 2, hero: str = "F7", case: Optional[Mapping[str, Any]] = None
                    ) -> dict[str, Any]:
    """Plan diff on the hero feeder: GIS membership vs M8-corrected membership, both played against the physics."""
    from wattsentry.agent.loop import default_components

    comps = comps or default_components(world)
    day = world.week_start
    moves = validated_moves(world.gis_flags(), day)
    truth = {s: w.feeder_id for s, w in world.sites.items()}
    case = dict(case) if case else pick_case(moves, truth, hero)
    if case is None:
        return {"found": False, "hero": hero, "n_moves": len(moves),
                "note": f"no correct M8 feeder move touches {hero} in this world"}
    gis = gis_members(world)
    fixed = apply_moves(gis, [case])
    spec = world.feeders[hero].spec
    sids = sorted(set(gis.get(hero, [])) | set(fixed.get(hero, [])) | set(world.feeders[hero].site_ids))
    inputs = _day_inputs(world, comps, d, sids)
    p_gis, pool_gis = plan_on(world, comps, spec, gis.get(hero, []), d, inputs)
    p_fix, pool_fix = plan_on(world, comps, spec, fixed.get(hero, []), d, inputs)
    sid = case["site_id"]
    sites = sorted(set(p_gis.schedule) | set(p_fix.schedule))
    diff = []
    for s in sites:
        a, b = p_gis.schedule.get(s), p_fix.schedule.get(s)
        da = None if a is None else _midday_kwh(a, world, d)
        db = None if b is None else _midday_kwh(b, world, d)
        dmax = max(abs((a[k] if a else 0.0) - (b[k] if b else 0.0)) for k in range(STEPS))
        if dmax > 1e-3:
            diff.append({"site_id": s, "midday_charge_kwh_gis": da, "midday_charge_kwh_m8": db,
                         "max_abs_delta_kw": round(dmax, 3), "moved": s == sid})
    diff.sort(key=lambda r: (not r["moved"], -r["max_abs_delta_kw"]))
    phys_gis = play_feeder(world, hero, d, p_gis.schedule, spec.g_kw)
    phys_fix = play_feeder(world, hero, d, p_fix.schedule, spec.g_kw)
    b = world.sites[sid].battery
    return {
        "found": True, "hero": hero, "day": d, "local_date": _local_date(world, d), "case": case["case"],
        "site_id": sid, "recorded_feeder": case["recorded_feeder"], "suggested_feeder": case["suggested_feeder"],
        "true_feeder": truth[sid], "score": case.get("score"), "message": case.get("message"),
        "verdict": case.get("verdict"), "site_has_battery": b is not None,
        "site_battery_kwh": None if b is None else b.capacity_kwh, "site_kwp": world.sites[sid].row.get("kwp_recorded"),
        "limit_kw": round(spec.g_kw, 2), "tau": spec.tau,
        "pool_gis": len(pool_gis), "pool_m8": len(pool_fix),
        "planned_peak_gis_kw": round(max(p_gis.export_curve), 2), "planned_peak_m8_kw": round(max(p_fix.export_curve), 2),
        "physical_gis": {k: v for k, v in phys_gis.items() if k != "export_kw"},
        "physical_m8": {k: v for k, v in phys_fix.items() if k != "export_kw"},
        "export_kw_gis": phys_gis["export_kw"], "export_kw_m8": phys_fix["export_kw"],
        "n_sites_changed": len(diff), "plan_diff": diff[:12], "n_moves": len(moves), "adapter": ADAPTER,
    }


def _local_date(world, d: int) -> str:
    from wattsentry.agent.audit import local_dt

    return local_dt(world.day_ts(d)).date().isoformat()
