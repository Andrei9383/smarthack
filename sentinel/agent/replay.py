"""Replay a test week through the agent, once per policy, into one SQLite file (card #17, plan §3.6 `replay`).

Same world, same predictions, same prices and limits for every policy; only decide/act differ. One `runs` row per
policy (label from the world, e.g. "synthetic RO"; components real vs fallback in `notes`), topology once
(feeders, parks, assets), then every row the loop writes. Prints the plan §2 replay metrics as LAB_METRIC lines.

Judge (#149): on by default for the Sentinel policy (`--judge off` to skip). It writes judge_verdicts (+ trust, and
outcomes once the J1 grader is in) and logs `t_judge_s` per replay day. `--judge-check` replays Sentinel a second
time with the judge off (rows discarded) and fails unless every KPI and every action id is identical: the judge only
records verdicts until the actions are specified (judge-design §10).
"""

from __future__ import annotations

import json
import math
import os
import subprocess
import sys
import time
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Optional, Sequence

from .audit import REPO_ROOT, AuditDB, Ts
from .loop import BASELINE_ALIASES, Components, LoopConfig, PolicyResult, default_components, log, run_policy
from .world import World, make_world

METRICS = ("peak_reduction", "minutes_over_limit", "p10_hit_rate", "comfort_breaches", "lei_per_home",
           "alerts_per_turbine_day")


class JudgeChangedDecisions(RuntimeError):
    """--judge-check: the Sentinel run with the judge on differs from the run with it off."""


class _NullSink:
    """Discards every row (the judge-off comparison run of --judge-check)."""

    def add(self, table, row):
        pass

    def add_many(self, table, rows):
        pass

    def has_table(self, table):
        return True

    def write_action(self, action, rows=None, *, ts=None, extra=None):
        pass


def _emit(name: str, value: float) -> None:
    from sentinel.infra.metrics import emit

    emit(name, value)


def _judge_metrics(summary: dict[str, Any], days: int) -> None:
    if not summary:
        return
    per_day = summary.get("t_judge_day_s") or {}
    total = float(summary.get("t_judge_s") or 0.0)
    _emit("t_judge_s", total)
    _emit("t_judge_per_day_s", total / max(1, days))
    for d, v in sorted(per_day.items()):
        _emit(f"t_judge_d{d}_s", v)
    counts = summary.get("verdicts") or {}
    n = sum(v for k, v in counts.items() if not k.startswith("rule:"))
    _emit("judge_verdicts", n)
    for v in ("use", "widen", "dont_use"):
        _emit(f"judge_{v}_share", counts.get(v, 0) / n if n else 0.0)
    _emit("judge_trust_rows", summary.get("trust_rows", 0))
    _emit("judge_outcomes", summary.get("outcomes", 0))
    _emit("judge_ledger_only", float(summary.get("mode") == "ledger-only"))
    print(f"# judge: {json.dumps({k: summary[k] for k in ('mode', 'source', 'ledger', 'verdicts', 'errors', 'tables_missing') if k in summary})}",
          file=sys.stderr)


@dataclass
class ReplayReport:
    out: Path
    results: dict[str, PolicyResult]
    metrics: dict[str, float]
    db: dict[str, Any]
    components: dict[str, str]
    seconds: float
    run_ids: dict[str, str] = field(default_factory=dict)
    judge_check: Optional[dict[str, Any]] = None


def _git_sha() -> Optional[str]:
    try:
        return subprocess.run(["git", "rev-parse", "HEAD"], cwd=REPO_ROOT, capture_output=True, text=True,
                              timeout=5).stdout.strip() or None
    except Exception:
        return None


def _predictor_report(preds: dict[str, Any]) -> dict[str, Any]:
    from .predictors import predictor_report

    return predictor_report({k: v for k, v in preds.items() if k != "zone"})


def feeder_line(head: tuple[float, float], points: Sequence[tuple[float, float]], max_points: int = 200
                ) -> dict[str, Any]:
    """GeoJSON LineString ([lon, lat]) for a radial LV feeder: a minimum spanning tree over the substation head and
    the feeder's sites (Prim, local equirectangular distance), walked depth-first from the head so one line draws
    every branch (back along a branch before the next). Falls back to sites ordered by distance from the head when
    the walk would exceed `max_points`. Deterministic (ties by input order)."""
    pts = [tuple(map(float, head))] + [tuple(map(float, p)) for p in points]
    n = len(pts)
    if n == 1:
        return {"type": "LineString", "coordinates": [list(pts[0]), list(pts[0])]}
    kx = math.cos(math.radians(pts[0][1]))

    def d2(a: int, b: int) -> float:
        return ((pts[a][0] - pts[b][0]) * kx) ** 2 + (pts[a][1] - pts[b][1]) ** 2

    parent = [-1] * n
    best = [d2(0, j) for j in range(n)]
    done = [False] * n
    done[0] = True
    children: dict[int, list[int]] = {i: [] for i in range(n)}
    for j in range(1, n):
        parent[j] = 0
    for _ in range(n - 1):
        j = min((k for k in range(n) if not done[k]), key=lambda k: best[k])
        done[j] = True
        children[parent[j]].append(j)
        for k in range(n):
            if not done[k] and d2(j, k) < best[k]:
                best[k], parent[k] = d2(j, k), j
    walk: list[int] = []

    def dfs(i: int) -> None:
        walk.append(i)
        for c in sorted(children[i], key=lambda c: d2(i, c)):
            dfs(c)
            walk.append(i)

    sys.setrecursionlimit(max(1000, 4 * n))
    dfs(0)
    while len(walk) > 1 and walk[-1] == walk[-2]:
        walk.pop()
    if len(walk) > max_points:
        walk = sorted(range(n), key=lambda k: d2(0, k))[:max_points]
    # trim the final return to the head: the line ends at the last branch tip
    while len(walk) > 2 and walk[-1] in walk[:-1]:
        walk.pop()
    return {"type": "LineString", "coordinates": [[round(pts[k][0], 5), round(pts[k][1], 5)] for k in walk]}


def write_topology(db: AuditDB, world: World) -> None:
    for fid, f in world.feeders.items():
        sites = [world.sites[s] for s in f.site_ids]
        geom = feeder_line((f.lon, f.lat), [(float(s.row["lon"]), float(s.row["lat"])) for s in sites])
        db.add("feeders", {
            "id": fid, "name": f"{fid} {f.locality}", "county": f.county,
            "locality": f.locality, "limit_kw": f.spec.g_kw, 
            "tau": f.spec.tau, "gis_trust": "low" if f.spec.tau < 0.8 else "ok", "n_sites": len(sites),
            "g_max_kw": f.spec.g_max_kw, "g_conservative_kw": f.spec.g_conservative_kw,
            "installed_pv_kw": round(sum(float(s.row.get("kwp_recorded") or 0) for s in sites), 2),
            "installed_batt_kwh": round(sum(s.battery.capacity_kwh for s in sites if s.battery), 2),
            "geometry_json": geom})
    for pid, p in world.parks.items():
        rated = sum(world.turbines[t].rated_kw for t in p.turbine_ids)
        db.add("parks", {"id": pid, "name": p.name, "county": p.county, "lat": p.lat,
                         "lon": p.lon, "turbine_count": len(p.turbine_ids), "rated_mw": rated / 1000})
    for sid, s in world.sites.items():
        r, b = s.row, s.battery
        db.add("assets", {
            "id": sid, "asset_kind": "site", "name": f"{sid} {r.get('locality', '')}", "lat": r["lat"],
            "lon": r["lon"], "county": r.get("county") or "", "locality": r.get("locality"),
            "feeder_id": r.get("gis_feeder_id") or r.get("feeder_id"), "park_id": None,
            "rated_kw": float(r.get("kwp_recorded") or 0.0), "has_battery": int(b is not None),
            "battery_kwh": b.capacity_kwh if b else None, "battery_kw": b.p_max_kw if b else None,
            "comfort_reserve_kwh": round(b.comfort_reserve_kwh, 4) if b else None,
            "recorded_kwp": r.get("kwp_recorded"), "true_kwp": r.get("kwp_true"),
            "recorded_tilt_deg": r.get("tilt_recorded"), "true_tilt_deg": r.get("tilt_true"),
            "recorded_azimuth_deg": r.get("azimuth_recorded"), "true_azimuth_deg": r.get("azimuth_true"),
            "gis_phase": r.get("gis_phase"),
            "gis_trust": "low" if world.feeders[s.feeder_id].spec.tau < 0.8 else "ok",
            "metadata_json": {k: v for k, v in r.items() if v is not None}})
    for tid, t in world.turbines.items():
        r = t.row
        db.add("assets", {
            "id": tid, "asset_kind": "turbine", "name": tid, "lat": r["lat"], "lon": r["lon"],
            "county": r.get("county") or "", "locality": None, "feeder_id": None, "park_id": t.park_id,
            "rated_kw": t.rated_kw, "has_battery": 0, "turbine_model": r.get("model"),
            "rotor_diameter_m": r.get("rotor_d_m"), "hub_height_m": r.get("hub_height_m"), "metadata_json": r})


KPI_UNITS = {"minutes_over_limit": "min", "lei_per_home": "lei/home/month", "alerts_per_turbine_day": "1/turbine-day",
             "energy_over_limit": "kWh", "peak_overshoot": "kW"}


def hero_feeder(world: World) -> Optional[str]:
    from .world import HERO_FEEDER_ID

    return HERO_FEEDER_ID if HERO_FEEDER_ID in world.feeders else None


def compute_metrics(world: World, results: dict[str, PolicyResult]) -> dict[str, float]:
    m: dict[str, float] = {}
    base, sent = results.get("baseline"), results.get("sentinel")
    n_fd = max(1, len(world.feeders) * world.days)
    for pol, r in results.items():
        m[f"minutes_over_limit_{pol}"] = r.minutes_over_limit
        m[f"minutes_over_limit_per_feeder_day_{pol}"] = r.minutes_over_limit / n_fd
        m[f"energy_over_limit_{pol}"] = r.energy_over_limit_kwh
        m[f"peak_overshoot_{pol}"] = r.peak_overshoot_kw
        hf = hero_feeder(world)
        for (f, d), v in sorted(r.feeder_days.items()):
            if f == hf:  # the hero feeder day by day (minutes can rise while energy over the limit falls)
                m[f"minutes_over_limit_{f}_d{d}_{pol}"] = v["minutes_over"]
                m[f"energy_over_limit_{f}_d{d}_{pol}"] = v.get("kwh_over", 0.0)
                m[f"peak_overshoot_{f}_d{d}_{pol}"] = v.get("peak_overshoot", 0.0)
        m[f"comfort_breaches_{pol}"] = float(r.comfort_breaches)
        m[f"alerts_per_turbine_day_{pol}"] = r.turbine_alerts / r.turbine_days if r.turbine_days else 0.0
        m[f"lei_cost_per_home_month_{pol}"] = r.cost_lei / max(1, r.n_homes) * 30 / max(1, world.days)
    if sent is not None:
        m["minutes_over_limit"] = sent.minutes_over_limit
        m["comfort_breaches"] = float(sent.comfort_breaches)
        m["alerts_per_turbine_day"] = m["alerts_per_turbine_day_sentinel"]
        if sent.p10_hit_rate is not None:
            m["p10_hit_rate"] = sent.p10_hit_rate
        m["clip_rate"] = sent.clipped_steps / sent.agent_steps if sent.agent_steps else 0.0
    if base is not None and sent is not None:
        red, red_c = [], []
        for key, b in base.feeder_days.items():
            s = sent.feeder_days[key]
            if b["peak_export"] > 1e-6:
                r = (b["peak_export"] - s["peak_export"]) / b["peak_export"]
                red.append(r)
                if b["minutes_over"] > 0:
                    red_c.append(r)
        m["peak_reduction"] = sum(red) / len(red) if red else 0.0
        if red_c:
            m["peak_reduction_congested"] = sum(red_c) / len(red_c)
        m["lei_per_home"] = (base.cost_lei - sent.cost_lei) / max(1, sent.n_homes) * 30 / max(1, world.days)
    return m


def replay(policies: Sequence[str] = ("baseline", "sentinel"), out: str | os.PathLike = "sentinel.sqlite", *,
           world: str | World = "mock", days: int = 7, seed: int = 0, scale: float = 1.0,
           schema_dir: Optional[Path] = None, resolve_hourly: bool = False,
           force_fallback: bool | Sequence[str] = False, run_id: Optional[str] = None,
           comps: Optional[Components] = None, print_metrics: bool = True, full: bool = False,
           strict: bool = True, judge: bool = True, judge_check: bool = False) -> ReplayReport:
    """Default output is the demo DB (R18 < 50 MB): no per-step pv/load prediction rows (asset_steps keeps the PV
    quantiles) and schedules only for Sentinel (baseline battery rows are not shown by the UI). `full=True`
    stores both. `strict=True` raises `audit.SchemaMismatch` on any field the schema cannot hold that is not in
    `audit.ALLOWED_DROPS`."""
    t_start = time.perf_counter()
    w = world if isinstance(world, World) else make_world(world, days=days, seed=seed,
                                                          **({"scale": scale} if world == "mock" else {}))
    pols = []
    for p in policies:
        if p not in BASELINE_ALIASES:
            raise ValueError(f"unknown policy {p!r}")
        if BASELINE_ALIASES[p] not in pols:
            pols.append(BASELINE_ALIASES[p])
    comps = comps or default_components(w, force_fallback=force_fallback, judge=judge)
    base_id = run_id or os.environ.get("LAB_RUN_ID") or f"{w.name}-s{w.seed}"
    db = AuditDB(out, schema_dir=schema_dir, strict=strict, vacuum=not full)
    log(f"world: {json.dumps(w.describe())}; schema: {db.schema_source}; out: {db.path}")
    write_topology(db, w)
    results: dict[str, PolicyResult] = {}
    run_ids = {}
    first_inputs = "sentinel" if "sentinel" in pols else pols[0]  # shared input rows go under one run
    for pol in pols:
        rid = f"{base_id}:{pol}"
        run_ids[pol] = rid
        started = int(datetime.now(timezone.utc).timestamp())
        t0 = time.perf_counter()
        cfg = LoopConfig(policy=pol, run_id=rid, resolve_hourly=resolve_hourly, seed=seed,
                         write_inputs=(pol == first_inputs), step_predictions=full,
                         write_schedules=full or pol == "sentinel", judge=judge)
        res = run_policy(w, comps, cfg, sink=db)
        results[pol] = res
        t_pol = time.perf_counter() - t0
        if print_metrics:
            _emit(f"t_policy_{pol}_s", t_pol)
        log(f"policy {pol}: {t_pol:.1f}s, actions {res.n_actions}, plans {res.plan_status}"
            + (f", judge {res.judge.get('t_judge_s')}s {res.judge.get('verdicts')}" if res.judge else ""))
        db.add("runs", {
            "id": rid, "kind": "replay", "policy": pol, "label": w.label, "git_sha": _git_sha(),
            "model_versions_json": {k: f"{getattr(p, 'model', k)}:{getattr(p, 'model_version', '?')}"
                                    for k, p in comps.predictors.items()},
            "created_at": Ts(started), "finished_at": Ts(int(datetime.now(timezone.utc).timestamp())),
            "week_start": w.week_start,
            "notes": {"components": comps.sources, "fallbacks": comps.fallbacks, "component_errors": comps.errors,
                      "predictors": _predictor_report(comps.predictors),
                      **(w.replay_notes() if hasattr(w, "replay_notes") else {}),
                      "world": w.describe(), "resolve_hourly": resolve_hourly, "schema": db.schema_source,
                      "judge": res.judge or {"mode": "off" if not judge or pol != "sentinel" else "unavailable"},
                      "inputs_hash": __import__("sentinel.contracts", fromlist=["input_hash"]).input_hash(
                          res.day_hashes), "plan_status": res.plan_status,
                      "commit_calibration": {d: {k: {"method": "online split conformal, past replay days",
                                                     "buckets": v.get("n_buckets"),
                                                     **{f: next(iter(v["members"].values()))[f]
                                                        for f in ("n_cal", "raw_coverage")}}
                                                 for k, v in c.items()} for d, c in res.conformal.items()},
                      "warning": ("FALLBACK components: " + ", ".join(comps.fallbacks)) if comps.fallbacks else ""}})
    metrics = compute_metrics(w, results)
    check = None
    if judge_check and "sentinel" in results:
        t0 = time.perf_counter()
        on = results["sentinel"]
        cfg = LoopConfig(policy="sentinel", run_id=run_ids["sentinel"], resolve_hourly=resolve_hourly, seed=seed,
                         write_inputs=(first_inputs == "sentinel"), step_predictions=full, write_schedules=True,
                         judge=False)
        off = run_policy(w, comps, cfg, sink=_NullSink())
        m_off = compute_metrics(w, {**results, "sentinel": off})
        diffs = {k: (metrics.get(k), m_off.get(k)) for k in set(metrics) | set(m_off)
                 if metrics.get(k) != m_off.get(k)}
        same_actions = on.action_ids == off.action_ids and on.n_actions == off.n_actions
        check = {"identical": not diffs and same_actions, "kpi_diffs": diffs, "actions_identical": same_actions,
                 "n_kpis": len(metrics), "n_actions": len(on.action_ids), "seconds": time.perf_counter() - t0}
        log(f"judge check (judge on vs off, Sentinel): {'IDENTICAL' if check['identical'] else 'DIFFERENT'}; "
            f"{len(metrics)} KPIs, {len(on.action_ids)} actions; diffs {diffs}")
    for pol, rid in run_ids.items():
        for name in METRICS + ("peak_reduction_congested", "clip_rate"):
            key = name if pol == "sentinel" else f"{name}_{pol}"
            val = metrics.get(key, metrics.get(f"{name}_{pol}"))
            if pol == "sentinel" and name in ("minutes_over_limit", "comfort_breaches", "alerts_per_turbine_day"):
                val = metrics.get(f"{name}_sentinel")
            if val is None or (isinstance(val, float) and math.isnan(val)):
                continue
            db.add("kpis", {"run_id": rid, "policy": pol, "name": name, "value": float(val), "label": w.label,
                            "unit": KPI_UNITS.get(name)})
        # energy over the limit + peak overshoot: all feeders x days, and the hero feeder day by day
        hf = hero_feeder(w)
        extra = ["energy_over_limit", "peak_overshoot"] + [f"{k}_{hf}_d{d}" for d in range(w.days) for k in
                                                           ("minutes_over_limit", "energy_over_limit", "peak_overshoot")
                                                           if hf]
        for name in extra:
            val = metrics.get(f"{name}_{pol}")
            if val is not None:
                db.add("kpis", {"run_id": rid, "policy": pol, "name": name, "value": float(val), "label": w.label,
                                "unit": KPI_UNITS.get(name.split(f"_{hf}_d")[0] if hf else name)})
    summary = db.summary()
    db.close()
    secs = time.perf_counter() - t_start
    report = ReplayReport(Path(out), results, metrics, summary, dict(comps.sources), secs, run_ids, check)
    if print_metrics:
        if comps.fallbacks:
            print(f"# FALLBACK components: {', '.join(comps.fallbacks)} (label: {w.label}; see runs.notes)")
        order = ["peak_reduction", "peak_reduction_congested", "minutes_over_limit"] + \
                [f"minutes_over_limit_{p}" for p in pols] + ["p10_hit_rate", "comfort_breaches"] + \
                [f"comfort_breaches_{p}" for p in pols] + ["lei_per_home", "alerts_per_turbine_day"] + \
                [f"alerts_per_turbine_day_{p}" for p in pols] + [f"energy_over_limit_{p}" for p in pols] + \
                [f"peak_overshoot_{p}" for p in pols]
        for k in dict.fromkeys(order):
            if k in metrics:
                print(f"LAB_METRIC {k}={metrics[k]:.4g}")
        if "sentinel" in results:
            _judge_metrics(results["sentinel"].judge, w.days)
        if check is not None:
            _emit("judge_kpi_identical", float(check["identical"]))
            _emit("judge_kpi_diffs", float(len(check["kpi_diffs"])))
            _emit("t_judge_check_s", check["seconds"])
        for t in ("judge_verdicts", "trust", "outcomes"):
            if t in summary["rows"]:
                _emit(f"rows_{t}", summary["rows"][t])
        print(f"# replay: {secs:.1f}s, {sum(summary['rows'].values())} rows -> {out}", file=sys.stderr)
        if summary["dropped_fields"]:
            print(f"# not stored (audit.ALLOWED_DROPS): {json.dumps(summary['dropped_fields'])}", file=sys.stderr)
        if summary["unexpected_drops"] or summary["missing_tables"] or summary["filled_not_null"]:
            print("# SCHEMA GAPS (non-strict run): " + json.dumps(
                {k: summary[k] for k in ("unexpected_drops", "missing_tables", "filled_not_null")}), file=sys.stderr)
    if check is not None and not check["identical"]:
        raise JudgeChangedDecisions(f"judge on vs off changed the Sentinel run: KPI diffs {check['kpi_diffs']}, "
                                    f"actions identical: {check['actions_identical']}")
    return report


def main(argv: Optional[Sequence[str]] = None) -> ReplayReport:
    import argparse

    ap = argparse.ArgumentParser(description="Replay a test week (baseline vs Sentinel) into SQLite")
    ap.add_argument("--policy", default="baseline,sentinel", help="comma list: baseline|threshold|sentinel")
    ap.add_argument("--out", default=os.path.join(os.environ.get("LAB_OUTPUT_DIR", "."), "sentinel.sqlite"))
    ap.add_argument("--world", default="mock", choices=["mock", "sim"])
    ap.add_argument("--days", type=int, default=7)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--scale", type=float, default=1.0, help="mock world size factor")
    ap.add_argument("--resolve-hourly", action="store_true", help="SHOULD: re-plan every hour (R4)")
    ap.add_argument("--fallback", default="", help="force built-ins: all | planner,qualify,triage,committer,demand")
    ap.add_argument("--run-id", default=None)
    ap.add_argument("--full", action="store_true",
                    help="also store per-step pv/load predictions and baseline schedules (DB > 50 MB)")
    ap.add_argument("--lite", action="store_true", help=argparse.SUPPRESS)  # old flag; lite is now the default
    ap.add_argument("--strict", dest="strict", action="store_true", default=True,
                    help="raise if a produced field has no column and is not in audit.ALLOWED_DROPS (default)")
    ap.add_argument("--no-strict", dest="strict", action="store_false", help="warn and drop instead")
    ap.add_argument("--judge", default="on", choices=["on", "off"],
                    help="Judge verdicts for the Sentinel policy (#149): judge_verdicts / trust / outcomes rows")
    ap.add_argument("--judge-check", action="store_true",
                    help="replay Sentinel again with the judge off and fail unless KPIs and actions are identical")
    a = ap.parse_args(argv)
    ff: bool | list[str] = True if a.fallback == "all" else [x for x in a.fallback.split(",") if x]
    return replay([p for p in a.policy.split(",") if p], a.out, world=a.world, days=a.days, seed=a.seed,
                  scale=a.scale, resolve_hourly=a.resolve_hourly, force_fallback=ff, run_id=a.run_id,
                  full=a.full and not a.lite, strict=a.strict, judge=a.judge == "on", judge_check=a.judge_check)
