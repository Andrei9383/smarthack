"""Robustness drills on the agent replay (card #78): clean vs drilled run, same world and seed, KPI deltas.

    python -m wattsentry.agent.drills --drill <name|all|a,b> [--world mock|sim] [--days N] [--seed S]
        [--policy sentinel] [--share 1.0] [--delivery-hour 8] [--out-dir DIR] [--report drills_report.json]

A drill corrupts only what the agent OBSERVES of each site's telemetry, with the semantics of
`wattsentry.data.drills.telemetry_drill` (gaps_10 / gaps_40 / missing_day / frozen_sensor / units_kw / units_x1000 /
telemetry_dropout; the function itself is applied to a long W table of the observed telemetry). Physics and KPIs stay
on the TRUE flows: battery model, feeder export, minutes over limit, comfort and realised flexibility all use the
world's true pv/load. The agent must not crash and must not commit flexibility it cannot deliver.

Observation vs truth in `loop.run_policy`: `world.observed_site_day` feeds every observation read (telemetry ages
for qualify at t0 and every tick, `pv_obs` -> qualify `actuals`, measurements rows, input hashes); `world.site_day`
is the physics (battery step, grid, feeder export, cost, realised flexibility, asset_steps.actual_kw). Predictors get
a bare `Window(t0, 96)`: no telemetry reaches them, so `human_review` cannot react to a drill yet.
Drills that only REMOVE readings (gaps_*, missing_day, telemetry_dropout) are also exact through
`site_day.telemetry_age_s` alone (observation-only); drills that change the VALUES the agent reads (frozen_sensor,
units_*) need the `observed_site_day` hook and are reported as skipped without it (`observation_hook_available`).

How a world is drilled: `drilled(world, name)` returns an instance of a subclass of the world's own class that shares
its state (same sites, feeders, caches of the true days) and overrides only `site_day` / `observed_site_day`;
everything else (prices, turbines, predictors, limits) is the original. Turbine SCADA is not drilled.

KPIs (sentinel policy): human_review_share = site-ticks whose per-tick qualify status is `human_review` / site-ticks
assessed (counted by wrapping `Components.qualify_many`; the t0 call is not counted), stale_share = site-ticks
excluded for stale telemetry / data gap, not_sure_share = both; p10_hit_rate, comfort_breaches, minutes_over_limit,
clip_rate from `replay.compute_metrics`; committed_kwh / shortfall_kwh (sum over committed steps of committed -
min(realised, committed)) from the run's `commitments` rows.
"""

from __future__ import annotations

import argparse
import contextlib
import inspect
import json
import math
import os
import sqlite3
import sys
import tempfile
import time
import traceback
import zlib
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Optional, Sequence

import numpy as np
import pandas as pd

from wattsentry.contracts import DAY_STEPS
from wattsentry.data.drills import DRILLS, telemetry_drill

from . import loop as _loop
from .loop import default_components, log
from .replay import replay
from .world import STEP_S, SiteDay, World, make_world

PRESENCE_DRILLS = ("gaps_10", "gaps_40", "missing_day", "telemetry_dropout")  # remove readings -> ages only
VALUE_DRILLS = ("frozen_sensor", "units_kw", "units_x1000")  # change the values the agent reads -> need the hook
DRILL_NAMES = PRESENCE_DRILLS[:3] + VALUE_DRILLS + PRESENCE_DRILLS[3:]
assert set(DRILL_NAMES) <= set(DRILLS)
OBS_HOOK = "observed_site_day"
HOOK_MESSAGE = ("skipped: needs the observation hook World.observed_site_day(site_id, d) -> SiteDay used by "
                "loop.run_policy for perception (#29 owns world.py/loop.py). Today run_policy reads SiteDay.pv_kw "
                "for both pv_obs (perceive) and the battery/grid physics, so corrupting the values would corrupt the "
                "truth.")
KPIS = ("human_review_share", "p10_hit_rate", "comfort_breaches", "minutes_over_limit", "clip_rate")
EXTRA_KPIS = ("not_sure_share", "stale_share", "human_review_actions", "commit_steps", "committed_kwh",
              "shortfall_kwh", "shortfall_share", "peak_reduction")
DT_H = STEP_S / 3600.0
STALE_S = 3600.0  # qualify DEFAULT_THRESHOLDS site.stale_minutes = 60


def observation_hook_available() -> bool:
    """True once world.World has `observed_site_day` AND loop.run_policy calls it (#29)."""
    if not callable(getattr(World, OBS_HOOK, None)):
        return False
    try:
        return OBS_HOOK in inspect.getsource(_loop.run_policy)
    except (OSError, TypeError):  # no source (frozen install): trust the World method
        return True


def drill_available(name: str, hook: Optional[bool] = None) -> tuple[bool, str]:
    if name in VALUE_DRILLS and not (observation_hook_available() if hook is None else hook):
        return False, HOOK_MESSAGE
    return True, ""


# =============================================================================================== observation
def build_observation(truth: Callable[[str, int], SiteDay], site_ids: Sequence[str], days: int,
                      tick_ts: Callable[[int, int], int], drill: str, rng: np.random.Generator,
                      sites: Optional[Sequence[str]] = None, delivery_hour_utc: int = 8
                      ) -> tuple[dict[tuple[str, int], SiteDay], dict[str, Any]]:
    """True site-days -> long telemetry in W (what the agent would receive) -> `telemetry_drill` -> observed site-days.

    A reading is received if the world delivered it (telemetry_age_s == 0) and the drill kept it (row present, pv and
    load not NaN). Observed telemetry_age_s = seconds since the last received reading + one step (the worlds' own
    convention), continuous across days; where the drill removed nothing in the current gap the world's own age is
    kept, so a no-op drill reproduces the clean ages exactly. Observed values = drilled W / 1000 (0 where missing).
    """
    sids = sorted(site_ids)
    S, n = len(sids), days * DAY_STEPS
    pv, ld, age = np.zeros((S, n)), np.zeros((S, n)), np.zeros((S, n))
    true: dict[tuple[str, int], SiteDay] = {}
    for i, s in enumerate(sids):
        for d in range(days):
            sd = true[(s, d)] = truth(s, d)
            j = slice(d * DAY_STEPS, (d + 1) * DAY_STEPS)
            pv[i, j], ld[i, j], age[i, j] = sd.pv_kw, sd.load_kw, sd.telemetry_age_s
    ts = np.array([tick_ts(d, k) for d in range(days) for k in range(DAY_STEPS)], dtype=np.int64)
    got = age == 0
    df = pd.DataFrame({"ts": pd.to_datetime(np.tile(ts, S), unit="s", utc=True),
                       "site_id": np.repeat(np.array(sids, dtype=object), n),
                       "pv_ac_w": np.where(got, pv * 1000.0, np.nan).ravel(),
                       "load_w": np.where(got, ld * 1000.0, np.nan).ravel()})
    out = telemetry_drill(df, drill, rng, sites=list(sites) if sites is not None else None,
                          delivery_hour_utc=delivery_hour_utc)
    pv_w, ld_w = np.full(S * n, np.nan), np.full(S * n, np.nan)
    idx = out.index.to_numpy()  # telemetry_dropout drops rows: absent = not received
    pv_w[idx], ld_w[idx] = out["pv_ac_w"].to_numpy(float), out["load_w"].to_numpy(float)
    pv_w, ld_w = pv_w.reshape(S, n), ld_w.reshape(S, n)
    got_d = got & np.isfinite(pv_w) & np.isfinite(ld_w)
    age_d = np.zeros_like(age)
    for i in range(S):
        last, dropped = int(ts[0]) - STEP_S, False
        for j in range(n):
            if got_d[i, j]:
                last, dropped = int(ts[j]), False
                continue
            dropped |= bool(got[i, j])  # the world sent it, the drill removed it
            age_d[i, j] = max(age[i, j], float(ts[j] - last + STEP_S)) if dropped else age[i, j]
    pv_o = np.where(got_d, pv_w / 1000.0, 0.0)
    ld_o = np.where(got_d, ld_w / 1000.0, 0.0)
    obs: dict[tuple[str, int], SiteDay] = {}
    for i, s in enumerate(sids):
        for d in range(days):
            j = slice(d * DAY_STEPS, (d + 1) * DAY_STEPS)
            sd = true[(s, d)]
            obs[(s, d)] = SiteDay(pv_o[i, j].tolist(), ld_o[i, j].tolist(), age_d[i, j].tolist(), list(sd.inv_status),
                                  list(sd.inv_voltage_v))
    both = got & got_d
    changed = both & ((np.abs(pv_o - pv) > 1e-9) | (np.abs(ld_o - ld) > 1e-9))
    stats = {"drill": drill, "site_ticks": int(S * n), "sites_hit": len(sites) if sites is not None else S,
             "sites": S, "missing_share_clean": float(1 - got.mean()) if n else 0.0,
             "missing_share": float(1 - got_d.mean()) if n else 0.0,
             "age_over_60min_share_clean": float((age > STALE_S).mean()) if n else 0.0,
             "age_over_60min_share": float((age_d > STALE_S).mean()) if n else 0.0,
             "value_changed_share": float(changed.sum() / max(1, both.sum()))}
    return obs, stats


@dataclass
class DrillSpec:
    name: str
    seed: int = 0
    share: float = 1.0
    delivery_hour_utc: int = 8
    sites: Optional[list[str]] = None  # None = every site (telemetry_drill default)
    via_hook: bool = False

    def rng(self) -> np.random.Generator:
        return np.random.default_rng([self.seed, zlib.crc32(self.name.encode()), 0])

    def describe(self) -> dict[str, Any]:
        return {"name": self.name, "seed": self.seed, "share": self.share, "delivery_hour_utc": self.delivery_hour_utc,
                "sites_hit": None if self.sites is None else len(self.sites),
                "path": "observed_site_day hook" if self.via_hook else "site_day.telemetry_age_s"}


class _DrillMixin:
    """Overrides only the observation path; `true_site_day` is the original world's day."""

    _drill: DrillSpec
    _drill_obs: Optional[dict[tuple[str, int], SiteDay]]
    drill_stats: dict[str, Any]

    def true_site_day(self, site_id: str, d: int) -> SiteDay:
        return super().site_day(site_id, d)  # type: ignore[misc]

    def observed_site_day(self, site_id: str, d: int) -> SiteDay:
        if self._drill_obs is None:
            spec = self._drill
            self._drill_obs, self.drill_stats = build_observation(
                self.true_site_day, list(self.sites), self.days, self.tick_ts,  # type: ignore[attr-defined]
                spec.name, spec.rng(), spec.sites, spec.delivery_hour_utc)
        return self._drill_obs[(site_id, d)]

    def site_day(self, site_id: str, d: int) -> SiteDay:
        true = self.true_site_day(site_id, d)
        if self._drill.via_hook:
            self.observed_site_day(site_id, d)  # build (cached) so drill_stats is populated either way
            return true  # the loop perceives through observed_site_day
        obs = self.observed_site_day(site_id, d)  # telemetry_age_s is observation-only in run_policy
        return SiteDay(true.pv_kw, true.load_kw, obs.telemetry_age_s, true.inv_status, true.inv_voltage_v)

    def describe(self) -> dict[str, Any]:
        return {**super().describe(), "drill": self._drill.describe()}  # type: ignore[misc]


_CLASSES: dict[type, type] = {}


def drilled(world: World, name: str, *, seed: int = 0, share: float = 1.0, delivery_hour_utc: int = 8,
            via_hook: Optional[bool] = None) -> World:
    """The same world with drill `name` applied to what the agent observes. Shares the world's state (no copy of the
    true days). `share` < 1 drills that fraction of the sites (seeded pick); default all, like telemetry_drill."""
    if name not in DRILLS:
        raise KeyError(f"unknown drill {name!r}; one of {list(DRILLS)}")
    base = type(world)
    if isinstance(world, _DrillMixin):
        raise TypeError("world is already drilled")
    cls = _CLASSES.get(base)
    if cls is None:
        cls = _CLASSES[base] = type(f"Drilled{base.__name__}", (_DrillMixin, base), {"__module__": __name__})
    w = cls.__new__(cls)
    w.__dict__.update(world.__dict__)
    sids = sorted(world.sites)
    picked = None
    if share < 1.0:
        k = int(round(max(0.0, share) * len(sids)))
        rng = np.random.default_rng([seed, zlib.crc32(name.encode()), 1])
        picked = sorted(rng.choice(np.array(sids, dtype=object), size=k, replace=False).tolist()) if k else []
    w._drill = DrillSpec(name, seed, share, delivery_hour_utc, picked,
                         observation_hook_available() if via_hook is None else via_hook)
    w._drill_obs = None
    w.drill_stats = {}
    return w


# =============================================================================================== runs
class _QualifyCounter:
    """Wraps Components.qualify_many; counts site statuses of the per-tick calls (actuals given; t0 call has None)."""

    def __init__(self, fn: Callable[..., list[Any]]):
        self.fn = fn
        self.n: Counter[str] = Counter()

    def __call__(self, assets, predictions, ages, thresholds=None, **kw):
        out = self.fn(assets, predictions, ages, thresholds, **kw)
        if kw.get("actuals") is not None:
            for x in out:
                if getattr(x, "asset_kind", "site") != "site":
                    continue
                st = x.status
                self.n["site_ticks"] += 1
                if st.status == "human_review":
                    self.n["human_review"] += 1
                elif st.status == "excluded" and st.reason in ("stale_telemetry", "data_gap"):
                    self.n["stale"] += 1
        return out


def _commitment_kpis(db: Path, run_id: str) -> dict[str, float]:
    conn = sqlite3.connect(db)
    try:
        rows = conn.execute("SELECT committed_kw, realised_kw FROM commitments WHERE run_id = ? AND committed_kw > 0",
                            (run_id,)).fetchall()
    finally:
        conn.close()
    com = sum(c for c, _ in rows) * DT_H
    got = sum(min(max(r or 0.0, 0.0), c) for c, r in rows) * DT_H
    return {"committed_kwh": com, "shortfall_kwh": com - got, "shortfall_share": (com - got) / com if com > 0 else 0.0}


def run_once(world: World, out: Path, run_id: str, *, policies: Sequence[str] = ("sentinel",), seed: int = 0,
             force_fallback: bool | Sequence[str] = False) -> dict[str, Any]:
    """One replay (clean or drilled world) -> KPIs. Uses wattsentry.agent.replay.replay with injected components."""
    t = time.perf_counter()
    comps = default_components(world, force_fallback=force_fallback)
    counter = _QualifyCounter(comps.qualify_many)
    comps.qualify_many = counter
    rep = replay(list(policies), out, world=world, seed=seed, comps=comps, run_id=run_id, print_metrics=False,
                 strict=False)
    m = rep.metrics
    kpis: dict[str, float] = {k: float(m[k]) for k in ("minutes_over_limit", "comfort_breaches", "p10_hit_rate",
                                                       "clip_rate", "peak_reduction") if k in m}
    res = rep.results.get("sentinel")
    if res is not None:
        n = counter.n
        st = max(1, n["site_ticks"])
        kpis.update(human_review_share=n["human_review"] / st, stale_share=n["stale"] / st,
                    not_sure_share=(n["human_review"] + n["stale"]) / st,
                    human_review_actions=float(res.n_actions.get("human_review", 0)),
                    commit_steps=float(res.commit_steps))
        kpis.update(_commitment_kpis(Path(out), rep.run_ids["sentinel"]))
    return {"run_id": rep.run_ids.get("sentinel") or next(iter(rep.run_ids.values()), run_id), "kpis": kpis,
            "seconds": round(time.perf_counter() - t, 2), "component_errors": dict(comps.errors),
            "plan_status": dict(res.plan_status) if res is not None else {},
            "site_ticks": int(counter.n["site_ticks"]), "components": dict(comps.sources)}


def _delta(a: dict[str, float], b: dict[str, float]) -> dict[str, Optional[float]]:
    return {k: (b[k] - a[k]) if (k in a and k in b) else None for k in dict.fromkeys([*a, *b])}


def run_drills(names: Sequence[str] = DRILL_NAMES, *, world: str | World = "mock", days: int = 7, seed: int = 0,
               scale: float = 1.0, policies: Sequence[str] = ("sentinel",), share: float = 1.0,
               delivery_hour_utc: int = 8, force_fallback: bool | Sequence[str] = False,
               out_dir: Optional[str | os.PathLike] = None, run_id: Optional[str] = None) -> dict[str, Any]:
    """Clean run, then each drill on the same world and seed. Never raises for a drill: errors are recorded.
    `report["clean"]["status"]` is "error" if the clean run failed (then no drill runs). SQLite files go to
    `out_dir` (kept) or a temporary directory (deleted)."""
    hook = observation_hook_available()
    base = run_id or os.environ.get("LAB_RUN_ID") or f"{world if isinstance(world, str) else world.name}-s{seed}"
    report: dict[str, Any] = {
        "card": 78, "run_id": os.environ.get("LAB_RUN_ID"), "seed": seed, "days": days, "policies": list(policies),
        "share": share, "delivery_hour_utc": delivery_hour_utc,
        "hook": {"name": f"World.{OBS_HOOK}", "available": hook, "needed_for": list(VALUE_DRILLS)},
        "kpis": list(KPIS + EXTRA_KPIS), "clean": {}, "drills": {}}
    with contextlib.ExitStack() as stack:
        if out_dir is None:
            d = Path(stack.enter_context(tempfile.TemporaryDirectory(prefix="drills-")))
        else:
            d = Path(out_dir)
            d.mkdir(parents=True, exist_ok=True)
        report["sqlite_dir"] = str(d) if out_dir is not None else None
        try:
            w = world if isinstance(world, World) else make_world(
                world, days=days, seed=seed, **({"scale": scale} if world == "mock" else {}))
            report["world"] = w.describe()
            report["days"] = w.days
            clean = run_once(w, d / "clean.sqlite", f"{base}-clean", policies=policies, seed=seed,
                             force_fallback=force_fallback)
            report["components"] = clean.pop("components")
            report["clean"] = {"status": "ok", **clean}
        except Exception as e:  # clean run failed: nothing to compare against
            report["clean"] = {"status": "error", "error": f"{type(e).__name__}: {e}",
                               "traceback": traceback.format_exc(limit=8)}
            log(f"drills: clean run FAILED: {type(e).__name__}: {e}")
            return report
        for name in names:
            ok, why = drill_available(name, hook)
            if not ok:
                report["drills"][name] = {"status": "skipped", "message": why}
                log(f"drills: {name} {why}")
                continue
            t = time.perf_counter()
            try:
                dw = drilled(w, name, seed=seed, share=share, delivery_hour_utc=delivery_hour_utc, via_hook=hook)
                r = run_once(dw, d / f"{name}.sqlite", f"{base}-drill-{name}", policies=policies, seed=seed,
                             force_fallback=force_fallback)
                r.pop("components", None)
                report["drills"][name] = {"status": "ok", **r, "delta": _delta(report["clean"]["kpis"], r["kpis"]),
                                          "observation": dict(dw.drill_stats), "spec": dw._drill.describe()}
            except Exception as e:  # never stop the batch on one drill
                report["drills"][name] = {"status": "error", "error": f"{type(e).__name__}: {e}",
                                          "traceback": traceback.format_exc(limit=8),
                                          "seconds": round(time.perf_counter() - t, 2)}
                log(f"drills: {name} FAILED ({type(e).__name__}: {e}); continuing")
    return report


# =============================================================================================== CLI
def _json_safe(x: Any) -> Any:
    if isinstance(x, float) and (math.isnan(x) or math.isinf(x)):
        return None
    if isinstance(x, dict):
        return {str(k): _json_safe(v) for k, v in x.items()}
    if isinstance(x, (list, tuple)):
        return [_json_safe(v) for v in x]
    return x


def _fmt(v: Optional[float]) -> Optional[str]:
    if v is None or (isinstance(v, float) and (math.isnan(v) or math.isinf(v))):
        return None
    return f"{v:.6g}"


def metric_lines(report: dict[str, Any]) -> list[str]:
    out = []
    order = KPIS + EXTRA_KPIS
    clean = report.get("clean", {})
    if clean.get("status") == "ok":
        for k in order:
            s = _fmt(clean["kpis"].get(k))
            if s is not None:
                out.append(f"LAB_METRIC drill_clean_{k}={s}")
    n = Counter(r["status"] for r in report["drills"].values())
    for name, r in report["drills"].items():
        if r["status"] == "skipped":
            out.append(f"LAB_METRIC drill_{name}_skipped=1")
            continue
        out.append(f"LAB_METRIC drill_{name}_ok={int(r['status'] == 'ok')}")  # 0 = the drill run raised
        if r["status"] != "ok":
            continue
        for k in order:
            s = _fmt(r["kpis"].get(k))
            if s is not None:
                out.append(f"LAB_METRIC drill_{name}_{k}={s}")
        for k in order:
            s = _fmt(r["delta"].get(k))
            if s is not None:
                out.append(f"LAB_METRIC drill_{name}_delta_{k}={s}")
    out += [f"LAB_METRIC drills_ok={n['ok']}", f"LAB_METRIC drills_skipped={n['skipped']}",
            f"LAB_METRIC drills_failed={n['error']}"]
    return out


def summary_lines(report: dict[str, Any]) -> list[str]:
    c = report.get("clean", {})
    if c.get("status") != "ok":
        return [f"# clean run FAILED: {c.get('error')}"]
    show = ("human_review_share", "not_sure_share", "p10_hit_rate", "comfort_breaches", "minutes_over_limit",
            "committed_kwh", "shortfall_share")
    lines = [f"# drills on {report['world'].get('world')} world, {report['days']} days, seed {report['seed']}, "
             f"hook {'present' if report['hook']['available'] else 'absent'}; clean {c['run_id']} ({c['seconds']}s): "
             + ", ".join(f"{k}={_fmt(c['kpis'].get(k))}" for k in show)]
    for name, r in report["drills"].items():
        if r["status"] == "ok":
            ob = r["observation"]
            lines.append(f"# {name}: ok {r['seconds']}s, missing {ob['missing_share']:.1%} (clean "
                         f"{ob['missing_share_clean']:.1%}), age>60min {ob['age_over_60min_share']:.1%}; "
                         + ", ".join(f"{k}={_fmt(r['kpis'].get(k))} ({_fmt(r['delta'].get(k))})" for k in show))
        else:
            lines.append(f"# {name}: " + (r["message"] if r["status"] == "skipped" else f"ERROR {r.get('error')}"))
    return lines


def main(argv: Optional[Sequence[str]] = None) -> int:
    ap = argparse.ArgumentParser(description="Robustness drills on the agent replay: clean vs drilled, KPI deltas "
                                             "(card #78)")
    ap.add_argument("--drill", default="all", help=f"all | comma list of {', '.join(DRILL_NAMES)}")
    ap.add_argument("--world", default="mock", choices=["mock", "sim"])
    ap.add_argument("--days", type=int, default=7)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--scale", type=float, default=1.0, help="mock world size factor")
    ap.add_argument("--policy", default="sentinel", help="comma list; KPIs are the sentinel policy's")
    ap.add_argument("--share", type=float, default=1.0, help="fraction of sites drilled (default all)")
    ap.add_argument("--delivery-hour", type=int, default=8,
                    help="telemetry_dropout: readings missing in the 6 h before this UTC hour each day")
    ap.add_argument("--fallback", default="", help="force built-ins: all | planner,qualify,triage,committer,demand")
    ap.add_argument("--out-dir", default=None, help="keep the per-run SQLite files here (default: temp dir, deleted)")
    ap.add_argument("--report", default=os.path.join(os.environ.get("LAB_OUTPUT_DIR", "."), "drills_report.json"))
    ap.add_argument("--run-id", default=None)
    a = ap.parse_args(argv)
    names = list(DRILL_NAMES) if a.drill == "all" else [x.strip() for x in a.drill.split(",") if x.strip()]
    bad = [x for x in names if x not in DRILLS]
    if bad:
        ap.error(f"unknown drill(s) {bad}; one of {list(DRILL_NAMES)}")
    if a.out_dir and Path(a.out_dir).resolve().is_relative_to(Path(__file__).resolve().parents[2] / "demo"):
        ap.error("--out-dir must not be inside demo/")
    ff: bool | list[str] = True if a.fallback == "all" else [x for x in a.fallback.split(",") if x]
    report = run_drills(names, world=a.world, days=a.days, seed=a.seed, scale=a.scale,
                        policies=[p for p in a.policy.split(",") if p], share=a.share,
                        delivery_hour_utc=a.delivery_hour, force_fallback=ff, out_dir=a.out_dir, run_id=a.run_id)
    for line in metric_lines(report):
        print(line)
    for line in summary_lines(report):
        print(line)
    try:
        p = Path(a.report)
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(json.dumps(_json_safe(report), indent=2, default=str, allow_nan=False))
        print(f"# report -> {p}", file=sys.stderr)
    except Exception as e:  # the metrics are already printed
        print(f"# could not write the report: {type(e).__name__}: {e}", file=sys.stderr)
    sys.stdout.flush()
    return 0 if report["clean"].get("status") == "ok" else 1


if __name__ == "__main__":
    sys.exit(main())
