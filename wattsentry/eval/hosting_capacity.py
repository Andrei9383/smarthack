"""Hosting capacity: how many more prosumers each feeder takes without a new cable.

    python -m wattsentry.eval.hosting_capacity [--world sim|mock] [--feeders F7,F3] [--workers 20] [--out DIR]

Per feeder, add n synthetic homes (`SyntheticPlan`: nested in n, PV / load of healthy donors on the same feeder
scaled to a fleet kWp draw, battery drawn from the fleet) and replay the test week through the K1 battery kernel
(`replay_week`, same physics as `agent.loop.run_policy`). The capacity of a policy (`self_consumption`, `threshold`,
`sentinel`) is the largest n whose minutes over the M8 limit G = tau*G_max + (1 - tau)*G_cons stay within a
tolerance (0 / 30 min/week), found by galloping + bisection (`capacity_search`, assumes minutes(n) non-decreasing).
`hosting_capacity_gain` = capacity(sentinel) - capacity(self_consumption). Every model read goes through `validated`.
Output: hosting_capacity.json (`hosting_capacity/v1`, see `assemble`) + .csv in --out, and LAB_METRIC lines.
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import sys
import time
import zlib
import multiprocessing
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import dataclass, field, replace
from pathlib import Path
from typing import Any, Callable, Mapping, Optional, Sequence

import numpy as np

from wattsentry.contracts import DAY_STEPS, DT_H, AssetStatus, BatterySpec, FeederSpec, Prediction, Window

SCHEMA = "hosting_capacity/v1"
LABEL = "synthetic RO"
POLICIES = ("self_consumption", "threshold", "sentinel")
BASELINE = "self_consumption"
TOLS_MIN = (0, 30)
VARIANTS = ("batteries", "no_batteries")
MODEL_IDS = {"pv": "pvnet", "load": "loadnet", "sitehealth": "sitehealth", "gridtwin": "gridtwin"}  # M1 M2 M4 M8
N_MAX = 300
EPS_KW = 1e-6  # same over-limit test as the loop: export > G + 1e-6
STEP_MIN = 15.0


def log(msg: str) -> None:
    print(f"[hosting] {msg}", file=sys.stderr, flush=True)


# =============================================================================================== Judge access
def judge_mode() -> str:
    from wattsentry.judge import adapter

    return f"C1 adapter, verdict source {type(adapter.default_judge()).__name__}"


def _local_day(t0: str) -> Any:
    from datetime import datetime

    return datetime.fromisoformat(t0.replace("Z", "+00:00"))  # tz-aware: the adapter keys on the Bucharest day


def validated(asset_id: str, model: str, t0: str, raw: Callable[[], Any]) -> tuple[Any, str]:
    """THE single place this module reads a model output: M1 pv, M2 load, M4 sitehealth (Predictions) and the M8
    gridtwin feeder limit (a FeederSpec). Returns (value, verdict); value None = unknown (Judge said dont_use and
    there is no substitute).

    Reads go through `wattsentry.judge.adapter.validated` (band scale applied, substitute swapped in). M8 has
    no quantiles: a dont_use verdict on the twin means the conservative limit (tau = 0)."""
    from wattsentry.judge.adapter import validated as judge_validated

    value = raw()
    day = _local_day(t0)
    if isinstance(value, FeederSpec):
        v = judge_validated(asset_id, model, day, raw=None)
        if v.verdict == "dont_use":
            value = FeederSpec(value.feeder_id, value.g_max_kw, value.g_conservative_kw, 0.0)
        return value, v.verdict
    v = judge_validated(asset_id, model, day, raw=value)
    if not v.known:
        return None, v.verdict
    if v.q50 is not None and isinstance(value, Prediction):
        value = replace(value, q10=list(v.q10), q50=list(v.q50), q90=list(v.q90),
                        **({"fault_probs": v.fault_probs} if v.fault_probs else {}))
    return value, v.verdict


# =============================================================================================== data
def battery_arrays(batteries: Sequence[Optional[BatterySpec]]) -> dict[str, np.ndarray]:
    """(S,) arrays: cap, pmax, eta_c, eta_d, e_rule (soc_min*E: the rule's physical floor), e_plan (e_min: the
    planner's floor incl. the comfort reserve), e_max. No battery -> zeros (cap 0: the kernel applies 0 kW)."""
    rows = []
    for b in batteries:
        if b is None or b.capacity_kwh <= 0 or b.p_max_kw <= 0:
            rows.append((0.0, 0.0, 1.0, 1.0, 0.0, 0.0, 0.0))
        else:
            rows.append((b.capacity_kwh, b.p_max_kw, b.eta_charge, b.eta_discharge, b.soc_min * b.capacity_kwh,
                         b.e_min, b.e_max))
    a = np.array(rows, dtype=float).reshape(-1, 7)
    keys = ("cap", "pmax", "eta_c", "eta_d", "e_rule", "e_plan", "e_max")
    return {k: np.ascontiguousarray(a[:, i]) for i, k in enumerate(keys)}


@dataclass
class FeederWeek:
    """One feeder over the test week, everything the replay needs. Shapes: S sites, D days, T = 96 steps.

    pv / load: realised kW (S, D, T); pv_q / load_q: forecast q10, q50, q90 (S, D, 3, T); statuses[d][site] =
    M4 qualify at t0 of day d; healthy (S,): usable as a donor for synthetic homes (no fault this week, not OOD).
    """

    feeder_id: str
    g_kw: float
    t0s: list[str]
    prices_kwh: np.ndarray  # (D, T) lei/kWh
    site_ids: list[str]
    batteries: list[Optional[BatterySpec]]
    kwp: np.ndarray  # (S,)
    pv: np.ndarray
    load: np.ndarray
    pv_q: np.ndarray
    load_q: np.ndarray
    statuses: list[dict[str, AssetStatus]]
    healthy: np.ndarray
    synthetic: np.ndarray = field(default_factory=lambda: np.zeros(0, bool))  # (S,) True for added homes
    meta: dict[str, Any] = field(default_factory=dict)  # src id, g_max, g_cons, tau, verdict counts, ...

    def __post_init__(self) -> None:
        if self.synthetic.shape != (len(self.site_ids),):
            self.synthetic = np.zeros(len(self.site_ids), bool)

    @property
    def n_sites(self) -> int:
        return len(self.site_ids)

    @property
    def n_days(self) -> int:
        return len(self.t0s)

    @property
    def n_existing(self) -> int:
        return int((~self.synthetic).sum())

    @property
    def n_batteries(self) -> int:
        return sum(b is not None for b in self.batteries)


@dataclass
class Fleet:
    """The fleet-wide distributions synthetic homes are drawn from (all sites of the world, not one feeder)."""

    kwp: np.ndarray  # kWp of every site
    batteries: list[BatterySpec]  # every battery in the fleet
    battery_share: float  # share of sites with a battery

    @classmethod
    def from_world(cls, world: Any) -> "Fleet":
        kwp = np.array([_kwp(s.row) for s in world.sites.values()], float)
        bats = [s.battery for s in world.sites.values() if s.battery is not None]
        return cls(kwp, bats, len(bats) / max(1, len(world.sites)))


def _kwp(row: Mapping[str, Any]) -> float:
    return float(row.get("kwp_true") or row.get("kwp_recorded") or 0.0)


def _eligible(sid: str) -> AssetStatus:
    return AssetStatus(sid, "eligible", 1.0, "ok")


def extract_feeders(world: Any, comps: Any, feeder_ids: Optional[Sequence[str]] = None) -> list[FeederWeek]:
    """FeederWeek per feeder: realised PV / load (world.site_day = physics), M1 / M2 forecasts and M4 statuses at
    t0 of each day (qualify exactly as the loop does), G from M8 (the world's FeederSpec, tau applied)."""
    from wattsentry.agent.loop import _safe_qualify

    fids = list(feeder_ids or world.feeders)
    sites = [s for f in fids for s in world.feeders[f].site_ids]
    D, T = world.days, DAY_STEPS
    preds = comps.predictors
    verdicts: dict[str, dict[str, int]] = {}

    def read(asset_id: str, key: str, t0: str, raw: Any) -> tuple[Any, str]:
        model = str(getattr(preds.get(key), "model", None) or MODEL_IDS[key])
        v, verdict = validated(asset_id, model, t0, lambda: raw)
        c = verdicts.setdefault(key, {})
        c[verdict] = c.get(verdict, 0) + 1
        if v is None:
            c["unknown"] = c.get("unknown", 0) + 1
        return v, verdict

    faulty = {f["asset_id"] for f in world.fault_truth() if f.get("asset_kind", "site") == "site"}
    faulty |= set(getattr(world, "ood_sites", set()) or ())
    idx = {s: i for i, s in enumerate(sites)}
    S = len(sites)
    pv, ld = np.zeros((S, D, T)), np.zeros((S, D, T))
    pv_q, ld_q = np.zeros((S, D, 3, T)), np.zeros((S, D, 3, T))
    statuses: list[dict[str, AssetStatus]] = []
    prices = np.zeros((D, T))
    t0s = []
    win = lambda t0: Window(t0=t0, horizon=T)  # noqa: E731
    for d in range(D):
        t0 = world.day_t0(d)
        t0s.append(t0)
        prices[d] = np.asarray(world.prices(d), float) / 1000.0
        hp = preds["sitehealth"]
        if hasattr(hp, "prefetch"):
            hp.prefetch(list(sites), t0, 0)
        hl, ages, unknown = {}, {}, set()
        for s in sites:
            row = world.sites[s].row
            sd = world.site_day(s, d)
            i = idx[s]
            pv[i, d], ld[i, d] = sd.pv_kw, sd.load_kw
            for key, arr in (("pv", pv_q), ("load", ld_q)):
                raw = preds[key].predict(row, win(t0))
                v, _ = read(s, key, t0, raw)
                if v is None:  # unknown: keep the raw curve for the rest-of-feeder forecast, out of the pool
                    unknown.add(s)
                    v = raw
                arr[i, d] = [v.q10, v.q50, v.q90]
            raw = hp.predict(row, win(t0))
            v, _ = read(s, "sitehealth", t0, raw)
            if v is None:
                unknown.add(s)
            hl[s] = raw if v is None else v
            ages[s] = world.observed_site_day(s, d).telemetry_age_s[0] / 60.0
        ass = _safe_qualify(comps, [world.sites[s].row for s in sites], hl, ages, None, {})
        st = {x.asset_id: x.status for x in ass}
        for s in unknown:  # no reliable source: human review, the site runs its own rule
            st[s] = AssetStatus(s, "human_review", 0.0, "judge:no_reliable_source", battery_ok=False)
        statuses.append({s: st.get(s) or _eligible(s) for s in sites})
    out = []
    for f in fids:
        feeder = world.feeders[f]
        spec, verdict = read(f, "gridtwin", t0s[0], feeder.spec)
        ids = list(feeder.site_ids)
        ii = np.array([idx[s] for s in ids], int)
        out.append(FeederWeek(
            feeder_id=f, g_kw=float(spec.g_kw), t0s=list(t0s), prices_kwh=prices.copy(), site_ids=ids,
            batteries=[world.sites[s].battery for s in ids], kwp=np.array([_kwp(world.sites[s].row) for s in ids]),
            pv=pv[ii], load=ld[ii], pv_q=pv_q[ii], load_q=ld_q[ii],
            statuses=[{s: day[s] for s in ids} for day in statuses],
            healthy=np.array([s not in faulty for s in ids], bool),
            meta={"src_feeder_id": getattr(world, "feeder_label_src", {}).get(f, f), "g_max_kw": spec.g_max_kw,
                  "g_conservative_kw": spec.g_conservative_kw, "tau": spec.tau, "gridtwin_verdict": verdict,
                  "week_start": getattr(world, "week_start", ""), "verdicts": verdicts}))
    return out


# =============================================================================================== synthetic homes
@dataclass
class SyntheticPlan:
    """n_max new homes for one feeder, drawn once: homes(n) = the first n (nested in n). Indexes are into the
    feeder's existing sites (donors); scale = new kWp / donor kWp."""

    pv_donor: np.ndarray
    load_donor: np.ndarray
    kwp: np.ndarray
    scale: np.ndarray
    batteries: list[Optional[BatterySpec]]

    @property
    def n_max(self) -> int:
        return len(self.kwp)


def plan_synthetic(fw: FeederWeek, fleet: Fleet, n_max: int = N_MAX, seed: int = 0) -> SyntheticPlan:
    rng = np.random.default_rng([seed, zlib.crc32(fw.feeder_id.encode())])
    donors = np.flatnonzero(fw.healthy & (fw.kwp > 0) & ~fw.synthetic)
    if donors.size == 0:
        donors = np.flatnonzero((fw.kwp > 0) & ~fw.synthetic)
    if donors.size == 0:
        raise ValueError(f"{fw.feeder_id}: no donor homes with kWp > 0")
    pv_d = rng.choice(donors, n_max)
    ld_d = rng.choice(donors, n_max)
    kwp = rng.choice(fleet.kwp[fleet.kwp > 0], n_max)
    has_b = rng.random(n_max) < fleet.battery_share
    pick = rng.integers(0, max(1, len(fleet.batteries)), n_max)
    bats = [fleet.batteries[k] if (h and fleet.batteries) else None for h, k in zip(has_b, pick)]
    return SyntheticPlan(pv_d, ld_d, kwp, kwp / fw.kwp[pv_d], bats)


def with_homes(fw: FeederWeek, plan: SyntheticPlan, n: int, batteries: bool = True) -> FeederWeek:
    """The feeder plus the first n synthetic homes (`batteries=False`: the same homes without a battery)."""
    if n <= 0:
        return fw
    if n > plan.n_max:
        raise ValueError(f"n={n} > plan n_max={plan.n_max}")
    pv_d, ld_d = plan.pv_donor[:n], plan.load_donor[:n]
    sc = plan.scale[:n, None, None]
    ids = [f"{fw.feeder_id}+{k:03d}" for k in range(n)]
    st = [{**day, **{s: _eligible(s) for s in ids}} for day in fw.statuses]
    return replace(
        fw, site_ids=fw.site_ids + ids,
        batteries=fw.batteries + (list(plan.batteries[:n]) if batteries else [None] * n),
        kwp=np.concatenate([fw.kwp, plan.kwp[:n]]),
        pv=np.concatenate([fw.pv, fw.pv[pv_d] * sc]), load=np.concatenate([fw.load, fw.load[ld_d]]),
        pv_q=np.concatenate([fw.pv_q, fw.pv_q[pv_d] * sc[..., None]]),
        load_q=np.concatenate([fw.load_q, fw.load_q[ld_d]]),
        statuses=st, healthy=np.concatenate([fw.healthy, np.ones(n, bool)]),
        synthetic=np.concatenate([fw.synthetic, np.ones(n, bool)]))


# =============================================================================================== replay
@dataclass
class WeekResult:
    minutes_over: float
    kwh_over: float
    peak_export_kw: float
    plan_status: dict[str, int] = field(default_factory=dict)
    net_kw: Optional[np.ndarray] = None  # (D*T,) feeder net export, + export

    def row(self) -> dict[str, Any]:
        return {"minutes_over": self.minutes_over, "kwh_over": round(self.kwh_over, 3),
                "peak_export_kw": round(self.peak_export_kw, 3)}


def _rollout(req: np.ndarray, e0: np.ndarray, ba: Mapping[str, np.ndarray], floor: np.ndarray
             ) -> tuple[np.ndarray, np.ndarray]:
    """K1 kernel (Rust when built): applied battery kW (S, T) and energy at the end (S,)."""
    from wattsentry.kernels import battery_rollout

    cap = ba["cap"]
    soc0 = np.divide(e0, cap, out=np.zeros_like(e0), where=cap > 0)
    soc, batt = battery_rollout(np.ascontiguousarray(req, dtype=np.float64), soc0, cap, ba["pmax"], ba["eta_c"],
                                ba["eta_d"], floor, ba["e_max"], DT_H)
    return batt, soc[:, -1] * cap


def thresholds(fw: FeederWeek, alpha: float) -> np.ndarray:
    """(S, D) per-home export threshold of each day: alpha * max_t(P50 PV - P50 load), as the loop's threshold rule."""
    return alpha * np.maximum(0.0, (fw.pv_q[:, :, 1] - fw.load_q[:, :, 1]).max(-1))


def rule_request(fw: FeederWeek, policy: str, alpha: float = 0.5) -> np.ndarray:
    """(S, D, T) battery request of a rule: self-consumption = surplus; threshold = surplus above theta, discharge
    to cover load, idle in between (`policies.threshold_step`). The kernel clips it to the battery."""
    surplus = fw.pv - fw.load
    if policy == "self_consumption":
        return surplus
    if policy == "threshold":
        th = thresholds(fw, alpha)[:, :, None]
        return np.where(surplus > th, surplus - th, np.where(surplus < 0, surplus, 0.0))
    raise ValueError(f"not a rule policy: {policy!r}")


def _prediction(sid: str, model: str, t0: str, q: np.ndarray) -> Prediction:
    return Prediction(sid, model, "hosting-capacity", t0, q.shape[-1], q[0], q[1], q[2])


Planner = Callable[..., Any]


def default_planner(spec, elig, rest, fc, g, prices, *, statuses, t0):
    """The loop's planner chain: wattsentry.opt.lp.plan_feeder, on error the aggregate planner."""
    from wattsentry.opt.lp import plan_feeder

    try:
        return plan_feeder(spec, elig, rest, fc, g, prices, statuses=statuses, t0=t0)
    except Exception as e:  # LPSolveError or a solver failure: the loop's backup
        from wattsentry.opt.aggregate import plan_feeder_aggregate

        log(f"LP failed on {spec.feeder_id} {t0} ({type(e).__name__}: {e}); aggregate planner")
        return plan_feeder_aggregate(spec, elig, rest, fc, g, prices, statuses=statuses, t0=t0)


def sentinel_day_request(fw: FeederWeek, d: int, energy: np.ndarray, planner: Planner = default_planner
                         ) -> tuple[np.ndarray, np.ndarray, str]:
    """Plan day d on the forecasts. Returns (pool mask (S,), requested kW (S, T) for the pool, solver status)."""
    from wattsentry.opt.policies import Site, rest_of_feeder_from_forecasts

    st = fw.statuses[d]
    t0 = fw.t0s[d]
    T = fw.pv.shape[-1]
    sites = [Site(s, b, float(energy[i]) if b is not None else None) for i, (s, b) in
             enumerate(zip(fw.site_ids, fw.batteries))]
    pool = np.array([b is not None and st[s].in_pool and st[s].battery_ok
                     for s, b in zip(fw.site_ids, fw.batteries)], bool)
    req = np.zeros((fw.n_sites, T))
    if not pool.any():
        return pool, req, "no_pool"
    fc = {s: {"pv": _prediction(s, "pvnet", t0, fw.pv_q[i, d]), "load": _prediction(s, "loadnet", t0, fw.load_q[i, d])}
          for i, s in enumerate(fw.site_ids)}
    rest_sites = [sites[i] for i in np.flatnonzero(~pool)]
    rest = rest_of_feeder_from_forecasts(rest_sites, fc, st, T) if rest_sites else None
    spec = FeederSpec(fw.feeder_id, fw.g_kw, fw.g_kw, 1.0)
    pl = planner(spec, [sites[i] for i in np.flatnonzero(pool)], rest, fc, fw.g_kw, fw.prices_kwh[d],
                 statuses=st, t0=t0)
    for i in np.flatnonzero(pool):
        sch = pl.schedule.get(fw.site_ids[i])
        if sch is not None:
            req[i] = sch
    return pool, req, str(pl.solver_status)


def replay_week(fw: FeederWeek, policy: str, *, alpha: float = 0.5, planner: Planner = default_planner,
                keep_net: bool = False) -> WeekResult:
    """The week on one feeder under one policy; minutes over = 15 x steps with net export > G (+1e-6)."""
    if policy not in POLICIES:
        raise ValueError(f"unknown policy {policy!r}; expected one of {POLICIES}")
    S, D, T = fw.pv.shape
    ba = battery_arrays(fw.batteries)
    e0 = ba["e_plan"].copy()  # the loop starts every battery at e_min
    status: dict[str, int] = {}
    if policy != "sentinel":
        req = rule_request(fw, policy, alpha).reshape(S, D * T)
        batt, _ = _rollout(req, e0, ba, ba["e_rule"])
    else:
        batt = np.zeros((S, D * T))
        energy = e0
        rule_req = rule_request(fw, "self_consumption")
        for d in range(D):
            pool, preq, solver = sentinel_day_request(fw, d, energy, planner)
            status[solver] = status.get(solver, 0) + 1
            req = np.where(pool[:, None], preq, rule_req[:, d])
            floor = np.where(pool, ba["e_plan"], ba["e_rule"])
            batt[:, d * T:(d + 1) * T], energy = _rollout(req, energy, ba, floor)
    net = (fw.pv - fw.load).reshape(S, D * T).sum(0) - batt.sum(0)
    over = net > fw.g_kw + EPS_KW
    return WeekResult(minutes_over=STEP_MIN * float(over.sum()),
                      kwh_over=float(np.maximum(0.0, net - fw.g_kw).sum() * DT_H),
                      peak_export_kw=float(net.max()) if net.size else 0.0, plan_status=status,
                      net_kw=net if keep_net else None)


# =============================================================================================== search
@dataclass
class Capacity:
    capacity: dict[int, int]  # tol (min/week) -> largest n with minutes(n) <= tol
    censored: dict[int, bool]  # True: never exceeded tol up to n_max (capacity >= n_max)
    curve: dict[int, float]  # every n evaluated -> minutes over
    monotone: bool


def capacity_search(minutes: Callable[[int], float], tols: Sequence[float] = TOLS_MIN, n_max: int = N_MAX,
                    step0: int = 4) -> Capacity:
    """Largest n in [0, n_max] with minutes(n) <= tol, per tol: galloping from the best known feasible n (step0,
    2*step0, ...) until a violation, then bisection. Evaluations are cached and shared by every tol (assumes
    minutes is non-decreasing in n; `monotone` reports whether the evaluated points were)."""
    cache: dict[int, float] = {}

    def m(n: int) -> float:
        if n not in cache:
            cache[n] = float(minutes(n))
        return cache[n]

    cap, cens = {}, {}
    for tol in sorted(tols):
        if m(0) > tol:
            cap[tol], cens[tol] = 0, False
            continue
        ok = [n for n, v in cache.items() if v <= tol]
        lo = max(ok)
        bad = [n for n, v in cache.items() if v > tol and n > lo]
        hi = min(bad) if bad else None
        step = step0
        while hi is None and lo < n_max:
            probe = min(n_max, lo + step)
            if m(probe) > tol:
                hi = probe
            else:
                lo = probe
                step *= 2
        if hi is None:
            cap[tol], cens[tol] = n_max, True
            continue
        while hi - lo > 1:
            mid = (lo + hi) // 2
            if m(mid) > tol:
                hi = mid
            else:
                lo = mid
        cap[tol], cens[tol] = lo, False
    pts = sorted(cache.items())
    mono = all(b[1] >= a[1] for a, b in zip(pts, pts[1:]))
    return Capacity(cap, cens, dict(pts), mono)


# =============================================================================================== jobs
@dataclass
class Job:
    fw: FeederWeek
    plan: SyntheticPlan
    policy: str
    variant: str
    tols: tuple[int, ...] = TOLS_MIN
    n_max: int = N_MAX
    alpha: float = 0.5


def run_job(job: Job, planner: Planner = default_planner) -> dict[str, Any]:
    """Capacity of one (feeder, policy, variant): JSON-ready dict."""
    t = time.perf_counter()
    batteries = job.variant == "batteries"
    details: dict[int, WeekResult] = {}
    status: dict[str, int] = {}

    def minutes(n: int) -> float:
        r = replay_week(with_homes(job.fw, job.plan, n, batteries), job.policy, alpha=job.alpha, planner=planner)
        details[n] = r
        for k, v in r.plan_status.items():
            status[k] = status.get(k, 0) + v
        return r.minutes_over

    c = capacity_search(minutes, job.tols, job.n_max)
    return {"feeder_id": job.fw.feeder_id, "policy": job.policy, "variant": job.variant,
            "capacity": {str(k): v for k, v in c.capacity.items()},
            "censored": {str(k): v for k, v in c.censored.items()}, "monotone": c.monotone,
            "curve": [[n, *details[n].row().values()] for n in sorted(details)],
            "evals": len(details), "plan_status": status, "seconds": round(time.perf_counter() - t, 2)}


def _job_cost(job: Job) -> float:  # schedule the expensive ones first (WattSentry LP on big feeders)
    return job.fw.n_sites * (50.0 if job.policy == "sentinel" else 1.0) * (2.0 if job.variant == "batteries" else 1.0)


_THREAD_VARS = ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "RAYON_NUM_THREADS")


def run_jobs(jobs: Sequence[Job], workers: int = 1) -> list[dict[str, Any]]:
    jobs = sorted(jobs, key=_job_cost, reverse=True)
    if workers <= 1:
        return [run_job(j) for j in jobs]
    out = []
    # spawn, not fork: the parent may already run Rust (rayon) / OpenMP threads, and forked workers
    # inherit their locked mutexes and hang. Thread caps are set before the workers start.
    saved = {k: os.environ.get(k) for k in _THREAD_VARS}
    os.environ.update({k: "1" for k in _THREAD_VARS})
    try:
        with ProcessPoolExecutor(max_workers=workers, initializer=_worker_init,
                                 mp_context=multiprocessing.get_context("spawn")) as ex:
            futs = {ex.submit(run_job, j): j for j in jobs}
            for f in as_completed(futs):
                r = f.result()
                log(f"{r['feeder_id']} {r['policy']} {r['variant']}: capacity {r['capacity']} "
                    f"({r['evals']} replays, {r['seconds']} s)")
                out.append(r)
    finally:
        for k, v in saved.items():
            if v is None:
                os.environ.pop(k, None)
            else:
                os.environ[k] = v
    return out


def _worker_init() -> None:
    for k in _THREAD_VARS:
        os.environ[k] = "1"


# =============================================================================================== results
def assemble(feeders: Sequence[FeederWeek], results: Sequence[Mapping[str, Any]], *, run_id: str,
             tols: Sequence[int] = TOLS_MIN, policies: Sequence[str] = POLICIES,
             variants: Sequence[str] = VARIANTS, n_max: int = N_MAX, extra: Optional[Mapping[str, Any]] = None
             ) -> dict[str, Any]:
    """The `hosting_capacity/v1` document (read by `dso_value.hosting_capacity`).

    feeders[i]: feeder_id, src_feeder_id, g_kw, g_max_kw, g_conservative_kw, tau, n_homes, n_batteries,
      capacity[variant][tol][policy] = homes that can be added, censored[...] (True = >= n_max, not reached),
      gain[variant][tol] = {vs_self_consumption, vs_threshold} (sentinel minus the policy),
      sentinel_ge_baseline[variant][tol], minutes_over_now[variant][policy] (n = 0), curve[variant][policy] =
      [[n, minutes_over, kwh_over, peak_export_kw], ...] (every replay of the search), monotone[variant][policy].
    total: capacity[variant][tol][policy] and gain[variant][tol] summed over feeders.
    """
    by = {(r["feeder_id"], r["variant"], r["policy"]): r for r in results}
    out_f = []
    tot_cap = {v: {str(t): {p: 0 for p in policies} for t in tols} for v in variants}
    below = 0
    for fw in feeders:
        f = fw.feeder_id
        rec: dict[str, Any] = {
            "feeder_id": f, "src_feeder_id": fw.meta.get("src_feeder_id", f), "g_kw": round(fw.g_kw, 3),
            "g_max_kw": fw.meta.get("g_max_kw"), "g_conservative_kw": fw.meta.get("g_conservative_kw"),
            "tau": fw.meta.get("tau"), "n_homes": fw.n_existing, "n_batteries": fw.n_batteries,
            "capacity": {}, "censored": {}, "gain": {}, "sentinel_ge_baseline": {}, "minutes_over_now": {},
            "curve": {}, "monotone": {}}
        for v in variants:
            rec["capacity"][v], rec["censored"][v], rec["gain"][v], rec["sentinel_ge_baseline"][v] = {}, {}, {}, {}
            rec["minutes_over_now"][v], rec["curve"][v], rec["monotone"][v] = {}, {}, {}
            for p in policies:
                r = by.get((f, v, p))
                if r is None:
                    continue
                rec["curve"][v][p] = r["curve"]
                rec["monotone"][v][p] = r["monotone"]
                rec["minutes_over_now"][v][p] = next((c[1] for c in r["curve"] if c[0] == 0), None)
            for t in tols:
                ts = str(t)
                caps = {p: by[(f, v, p)]["capacity"][ts] for p in policies if (f, v, p) in by}
                rec["capacity"][v][ts] = caps
                rec["censored"][v][ts] = {p: by[(f, v, p)]["censored"][ts] for p in caps}
                for p, c in caps.items():
                    tot_cap[v][ts][p] += c
                if "sentinel" in caps:
                    rec["gain"][v][ts] = {f"vs_{p}": caps["sentinel"] - caps[p] for p in caps if p != "sentinel"}
                    if BASELINE in caps:
                        ok = caps["sentinel"] >= caps[BASELINE]
                        rec["sentinel_ge_baseline"][v][ts] = ok
                        below += not ok
        out_f.append(rec)
    tot_gain = {v: {t: {f"vs_{p}": c["sentinel"] - c[p] for p in c if p != "sentinel"} if "sentinel" in c else {}
                    for t, c in tv.items()} for v, tv in tot_cap.items()}
    return {
        "schema": SCHEMA, "label": LABEL, "run_id": run_id, "judge_mode": judge_mode(),
        "week_start": feeders[0].meta.get("week_start") if feeders else "", "days": feeders[0].n_days if feeders else 0,
        "policies": list(policies), "tols_min_per_week": list(tols), "variants": list(variants), "n_max": n_max,
        "assumptions": {
            "new_homes": "PV of a healthy donor on the same feeder scaled to a fleet kWp draw; load of another donor; "
                         "battery drawn from the fleet with the fleet's battery share (variant no_batteries: none)",
            "new_home_forecasts": "donor M1/M2 forecasts scaled like the PV (proxy)",
            "limit": "G = tau*G_max + (1 - tau)*G_cons (M8), unchanged when homes are added (same cable)",
            "physics": "K1 battery_rollout; batteries start at e_min and carry energy across days; Sentinel = "
                       "day-ahead LP per day + comfort-reserve clip, no intraday re-qualify / hourly re-solve",
            "threshold_alpha": (extra or {}).get("alpha", 0.5)},
        "feeders": out_f,
        "total": {"capacity": tot_cap, "gain": tot_gain},
        "sentinel_below_baseline": below,
        **({"run": dict(extra)} if extra else {}),
    }


def write_outputs(doc: Mapping[str, Any], out_dir: Path) -> tuple[Path, Path]:
    out_dir.mkdir(parents=True, exist_ok=True)
    jp = out_dir / "hosting_capacity.json"
    jp.write_text(json.dumps(doc, indent=1, default=float))
    cp = out_dir / "hosting_capacity.csv"
    with cp.open("w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["feeder_id", "variant", "tol_min", "policy", "capacity", "censored", "gain_vs_self_consumption",
                    "label", "run_id"])
        for f in doc["feeders"]:
            for v, tv in f["capacity"].items():
                for t, caps in tv.items():
                    for p, c in caps.items():
                        g = c - caps[BASELINE] if BASELINE in caps else ""
                        w.writerow([f["feeder_id"], v, t, p, c, int(f["censored"][v][t][p]), g, doc["label"],
                                    doc["run_id"]])
    return jp, cp


def emit_metrics(doc: Mapping[str, Any]) -> None:
    """Headline: hosting_capacity_gain = total sentinel - self_consumption, batteries variant, 30 min/week."""
    from wattsentry.infra.metrics import emit

    tot = doc["total"]
    for v, tv in tot["gain"].items():
        sfx = "" if v == "batteries" else "_nobatt"
        for t, g in tv.items():
            if f"vs_{BASELINE}" in g:
                name = "hosting_capacity_gain" + ("" if t == "30" else f"_tol{t}") + sfx
                emit(name, g[f"vs_{BASELINE}"])
            if "vs_threshold" in g:
                emit(f"hosting_capacity_gain_vs_threshold_tol{t}{sfx}", g["vs_threshold"])
    for v, tv in tot["capacity"].items():
        sfx = "" if v == "batteries" else "_nobatt"
        for t, caps in tv.items():
            for p, c in caps.items():
                emit(f"hosting_capacity_{p}_tol{t}{sfx}", c)
    for f in doc["feeders"]:
        for v, tv in f["gain"].items():
            sfx = "" if v == "batteries" else "_nobatt"
            for t, g in tv.items():
                if f"vs_{BASELINE}" in g:
                    emit(f"hosting_capacity_gain_{f['feeder_id']}_tol{t}{sfx}", g[f"vs_{BASELINE}"])
    emit("hosting_capacity_sentinel_below_baseline", doc["sentinel_below_baseline"])


def summary_table(doc: Mapping[str, Any]) -> str:
    lines = []
    for v in doc["variants"]:
        for t in map(str, doc["tols_min_per_week"]):
            lines.append(f"-- {v}, tol {t} min/week ({doc['label']}, run {doc['run_id']})")
            lines.append(f"{'feeder':>8} {'homes':>5} {'G kW':>8} " + " ".join(f"{p:>16}" for p in doc["policies"])
                         + f" {'gain':>6}")
            for f in doc["feeders"]:
                caps = f["capacity"][v][t]
                cens = f["censored"][v][t]
                cells = " ".join(f"{str(caps.get(p, '-')) + ('+' if cens.get(p) else ''):>16}" for p in doc["policies"])
                g = f["gain"][v].get(t, {}).get(f"vs_{BASELINE}", "-")
                lines.append(f"{f['feeder_id']:>8} {f['n_homes']:>5} {f['g_kw']:>8.1f} {cells} {g:>6}")
            caps = doc["total"]["capacity"][v][t]
            g = doc["total"]["gain"][v][t].get(f"vs_{BASELINE}", "-")
            lines.append(f"{'total':>8} {'':>5} {'':>8} " + " ".join(f"{caps.get(p, '-'):>16}" for p in doc["policies"])
                         + f" {g:>6}")
    return "\n".join(lines)


# =============================================================================================== CLI
def _world(kind: str, days: int, seed: int) -> Any:
    from wattsentry.agent.world import MockWorld, SimWorld, find_windows

    if kind == "mock":
        return MockWorld(days=days, seed=seed, with_wind=False)
    win = find_windows()
    allf = sorted({s["feeder_id"] for s in json.loads((Path(win) / "sites.json").read_text())}) if win else None
    w = SimWorld(days=days, seed=seed, feeders=allf, wind=False)
    w.feeder_label_src = {lbl: src for src, lbl in w.feeder_label.items() if lbl in w.feeders}
    return w


def main(argv: Optional[Sequence[str]] = None) -> dict[str, Any]:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--world", default="sim", choices=("sim", "mock"))
    ap.add_argument("--days", type=int, default=7)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--feeders", default="", help="comma list of feeder ids (world labels, e.g. F7); default all")
    ap.add_argument("--policies", default=",".join(POLICIES))
    ap.add_argument("--variants", default=",".join(VARIANTS))
    ap.add_argument("--tols", default=",".join(map(str, TOLS_MIN)), help="minutes over limit per week")
    ap.add_argument("--n-max", type=int, default=N_MAX)
    ap.add_argument("--alpha", type=float, default=0.5, help="threshold rule alpha (the replay's default)")
    ap.add_argument("--workers", type=int, default=min(20, os.cpu_count() or 1))
    ap.add_argument("--out", default=os.environ.get("LAB_OUTPUT_DIR", "out/hosting_capacity"))
    a = ap.parse_args(argv)
    t_start = time.perf_counter()
    from wattsentry.agent.loop import default_components
    from wattsentry.kernels import backend

    world = _world(a.world, a.days, a.seed)
    comps = default_components(world)
    fids = [f for f in a.feeders.split(",") if f] or list(world.feeders)
    t = time.perf_counter()
    fws = extract_feeders(world, comps, None)  # every feeder: the fleet distribution needs all of them anyway
    fws = [fw for fw in fws if fw.feeder_id in fids]
    log(f"extracted {len(fws)} feeders in {time.perf_counter() - t:.1f}s (kernels: {backend()})")
    fleet = Fleet.from_world(world)
    tols = tuple(int(x) for x in a.tols.split(","))
    pols = [p for p in a.policies.split(",") if p]
    vars_ = [v for v in a.variants.split(",") if v]
    jobs = [Job(fw, plan_synthetic(fw, fleet, a.n_max, a.seed), p, v, tols, a.n_max, a.alpha)
            for fw in fws for p in pols for v in vars_]
    t = time.perf_counter()
    results = run_jobs(jobs, a.workers)
    t_search = time.perf_counter() - t
    run_id = os.environ.get("LAB_RUN_ID") or f"local-{world.name}-s{a.seed}"
    extra = {"world": world.name, "seed": a.seed, "alpha": a.alpha, "workers": a.workers, "kernels": backend(),
             "components": comps.sources, "fallbacks": comps.fallbacks, "t_search_s": round(t_search, 1),
             "fleet": {"n_sites": len(world.sites), "battery_share": round(fleet.battery_share, 4),
                       "kwp_mean": round(float(fleet.kwp.mean()), 3)},
             "verdicts": fws[0].meta.get("verdicts") if fws else {}}
    doc = assemble(fws, results, run_id=run_id, tols=tols, policies=pols, variants=vars_, n_max=a.n_max,
                   extra=extra)
    jp, cp = write_outputs(doc, Path(a.out))
    print(summary_table(doc), flush=True)
    emit_metrics(doc)
    from wattsentry.infra.metrics import emit

    emit("t_hosting_search_s", t_search)
    emit("t_hosting_total_s", time.perf_counter() - t_start)
    log(f"wrote {jp} and {cp}")
    return doc


if __name__ == "__main__":
    main()
