"""Per-feeder day-ahead LP (card #18, plan §3.5): cvxpy + HiGHS, vectorised over sites (matrix variables).

    plan_feeder(feeder, eligible_sites, rest_of_feeder, forecasts, G=None, prices=None, *,
                statuses=None, params=LPParams(), t0=None, peak_mask=None) -> DetailedPlan

For feeder f, steps t = 1..T (T = forecast horizon, 96 for a day), Δt = 0.25 h, controlled sites i
(all variables >= 0):

    min  Σ_t Σ_i [ π_t·m_it·Δt − β·π_t·x_it·Δt + c_deg·(c_it + d_it)·Δt ]  +  M·Σ_t s_t·Δt  +  λ·z
         + M·w·max_t s_t          (w = slack_peak_weight; our addition: flattens unavoidable violations)
    site balance (P50):  P̂50_it − L̂50_it − c_it + d_it = x_it − m_it
    battery:             e_i,t+1 = e_it + η_c·c_it·Δt − d_it·Δt/η_d ;  R_i ≤ e_it ≤ E_i ;  c_it, d_it ≤ P_i ;
                         e_i,0 = e0_i ;  e_i,T ≥ e0_i
    export limit:        Σ_i (P̂90_it − L̂10_it − c_it + d_it) + X_t^rest ≤ G + s_t      (worst plausible export)
    evening peak:        z ≥ Σ_i (m_it − x_it) + M_t^rest     for t in 17:00–22:00 Europe/Bucharest

`R_i` is `BatterySpec.e_min = max(soc_min·E, comfort reserve)`; `E_i` is `e_max`. If a battery starts below its
reserve, its floor is e0 (the planner never takes it lower, and is not forced up).

Who is controlled: sites in `eligible_sites` that are in the pool (`AssetStatus.in_pool`), `battery_ok`, and have a
battery. Derated sites have their PV quantiles scaled by `headroom_factor` (PV fault: forecast cut by the
estimated loss; battery stays in if battery_ok). Any other site passed in `eligible_sites` is *folded* into the
rest of the feeder: it runs the self-consumption rule and is simulated on its forecasts (R3). `rest_of_feeder`
covers the feeder's other (non-eligible) sites: a `RestOfFeeder` (see `policies.rest_of_feeder_from_forecasts`),
a plain per-step net-export sequence (used for both P50 and worst), or None.

Output (`DetailedPlan`, a contract `FeederPlan`): schedule[site] = battery kW per step (+ charge, − discharge),
already netted (c − d) and replayed through `BatterySpec.step` with the comfort-reserve floor, so it is feasible
for the battery model; export_curve = P50 feeder net export of that schedule (+ rest); slack_t = max(0, worst
export − G) of that schedule (kW the plan could not remove). Diagnostics: soc_kwh, worst_export_curve,
peak_import_kw, simultaneous_steps, solve_time_s (HiGHS, all solves), wall_time_s, extra.

Simultaneous charge/discharge: the LP 'burns' energy (c and d at once) when the battery is full and the limit still
binds (slack > 0), or with negative prices. `solve_with_repair` then forbids discharge at those steps and re-solves
(still an LP, ≤ `repair_iters` times); `extra["simultaneous_before_repair"]` / `simultaneous_steps` report it
before / after. Any remainder is netted by the clip, so the reported slack is always what the battery can do.

Units: kW, kWh, prices lei/kWh. Weights must satisfy M ≫ λ ≫ π (checked loosely). LP only: no binaries.
Raises `LPSolveError` if HiGHS does not return an optimal solution; callers (#17) then use
`aggregate.plan_feeder_aggregate` or `policies.self_consumption_plan`.
"""

from __future__ import annotations

import time
from dataclasses import dataclass
from datetime import timedelta
from typing import Any, Mapping, Optional, Sequence, Union

import cvxpy as cp
import numpy as np

from sentinel.contracts import DT_H, AssetStatus, BatterySpec, FeederSpec, parse_utc

from .policies import (
    DEFAULT_BETA,
    ArrayLike,
    DetailedPlan,
    Forecasts,
    RestLike,
    RestOfFeeder,
    Site,
    as_rest,
    as_site,
    battery_from_row,
    feeder_id_and_g,
    horizon_of,
    is_controllable,
    price_vector,
    pv_factor,
    rest_of_feeder_from_forecasts,
    site_from_row,
    site_quantiles,
    t0_of,
)

__all__ = [
    "LPParams",
    "LPSolveError",
    "LPSolution",
    "Site",
    "RestOfFeeder",
    "DetailedPlan",
    "battery_from_row",
    "site_from_row",
    "peak_mask_for",
    "solve_lp",
    "solve_with_repair",
    "clip_schedule",
    "plan_feeder",
]

try:  # Europe/Bucharest for the evening window; fall back to a fixed-offset approximation without tzdata.
    from zoneinfo import ZoneInfo

    _TZ: Any = ZoneInfo("Europe/Bucharest")
except Exception:  # pragma: no cover
    _TZ = None


class LPSolveError(RuntimeError):
    """HiGHS did not return an optimal solution (status in the message)."""


@dataclass(frozen=True)
class LPParams:
    """Objective weights and options. Defaults: β export remuneration ratio (to verify); c_deg ≈ €0.025/kWh per
    direction in lei; λ lei per kW of evening peak; M lei per kWh above the limit (M ≫ λ ≫ π)."""

    beta: float = DEFAULT_BETA
    c_deg: float = 0.12  # lei/kWh per direction (C_pack / (2·N_cycles·E_usable) ≈ €0.02–0.03)
    lam: float = 20.0  # lei per kW of evening feeder import peak z
    big_m: float = 1.0e4  # lei per kWh of slack
    slack_peak_weight: float = 1.0  # + big_m * weight * max_t s_t: flattens unavoidable violations (1 kW ~ 1 kWh)
    peak_start_h: float = 17.0
    peak_end_h: float = 22.0
    simultaneous_tol_kw: float = 1e-3
    repair_iters: int = 4  # re-solves that forbid discharge at steps where the LP charged + discharged at once
    time_limit_s: float = 60.0
    verbose: bool = False

    def __post_init__(self) -> None:
        if not (self.big_m > self.lam > 0 and self.c_deg >= 0 and 0 <= self.beta <= 1):
            raise ValueError("need big_m > lam > 0, c_deg >= 0, 0 <= beta <= 1")


def _local_hour(ts) -> float:
    if _TZ is not None:
        loc = ts.astimezone(_TZ)
        return loc.hour + loc.minute / 60.0
    off = 3 if 4 <= ts.month <= 10 else 2  # pragma: no cover
    return (ts.hour + ts.minute / 60.0 + off) % 24  # pragma: no cover


def peak_mask_for(t0: str, horizon: int, start_h: float = 17.0, end_h: float = 22.0,
                  step_minutes: int = 15) -> np.ndarray:
    """Boolean [horizon]: step starts in [start_h, end_h) local Europe/Bucharest time (DST-aware)."""
    start = parse_utc(t0)
    return np.array([
        start_h <= _local_hour(start + timedelta(minutes=step_minutes * k)) < end_h for k in range(horizon)
    ])


@dataclass
class LPSolution:
    """Raw LP solution (arrays [N, T] / [N, T+1] / [T]) from `solve_lp`."""

    c: np.ndarray
    d: np.ndarray
    e: np.ndarray
    s: np.ndarray
    z: float
    status: str
    objective: Optional[float]
    solve_time_s: float

    @property
    def simultaneous(self) -> np.ndarray:
        return np.minimum(self.c, self.d)


def solve_lp(
    net50: np.ndarray,
    worst: np.ndarray,
    p_max: np.ndarray,
    e_lo: np.ndarray,
    e_hi: np.ndarray,
    e0: np.ndarray,
    eta_c: np.ndarray,
    eta_d: np.ndarray,
    rest_worst: np.ndarray,
    rest_net: np.ndarray,
    g_kw: float,
    prices: np.ndarray,
    peak_mask: np.ndarray,
    params: LPParams = LPParams(),
    dt_h: float = DT_H,
    no_discharge: Optional[np.ndarray] = None,
) -> LPSolution:
    """The §3.5 LP on arrays. net50 = P50 PV − P50 load [N, T]; worst = P90 PV − P10 load [N, T]; per-site
    vectors [N]; rest_* / prices / peak_mask [T]; no_discharge [T] bool forces d = 0 at those steps (repair).
    Shared by `plan_feeder` and the aggregate fallback (N = 1)."""
    N, T = net50.shape
    col = lambda v: np.asarray(v, float).reshape(N, 1)  # noqa: E731
    c = cp.Variable((N, T), nonneg=True)
    d = cp.Variable((N, T), nonneg=True)
    x = cp.Variable((N, T), nonneg=True)
    m = cp.Variable((N, T), nonneg=True)
    e = cp.Variable((N, T + 1))
    s = cp.Variable(T, nonneg=True)
    z = cp.Variable(nonneg=True)
    s_max = cp.Variable(nonneg=True)
    P = np.repeat(col(p_max), T, axis=1)
    ec = np.repeat(col(eta_c) * dt_h, T, axis=1)
    ed = np.repeat(dt_h / col(eta_d), T, axis=1)
    lo = np.repeat(col(e_lo), T, axis=1)
    hi = np.repeat(col(e_hi), T, axis=1)
    cons = [
        x - m == net50 - c + d,
        c <= P,
        d <= (P if no_discharge is None else P * ~np.asarray(no_discharge, bool)[None, :]),
        e[:, 0] == e0,
        e[:, 1:] == e[:, :-1] + cp.multiply(ec, c) - cp.multiply(ed, d),
        e[:, 1:] >= lo,
        e[:, 1:] <= hi,
        e[:, T] >= e0,
        cp.sum(worst - c + d, axis=0) + rest_worst <= g_kw + s,
        s <= s_max,
    ]
    idx = np.flatnonzero(peak_mask)
    if idx.size:
        cons.append(z >= cp.sum(m[:, idx] - x[:, idx], axis=0) - rest_net[idx])
    pi = np.broadcast_to(prices, (N, T))
    obj = (
        dt_h * cp.sum(cp.multiply(pi, m))
        - params.beta * dt_h * cp.sum(cp.multiply(pi, x))
        + params.c_deg * dt_h * cp.sum(c + d)
        + params.big_m * dt_h * cp.sum(s)
        + params.big_m * params.slack_peak_weight * s_max
        + params.lam * z
    )
    prob = cp.Problem(cp.Minimize(obj), cons)
    try:
        prob.solve(solver=cp.HIGHS, verbose=params.verbose, time_limit=params.time_limit_s)
    except cp.error.SolverError as exc:
        raise LPSolveError(f"HiGHS failed: {exc}") from exc
    if prob.status not in (cp.OPTIMAL, cp.OPTIMAL_INACCURATE):
        raise LPSolveError(f"LP status {prob.status}")
    st = prob.solver_stats
    solve_t = float(st.solve_time) if st is not None and st.solve_time is not None else float("nan")
    return LPSolution(
        c=np.maximum(0.0, np.asarray(c.value)), d=np.maximum(0.0, np.asarray(d.value)), e=np.asarray(e.value),
        s=np.maximum(0.0, np.asarray(s.value)), z=float(z.value), status=str(prob.status),
        objective=float(prob.value), solve_time_s=solve_t,
    )


def clip_schedule(battery: BatterySpec, e0: float, batt_kw: ArrayLike, floor_kwh: Optional[float] = None,
                  dt_h: float = DT_H) -> tuple[np.ndarray, np.ndarray]:
    """Per-site feasibility clip: replay requested kW through `BatterySpec.step`, never discharging below
    `floor_kwh` (default `battery.e_min`, the comfort reserve; or e0 if it starts lower).
    Returns (applied kW [T], energy kWh [T+1])."""
    floor = battery.e_min if floor_kwh is None else floor_kwh
    floor = min(floor, e0)
    p_req = np.asarray(batt_kw, float)
    out = np.zeros_like(p_req)
    e = np.zeros(len(p_req) + 1)
    e[0] = e0
    for t, p in enumerate(p_req):
        if p < 0:  # keep the planner's floor (step() itself only knows the soc_min floor)
            p = max(p, -max(0.0, (e[t] - floor) * battery.eta_discharge / dt_h))
        out[t], e[t + 1] = battery.step(e[t], float(p), dt_h)
    return out, e


def solve_with_repair(lp_args: dict, params: LPParams) -> tuple[LPSolution, np.ndarray, dict]:
    """`solve_lp` + LP-only repair of simultaneous charge/discharge (no binaries).

    The LP can 'burn' energy (c and d at once lose (1 − η_c·η_d) of it) to cut export when the battery is full and
    the limit still binds. Discharging during such a step only adds export, so at steps where any site does both
    we forbid discharge for all sites and re-solve (up to `params.repair_iters` times). Returns
    (solution, remaining simultaneous mask [N, T], info)."""
    T = len(lp_args["prices"])
    banned = np.zeros(T, bool)
    sol = solve_lp(**lp_args, params=params)
    total_t = sol.solve_time_s
    first = int((sol.simultaneous > params.simultaneous_tol_kw).sum())
    iters = 0
    while iters < params.repair_iters:
        simul = sol.simultaneous > params.simultaneous_tol_kw
        new = simul.any(axis=0) & ~banned
        if not new.any():
            break
        banned |= new
        sol = solve_lp(**lp_args, params=params, no_discharge=banned)
        total_t += sol.solve_time_s
        iters += 1
    simul = sol.simultaneous > params.simultaneous_tol_kw
    info = {"repair_iters": iters, "simultaneous_before_repair": first, "no_discharge_steps": int(banned.sum()),
            "solve_time_total_s": total_t}
    return sol, simul, info


@dataclass
class _Prepared:
    fid: str
    g: float
    t0: str
    T: int
    sites: list[Site]  # controlled
    folded: list[str]
    q: dict[str, dict[str, np.ndarray]]
    rest: RestOfFeeder
    prices: np.ndarray
    mask: np.ndarray


def _prepare(feeder, eligible_sites, rest_of_feeder, forecasts, G, prices, statuses, t0, peak_mask,
             params) -> _Prepared:
    fid, g = feeder_id_and_g(feeder, G)
    statuses = statuses or {}
    sites = [as_site(s) for s in eligible_sites]
    ids = [s.site_id for s in sites]
    T = horizon_of(forecasts, ids)
    if T is None:
        if rest_of_feeder is None:
            raise ValueError("no forecasts for eligible sites and no rest_of_feeder: horizon unknown")
        T = len(rest_of_feeder.net_kw) if isinstance(rest_of_feeder, RestOfFeeder) else len(rest_of_feeder)
    rest = as_rest(rest_of_feeder, T)
    ok = [is_controllable(s, statuses.get(s.site_id)) for s in sites]
    controlled = [s for s, k in zip(sites, ok) if k]
    folded = [s for s, k in zip(sites, ok) if not k]
    if folded:
        rest = rest + rest_of_feeder_from_forecasts(folded, forecasts, statuses, T)
    q = {s.site_id: site_quantiles(forecasts, s.site_id, pv_factor(statuses.get(s.site_id))) for s in controlled}
    for sid, v in q.items():
        if len(v["pv50"]) != T:
            raise ValueError(f"{sid}: horizon {len(v['pv50'])} != {T}")
    t0 = t0 or t0_of(forecasts, ids) or ""
    if peak_mask is not None:
        mask = np.asarray(peak_mask, bool)
    elif t0:
        mask = peak_mask_for(t0, T, params.peak_start_h, params.peak_end_h)
    else:
        mask = np.zeros(T, bool)
    return _Prepared(fid, g, t0, T, controlled, [s.site_id for s in folded], q, rest,
                     price_vector(prices, T), mask)


def _finish(pp: _Prepared, applied: dict[str, np.ndarray], soc: dict[str, np.ndarray], *, method: str,
            status: str, objective: Optional[float], simultaneous: int, solve_time: float, wall_start: float,
            extra: Optional[dict] = None) -> DetailedPlan:
    """Export curve / slack / peak recomputed from the applied (clipped) schedule."""
    T = pp.T
    net = np.asarray(pp.rest.net_kw, float).copy()
    worst = np.asarray(pp.rest.worst_kw, float).copy()
    for s in pp.sites:
        p = applied[s.site_id]
        v = pp.q[s.site_id]
        net += v["pv50"] - v["l50"] - p
        worst += v["pv90"] - v["l10"] - p
    slack = np.maximum(0.0, worst - pp.g)
    slack[slack < 1e-6] = 0.0
    peak = float(max(0.0, (-net[pp.mask]).max())) if pp.mask.any() else 0.0
    return DetailedPlan(
        feeder_id=pp.fid, t0=pp.t0, policy="sentinel", g_kw=pp.g,
        schedule={sid: list(map(float, p)) for sid, p in applied.items()},
        export_curve=list(map(float, net)), slack=list(map(float, slack)), horizon=T,
        solver_status=status, objective=objective, method=method,
        soc_kwh={sid: list(map(float, e)) for sid, e in soc.items()},
        worst_export_curve=list(map(float, worst)), peak_import_kw=peak, simultaneous_steps=int(simultaneous),
        folded_sites=list(pp.folded), n_sites=len(pp.sites), solve_time_s=float(solve_time),
        wall_time_s=time.perf_counter() - wall_start, extra=dict(extra or {}),
    )


def plan_feeder(
    feeder: Union[FeederSpec, str],
    eligible_sites: Sequence[Union[Site, Mapping[str, Any]]],
    rest_of_feeder: RestLike,
    forecasts: Forecasts,
    G: Optional[float] = None,
    prices: Union[float, ArrayLike, None] = None,
    *,
    statuses: Optional[Mapping[str, AssetStatus]] = None,
    params: LPParams = LPParams(),
    t0: Optional[str] = None,
    peak_mask: Optional[ArrayLike] = None,
) -> DetailedPlan:
    """Solve the per-feeder LP (module docstring). `G` defaults to `feeder.g_kw` (FeederSpec); `prices` lei/kWh
    as scalar, per step or hourly (None -> 0.6 flat). `eligible_sites`: `Site` or `sites` rows (dicts)."""
    wall = time.perf_counter()
    pp = _prepare(feeder, eligible_sites, rest_of_feeder, forecasts, G, prices, statuses, t0, peak_mask, params)
    if not pp.sites:
        return _finish(pp, {}, {}, method="lp", status="no_controllable_sites", objective=None,
                       simultaneous=0, solve_time=0.0, wall_start=wall)
    if params.lam <= float(np.max(np.abs(pp.prices))):
        raise ValueError("need lam >> prices (M >> lambda >> pi)")
    N = len(pp.sites)
    bats = [s.battery for s in pp.sites]
    e0 = np.array([s.initial_energy() for s in pp.sites])
    e_min = np.array([b.e_min for b in bats])
    sol, simul, info = solve_with_repair(
        dict(
            net50=np.stack([pp.q[s.site_id]["pv50"] - pp.q[s.site_id]["l50"] for s in pp.sites]),
            worst=np.stack([pp.q[s.site_id]["pv90"] - pp.q[s.site_id]["l10"] for s in pp.sites]),
            p_max=np.array([b.p_max_kw for b in bats]),
            e_lo=np.minimum(e_min, e0),
            e_hi=np.array([b.e_max for b in bats]),
            e0=e0,
            eta_c=np.array([b.eta_charge for b in bats]),
            eta_d=np.array([b.eta_discharge for b in bats]),
            rest_worst=np.asarray(pp.rest.worst_kw, float),
            rest_net=np.asarray(pp.rest.net_kw, float),
            g_kw=pp.g, prices=pp.prices, peak_mask=pp.mask,
        ),
        params,
    )
    applied, soc = {}, {}
    for i, s in enumerate(pp.sites):
        applied[s.site_id], soc[s.site_id] = clip_schedule(s.battery, e0[i], sol.c[i] - sol.d[i],
                                                           floor_kwh=min(e_min[i], e0[i]))
    extra = {"lp_slack_max_kw": float(sol.s.max()), "lp_peak_kw": sol.z,
             "n_vars": N * (5 * pp.T + 1) + pp.T + 1, **info}
    if simul.any():
        extra["simultaneous_sites"] = sorted({pp.sites[i].site_id for i in np.flatnonzero(simul.any(axis=1))})
    return _finish(pp, applied, soc, method="lp", status=sol.status, objective=sol.objective,
                   simultaneous=int(simul.sum()), solve_time=info["solve_time_total_s"], wall_start=wall,
                   extra=extra)


def _timing(n_sites: int = 100, t0: str = "2025-06-21T00:00:00Z") -> dict[str, float]:  # pragma: no cover
    """Benchmark helper: `python -m sentinel.opt.lp [N]` plans one congested mock feeder of N sites."""
    from sentinel.contracts import MockPredictor

    pv, ld = MockPredictor("pv"), MockPredictor("load")
    sites, fc = [], {}
    for k in range(n_sites):
        sid = f"S{k:04d}"
        row = {"site_id": sid, "kwp_true": 4 + 6 * ((k * 7) % 10) / 10, "lat": 44.4, "archetype": "base",
               "battery_kwh": 10.0, "battery_kw": 5.0, "comfort_reserve_kwh": 2.0}
        sites.append(row)
        fc[sid] = {"pv": pv.predict(row, t0=t0), "load": ld.predict(row, t0=t0)}
    worst = sum(np.asarray(f["pv"].q90) - np.asarray(f["load"].q10) for f in fc.values())
    plan = plan_feeder("F_bench", sites, None, fc, G=0.7 * float(worst.max()), prices=0.6)
    return {"n_sites": n_sites, "solve_time_s": plan.solve_time_s, "wall_time_s": plan.wall_time_s,
            "max_slack_kw": plan.max_slack_kw, "simultaneous_steps": plan.simultaneous_steps}


if __name__ == "__main__":  # pragma: no cover
    import sys

    for n in [int(a) for a in sys.argv[1:]] or [10, 50, 100]:
        print(_timing(n))
