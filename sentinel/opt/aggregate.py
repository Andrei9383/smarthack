"""Fallback planner (card #18, plan §3.5): one virtual battery per feeder, split to sites by headroom.

    plan_feeder_aggregate(feeder, eligible_sites, rest_of_feeder, forecasts, G=None, prices=None, *,
                          statuses=None, params=LPParams(), t0=None, peak_mask=None) -> DetailedPlan

Same signature, same controlled / folded site rules and same output conventions as `lp.plan_feeder`
(`method="aggregate"`). Steps:

1. Virtual battery: capacity, power, energy bounds and e0 are sums over controlled sites; efficiencies are
   capacity-weighted means. Feeder P50 / worst-case net export are the site sums.
2. Solve the same LP (`lp.solve_lp`) with N = 1 (a few hundred variables, milliseconds).
3. Split the aggregate net power c − d at each step across sites by headroom: charging in proportion to free energy
   (E_i − e_i), discharging in proportion to energy above the reserve (e_i − R_i), each capped by the site's power
   and energy limits (water-filling redistributes what a capped site can't take).
4. Per-site feasibility clip through `BatterySpec.step` with the comfort-reserve floor. Export curve and slack are
   recomputed from the applied schedule, so any split shortfall shows up as slack, never as a hidden breach.
"""

from __future__ import annotations

import time
from typing import Any, Mapping, Optional, Sequence, Union

import numpy as np

from sentinel.contracts import DT_H, AssetStatus, FeederSpec

from .lp import LPParams, _finish, _prepare, solve_with_repair
from .policies import ArrayLike, DetailedPlan, Forecasts, RestLike, Site

__all__ = ["plan_feeder_aggregate", "split_by_headroom"]


def _waterfill(total: float, weights: np.ndarray, caps: np.ndarray) -> np.ndarray:
    """Split `total` >= 0 in proportion to `weights`, each share <= its cap (redistributing the excess)."""
    out = np.zeros_like(caps)
    free = caps > 1e-12
    remaining = min(total, float(caps.sum()))
    for _ in range(len(caps) + 1):
        if remaining <= 1e-12 or not free.any():
            break
        w = np.where(free, np.maximum(weights, 0.0), 0.0)
        if w.sum() <= 1e-12:
            w = free.astype(float)
        share = remaining * w / w.sum()
        room = caps - out
        take = np.minimum(share, room)
        out += take
        remaining -= float(take.sum())
        free &= (caps - out) > 1e-12
    return out


def split_by_headroom(
    sites: Sequence[Site], e0: np.ndarray, floors: np.ndarray, p_total: np.ndarray, dt_h: float = DT_H
) -> tuple[dict[str, np.ndarray], dict[str, np.ndarray]]:
    """Split an aggregate battery schedule p_total[T] (+ charge) across sites, then step each battery.
    Returns (applied kW per site [T], energy per site [T+1])."""
    N, T = len(sites), len(p_total)
    bats = [s.battery for s in sites]
    p_max = np.array([b.p_max_kw for b in bats])
    e_max = np.array([b.e_max for b in bats])
    eta_c = np.array([b.eta_charge for b in bats])
    eta_d = np.array([b.eta_discharge for b in bats])
    e = np.zeros((N, T + 1))
    e[:, 0] = e0
    out = np.zeros((N, T))
    for t in range(T):
        p = float(p_total[t])
        if p > 0:
            room = np.maximum(0.0, e_max - e[:, t])
            req = _waterfill(p, room, np.minimum(p_max, room / (eta_c * dt_h)))
        elif p < 0:
            avail = np.maximum(0.0, e[:, t] - floors)
            req = -_waterfill(-p, avail, np.minimum(p_max, avail * eta_d / dt_h))
        else:
            req = np.zeros(N)
        for i, b in enumerate(bats):
            out[i, t], e[i, t + 1] = b.step(e[i, t], float(req[i]), dt_h)
            if e[i, t + 1] < floors[i] - 1e-9:  # never happens given the caps above; belt and braces
                raise AssertionError(f"{sites[i].site_id}: split breached the reserve")
    return ({s.site_id: out[i] for i, s in enumerate(sites)}, {s.site_id: e[i] for i, s in enumerate(sites)})


def plan_feeder_aggregate(
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
    """Virtual-battery fallback for `lp.plan_feeder` (module docstring)."""
    wall = time.perf_counter()
    pp = _prepare(feeder, eligible_sites, rest_of_feeder, forecasts, G, prices, statuses, t0, peak_mask, params)
    if not pp.sites:
        return _finish(pp, {}, {}, method="aggregate", status="no_controllable_sites", objective=None,
                       simultaneous=0, solve_time=0.0, wall_start=wall)
    bats = [s.battery for s in pp.sites]
    cap = np.array([b.capacity_kwh for b in bats])
    e0 = np.array([s.initial_energy() for s in pp.sites])
    floors = np.minimum(np.array([b.e_min for b in bats]), e0)
    w = cap / cap.sum()
    net50 = sum(pp.q[s.site_id]["pv50"] - pp.q[s.site_id]["l50"] for s in pp.sites)
    worst = sum(pp.q[s.site_id]["pv90"] - pp.q[s.site_id]["l10"] for s in pp.sites)
    sol, simul_mask, info = solve_with_repair(dict(
        net50=np.asarray(net50)[None, :], worst=np.asarray(worst)[None, :],
        p_max=np.array([sum(b.p_max_kw for b in bats)]),
        e_lo=np.array([floors.sum()]), e_hi=np.array([sum(b.e_max for b in bats)]), e0=np.array([e0.sum()]),
        eta_c=np.array([float(w @ [b.eta_charge for b in bats])]),
        eta_d=np.array([float(w @ [b.eta_discharge for b in bats])]),
        rest_worst=np.asarray(pp.rest.worst_kw, float), rest_net=np.asarray(pp.rest.net_kw, float),
        g_kw=pp.g, prices=pp.prices, peak_mask=pp.mask,
    ), params)
    p_total = sol.c[0] - sol.d[0]
    applied, soc = split_by_headroom(pp.sites, e0, floors, p_total)
    simul = int(simul_mask.sum())
    shortfall = float(np.abs(p_total - sum(applied.values())).max())
    return _finish(pp, applied, soc, method="aggregate", status=sol.status, objective=sol.objective,
                   simultaneous=simul, solve_time=info["solve_time_total_s"], wall_start=wall,
                   extra={"lp_slack_max_kw": float(sol.s.max()), "split_shortfall_kw": shortfall, **info})
