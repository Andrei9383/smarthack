"""P10 feeder commitment (plan §3.4, §3.5 "P10 commitment"; contract team/contracts/opt.md).

    scenarios = make_scenarios(pv_preds, load_preds, n=200, batteries=..., soc0=...)
    c = commit(feeder, schedule, scenarios, haircut=0.05)          # -> Commitment
    c = realise(c, schedule, pv_actual, load_actual, ...)          # next tick
    hit_stats([c, ...])                                            # p10_hit_rate, p10_miss_kw

1. Scenarios (~200 per feeder-day): each scenario draws one ensemble member, ONE PV quantile level shared by every
   site of the feeder (weather is common; Latin-hypercube over scenarios) and an independent load level per site.
   A level u maps to a value through the member's (calibrated) q10/q50/q90 — piecewise linear in the normal score
   z = Phi^-1(u), split-normal tails past q10/q90 — clipped at 0 (and the PV cap if given). One level per
   scenario-day, so every scenario is a coherent day.
2. `rollout` replays the planned battery schedule through the battery model with clipping (power limits, SoC
   limits, comfort reserve) for every scenario at once, (N scenarios x S sites x T steps) in numpy. The baseline
   self-consumption rule runs through the same kernel (request = pv - load, floor = soc_min only, as
   `BatterySpec.step`), so the two are compared under the same weather and load.
3. deliverable_t = sum over pool sites of (baseline export - planned export) = export reduction vs the rule.
   nominal_t   = deliverable on the P50 forecast (what the plan says it moves), >= 0
   available_t = min(nominal_t, Q50(deliverable_t))
   committed_t = min(available_t, (1 - h) * Q10(deliverable_t)),  h = telemetry-loss haircut
   so nominal >= available >= committed >= 0 always (asserted). Sites outside the pool run the rule in both
   worlds and cancel, so only scheduled sites enter.
"""

from __future__ import annotations

from dataclasses import dataclass, field, replace
from typing import Any, Mapping, NamedTuple, Optional, Sequence, Union

import numpy as np
from scipy.special import ndtri

from sentinel.contracts import DAY_STEPS, DT_H, BatterySpec, Commitment, FeederPlan, FeederSpec, Prediction
from sentinel.eval.uncertainty.conformal import ConformalCalibration, members_of, quantile_at, sort_quantiles
from sentinel.kernels import battery_rollout

N_SCENARIOS = 200
DEFAULT_HAIRCUT = 0.05
LEVEL_CLIP = 1e-3  # u in [0.001, 0.999]: |z| <= 3.09
MIN_COMMIT_KW = 1e-3

Triple = tuple[np.ndarray, np.ndarray, np.ndarray]
ForecastIn = Union[Prediction, Triple, Sequence[Triple]]


# --- battery kernel ------------------------------------------------------------------------------------------
class BatteryParams(NamedTuple):
    """Per-site battery arrays, shape (S,). Energies kWh, powers kW, efficiencies one-way.

    e_min_kwh is the floor the rollout clips to: max(soc_min·E, comfort reserve) for a planned schedule,
    soc_min·E for the self-consumption rule (see `battery_params`).
    """

    capacity_kwh: np.ndarray
    p_max_kw: np.ndarray
    eta_charge: np.ndarray
    eta_discharge: np.ndarray
    e_min_kwh: np.ndarray
    e_max_kwh: np.ndarray


def battery_params(specs: Sequence[Optional[BatterySpec]], respect_reserve: bool = True) -> BatteryParams:
    """Stack BatterySpecs (None = no battery) into arrays. respect_reserve=False gives the physical floor only."""
    rows = []
    for b in specs:
        if b is None or b.capacity_kwh == 0:
            rows.append((0.0, 0.0, 1.0, 1.0, 0.0, 0.0))
            continue
        floor = b.e_min if respect_reserve else b.soc_min * b.capacity_kwh
        rows.append((b.capacity_kwh, b.p_max_kw, b.eta_charge, b.eta_discharge, floor, b.e_max))
    a = np.array(rows, dtype=float).reshape(-1, 6)
    return BatteryParams(*(a[:, i] for i in range(6)))


_battery_params = battery_params  # `rollout` takes an argument with the same name


def rollout(schedule: np.ndarray, soc0: np.ndarray, battery_params: Union[BatteryParams, Sequence[BatterySpec]],
            pv: np.ndarray, load: np.ndarray, dt_h: float = DT_H) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Replay a requested battery power through the battery model, vectorised over scenarios.

    Args (all kW unless noted; the last two axes are (S sites, T steps), leading axes broadcast, e.g. N scenarios):
      schedule  requested battery power (+ charge, - discharge), (S, T) or (N, S, T)
      soc0      initial state of charge as a FRACTION of capacity, (S,) or (N, S)
      battery_params  BatteryParams (S,) arrays, or a sequence of BatterySpec (planner floor incl. reserve)
      pv, load  (S, T) or (N, S, T)
    Returns (soc, batt_power, net_export), each of the broadcast shape (..., S, T):
      soc         fraction of capacity at the END of step t (0 for sites without a battery)
      batt_power  applied kW after clipping to ±p_max, the energy ceiling e_max and the floor e_min
      net_export  pv - load - batt_power (+ export, - import)
    Per step: p = clip(req, -P, P); p <= (e_max - e)/(eta_c·dt); p >= -(e - e_min)·eta_d/dt;
    e += eta_c·p·dt if p >= 0 else p·dt/eta_d. A battery that starts below e_min may charge but not discharge.
    Matches `BatterySpec.step` / `self_consumption_step` exactly when e0 >= e_min.
    """
    bp = battery_params if isinstance(battery_params, BatteryParams) else _battery_params(
        battery_params, respect_reserve=True)
    req = np.asarray(schedule, dtype=float)
    pv = np.asarray(pv, dtype=float)
    load = np.asarray(load, dtype=float)
    shape = np.broadcast_shapes(req.shape, pv.shape, load.shape)
    lead = shape[:-1]  # (..., S)
    t_steps = shape[-1]
    req = np.broadcast_to(req, shape)
    cap, pmax, eta_c, eta_d, e_lo, e_hi = (np.broadcast_to(x, lead) for x in bp)
    e = np.broadcast_to(np.asarray(soc0, dtype=float), lead) * cap
    e = np.array(e, dtype=float)  # writable copy
    inv_cap = np.where(cap > 0, 1.0 / np.where(cap > 0, cap, 1.0), 0.0)

    soc = np.empty(shape)
    batt = np.empty(shape)
    for t in range(t_steps):
        p = np.clip(req[..., t], -pmax, pmax)
        room_c = np.maximum(0.0, (e_hi - e) / (eta_c * dt_h))
        room_d = np.maximum(0.0, (e - e_lo) * eta_d / dt_h)
        p = np.where(p >= 0, np.minimum(p, room_c), np.maximum(p, -room_d))
        e_next = e + np.where(p >= 0, eta_c * p * dt_h, p * dt_h / eta_d)
        e = np.clip(e_next, np.minimum(e_lo, e), e_hi)
        batt[..., t] = p
        soc[..., t] = e * inv_cap
    net = np.broadcast_to(pv, shape) - np.broadcast_to(load, shape) - batt
    return soc, batt, net


def rollout_baseline(soc0: np.ndarray, specs: Union[BatteryParams, Sequence[Optional[BatterySpec]]],
                     pv: np.ndarray, load: np.ndarray, dt_h: float = DT_H):
    """The self-consumption rule through the same kernel (floor = soc_min, no reserve, as BatterySpec.step)."""
    bp = specs if isinstance(specs, BatteryParams) else battery_params(specs, respect_reserve=False)
    pv = np.asarray(pv, dtype=float)
    load = np.asarray(load, dtype=float)
    return rollout(pv - load, soc0, bp, pv, load, dt_h)


# --- scenarios -----------------------------------------------------------------------------------------------
@dataclass
class Scenarios:
    """N coherent feeder-days. pv/load (N, S, T) kW, site order = site_ids. pv_p50/load_p50 (S, T) = member-mean
    calibrated q50 (used for `nominal`). batteries / soc0 (fraction) travel with the scenarios so that
    `commit(feeder, schedule, scenarios)` keeps the contract's three-argument form."""

    site_ids: list[str]
    pv: np.ndarray
    load: np.ndarray
    member: np.ndarray  # (N,) ensemble member index
    pv_level: np.ndarray  # (N,) shared PV quantile level u
    load_level: np.ndarray  # (N, S) per-site load levels
    pv_p50: np.ndarray
    load_p50: np.ndarray
    feeder_id: str = ""
    t0: str = ""
    batteries: dict[str, Optional[BatterySpec]] = field(default_factory=dict)
    soc0: dict[str, float] = field(default_factory=dict)

    @property
    def n(self) -> int:
        return int(self.pv.shape[0])

    def index(self, site_ids: Sequence[str]) -> np.ndarray:
        pos = {s: i for i, s in enumerate(self.site_ids)}
        missing = [s for s in site_ids if s not in pos]
        if missing:
            raise KeyError(f"sites not in scenarios: {missing[:5]}")
        return np.array([pos[s] for s in site_ids], dtype=int)


def _members(f: ForecastIn) -> list[Triple]:
    if isinstance(f, Prediction):
        return members_of(f)
    if isinstance(f, tuple) and len(f) == 3 and not isinstance(f[0], tuple):
        return [tuple(np.asarray(a, dtype=float) for a in f)]  # type: ignore[list-item]
    return [tuple(np.asarray(a, dtype=float) for a in m) for m in f]  # type: ignore[union-attr]


def _stack(fc: Mapping[str, ForecastIn], site_ids: Sequence[str], cal: Optional[ConformalCalibration],
           cap: Optional[Mapping[str, float]]) -> np.ndarray:
    """(M, 3, S, T): members x (q10, q50, q90) x sites x steps; sites with fewer members repeat them (modulo)."""
    per_site = []
    for s in site_ids:
        ms = _members(fc[s])
        c = None if cap is None or s not in cap else float(cap[s])
        if cal is not None:
            ms = cal.apply_members(ms, floor=0.0, cap=c)
        else:
            ms = [sort_quantiles(*m, floor=0.0, cap=c) for m in ms]
        per_site.append(ms)
    m_max = max(len(ms) for ms in per_site)
    t = per_site[0][0][0].shape[-1]
    out = np.empty((m_max, 3, len(site_ids), t))
    for j, ms in enumerate(per_site):
        for k in range(m_max):
            out[k, :, j, :] = np.stack(ms[k % len(ms)])
    return out


def make_scenarios(pv: Mapping[str, ForecastIn], load: Mapping[str, ForecastIn], n: int = N_SCENARIOS,
                   seed: int = 0, *, site_ids: Optional[Sequence[str]] = None,
                   pv_calibration: Optional[ConformalCalibration] = None,
                   load_calibration: Optional[ConformalCalibration] = None,
                   pv_cap: Optional[Mapping[str, float]] = None,
                   batteries: Optional[Mapping[str, Optional[BatterySpec]]] = None,
                   soc0: Optional[Mapping[str, float]] = None,
                   feeder_id: str = "", t0: str = "") -> Scenarios:
    """~200 scenarios for one feeder-day.

    pv / load: {site_id: Prediction | (q10, q50, q90) | [(q10, q50, q90), ...members]}. Calibrations (from
    `ConformalCalibration.fit`) are applied per member before sampling. pv_cap: {site_id: kW} upper clip (e.g.
    kWp). Sampling: member ~ U{0..M-1}; PV level u ~ Latin hypercube over scenarios, shared by all sites; load
    levels iid U(0,1) per (scenario, site); u clipped to [0.001, 0.999].
    """
    site_ids = list(site_ids if site_ids is not None else pv.keys())
    rng = np.random.default_rng(seed)
    pv_q = _stack(pv, site_ids, pv_calibration, pv_cap)  # (M, 3, S, T)
    ld_q = _stack(load, site_ids, load_calibration, None)
    m_count = max(pv_q.shape[0], ld_q.shape[0])
    member = rng.integers(0, m_count, size=n)
    pv_level = (rng.permutation(n) + rng.random(n)) / n
    load_level = rng.random((n, len(site_ids)))
    pv_level = np.clip(pv_level, LEVEL_CLIP, 1 - LEVEL_CLIP)
    load_level = np.clip(load_level, LEVEL_CLIP, 1 - LEVEL_CLIP)

    def draw(q: np.ndarray, z: np.ndarray, cap: Optional[np.ndarray]) -> np.ndarray:
        g = q[member % q.shape[0]]  # (N, 3, S, T)
        v = quantile_at(g[:, 0], g[:, 1], g[:, 2], z)
        v = np.maximum(v, 0.0)
        return np.minimum(v, cap) if cap is not None else v

    cap_arr = None
    if pv_cap is not None:
        cap_arr = np.array([float(pv_cap.get(s, np.inf)) for s in site_ids])[None, :, None]
    pv_s = draw(pv_q, ndtri(pv_level)[:, None, None], cap_arr)
    ld_s = draw(ld_q, ndtri(load_level)[:, :, None], None)
    return Scenarios(
        site_ids=site_ids, pv=pv_s, load=ld_s, member=member, pv_level=pv_level, load_level=load_level,
        pv_p50=pv_q[:, 1].mean(axis=0), load_p50=ld_q[:, 1].mean(axis=0), feeder_id=feeder_id, t0=t0,
        batteries=dict(batteries or {}), soc0=dict(soc0 or {}),
    )


# --- commitment ----------------------------------------------------------------------------------------------
def _schedule_matrix(schedule: Union[Mapping[str, Sequence[float]], FeederPlan]) -> tuple[list[str], np.ndarray]:
    if isinstance(schedule, FeederPlan):
        schedule = schedule.schedule
    ids = list(schedule.keys())
    if not ids:
        return [], np.zeros((0, DAY_STEPS))
    return ids, np.array([np.asarray(schedule[s], dtype=float) for s in ids])


def _direction(direction: Any, t: int) -> np.ndarray:
    if direction is None or (isinstance(direction, str) and direction == "export"):
        return np.ones(t)
    d = np.asarray(direction, dtype=float)
    if d.shape != (t,) or not np.isin(d, (-1.0, 0.0, 1.0)).all():
        raise ValueError("direction must be 'export' or an array of -1/0/+1 per step")
    return d


def _net_export(req: np.ndarray, soc0: np.ndarray, bp: BatteryParams, pv: np.ndarray,
                load: np.ndarray) -> np.ndarray:
    """Third output of `rollout` (pv - load - batt) via the dispatched K1 kernel (Rust if built, else `rollout`).

    Bit-identical to `rollout(req, soc0, bp, pv, load)[2]`. A request shared by every scenario ((S, T) with soc0
    (S,)) is rolled out once and broadcast: the battery does not see pv/load, so each scenario would repeat it.
    Not dispatched inside `rollout` itself: the kernel's Python reference calls `rollout`.
    """
    req = np.asarray(req, dtype=float)
    pv = np.asarray(pv, dtype=float)
    load = np.asarray(load, dtype=float)
    s0 = np.asarray(soc0, dtype=float)
    shape = np.broadcast_shapes(req.shape, pv.shape, load.shape)
    if req.ndim == 2 and s0.ndim <= 1:
        _, batt = battery_rollout(np.broadcast_to(req, shape[-2:]), np.broadcast_to(s0, shape[-2:-1]), *bp,
                                  dt_h=DT_H)
    else:
        r3 = np.broadcast_to(req, shape).reshape((-1,) + shape[-2:])
        _, batt = battery_rollout(r3, np.broadcast_to(s0, shape[:-1]).reshape(r3.shape[:-1]), *bp, dt_h=DT_H)
        batt = batt.reshape(shape)
    return np.broadcast_to(pv, shape) - np.broadcast_to(load, shape) - np.broadcast_to(batt, shape)


def _deliverable(sched: np.ndarray, soc0: np.ndarray, specs: Sequence[Optional[BatterySpec]],
                 pv: np.ndarray, load: np.ndarray) -> np.ndarray:
    """Sum over sites of baseline export - planned export, shape pv.shape[:-2] + (T,).

    Same result as `rollout` / `rollout_baseline`, through the K1 kernel (sentinel.kernels.battery_rollout)."""
    pv = np.asarray(pv, dtype=float)
    load = np.asarray(load, dtype=float)
    x_plan = _net_export(sched, soc0, battery_params(specs, respect_reserve=True), pv, load)
    x_base = _net_export(pv - load, soc0, battery_params(specs, respect_reserve=False), pv, load)
    return (x_base - x_plan).sum(axis=-2)


def commit(feeder: Union[FeederSpec, FeederPlan, str], schedule: Union[Mapping[str, Sequence[float]], FeederPlan,
                                                                     None],
           scenarios: Scenarios, haircut: float = DEFAULT_HAIRCUT, *,
           batteries: Optional[Mapping[str, Optional[BatterySpec]]] = None,
           soc0: Optional[Mapping[str, float]] = None, quantile: float = 0.10, direction: Any = None,
           t0: Optional[str] = None, min_commit_kw: float = MIN_COMMIT_KW) -> Commitment:
    """Feeder commitment from a battery schedule and scenarios (plan §3.5).

    feeder: FeederSpec, FeederPlan or feeder id. schedule: {site_id: batt_kw[T]} or a FeederPlan (None → the
    FeederPlan passed as `feeder`). batteries / soc0 (fraction of capacity) default to the ones carried by
    `scenarios`. direction: None/"export" (deliverable = export reduction) or a per-step ±1/0 array (−1 = the
    plan's import reduction in the evening). Returns Commitment with extra = {direction, q10, q50, mean
    deliverable, n_scenarios, quantile, sites}.
    """
    if schedule is None:
        if not isinstance(feeder, FeederPlan):
            raise ValueError("schedule is None and feeder is not a FeederPlan")
        schedule = feeder
    feeder_id = feeder if isinstance(feeder, str) else feeder.feeder_id
    t0 = t0 or getattr(feeder, "t0", None) or (schedule.t0 if isinstance(schedule, FeederPlan) else "") or \
        scenarios.t0
    if not 0.0 <= haircut < 1.0:
        raise ValueError("haircut must be in [0, 1)")
    batteries = dict(batteries if batteries is not None else scenarios.batteries)
    soc0_map = dict(soc0 if soc0 is not None else scenarios.soc0)
    ids, sched = _schedule_matrix(schedule)
    t_steps = scenarios.pv.shape[-1]
    if not ids:
        z = [0.0] * t_steps
        return Commitment(feeder_id=feeder_id, t0=t0, nominal=z, available=z, committed=z, haircut=haircut,
                          horizon=t_steps, extra={"n_scenarios": scenarios.n, "sites": []})
    if sched.shape[1] != t_steps:
        raise ValueError(f"schedule has {sched.shape[1]} steps, scenarios {t_steps}")
    idx = scenarios.index(ids)
    specs = [batteries.get(s) for s in ids]
    s0 = np.array([float(soc0_map.get(s, 0.5)) for s in ids])
    dirv = _direction(direction, t_steps)

    d = dirv * _deliverable(sched, s0, specs, scenarios.pv[:, idx], scenarios.load[:, idx])  # (N, T)
    d_nom = dirv * _deliverable(sched, s0, specs, scenarios.pv_p50[idx], scenarios.load_p50[idx])  # (T,)
    q_lo = np.quantile(d, quantile, axis=0)
    q50 = np.quantile(d, 0.5, axis=0)
    nominal = np.maximum(0.0, d_nom)
    available = np.minimum(nominal, np.maximum(0.0, q50))
    committed = np.minimum(available, (1.0 - haircut) * np.maximum(0.0, q_lo))
    committed = np.where(committed >= min_commit_kw, committed, 0.0)
    assert np.all(nominal + 1e-9 >= available) and np.all(available + 1e-9 >= committed) and np.all(committed >= 0)
    return Commitment(
        feeder_id=feeder_id, t0=t0, nominal=nominal.tolist(), available=available.tolist(),
        committed=committed.tolist(), haircut=float(haircut), horizon=t_steps,
        extra={"direction": dirv.tolist(), "q10_deliverable": q_lo.tolist(), "q50_deliverable": q50.tolist(),
               "mean_deliverable": d.mean(axis=0).tolist(), "n_scenarios": scenarios.n, "quantile": quantile,
               "sites": ids},
    )


# --- realised + scoring --------------------------------------------------------------------------------------
def realised_deliverable(schedule: Union[Mapping[str, Sequence[float]], FeederPlan],
                         pv_actual: Mapping[str, Sequence[float]], load_actual: Mapping[str, Sequence[float]],
                         batteries: Mapping[str, Optional[BatterySpec]], soc0: Mapping[str, float],
                         direction: Any = None) -> np.ndarray:
    """Realised export reduction (T,) from measured pv/load: the schedule as executed (with clipping) vs the
    rule's counterfactual under the same actuals. pv_actual/load_actual may also be (N, T) arrays per site to
    score N actual days at once (returns (N, T))."""
    ids, sched = _schedule_matrix(schedule)
    if not ids:
        return np.zeros(DAY_STEPS)
    pv = np.stack([np.asarray(pv_actual[s], dtype=float) for s in ids], axis=-2)
    ld = np.stack([np.asarray(load_actual[s], dtype=float) for s in ids], axis=-2)
    specs = [batteries.get(s) for s in ids]
    s0 = np.array([float(soc0.get(s, 0.5)) for s in ids])
    return _direction(direction, sched.shape[1]) * _deliverable(sched, s0, specs, pv, ld)


def realise(c: Commitment, schedule: Union[Mapping[str, Sequence[float]], FeederPlan],
            pv_actual: Mapping[str, Sequence[float]], load_actual: Mapping[str, Sequence[float]],
            batteries: Mapping[str, Optional[BatterySpec]], soc0: Mapping[str, float]) -> Commitment:
    """Copy of the commitment with `realised` filled from actuals (same direction as when committed)."""
    r = realised_deliverable(schedule, pv_actual, load_actual, batteries, soc0, c.extra.get("direction"))
    return replace(c, realised=[float(x) for x in r], extra=dict(c.extra))


def hit_stats(commitments: Union[Commitment, Sequence[Commitment], None] = None, *,
              committed: Optional[np.ndarray] = None, realised: Optional[np.ndarray] = None) -> dict[str, float]:
    """P10 reliability over feeder-steps with committed > 0: p10_hit_rate (realised >= committed), p10_miss_kw
    (mean shortfall on missed steps, 0 if none), p10_miss_kw_max, p10_shortfall_share (missed kWh / committed
    kWh), n_steps. Pass Commitments with `realised`, or aligned committed / realised arrays."""
    if commitments is not None:
        cs = [commitments] if isinstance(commitments, Commitment) else list(commitments)
        cm, rl = [], []
        for c in cs:
            if c.realised is None:
                continue
            for r, k in zip(c.realised, c.committed):
                if r is not None:
                    cm.append(k)
                    rl.append(r)
        committed, realised = np.array(cm, dtype=float), np.array(rl, dtype=float)
    if committed is None or realised is None:
        raise ValueError("pass commitments or committed + realised")
    c = np.asarray(committed, dtype=float)
    r = np.broadcast_to(np.asarray(realised, dtype=float), np.broadcast_shapes(c.shape, np.shape(realised)))
    c = np.broadcast_to(c, r.shape)
    live = c > 0
    if not live.any():
        return {"p10_hit_rate": float("nan"), "p10_miss_kw": 0.0, "p10_miss_kw_max": 0.0,
                "p10_shortfall_share": 0.0, "n_steps": 0}
    cc, rr = c[live], r[live]
    hit = rr + 1e-6 >= cc
    short = np.maximum(0.0, cc - rr)
    return {
        "p10_hit_rate": float(hit.mean()),
        "p10_miss_kw": float(short[~hit].mean()) if (~hit).any() else 0.0,
        "p10_miss_kw_max": float(short.max()),
        "p10_shortfall_share": float(short.sum() / cc.sum()),
        "n_steps": int(live.sum()),
    }
