"""Baseline policies + the site / plan types shared by the optimiser (card #18, plan §3.5, R3).

Policies (all return a `DetailedPlan`, i.e. a contract `FeederPlan` with diagnostics):

- `self_consumption_plan(...)`: the baseline. Every site runs `contracts.self_consumption_step` (one definition,
  shared with the simulator #11 and with the non-pool sites inside the LP).
- `threshold_plan(..., alpha)`: per-home export-threshold rule. Each home only charges the part of its PV surplus
  above `theta_i = alpha * max_t(P50 PV - P50 load)` (so its battery still has room at the midday peak) and
  discharges to cover load like the baseline. `tune_threshold(days)` picks `alpha` on validation days.

Shared building blocks (used by `lp.py` and `aggregate.py`):

- `Site(site_id, battery, e0_kwh)`, built from a `sites` row with `site_from_row` (data.md battery columns).
- `RestOfFeeder(net_kw, worst_kw)`: forecast net export of the sites the planner does not control (R3).
  `rest_of_feeder_from_forecasts(sites, forecasts)` simulates the rule on P50 and on worst-case inputs.
- `simulate_rule(battery, e0, pv, load)` / `simulate_fleet(...)`: full-day rule simulation.

Conventions (same for every policy so plans are comparable): kW per 15-min step, + export; battery + charge.
`export_curve` = planned feeder net export on P50 forecasts; `slack_t = max(0, worst_t - G)` where `worst_t` is the
worst plausible export (P90 PV, P10 load; the rule reacts to that PV, an LP schedule does not).
Prices are lei/kWh (the `prices` table is lei/MWh: divide by 1000).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Callable, Mapping, Optional, Sequence, Union

import numpy as np

from sentinel.contracts import DT_H, AssetStatus, BatterySpec, FeederPlan, FeederSpec, Prediction, Status
from sentinel.contracts import self_consumption_step

ArrayLike = Union[Sequence[float], np.ndarray]
Forecasts = Mapping[str, Mapping[str, Prediction]]  # {site_id: {"pv": Prediction, "load": Prediction}}

DEFAULT_BETA = 0.5  # export remuneration ratio (plan §3.5, "to verify")


# --------------------------------------------------------------------------------------------------------------
# Shared types
# --------------------------------------------------------------------------------------------------------------
@dataclass
class Site:
    """One prosumer site as the planner sees it. `battery=None` (or capacity 0) = no battery.

    `e0_kwh`: battery energy at t0 (from telemetry soc * capacity). None -> `battery.e_min` (the comfort reserve).
    """

    site_id: str
    battery: Optional[BatterySpec] = None
    e0_kwh: Optional[float] = None

    @property
    def has_battery(self) -> bool:
        return self.battery is not None and self.battery.capacity_kwh > 0 and self.battery.p_max_kw > 0

    def initial_energy(self) -> float:
        """e0 clamped into the physical range [soc_min * E, e_max]."""
        if not self.has_battery:
            return 0.0
        b = self.battery
        e0 = b.e_min if self.e0_kwh is None else float(self.e0_kwh)
        return float(min(b.e_max, max(b.soc_min * b.capacity_kwh, e0)))


def battery_from_row(row: Mapping[str, Any]) -> Optional[BatterySpec]:
    """`sites` row -> BatterySpec (data.md: battery_kwh, battery_kw, eta_*, soc_min/max, comfort_reserve_kwh)."""
    cap = float(row.get("battery_kwh") or 0.0)
    if cap <= 0:
        return None
    kw = {
        "capacity_kwh": cap,
        "p_max_kw": float(row.get("battery_kw") or 0.0),
        "comfort_reserve_kwh": float(row.get("comfort_reserve_kwh") or 0.0),
    }
    for k in ("eta_charge", "eta_discharge", "soc_min", "soc_max"):
        if row.get(k) is not None:
            kw[k] = float(row[k])
    return BatterySpec(**kw)


def site_from_row(row: Mapping[str, Any]) -> Site:
    """`sites` row (+ optional `soc` 0..1 or `e0_kwh` from the latest telemetry) -> Site."""
    b = battery_from_row(row)
    e0 = row.get("e0_kwh")
    if e0 is None and row.get("soc") is not None and b is not None:
        e0 = float(row["soc"]) * b.capacity_kwh
    return Site(site_id=str(row["site_id"]), battery=b, e0_kwh=None if e0 is None else float(e0))


def as_site(x: Union[Site, Mapping[str, Any]]) -> Site:
    return x if isinstance(x, Site) else site_from_row(x)


@dataclass
class RestOfFeeder:
    """Forecast net export (kW, + export) of the feeder's sites outside the planner's control, which keep running
    the self-consumption rule (R3). `net_kw` on P50 inputs (-> export curve, evening import M_rest = -net);
    `worst_kw` on worst plausible inputs P90 PV / P10 load (-> X_rest in the export limit). worst defaults to net.
    """

    net_kw: list[float]
    worst_kw: Optional[list[float]] = None

    def __post_init__(self) -> None:
        self.net_kw = [float(v) for v in self.net_kw]
        self.worst_kw = list(self.net_kw) if self.worst_kw is None else [float(v) for v in self.worst_kw]
        if len(self.worst_kw) != len(self.net_kw):
            raise ValueError("net_kw and worst_kw lengths differ")

    def __add__(self, other: "RestOfFeeder") -> "RestOfFeeder":
        return RestOfFeeder(
            list(np.add(self.net_kw, other.net_kw)), list(np.add(self.worst_kw, other.worst_kw))
        )


RestLike = Union[None, RestOfFeeder, ArrayLike]


def as_rest(rest: RestLike, horizon: int) -> RestOfFeeder:
    """None -> zeros; a plain sequence -> used for both net and worst; RestOfFeeder as is."""
    if rest is None:
        return RestOfFeeder([0.0] * horizon)
    if not isinstance(rest, RestOfFeeder):
        rest = RestOfFeeder(list(np.asarray(rest, dtype=float)))
    if len(rest.net_kw) != horizon:
        raise ValueError(f"rest_of_feeder has {len(rest.net_kw)} steps, horizon is {horizon}")
    return rest


@dataclass
class DetailedPlan(FeederPlan):
    """A contract `FeederPlan` plus diagnostics (all optional for consumers; `to_dict()` includes them).

    method: "lp" | "aggregate" | "rule". soc_kwh[site] has horizon + 1 values (energy at step boundaries).
    worst_export_curve: worst plausible feeder export (the curve `slack` is measured on). peak_import_kw: max P50
    feeder import in the evening window. simultaneous_steps: (site, step) pairs where the LP charged and
    discharged at once (> tol); the returned schedule is always the netted, battery-model-clipped one.
    folded_sites: sites passed as eligible that run the rule instead (not in pool / battery fault / no battery).
    """

    method: str = "rule"
    soc_kwh: dict[str, list[float]] = field(default_factory=dict)
    worst_export_curve: list[float] = field(default_factory=list)
    peak_import_kw: float = 0.0
    simultaneous_steps: int = 0
    folded_sites: list[str] = field(default_factory=list)
    n_sites: int = 0
    solve_time_s: float = 0.0  # solver time (HiGHS)
    wall_time_s: float = 0.0  # build + solve + post-processing
    extra: dict[str, Any] = field(default_factory=dict)

    @property
    def max_slack_kw(self) -> float:
        return max(self.slack) if self.slack else 0.0

    @property
    def over_limit_steps(self) -> int:
        return sum(s > 1e-6 for s in self.slack)


# --------------------------------------------------------------------------------------------------------------
# Forecast helpers
# --------------------------------------------------------------------------------------------------------------
def site_quantiles(forecasts: Forecasts, site_id: str, pv_factor: float = 1.0) -> dict[str, np.ndarray]:
    """{pv10, pv50, pv90, l10, l50, l90} arrays for one site; PV scaled by `pv_factor` (derated sites)."""
    try:
        f = forecasts[site_id]
        pv, ld = f["pv"], f["load"]
    except KeyError as exc:
        raise KeyError(f"forecasts[{site_id!r}] needs 'pv' and 'load' Predictions") from exc
    out = {
        "pv10": np.asarray(pv.q10, float) * pv_factor,
        "pv50": np.asarray(pv.q50, float) * pv_factor,
        "pv90": np.asarray(pv.q90, float) * pv_factor,
        "l10": np.asarray(ld.q10, float),
        "l50": np.asarray(ld.q50, float),
        "l90": np.asarray(ld.q90, float),
    }
    if len(out["pv50"]) != len(out["l50"]):
        raise ValueError(f"{site_id}: pv and load horizons differ")
    return out


def pv_factor(status: Optional[AssetStatus]) -> float:
    """Derated (PV fault) -> PV forecast scaled by headroom_factor; otherwise 1."""
    if status is not None and status.status == Status.DERATED.value:
        return status.headroom_factor
    return 1.0


def is_controllable(site: Site, status: Optional[AssetStatus]) -> bool:
    """The planner may schedule this battery: in pool (eligible / derated), battery_ok, and a battery exists."""
    if not site.has_battery:
        return False
    if status is None:
        return True
    return status.in_pool and status.battery_ok


def horizon_of(forecasts: Forecasts, site_ids: Sequence[str]) -> Optional[int]:
    for sid in site_ids:
        if sid in forecasts:
            return len(forecasts[sid]["pv"].q50)
    return None


def t0_of(forecasts: Forecasts, site_ids: Sequence[str]) -> Optional[str]:
    for sid in site_ids:
        if sid in forecasts:
            return forecasts[sid]["pv"].t0
    return None


def price_vector(prices: Union[float, ArrayLike, None], horizon: int) -> np.ndarray:
    """Scalar, per-step [horizon] or hourly [horizon / 4] prices (lei/kWh) -> per-step array."""
    if prices is None:
        prices = 0.6
    p = np.atleast_1d(np.asarray(prices, dtype=float))
    if p.size == 1:
        return np.full(horizon, float(p[0]))
    if p.size == horizon:
        return p
    if p.size * 4 == horizon:
        return np.repeat(p, 4)
    raise ValueError(f"prices has {p.size} values; expected 1, {horizon} or {horizon // 4}")


def feeder_id_and_g(feeder: Union[FeederSpec, str], G: Optional[float]) -> tuple[str, float]:
    if isinstance(feeder, FeederSpec):
        return feeder.feeder_id, float(feeder.g_kw if G is None else G)
    if G is None:
        raise ValueError("G is required when feeder is a plain id")
    return str(feeder), float(G)


# --------------------------------------------------------------------------------------------------------------
# Rule simulation
# --------------------------------------------------------------------------------------------------------------
StepFn = Callable[[Optional[BatterySpec], float, float, float], tuple[float, float, float]]


def threshold_step(
    battery: Optional[BatterySpec], e_kwh: float, pv_kw: float, load_kw: float, threshold_kw: float,
    dt_h: float = DT_H,
) -> tuple[float, float, float]:
    """Export-threshold rule: charge only the surplus above `threshold_kw`; discharge to cover load (as the
    baseline); in between, idle. threshold 0 = `self_consumption_step`. Returns (batt_kw, e_next, grid_kw)."""
    surplus = pv_kw - load_kw
    if battery is None or battery.capacity_kwh == 0:
        return 0.0, e_kwh, surplus
    if surplus > threshold_kw:
        req = surplus - threshold_kw
    elif surplus < 0:
        req = surplus
    else:
        req = 0.0
    batt_kw, e_next = battery.step(e_kwh, req, dt_h)
    return batt_kw, e_next, surplus - batt_kw


def simulate_rule(
    battery: Optional[BatterySpec], e0_kwh: float, pv_kw: ArrayLike, load_kw: ArrayLike,
    step: Optional[StepFn] = None, dt_h: float = DT_H,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Run a per-step rule over a day. Returns (batt_kw[T], e_kwh[T+1], grid_kw[T]); default rule = baseline."""
    step = step or (lambda b, e, pv, ld: self_consumption_step(b, e, pv, ld, dt_h))
    pv = np.asarray(pv_kw, float)
    ld = np.asarray(load_kw, float)
    T = len(pv)
    batt = np.zeros(T)
    grid = np.zeros(T)
    e = np.zeros(T + 1)
    e[0] = e0_kwh
    for t in range(T):
        batt[t], e[t + 1], grid[t] = step(battery, e[t], float(pv[t]), float(ld[t]))
    return batt, e, grid


@dataclass
class FleetTrace:
    """Per-site rule simulation over a day. net_kw = feeder sum of grid_kw (+ export)."""

    batt_kw: dict[str, np.ndarray]
    soc_kwh: dict[str, np.ndarray]
    grid_kw: dict[str, np.ndarray]
    net_kw: np.ndarray


def simulate_fleet(
    sites: Sequence[Union[Site, Mapping[str, Any]]],
    pv: Mapping[str, ArrayLike],
    load: Mapping[str, ArrayLike],
    *,
    thresholds: Optional[Mapping[str, float]] = None,
    dt_h: float = DT_H,
) -> FleetTrace:
    """Full-day fleet simulation of the baseline rule (thresholds=None) or the per-home threshold rule."""
    sites = [as_site(s) for s in sites]
    batt, soc, grid = {}, {}, {}
    T = len(next(iter(pv.values()))) if pv else 0
    net = np.zeros(T)
    for s in sites:
        th = None if thresholds is None else float(thresholds.get(s.site_id, 0.0))
        step = None
        if th is not None:
            step = lambda b, e, p, l, th=th: threshold_step(b, e, p, l, th, dt_h)  # noqa: E731
        b, e, g = simulate_rule(s.battery if s.has_battery else None, s.initial_energy(), pv[s.site_id],
                                load[s.site_id], step, dt_h)
        batt[s.site_id], soc[s.site_id], grid[s.site_id] = b, e, g
        net += g
    return FleetTrace(batt, soc, grid, net)


def rest_of_feeder_from_forecasts(
    sites: Sequence[Union[Site, Mapping[str, Any]]],
    forecasts: Forecasts,
    statuses: Optional[Mapping[str, AssetStatus]] = None,
    horizon: Optional[int] = None,
) -> RestOfFeeder:
    """X_rest / M_rest for non-eligible sites: the baseline rule simulated on P50 (net) and on P90 PV / P10 load
    (worst). PV of derated sites is scaled by headroom_factor."""
    sites = [as_site(s) for s in sites]
    statuses = statuses or {}
    T = horizon or horizon_of(forecasts, [s.site_id for s in sites]) or 0
    net = np.zeros(T)
    worst = np.zeros(T)
    for s in sites:
        q = site_quantiles(forecasts, s.site_id, pv_factor(statuses.get(s.site_id)))
        b = s.battery if s.has_battery else None
        net += simulate_rule(b, s.initial_energy(), q["pv50"], q["l50"])[2]
        worst += simulate_rule(b, s.initial_energy(), q["pv90"], q["l10"])[2]
    return RestOfFeeder(list(net), list(worst))


# --------------------------------------------------------------------------------------------------------------
# Policies as feeder plans
# --------------------------------------------------------------------------------------------------------------
def _rule_plan(
    policy: str, feeder, sites, rest_of_feeder, forecasts, G, *, statuses, thresholds, t0,
) -> DetailedPlan:
    fid, g = feeder_id_and_g(feeder, G)
    sites = [as_site(s) for s in sites]
    statuses = statuses or {}
    ids = [s.site_id for s in sites]
    T = horizon_of(forecasts, ids)
    if T is None:
        T = len(rest_of_feeder.net_kw) if isinstance(rest_of_feeder, RestOfFeeder) else len(rest_of_feeder or [])
    rest = as_rest(rest_of_feeder, T)
    q = {s.site_id: site_quantiles(forecasts, s.site_id, pv_factor(statuses.get(s.site_id))) for s in sites}
    th = thresholds(q) if callable(thresholds) else thresholds
    p50 = simulate_fleet(sites, {k: v["pv50"] for k, v in q.items()}, {k: v["l50"] for k, v in q.items()},
                         thresholds=th)
    worst = simulate_fleet(sites, {k: v["pv90"] for k, v in q.items()}, {k: v["l10"] for k, v in q.items()},
                           thresholds=th)
    export = p50.net_kw + np.asarray(rest.net_kw)
    worst_exp = worst.net_kw + np.asarray(rest.worst_kw)
    slack = np.maximum(0.0, worst_exp - g)
    t0 = t0 or t0_of(forecasts, ids) or ""
    from .lp import peak_mask_for  # local import: lp imports this module

    mask = peak_mask_for(t0, T) if t0 else np.zeros(T, bool)
    peak = float(max(0.0, (-export[mask]).max())) if mask.any() else 0.0
    return DetailedPlan(
        feeder_id=fid, t0=t0, policy=policy, g_kw=g,
        schedule={s.site_id: list(p50.batt_kw[s.site_id]) for s in sites if s.has_battery},
        export_curve=list(export), slack=list(slack), horizon=T, solver_status="rule", objective=None,
        method="rule", soc_kwh={s.site_id: list(p50.soc_kwh[s.site_id]) for s in sites if s.has_battery},
        worst_export_curve=list(worst_exp), peak_import_kw=peak, n_sites=len(sites),
        extra={"thresholds_kw": dict(th)} if th else {},
    )


def self_consumption_plan(
    feeder: Union[FeederSpec, str],
    sites: Sequence[Union[Site, Mapping[str, Any]]],
    rest_of_feeder: RestLike,
    forecasts: Forecasts,
    G: Optional[float] = None,
    prices: Union[float, ArrayLike, None] = None,
    *,
    statuses: Optional[Mapping[str, AssetStatus]] = None,
    t0: Optional[str] = None,
) -> DetailedPlan:
    """Baseline: every site runs the self-consumption rule. Same signature as `lp.plan_feeder` (prices unused)."""
    return _rule_plan("self_consumption", feeder, sites, rest_of_feeder, forecasts, G,
                      statuses=statuses, thresholds=None, t0=t0)


def home_thresholds(q: Mapping[str, Mapping[str, np.ndarray]], alpha: float) -> dict[str, float]:
    """Per-home threshold theta_i = alpha * max_t(P50 PV - P50 load) (kW, >= 0)."""
    return {sid: float(alpha * max(0.0, float(np.max(v["pv50"] - v["l50"])))) for sid, v in q.items()}


def threshold_plan(
    feeder: Union[FeederSpec, str],
    sites: Sequence[Union[Site, Mapping[str, Any]]],
    rest_of_feeder: RestLike,
    forecasts: Forecasts,
    G: Optional[float] = None,
    prices: Union[float, ArrayLike, None] = None,
    *,
    alpha: float = 0.5,
    statuses: Optional[Mapping[str, AssetStatus]] = None,
    t0: Optional[str] = None,
) -> DetailedPlan:
    """Per-home export-threshold rule with theta_i = alpha * (home's forecast peak surplus); alpha from
    `tune_threshold`."""
    return _rule_plan("threshold", feeder, sites, rest_of_feeder, forecasts, G, statuses=statuses,
                      thresholds=lambda q: home_thresholds(q, alpha), t0=t0)


@dataclass
class ValidationDay:
    """One feeder-day for tuning. pv / load = realised kW per site (what the rule actually sees);
    forecast_pv / forecast_load (P50) set each home's threshold, default = realised."""

    sites: Sequence[Union[Site, Mapping[str, Any]]]
    pv: Mapping[str, ArrayLike]
    load: Mapping[str, ArrayLike]
    g_kw: float
    rest_net_kw: Optional[ArrayLike] = None
    prices: Union[float, ArrayLike, None] = None
    forecast_pv: Optional[Mapping[str, ArrayLike]] = None
    forecast_load: Optional[Mapping[str, ArrayLike]] = None


@dataclass
class ThresholdTuning:
    alpha: float
    scores: dict[float, dict[str, float]]  # alpha -> {over_kwh, cost_lei, score}


def evaluate_threshold(day: ValidationDay, alpha: float, beta: float = DEFAULT_BETA,
                       dt_h: float = DT_H) -> dict[str, float]:
    """kWh above the limit and energy cost (import - beta * export revenue, lei) of the threshold rule on a day."""
    sites = [as_site(s) for s in day.sites]
    fpv = day.forecast_pv or day.pv
    fld = day.forecast_load or day.load
    q = {s.site_id: {"pv50": np.asarray(fpv[s.site_id], float), "l50": np.asarray(fld[s.site_id], float)}
         for s in sites}
    trace = simulate_fleet(sites, day.pv, day.load, thresholds=home_thresholds(q, alpha), dt_h=dt_h)
    T = len(trace.net_kw)
    rest = np.zeros(T) if day.rest_net_kw is None else np.asarray(day.rest_net_kw, float)
    over = float(np.maximum(0.0, trace.net_kw + rest - day.g_kw).sum() * dt_h)
    price = price_vector(day.prices, T)
    cost = 0.0
    for g in trace.grid_kw.values():
        cost += float((price * (np.maximum(0.0, -g) - beta * np.maximum(0.0, g))).sum() * dt_h)
    return {"over_kwh": over, "cost_lei": cost}


def tune_threshold(
    days: Sequence[ValidationDay],
    alphas: Optional[Sequence[float]] = None,
    beta: float = DEFAULT_BETA,
    over_weight: float = 100.0,
) -> ThresholdTuning:
    """Pick alpha on validation days: minimise `over_weight * kWh above limit + energy cost` summed over days
    (ties -> smallest alpha, i.e. closest to the baseline). over_weight in lei/kWh; default treats the limit
    as ~100x more important than the energy bill."""
    alphas = list(alphas) if alphas is not None else [round(a, 2) for a in np.linspace(0.0, 0.95, 20)]
    if not days:
        raise ValueError("need at least one validation day")
    scores: dict[float, dict[str, float]] = {}
    for a in alphas:
        over = cost = 0.0
        for d in days:
            r = evaluate_threshold(d, a, beta)
            over += r["over_kwh"]
            cost += r["cost_lei"]
        scores[a] = {"over_kwh": over, "cost_lei": cost, "score": over_weight * over + cost}
    best = min(alphas, key=lambda a: (round(scores[a]["score"], 9), a))
    return ThresholdTuning(alpha=best, scores=scores)
