"""Imbalance level 3: the supplier dispatches enrolled home batteries in the 15:00 LP.

Level 2 (``sentinel_id``) re-forecasts 16:00-24:00 local at 15:00 and trades the gap to the day-ahead purchase
on PI. Level 3 solves the same decision as one LP that may also schedule the batteries of enrolled homes:

    min  sum_t traded_t * (pi_hat_t - pzu_t)                                    intraday term (PI price forecast)
       + sum_t (1/M) sum_m [a |pzu_t| short_tm + b |pzu_t| long_tm]             expected imbalance
    s.t. short_tm - long_tm = y_tm - bought_t - traded_t + kappa * flex_t       M quantile nodes y of the 15:00 band
         flex_t = KW_TO_MWH * sum_i (c_it - d_it - bhat_it)                     change in enrolled net import
         e_it = e_i,t-1 + eta_c c_it dt - d_it dt / eta_d,   lb_i <= e_it <= hi_i,  0 <= c, d <= p_max
         e_i,end = ebase_i,end                                                   the customer gets the energy back
         sum_{i in feeder} (bhat_it - c_it + d_it) <= headroom_ft                feeder export limit (hard)

- ``bhat`` / ``ebase`` = what the enrolled battery would have done under its own self-consumption rule (forecast from
  live SoC + persistence of yesterday's PV/load), so flex = 0 (leave the batteries alone) is always feasible and the
  LP is never worse than its own no-battery version on the plan.
- ``lb_i`` = the comfort floor max(soc_min * E, comfort_reserve), lowered to what the rule
  itself would reach, so following the rule stays feasible.
- ``kappa`` = the P10 promise: the plan counts only the 10th percentile of the share of enrolled homes that comply
  (each home-day opts out with probability ``noncompliance_prob``); the rest of the gap is traded on PI.
- Per-step objective = the newsvendor expected cost on the marginal 15:00 distribution (split-normal through the
  calibrated P10/P50/P90), so it needs no scenario coupling across steps.

Pure numpy + scipy (HiGHS through ``scipy.optimize.linprog``), no cvxpy: runs in the check sandbox and locally.

Labels: homes, batteries, feeders = synthetic RO; PZU/PI = real RO (OPCOM); a, b = proxy;
enrolment share and noncompliance = assumptions (share_enrolled 0.10, noncompliance_prob 0.10).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Mapping, Optional, Sequence

import numpy as np

DT_H = 0.25
KW_TO_MWH = DT_H / 1000.0
Z10 = 1.2815515655446004  # standard-normal 90th percentile
EPS_CYCLE = 1e-4  # lei per kW per step on charge + discharge: no simultaneous charge/discharge, no idle cycling


# ============================================================================================== enrolled homes
@dataclass
class Enrolled:
    """The enrolled batteries, per home (N,). Energies kWh, powers kW; floor = planner floor (comfort reserve)."""

    idx: np.ndarray  # positions into the source's site list
    site_ids: np.ndarray
    capacity_kwh: np.ndarray
    p_max_kw: np.ndarray
    eta_c: np.ndarray
    eta_d: np.ndarray
    e_phys_kwh: np.ndarray  # soc_min * E: the physical floor of the self-consumption rule
    e_floor_kwh: np.ndarray  # max(soc_min * E, comfort_reserve): what the supplier may use
    e_hi_kwh: np.ndarray
    feeder: np.ndarray  # (N,) int code into ``feeders``
    feeders: list = field(default_factory=list)

    def __len__(self) -> int:
        return len(self.idx)

    def take(self, sel: Any) -> "Enrolled":
        sel = np.asarray(sel)
        return Enrolled(*(np.asarray(getattr(self, k))[sel] for k in
                          ("idx", "site_ids", "capacity_kwh", "p_max_kw", "eta_c", "eta_d", "e_phys_kwh",
                           "e_floor_kwh", "e_hi_kwh", "feeder")), feeders=list(self.feeders))


def enrolment_order(rows: Sequence[Mapping[str, Any]], seed: int = 0) -> np.ndarray:
    """Battery homes (battery_kwh > 0) in a fixed random order: the first round(share * n) are enrolled, so a
    larger share always contains a smaller one (nested sweep)."""
    cap = np.array([float(r.get("battery_kwh") or 0.0) for r in rows])
    bat = np.flatnonzero(cap > 0)
    return bat[np.random.default_rng(seed + 178).permutation(len(bat))]



def enrolled_from_rows(rows: Sequence[Mapping[str, Any]], idx: Sequence[int], feeder_key: str = "feeder_id"
                       ) -> Enrolled:
    idx = np.asarray(idx, dtype=np.int64)
    sel = [rows[i] for i in idx]

    def col(k: str, default: float) -> np.ndarray:
        return np.array([float(r.get(k) if r.get(k) is not None else default) for r in sel], dtype=np.float64)

    cap = col("battery_kwh", 0.0)
    e_phys = col("soc_min", 0.05) * cap
    names = [str(r.get(feeder_key) or "?") for r in sel]
    feeders = sorted(set(names))
    code = {n: k for k, n in enumerate(feeders)}
    return Enrolled(idx, np.array([str(r.get("site_id")) for r in sel], dtype=object), cap,
                    np.where(cap > 0, col("battery_kw", 0.0), 0.0), col("eta_charge", 0.95),
                    col("eta_discharge", 0.95), e_phys, np.maximum(e_phys, np.minimum(col("comfort_reserve_kwh", 0.0),
                                                                                      cap)),
                    col("soc_max", 1.0) * cap, np.array([code[n] for n in names], dtype=np.int64), feeders)


def commit_factor(n: int, p_noncompliance: float, z: float = Z10) -> float:
    """P10 of the share of n enrolled homes that follow the schedule (binomial, normal approximation)."""
    p = float(p_noncompliance)
    if n <= 0:
        return 0.0
    if p <= 0:
        return 1.0
    return float(np.clip(1.0 - p - z * np.sqrt(p * (1.0 - p) / n), 0.0, 1.0))


# ============================================================================================== forecasts
def quantile_nodes(bands: np.ndarray, m: int = 20) -> np.ndarray:
    """(T, 3) P10/P50/P90 -> (T, m) values at levels (j + 0.5) / m of a split normal through the three quantiles."""
    from scipy.special import ndtri

    bands = np.sort(np.asarray(bands, dtype=np.float64), axis=-1)
    z = ndtri((np.arange(m) + 0.5) / m) / Z10
    lo = (bands[:, 1] - bands[:, 0])[:, None]
    hi = (bands[:, 2] - bands[:, 1])[:, None]
    return bands[:, 1:2] + np.where(z[None, :] < 0, z[None, :] * lo, z[None, :] * hi)


def self_consumption_rollout(pv: np.ndarray, load: np.ndarray, e0: np.ndarray, en: Enrolled
                             ) -> tuple[np.ndarray, np.ndarray]:
    """The self-consumption rule (physical floor soc_min) on (N, T) kW from e0 (N,) kWh -> (batt kW + charge,
    e kWh at the end of each step). Same rule as ``wattsentry.contracts.self_consumption_step``."""
    return track_schedule(np.asarray(pv, float) - np.asarray(load, float), e0, en, floor=en.e_phys_kwh)


def track_schedule(req: np.ndarray, e0: np.ndarray, en: Enrolled, floor: Optional[np.ndarray] = None
                   ) -> tuple[np.ndarray, np.ndarray]:
    """Apply requested battery power (N, T) kW (+ charge) through the battery physics from e0 (N,) kWh, clipped to
    +-p_max, the ceiling e_hi and ``floor`` (default physical). A battery below the floor may charge, not discharge."""
    req = np.asarray(req, dtype=np.float64)
    n, t_steps = req.shape
    floor = en.e_phys_kwh if floor is None else np.asarray(floor, np.float64)
    e = np.array(e0, dtype=np.float64)
    batt = np.zeros((n, t_steps))
    soc = np.zeros((n, t_steps))
    for t in range(t_steps):
        p = np.clip(req[:, t], -en.p_max_kw, en.p_max_kw)
        room_c = np.maximum(0.0, (en.e_hi_kwh - e) / (en.eta_c * DT_H))
        room_d = np.maximum(0.0, (e - floor) * en.eta_d / DT_H)
        p = np.minimum(p, room_c)
        p = np.maximum(p, -room_d)
        e = e + np.where(p >= 0, en.eta_c * p * DT_H, p * DT_H / en.eta_d)
        batt[:, t] = p
        soc[:, t] = e
    return batt, soc


def trailing_spread(local_hour_hist: np.ndarray, spread_hist: np.ndarray, local_hour: np.ndarray) -> np.ndarray:
    """PI - PZU forecast per step = mean spread of the trailing history at the same local clock hour (0 if none)."""
    hh = np.floor(np.asarray(local_hour_hist, float)).astype(int)
    sp = np.asarray(spread_hist, float)
    ok = np.isfinite(sp)
    s = np.bincount(hh[ok], weights=sp[ok], minlength=24)
    c = np.bincount(hh[ok], minlength=24)
    prof = np.where(c > 0, s / np.maximum(c, 1), 0.0)
    return prof[np.floor(np.asarray(local_hour, float)).astype(int) % 24]


# ============================================================================================== the LP
@dataclass
class Level3Plan:
    traded: np.ndarray  # (T,) MWh PI trade (+ buy), 0 outside the window
    batt_kw: np.ndarray  # (N, W) planned battery power on the window steps (+ charge)
    e_kwh: np.ndarray  # (N, W) planned energy at the end of each window step
    flex_mwh: np.ndarray  # (T,) planned change of the enrolled net import vs the rule (- = more discharge)
    window: np.ndarray  # (W,) step indices
    status: str
    objective: float
    solve_s: float = 0.0


def plan_level3(nodes: np.ndarray, bought: np.ndarray, pzu: np.ndarray, pi_hat: np.ndarray, mask: np.ndarray,
                en: Optional[Enrolled] = None, e0: Optional[np.ndarray] = None, bhat: Optional[np.ndarray] = None,
                e_end_min: Optional[np.ndarray] = None, e_lb: Optional[np.ndarray] = None, *, a: float, b: float,
                kappa: float = 1.0, headroom_kw: Optional[np.ndarray] = None,
                trade_bound: Optional[float] = None) -> Level3Plan:
    """One day's 15:00 decision. nodes (T, M) MWh = the portfolio net import at M equally likely levels under the
    rule; bought, pzu, pi_hat (T,); mask (T,) = window steps. Enrolled inputs on the window steps: e0 (N,) kWh at
    the window start, bhat (N, W) kW = the rule's battery power, e_end_min (N,) kWh, e_lb (N,) kWh floor (default
    en.e_floor_kwh). headroom_kw (F, W) = extra export each feeder can take (None = unlimited). en=None or empty =
    the same LP without batteries (level 2 as an LP)."""
    import time

    from scipy import sparse
    from scipy.optimize import linprog

    t_start = time.perf_counter()
    mask = np.asarray(mask, bool)
    win = np.flatnonzero(mask)
    T = len(mask)
    nw = len(win)
    nodes = np.asarray(nodes, np.float64)
    M = nodes.shape[1]
    bought = np.asarray(bought, np.float64)
    pzu = np.asarray(pzu, np.float64)
    pi_hat = np.asarray(pi_hat, np.float64)
    N = 0 if en is None else len(en)
    if nw == 0:
        return Level3Plan(np.zeros(T), np.zeros((N, 0)), np.zeros((N, 0)), np.zeros(T), win, "empty", 0.0)
    y = nodes[win]  # (W, M)
    bw, pz, pih = bought[win], np.abs(pzu[win]), pi_hat[win]
    bhat = np.zeros((N, nw)) if N == 0 else np.asarray(bhat, np.float64).reshape(N, nw)
    k = float(kappa) * KW_TO_MWH

    # variable layout: traded (W) | c (N*W) | d (N*W) | e (N*W) | short (W*M) | long (W*M)
    o_c, o_d, o_e = nw, nw + N * nw, nw + 2 * N * nw
    o_s = nw + 3 * N * nw
    o_l = o_s + nw * M
    nvar = o_l + nw * M
    cost = np.zeros(nvar)
    cost[:nw] = pih - pzu[win]
    cost[o_c:o_e] = EPS_CYCLE
    cost[o_s:o_l] = np.repeat(a * pz / M, M)
    cost[o_l:] = np.repeat(b * pz / M, M)

    rows, cols, vals = [], [], []
    beq = []
    r = 0
    hw = np.arange(nw)
    # dev rows: short - long + traded - k sum_i c + k sum_i d = y - bought - k sum_i bhat
    rhs_flex = k * bhat.sum(axis=0) if N else np.zeros(nw)
    for j in range(M):
        rr = r + hw
        rows += [rr, rr, rr]
        cols += [o_s + hw * M + j, o_l + hw * M + j, hw]
        vals += [np.ones(nw), -np.ones(nw), np.ones(nw)]
        if N:
            for i in range(N):
                rows += [rr, rr]
                cols += [o_c + i * nw + hw, o_d + i * nw + hw]
                vals += [np.full(nw, -k), np.full(nw, k)]
        beq.append(y[:, j] - bw - rhs_flex)
        r += nw
    # SoC rows: e_t - e_{t-1} - eta_c dt c_t + dt/eta_d d_t = (e0 at t = 0)
    if N:
        e0 = np.asarray(e0, np.float64)
        for i in range(N):
            rr = r + hw
            rows += [rr, rr, rr]
            cols += [o_e + i * nw + hw, o_c + i * nw + hw, o_d + i * nw + hw]
            vals += [np.ones(nw), np.full(nw, -en.eta_c[i] * DT_H), np.full(nw, DT_H / en.eta_d[i])]
            if nw > 1:
                rows.append(rr[1:])
                cols.append(o_e + i * nw + hw[:-1])
                vals.append(-np.ones(nw - 1))
            b0 = np.zeros(nw)
            b0[0] = e0[i]
            beq.append(b0)
            r += nw
    A_eq = sparse.csr_matrix((np.concatenate(vals), (np.concatenate(rows), np.concatenate(cols))), shape=(r, nvar))
    b_eq = np.concatenate(beq)

    A_ub = b_ub = None
    if N and headroom_kw is not None:
        hk = np.asarray(headroom_kw, np.float64).reshape(len(en.feeders), nw)
        ur, uc, uv, ub = [], [], [], []
        ru = 0
        for f in range(hk.shape[0]):
            members = np.flatnonzero(en.feeder == f)
            if members.size == 0:
                continue
            rr = ru + hw
            for i in members:  # extra export = sum (bhat - c + d) <= headroom
                ur += [rr, rr]
                uc += [o_c + i * nw + hw, o_d + i * nw + hw]
                uv += [-np.ones(nw), np.ones(nw)]
            ub.append(hk[f] - bhat[members].sum(axis=0))
            ru += nw
        if ru:
            A_ub = sparse.csr_matrix((np.concatenate(uv), (np.concatenate(ur), np.concatenate(uc))), shape=(ru, nvar))
            b_ub = np.concatenate(ub)

    if trade_bound is None:
        trade_bound = float(np.abs(y - bw[:, None]).max()) + (2 * k * en.p_max_kw.sum() if N else 0.0) + 1e-3
    bounds = [(-trade_bound, trade_bound)] * nw
    if N:
        lb = en.e_floor_kwh if e_lb is None else np.asarray(e_lb, np.float64)
        lb = np.minimum(lb, e0)  # a battery that starts below the floor may not go lower
        end = np.full(N, -np.inf) if e_end_min is None else np.asarray(e_end_min, np.float64)
        for i in range(N):
            bounds += [(0.0, float(en.p_max_kw[i]))] * nw
        for i in range(N):
            bounds += [(0.0, float(en.p_max_kw[i]))] * nw
        for i in range(N):  # the window ends at the rule's energy: the customer's battery is handed back as it was
            lo, hi = float(lb[i]), float(max(en.e_hi_kwh[i], lb[i]))
            last = (float(np.clip(end[i], lo, hi)),) * 2 if np.isfinite(end[i]) else (lo, hi)
            bounds += [(lo, hi)] * (nw - 1) + [last]
    bounds += [(0.0, None)] * (2 * nw * M)
    res = linprog(cost, A_ub=A_ub, b_ub=b_ub, A_eq=A_eq, b_eq=b_eq, bounds=bounds, method="highs")
    solve_s = time.perf_counter() - t_start
    traded = np.zeros(T)
    flex = np.zeros(T)
    if res.status != 0 or res.x is None:
        return Level3Plan(traded, bhat.copy(), np.zeros((N, nw)), flex, win, f"failed:{res.message}", float("nan"),
                          solve_s)
    x = res.x
    traded[win] = x[:nw]
    c = x[o_c:o_d].reshape(N, nw) if N else np.zeros((0, nw))
    d = x[o_d:o_e].reshape(N, nw) if N else np.zeros((0, nw))
    e = x[o_e:o_s].reshape(N, nw) if N else np.zeros((0, nw))
    batt = c - d
    if N:
        flex[win] = KW_TO_MWH * (batt - bhat).sum(axis=0)
    return Level3Plan(traded, batt, e, flex, win, "optimal", float(res.fun), solve_s)


# ============================================================================================== one replay day
@dataclass
class DayResult:
    traded: np.ndarray  # (T,) MWh
    delta_mwh: np.ndarray  # (T,) realised change of the portfolio net import from the dispatch (MWh)
    plan: Level3Plan
    diag: dict


def run_day(*, nodes: np.ndarray, bought: np.ndarray, pzu: np.ndarray, pi_hat: np.ndarray, mask: np.ndarray,
            en: Enrolled, pv: np.ndarray, load: np.ndarray, batt: np.ndarray, soc_kwh: np.ndarray,
            pv_prev: np.ndarray, load_prev: np.ndarray, compliant: np.ndarray, a: float, b: float, kappa: float,
            headroom_kw: Optional[np.ndarray] = None, lead_steps: int = 4) -> DayResult:
    """Plan at 15:00 and execute one day for the enrolled homes. Arrays per enrolled home (N, T) kW for the day:
    pv, load, batt (realised, rule), soc_kwh (realised energy at the END of each step), pv_prev/load_prev = the
    same steps yesterday (persistence forecast). The plan starts ``lead_steps`` before the window (15:00 for a
    16:00 window) from the live SoC then; the 15:00-16:00 steps are forecast with the rule, the window's first SoC
    is that forecast, and the execution re-applies the schedule to the real SoC (physics clips any shortfall)."""
    mask = np.asarray(mask, bool)
    win = np.flatnonzero(mask)
    T = len(mask)
    N = len(en)
    if N == 0 or win.size == 0:
        plan = plan_level3(nodes, bought, pzu, pi_hat, mask, None, a=a, b=b)
        return DayResult(plan.traded, np.zeros(T), plan, {"enrolled": 0})
    if not np.all(np.diff(win) == 1):
        raise ValueError("window steps must be contiguous")
    s0 = max(int(win[0]) - lead_steps, 0)
    e_live = soc_kwh[:, s0 - 1] if s0 > 0 else soc_kwh[:, 0]  # SoC known at 15:00 (telemetry of enrolled homes)
    span = np.arange(s0, int(win[-1]) + 1)
    bh_all, eb_all = self_consumption_rollout(pv_prev[:, span], load_prev[:, span], e_live, en)
    off = int(win[0]) - s0
    e0 = eb_all[:, off - 1] if off > 0 else e_live
    bhat = bh_all[:, off:]
    ebase = eb_all[:, off:]
    e_lb = np.minimum(en.e_floor_kwh, ebase.min(axis=1))
    plan = plan_level3(nodes, bought, pzu, pi_hat, mask, en, e0, bhat, ebase[:, -1], e_lb, a=a, b=b, kappa=kappa,
                       headroom_kw=headroom_kw)
    real0 = soc_kwh[:, win[0] - 1] if win[0] > 0 else e_live
    if not plan.status == "optimal":
        return DayResult(plan.traded, np.zeros(T), plan, {"enrolled": N, "failed": 1})
    floor = np.minimum(np.minimum(en.e_floor_kwh, ebase.min(axis=1)), real0)
    applied, e_run = track_schedule(plan.batt_kw, real0, en, floor=floor)
    comp = np.asarray(compliant, bool)
    applied = np.where(comp[:, None], applied, batt[:, win])
    e_end = np.where(comp, e_run[:, -1], soc_kwh[:, win[-1]])
    delta = np.zeros(T)
    delta[win] = KW_TO_MWH * (applied - batt[:, win]).sum(axis=0)
    extra_export_kw = batt[:, win] - applied  # (N, W) + = more export than the rule
    diag = {"enrolled": N, "compliant": int(comp.sum()), "failed": 0,
            "discharge_mwh": float(KW_TO_MWH * np.clip(-applied, 0, None).sum()),
            "discharge_rule_mwh": float(KW_TO_MWH * np.clip(-batt[:, win], 0, None).sum()),
            "planned_flex_abs_mwh": float(np.abs(plan.flex_mwh).sum()),
            "tracking_error_mwh": float(KW_TO_MWH * np.abs(applied - plan.batt_kw)[comp].sum()),
            "terminal_gap_mwh": float((e_end - soc_kwh[:, win[-1]]).sum() / 1000.0),
            "extra_export_kw": extra_export_kw, "solve_s": plan.solve_s}
    return DayResult(plan.traded, delta, plan, diag)


def feeder_headroom(g_max_kw: np.ndarray, export_hist_kw: np.ndarray) -> np.ndarray:
    """Extra export each feeder can take (F, W): g_max - the max of the trailing days' net export at the same step
    (export_hist_kw (F, D_hist, W), + = export; NaN ignored). Never negative."""
    hist = np.asarray(export_hist_kw, np.float64)
    worst = np.nanmax(hist, axis=1) if hist.shape[1] else np.zeros(hist[:, 0].shape)
    worst = np.nan_to_num(worst, nan=0.0)
    return np.clip(np.asarray(g_max_kw, np.float64)[:, None] - worst, 0.0, None)


# ============================================================================================== imbalance replay glue
L3_POLICIES = ("sentinel_id_lp", "sentinel_l3", "sentinel_l3_oracle", "sentinel_l3_all")
L3_DOC = {
    "sentinel_id_lp": "level 2 as the 15:00 LP without batteries (PI price forecast = trailing 28-day spread)",
    "sentinel_l3": "level 3: + enrolled batteries (share of battery homes = --l3-share), same PI forecast",
    "sentinel_l3_oracle": "as sentinel_l3 but the LP sees the realised PI prices (upper bound, not deployable)",
    "sentinel_l3_all": "as sentinel_l3 with every battery home enrolled (share 1.0)",
}


def _local_hours(ts_utc: np.ndarray, tz: str = "Europe/Bucharest") -> np.ndarray:
    import pandas as pd

    idx = pd.DatetimeIndex(np.asarray(ts_utc).ravel())
    idx = idx.tz_localize("UTC") if idx.tz is None else idx.tz_convert("UTC")
    loc = idx.tz_convert(tz)
    return (loc.hour + loc.minute / 60.0).to_numpy().reshape(np.shape(ts_utc))


def _feeder_data(ts_days: np.ndarray, log=print):
    """(feeder ids, g_max (F,), export (F, n_days, T) kW realised, hist (F, n_days, 7, T) the 7 previous days) from
    the fleet's feeders.parquet + feeder_ts (physical feeders, + = export), or None when the files are missing."""
    try:
        import pandas as pd
        import polars as pl

        from wattsentry.eval.baselines_portfolio import find_fleet_dirs

        fleet, _ = find_fleet_dirs()
        fd = pl.read_parquet(fleet / "feeders.parquet", columns=["feeder_id", "g_max_kw"])
        ids = [str(x) for x in fd["feeder_id"].to_list()]
        g_max = fd["g_max_kw"].to_numpy().astype(np.float64)
        flat = pd.DatetimeIndex(np.asarray(ts_days).ravel())
        lo, hi = flat.min() - pd.Timedelta(days=8), flat.max() + pd.Timedelta(minutes=15)
        years = sorted({lo.year, hi.year, (lo + pd.Timedelta(days=2)).year})
        parts = [pl.read_parquet(p, columns=["ts", "feeder_id", "net_export_w"])
                 for y in years for p in [fleet / "feeder_ts" / f"year={y}.parquet"] if p.exists()]
        if not parts:
            return None
        df = pl.concat(parts).with_columns(pl.col("ts").dt.convert_time_zone("UTC").dt.replace_time_zone(None))
        grid = pd.date_range(lo.floor("D"), hi.ceil("D"), freq="15min")
        key = {f: k for k, f in enumerate(ids)}
        arr = np.full((len(ids), len(grid)), np.nan)
        fi = np.array([key.get(str(f), -1) for f in df["feeder_id"].to_list()])
        ti = grid.get_indexer(pd.DatetimeIndex(df["ts"].to_numpy()))
        ok = (fi >= 0) & (ti >= 0)
        arr[fi[ok], ti[ok]] = df["net_export_w"].to_numpy()[ok] / 1000.0
        pos = grid.get_indexer(flat).reshape(np.shape(ts_days))
        exp = arr[:, pos]  # (F, n, T)
        hist = np.stack([arr[:, np.clip(pos - k * 96, 0, None)] for k in range(1, 8)], axis=2)  # (F, n, 7, T)
        return ids, g_max, exp, hist
    except Exception as e:  # noqa: BLE001  feeder files are optional: the LP then runs without the export limit
        log(f"[level3] feeder data unavailable ({type(e).__name__}: {e}); no feeder export limit")
        return None


def replay_level3(src: Any, sites: np.ndarray, days: np.ndarray, *, ts: np.ndarray, bought: np.ndarray,
                  id_cal: np.ndarray, mask: np.ndarray, pzu: np.ndarray, pi: np.ndarray, policies: Sequence[str],
                  a: float, b: float, share: float = 0.10, p_noncompliance: float = 0.10, n_nodes: int = 20,
                  spread_days: int = 28, seed: int = 0, feeders: bool = True, log=print) -> dict[str, Any]:
    """Level-3 variants on the imbalance replay's days. days (n,) window day indices; ts (n, T) naive UTC; bought,
    mask (n, T); id_cal (n, T, 3) the calibrated 15:00 band; pzu, pi (n, T) realised. feeders=False skips the
    feeder export limit (no fleet files). Returns {policy: {traded (n, T), delta (n, T) MWh change of the metered
    portfolio import}, "diag": {...}}."""
    import time

    import pandas as pd

    from wattsentry.data.prices import prices_at
    from wattsentry.kernels import window_gather

    t_all = time.perf_counter()
    pols = [p for p in policies if p in L3_POLICIES]
    n, T = bought.shape
    rows = [src.sites[int(i)] for i in sites]
    order = enrolment_order(rows, seed)
    n_bat = len(order)
    shares = {"sentinel_id_lp": 0.0, "sentinel_l3": share, "sentinel_l3_oracle": share, "sentinel_l3_all": 1.0}
    top = max(int(round(shares[p] * n_bat)) for p in pols) if pols else 0
    union = np.sort(order[:top])
    en_all = enrolled_from_rows(rows, union)
    fd = _feeder_data(ts, log=log) if (top and feeders) else None
    if fd is not None:
        f_ids, g_max, f_exp, f_hist = fd
        fcode = {f: k for k, f in enumerate(f_ids)}
        f_of = np.array([fcode.get(str(rows[i].get("feeder_id")), -1) for i in union], dtype=np.int64)
    # window arrays for the union of enrolled homes, today and yesterday (persistence forecast)
    gs = np.asarray(sites, np.int64)[union]
    arrs = {}
    if top:
        for name in ("pv_ac", "load", "batt", "soc"):
            src_arr = src.array(name)
            for lag, tag in ((0, ""), (1, "_prev")):
                s_, d_ = np.meshgrid(gs, np.asarray(days, np.int64) - lag, indexing="xy")
                g = window_gather(src_arr, s_.ravel(), d_.ravel() * T, 0, T, np.float64)
                arrs[name + tag] = np.nan_to_num(g.reshape(n, len(gs), T))
    soc_kwh = arrs["soc"] * en_all.capacity_kwh[None, :, None] if top else None
    # PI forecast: trailing mean spread by local hour (prices before the day only)
    hist_ts = pd.date_range(pd.Timestamp(ts[0, 0]) - pd.Timedelta(days=spread_days), pd.Timestamp(ts[-1, -1]),
                            freq="15min")
    hp, hi_ = prices_at(hist_ts.to_numpy(), strict=False)
    h_spread = hi_ - hp
    h_hour = _local_hours(hist_ts.to_numpy())
    h_sec = hist_ts.asi8 // 10**9
    hours = _local_hours(ts)
    rng = np.random.default_rng(seed + 9178)
    out = {p: {"traded": np.zeros((n, T)), "delta": np.zeros((n, T))} for p in pols}
    acc = {p: {"discharge_mwh": 0.0, "discharge_rule_mwh": 0.0, "planned_flex_abs_mwh": 0.0,
               "tracking_error_mwh": 0.0, "terminal_gap_mwh": 0.0, "failed_days": 0, "solve_s": 0.0,
               "feeder_over_limit_steps": 0, "feeder_over_limit_steps_rule": 0, "compliant_home_days": 0,
               "enrolled": int(round(shares[p] * n_bat)), "kappa": 0.0} for p in pols}
    for j in range(n):
        d0 = int(pd.Timestamp(ts[j, 0]).value // 10**9)
        sel = (h_sec >= d0 - spread_days * 86400) & (h_sec < d0)
        pi_fc = pzu[j] + trailing_spread(h_hour[sel], h_spread[sel], hours[j])
        nodes = quantile_nodes(id_cal[j], n_nodes)
        comp_all = rng.random(len(union)) >= p_noncompliance
        win = np.flatnonzero(mask[j])
        for p in pols:
            k_en = int(round(shares[p] * n_bat))
            pih = pi[j] if p == "sentinel_l3_oracle" else pi_fc
            if k_en == 0:
                plan = plan_level3(nodes, bought[j], pzu[j], pih, mask[j], None, a=a, b=b)
                out[p]["traded"][j] = plan.traded
                acc[p]["failed_days"] += int(plan.status not in ("optimal", "empty"))
                acc[p]["solve_s"] += plan.solve_s
                continue
            members = np.flatnonzero(np.isin(union, order[:k_en]))
            en = en_all.take(members)
            kappa = commit_factor(k_en, p_noncompliance)
            acc[p]["kappa"] = kappa
            head = None
            if fd is not None and win.size:
                fsel = f_of[members]
                uniq = sorted(set(int(f) for f in fsel if f >= 0))
                remap = {f: k for k, f in enumerate(uniq)}
                en.feeder = np.array([remap.get(int(f), len(uniq)) for f in fsel], dtype=np.int64)
                en.feeders = [f_ids[f] for f in uniq] + (["?"] if (fsel < 0).any() else [])
                hr = (feeder_headroom(g_max[uniq], f_hist[uniq][:, j][:, :, win]) if uniq
                      else np.zeros((0, win.size)))
                if (fsel < 0).any():
                    hr = np.vstack([hr, np.full((1, win.size), 1e9)])
                head = hr
            take = lambda key: arrs[key][j][members]  # noqa: E731
            res = run_day(nodes=nodes, bought=bought[j], pzu=pzu[j], pi_hat=pih, mask=mask[j], en=en,
                          pv=take("pv_ac"), load=take("load"), batt=take("batt"), soc_kwh=soc_kwh[j][members],
                          pv_prev=take("pv_ac_prev"), load_prev=take("load_prev"), compliant=comp_all[members],
                          a=a, b=b, kappa=kappa, headroom_kw=head)
            out[p]["traded"][j] = res.traded
            out[p]["delta"][j] = res.delta_mwh
            dg = res.diag
            acc[p]["failed_days"] += int(dg.get("failed", 0))
            acc[p]["solve_s"] += float(dg.get("solve_s", res.plan.solve_s))
            if dg.get("failed", 0):
                continue
            acc[p]["compliant_home_days"] += int(dg["compliant"])
            for key in ("discharge_mwh", "discharge_rule_mwh", "planned_flex_abs_mwh", "tracking_error_mwh",
                        "terminal_gap_mwh"):
                acc[p][key] += float(dg[key])
            if fd is not None and win.size:
                for f in sorted(set(int(x) for x in f_of[members] if x >= 0)):
                    base = f_exp[f, j, win]
                    extra = dg["extra_export_kw"][f_of[members] == f].sum(axis=0)
                    acc[p]["feeder_over_limit_steps_rule"] += int((base > g_max[f]).sum())
                    acc[p]["feeder_over_limit_steps"] += int((base + extra > g_max[f]).sum())
        if (j + 1) % 50 == 0 or j == n - 1:
            log(f"[level3] {j + 1}/{n} days  " + "  ".join(
                f"{p}: solve {acc[p]['solve_s']:.0f}s fail {acc[p]['failed_days']}" for p in pols), flush=True)
    diag = {"battery_homes": n_bat, "share": share, "p_noncompliance": p_noncompliance, "n_nodes": n_nodes,
            "spread_days": spread_days, "feeder_limit": fd is not None, "per_policy": acc,
            "doc": {p: L3_DOC[p] for p in pols}, "t_level3_s": time.perf_counter() - t_all}
    return {**out, "diag": diag}
