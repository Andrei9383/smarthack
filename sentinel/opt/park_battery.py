"""W5 Wind D4 (card #168, spec W step 7): the park battery (45.72 MW / 91.44 MWh) covers forecast errors.

    python -m sentinel.opt.park_battery [--positions <W3 wind_positions.parquet>] [--policy sentinel] [--workers N]

The park sold `sold = sold_da + sold_id` (W3's "sentinel" positions, card #166) and delivers
`delivered = actual - C + D` (C = MWh charged from the park's own output, D = MWh discharged, per 15-min step).
dev = sold - delivered (> 0 short). Three ways to run the battery (spec W §7), all on the same steps and SoC model:

| variant | what it does                                                                                              |
|---------|-----------------------------------------------------------------------------------------------------------|
| `none`  | no battery                                                                                                |
| `rule`  | every quarter-hour cover what it can of the actual gap; when the gap is ~0 (|gap| < deadband) drift toward |
|         | 50% SoC, never moving the deviation outside the deadband                                                  |
| `lp`    | MPC: at every full hour, cvxpy + HiGHS plans the rest of the UTC day on the *updated* forecast, then the  |
|         | next 4 quarter-hours are run                                                                               |

**The LP** (`MPCPlanner`, one compiled DPP problem per process, horizon padded to 96 steps):

    min  sum_t a|pzu_t| s_t + b|pzu_t| l_t + pzu_t (C_t - D_t) + c_deg (C_t + D_t)  -  lam * e_T
    s_t >= dev_t,  l_t >= -dev_t,  dev_t = sold_t - fc_t + C_t - D_t   (planned deviation on the forecast)
    e_{t+1} = e_t + eta C_t - D_t / eta,  soc_min E <= e <= soc_max E,  C_t <= min(P dt, fc_t),  D_t <= P dt

a, b = X1's frozen imbalance factors (same cost as W3: `wind_cost_steps`). `pzu (C - D)` is the energy the battery
takes out of the sale (losses, and charging when PZU is low): without it the LP would charge for free. `c_deg` =
20 lei per MWh moved (charge or discharge), a **stated assumption** (spec W §7), not a measured wear cost.
`lam = k x mean|pzu|` over the horizon values the energy left at the end of the day (the plan stops at midnight);
k is picked on validation days from `K_GRID` and only the chosen k is reported on test.

`cover_only` (default, the headline): the battery only ever *shrinks* a deviation (never charges while short or
discharges while long). This is the card's "covers forecast errors". Without it (`lp_free`) the LP also moves
energy between hours on the imbalance price (charge while long at low PZU, discharge while long at high PZU): that
gain comes from the **proxy** imbalance price (PZU x (1 -+ b)), so it is reported separately, not as the headline.

**Execution.** The LP plans a deliberate deviation `dev_plan_t` per step. In real time the battery, like the rule,
sees the actual park output of the step and runs `u_t = (sold_t - actual_t) - dev_plan_t` (net discharge), clipped
to power / SoC / charge-from-own-output limits (and to the gap's sign when cover_only). So both get the same
real-time information; the LP's only extra is the plan (which errors to leave, when to charge, what SoC to keep).
With a perfect forecast this is exactly the plan.

**Updated forecast** (the MPC input). At the hour start t0 the actual park output up to t0 is known. The day-ahead
P50 (W3 `fc_p50_mwh`) is moved by the latest measured residuals: fc_t = clip(p50_t + beta_k . (r_last, r_hour), 0,
cap dt), r = actual - p50 at the last measured step and averaged over the last hour (scored steps only), k = steps
ahead. beta_k [2] is fitted on validation days (least squares per lead): the same mechanism as M5's
`nowcast_adjust`, but on the park total. Spec W §7: the LP
only beats the rule if the updated forecast beats persistence; the skill table (`forecast_skill`) reports MAE of
day-ahead, updated and persistence forecasts on the rest-of-day horizons the MPC uses.

Rows also run: `lp_da` (MPC on the day-ahead P50 only: what the update adds), `lp_24h` (horizon the next 24 h,
crossing midnight once tomorrow's position and PZU are known, from 12:00 UTC), `lp_oracle` (perfect foresight of the
rest of the day: the most an MPC could get), `lp_free` (no cover_only).

Steps W3 does not score (curtailed / data gap / unpriced: `score_ok` false) are idle for the battery and left out
of every metric. SoC carries over consecutive days of a park and restarts at 50% after a gap. Each park gets its own
battery (the spec names one battery at one park; three runs give three samples).

Metrics (LAB_METRIC, test days, all parks): `battery_cover_share_{v}` = 1 - sum|dev with| / sum|dev none|,
`battery_cost_cut_lei_{v}` / `_pct_{v}` = net cost (imbalance + energy + wear) cut vs no battery,
`battery_avoidable_cut_pct_{v}` (imbalance only, comparable to W3), `battery_cycles_per_day_{v}`
(= (sum C + sum D) / 2 / E per park-day), `lp_minus_rule_lei` (< 0: the LP is cheaper), `lp_beats_rule`;
unsuffixed `battery_cover_share` = lp. Labels: synthetic RO parks x real RO PZU x proxy imbalance price.

Output `$LAB_OUTPUT_DIR/park_battery.parquet`: one row per (variant, park, step), see `OUT_COLUMNS`.

Hot loops (team/infra.md): the per-step simulation is a numpy/scalar reference; profiles show the LP solves
dominate (>90% of the stage), so no Rust copy. Parks x variants run in a process pool.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from dataclasses import asdict, dataclass, replace
from pathlib import Path
from typing import Optional, Sequence

import numpy as np

LABEL = "synthetic RO parks (ERA5-driven #22/#23) x real RO prices (OPCOM PZU) x proxy imbalance price"
H = 96
K_GRID = (0.0, 0.5, 1.0, 2.0, 3.0)
DA_KNOWN_STEP = 48  # tomorrow's PZU + position are known from 12:00 UTC (PZU results ~13:00 CET, W3 sells at 10:00 UTC)
OUT_COLUMNS = ("variant", "ts", "park_id", "split", "score_ok", "holdout", "sold_mwh", "actual_mwh", "fc_p50_mwh",
               "fc_used_mwh", "charge_mwh", "discharge_mwh", "soc_mwh", "delivered_mwh", "dev_none_mwh", "dev_mwh",
               "pzu_lei_mwh", "avoidable_lei", "avoidable_none_lei", "energy_lei", "wear_lei", "net_cost_lei")


@dataclass(frozen=True)
class Battery:
    """PPC's Dobrogea battery (spec W §1/§7). eta is one-way (round trip eta^2 = 0.88)."""
    p_mw: float = 45.72
    e_mwh: float = 91.44
    eta: float = 0.94
    soc_min: float = 0.1
    soc_max: float = 0.9
    soc0: float = 0.5
    c_deg: float = 20.0  # lei per MWh charged or discharged: ASSUMPTION (spec W §7)
    dt: float = 0.25

    @property
    def step_mwh(self) -> float:
        return self.p_mw * self.dt

    @property
    def e_lo(self) -> float:
        return self.soc_min * self.e_mwh

    @property
    def e_hi(self) -> float:
        return self.soc_max * self.e_mwh


@dataclass(frozen=True)
class Variant:
    name: str
    policy: str  # none | rule | lp
    horizon: str = "day"  # day | 24h
    forecast: str = "nowcast"  # nowcast | da | oracle
    cover_only: bool = True
    k: float = 0.5  # terminal value lam = k x mean|pzu|


LP_FAMILIES = {"lp": Variant("lp", "lp"), "lp_da": Variant("lp_da", "lp", forecast="da"),
               "lp_24h": Variant("lp_24h", "lp", horizon="24h"),
               "lp_oracle": Variant("lp_oracle", "lp", forecast="oracle"),
               "lp_free": Variant("lp_free", "lp", cover_only=False)}
BASE = {"none": Variant("none", "none"), "rule": Variant("rule", "rule")}


# ================================================================================================ battery physics
def limits(e: float, prod: float, bat: Battery) -> tuple[float, float]:
    """(u_min, u_max) of the net discharge u = D - C this step: power, SoC window, charge only from own output."""
    u_max = min(bat.step_mwh, max(0.0, (e - bat.e_lo) * bat.eta))
    u_min = -min(bat.step_mwh, max(0.0, prod), max(0.0, (bat.e_hi - e) / bat.eta))
    return u_min, u_max


def apply(e: float, u: float, bat: Battery) -> float:
    """SoC after net discharge u (MWh at the grid side; < 0 = charge)."""
    return e - u / bat.eta if u >= 0 else e - u * bat.eta


def rule_u(gap: float, e: float, prod: float, bat: Battery, deadband: float) -> float:
    """Spec W §7 baseline: cover what it can of gap = sold - actual (> 0 short: discharge); when |gap| < deadband,
    also drift toward 50% SoC, keeping |dev| <= deadband."""
    lo, hi = limits(e, prod, bat)
    if abs(gap) >= deadband:
        return min(max(gap, lo), hi)
    d = min(max(e - bat.soc0 * bat.e_mwh, -deadband), deadband)  # > 0: above 50% -> discharge a little more
    return min(max(gap + d, lo), hi)


def lp_u(gap: float, dev_plan: float, e: float, prod: float, bat: Battery, cover_only: bool) -> float:
    """Real-time step of the MPC: leave the planned deviation, cover the rest of the actual gap."""
    lo, hi = limits(e, prod, bat)
    if cover_only:
        lo, hi = max(lo, min(0.0, gap)), min(hi, max(0.0, gap))
    return min(max(gap - dev_plan, lo), hi)


# ================================================================================================ LP (cvxpy + HiGHS)
class MPCPlanner:
    """One parametrised (DPP) LP over a fixed 96-step horizon; steps past the real horizon are switched off
    (zero weights and zero power), so the problem compiles once and every hourly re-solve only swaps numbers."""

    def __init__(self, bat: Battery, a: float, b: float, L: int = H):
        import cvxpy as cp

        self.bat, self.a, self.b, self.L = bat, a, b, L
        self.g = cp.Parameter(L)  # sold - forecast (> 0: short before the battery)
        self.pz = cp.Parameter(L)
        self.wa = cp.Parameter(L, nonneg=True)
        self.wb = cp.Parameter(L, nonneg=True)
        self.cmax = cp.Parameter(L, nonneg=True)
        self.dmax = cp.Parameter(L, nonneg=True)
        self.e0 = cp.Parameter()
        self.lam = cp.Parameter(nonneg=True)
        C, D = cp.Variable(L, nonneg=True), cp.Variable(L, nonneg=True)
        s, l = cp.Variable(L, nonneg=True), cp.Variable(L, nonneg=True)
        e = cp.Variable(L + 1)
        dev = self.g + C - D
        cons = [e[0] == self.e0, e[1:] == e[:-1] + bat.eta * C - D / bat.eta, e >= bat.e_lo, e <= bat.e_hi,
                C <= self.cmax, D <= self.dmax, s >= dev, l >= -dev]
        obj = (self.wa @ s + self.wb @ l + self.pz @ (C - D) + bat.c_deg * cp.sum(C + D) - self.lam * e[L])
        self.prob = cp.Problem(cp.Minimize(obj), cons)
        assert self.prob.is_dpp()
        self.C, self.D, self.e = C, D, e
        self.n_solves, self.n_fail, self.t_solve = 0, 0, 0.0

    def plan(self, fc: np.ndarray, sold: np.ndarray, pzu: np.ndarray, e0: float, k: float,
             cover_only: bool = True) -> Optional[dict]:
        """Plan over n = len(fc) <= L steps. Returns {C, D, dev, e} (n steps) or None if HiGHS failed."""
        import cvxpy as cp

        bat, L, n = self.bat, self.L, len(fc)
        pad = lambda x: np.concatenate([np.asarray(x, np.float64), np.zeros(L - n)])  # noqa: E731
        fc_ = np.clip(np.asarray(fc, np.float64), 0, None)
        g = np.asarray(sold, np.float64) - fc_
        cmax = np.minimum(bat.step_mwh, fc_)
        dmax = np.full(n, bat.step_mwh)
        if cover_only:
            cmax = np.minimum(cmax, np.maximum(0.0, -g))
            dmax = np.minimum(dmax, np.maximum(0.0, g))
        ap = np.abs(np.asarray(pzu, np.float64))
        self.g.value, self.pz.value = pad(g), pad(pzu)
        self.wa.value, self.wb.value = pad(self.a * ap), pad(self.b * ap)
        self.cmax.value, self.dmax.value = pad(cmax), pad(dmax)
        self.e0.value = float(min(max(e0, bat.e_lo), bat.e_hi))
        self.lam.value = float(k * ap.mean()) if n else 0.0
        t = time.perf_counter()
        try:
            self.prob.solve(solver=cp.HIGHS)
            ok = self.prob.status in ("optimal", "optimal_inaccurate") and self.C.value is not None
        except cp.error.SolverError:
            ok = False
        self.t_solve += time.perf_counter() - t
        self.n_solves += 1
        if not ok:
            self.n_fail += 1
            return None
        C, D = np.clip(self.C.value[:n], 0, None), np.clip(self.D.value[:n], 0, None)
        return {"C": C, "D": D, "dev": g + C - D, "e": self.e.value[: n + 1]}


# ================================================================================================ updated forecast
def origin_residuals(resid: np.ndarray, ok: np.ndarray, block: np.ndarray, t0: int, w: int = 4) -> np.ndarray:
    """(last measured step, mean of the last hour) of actual - P50 before t0, scored steps of the same contiguous
    block only; 0 where nothing is measured (then the forecast stays the day-ahead P50)."""
    s = max(0, t0 - w)
    if t0 == 0:
        return np.zeros(2)
    m = ok[s:t0] & (block[s:t0] == block[t0])
    if not m.any():
        return np.zeros(2)
    r = resid[s:t0][m]
    return np.array([r[-1], r.mean()])


def fit_nowcast(resid: np.ndarray, ok: np.ndarray, block: np.ndarray, fit: np.ndarray, L: int = H,
                ridge: float = 1e-6) -> np.ndarray:
    """beta [L, 2]: per lead k = 1..L (steps after the last measured one), least squares (no intercept, tiny ridge)
    of resid[t0+k-1] on origin_residuals(t0), over hour starts t0 in `fit` (validation steps)."""
    T = len(resid)
    xtx, xty = np.zeros((L, 2, 2)), np.zeros((L, 2))
    for t0 in range(4, T, 4):
        if not fit[t0]:
            continue
        x = origin_residuals(resid, ok, block, t0)
        if not x.any():
            continue
        e = min(T, t0 + L)
        m = ok[t0:e] & (block[t0:e] == block[t0]) & fit[t0:e]
        xtx[: e - t0] += m[:, None, None] * np.outer(x, x)[None]
        xty[: e - t0] += (m * resid[t0:e])[:, None] * x[None]
    scale = np.maximum(np.trace(xtx, axis1=1, axis2=2), 1e-12)[:, None, None]
    return np.linalg.solve(xtx + ridge * scale * np.eye(2)[None] + 1e-12 * np.eye(2)[None], xty[..., None])[..., 0]


def nowcast_fc(s: "Series", t0: int, e: int, beta: np.ndarray) -> np.ndarray:
    """Updated forecast for steps t0..e-1 made at t0: day-ahead P50 + beta_k . origin residuals, in [0, cap dt]."""
    x = origin_residuals(s.actual - s.p50, s.ok, s.block, t0)
    return np.clip(s.p50[t0:e] + beta[: e - t0] @ x, 0.0, s.cap_mwh_step)


# ================================================================================================ one park series
@dataclass
class Series:
    """One park, steps in time order (val + test days). sold = sold_da + sold_id."""
    park_id: str
    cap_mwh_step: float
    ts: np.ndarray
    sold: np.ndarray
    sold_da: np.ndarray
    sold_id: np.ndarray
    actual: np.ndarray
    p50: np.ndarray
    pzu: np.ndarray
    pi: np.ndarray
    ok: np.ndarray
    test: np.ndarray
    holdout: bool
    sid: np.ndarray  # step of the UTC day 0..95
    block: np.ndarray  # contiguous-days block id

    @property
    def T(self) -> int:
        return len(self.ts)


def horizon_end(s: Series, t0: int, horizon: str) -> int:
    """Exclusive end step of the plan made at t0: end of the UTC day, or (24h) t0 + 96 once tomorrow is known."""
    day_end = t0 - s.sid[t0] + H
    if horizon == "24h" and s.sid[t0] >= DA_KNOWN_STEP:
        e = min(s.T, t0 + H)
        same = np.flatnonzero(s.block[t0:e] != s.block[t0])
        return t0 + int(same[0]) if len(same) else e
    return min(day_end, s.T)


def forecast_at(s: Series, t0: int, e: int, how: str, beta: np.ndarray) -> np.ndarray:
    """MPC input for steps t0..e-1 made at t0: day-ahead P50, updated (nowcast) or the actual (oracle)."""
    if how == "oracle":
        return s.actual[t0:e].copy()
    if how == "da":
        return s.p50[t0:e].copy()
    return nowcast_fc(s, t0, e, beta)


def simulate(s: Series, v: Variant, bat: Battery, beta: np.ndarray, a: float, b: float,
             deadband: Optional[float] = None, planner: Optional[MPCPlanner] = None) -> dict:
    """Runs one variant on one park. Returns per-step u (net discharge), C, D, soc (after the step), fc_used."""
    T = s.T
    db = 0.01 * bat.step_mwh if deadband is None else deadband
    u = np.zeros(T)
    soc = np.zeros(T)
    fc_used = s.p50.copy()
    e = bat.soc0 * bat.e_mwh
    dev_plan = np.zeros(T)
    if v.policy == "lp" and planner is None:
        planner = MPCPlanner(bat, a, b)
    for t in range(T):
        if t > 0 and s.block[t] != s.block[t - 1]:
            e = bat.soc0 * bat.e_mwh  # gap in the days: restart at 50%
        if v.policy == "lp" and s.sid[t] % 4 == 0:
            end = horizon_end(s, t, v.horizon)
            fc = forecast_at(s, t, end, v.forecast, beta)
            p = planner.plan(fc, s.sold[t:end], s.pzu[t:end], e, v.k, v.cover_only)
            n = min(4, end - t)
            fc_used[t:t + n] = fc[:n]
            dev_plan[t:t + n] = 0.0 if p is None else p["dev"][:n]  # solver failure: cover everything
        gap = s.sold[t] - s.actual[t]
        if not s.ok[t] or v.policy == "none":
            ut = 0.0
        elif v.policy == "rule":
            ut = rule_u(gap, e, s.actual[t], bat, db)
        else:
            ut = lp_u(gap, dev_plan[t], e, s.actual[t], bat, v.cover_only)
        e = apply(e, ut, bat)
        u[t], soc[t] = ut, e
    out = {"u": u, "C": np.maximum(-u, 0.0), "D": np.maximum(u, 0.0), "soc": soc, "fc_used": fc_used}
    if planner is not None:
        out.update(n_solves=planner.n_solves, n_fail=planner.n_fail, t_solve=planner.t_solve)
    return out


# ================================================================================================ metrics
def step_costs(s: Series, sim: dict, bat: Battery, a: float, b: float) -> dict:
    """Per-step lei: avoidable (imbalance + intraday as W3, on delivered = actual + u), energy = pzu (C - D),
    wear = c_deg (C + D), net = avoidable + energy + wear; plus avoidable without the battery and deviations."""
    from sentinel.eval.wind_portfolio import wind_cost_steps

    delivered = s.actual + sim["u"]
    w = wind_cost_steps(delivered, s.sold_da, s.sold_id, s.pzu, s.pi, a, b)
    w0 = wind_cost_steps(s.actual, s.sold_da, s.sold_id, s.pzu, s.pi, a, b)
    energy = s.pzu * (sim["C"] - sim["D"])
    wear = bat.c_deg * (sim["C"] + sim["D"])
    return {"delivered": delivered, "dev": w["dev"], "dev_none": w0["dev"], "avoidable": w["avoidable"],
            "avoidable_none": w0["avoidable"], "energy": energy, "wear": wear,
            "net": w["avoidable"] + energy + wear}


def summarize(costs: Sequence[dict], sims: Sequence[dict], series: Sequence[Series], masks: Sequence[np.ndarray],
              bat: Battery) -> dict:
    """Totals over the masked steps of several parks."""
    tot = {k: 0.0 for k in ("abs_dev", "abs_dev_none", "avoidable", "avoidable_none", "energy", "wear", "net",
                            "throughput", "produced")}
    park_days = 0
    for c, sim, s, m in zip(costs, sims, series, masks):
        tot["abs_dev"] += float(np.abs(c["dev"])[m].sum())
        tot["abs_dev_none"] += float(np.abs(c["dev_none"])[m].sum())
        for k in ("avoidable", "avoidable_none", "energy", "wear", "net"):
            tot[k] += float(c[k][m].sum())
        tot["throughput"] += float((sim["C"] + sim["D"])[m].sum())
        tot["produced"] += float(s.actual[m].sum())
        park_days += len(np.unique(s.ts[m].astype("datetime64[D]"))) if m.any() else 0
    out = dict(tot)
    out["park_days"] = park_days
    out["cover_share"] = 1 - tot["abs_dev"] / tot["abs_dev_none"] if tot["abs_dev_none"] > 0 else float("nan")
    out["cost_cut_lei"] = tot["avoidable_none"] - tot["net"]
    out["cost_cut_pct"] = 100 * out["cost_cut_lei"] / tot["avoidable_none"] if tot["avoidable_none"] else float("nan")
    out["avoidable_cut_pct"] = (100 * (1 - tot["avoidable"] / tot["avoidable_none"]) if tot["avoidable_none"]
                                else float("nan"))
    out["cycles_per_day"] = tot["throughput"] / 2 / bat.e_mwh / park_days if park_days else float("nan")
    out["lei_per_mwh_produced"] = tot["net"] / tot["produced"] if tot["produced"] else float("nan")
    return out


def forecast_skill(s: Series, beta: np.ndarray, mask: np.ndarray) -> dict:
    """MAE (MWh/step) on the rest-of-day horizons the MPC uses (every hour start in mask): day-ahead P50, updated,
    persistence (last measured hour held). Per lead bucket (<= 1 h, 1-6 h, > 6 h) and all."""
    buckets = {"le1h": (1, 4), "1to6h": (5, 24), "gt6h": (25, H), "all": (1, H)}
    acc = {f"{f}_{b}": [0.0, 0] for f in ("da", "nowcast", "persistence") for b in buckets}
    for t0 in range(0, s.T, 4):
        if not mask[t0]:
            continue
        end = horizon_end(s, t0, "day")
        pm = s.ok[max(0, t0 - 4):t0] & (s.block[max(0, t0 - 4):t0] == s.block[t0]) if t0 else np.zeros(0, bool)
        pers = float(s.actual[max(0, t0 - 4):t0][pm].mean()) if pm.any() else np.nan
        k = np.arange(1, end - t0 + 1)
        m = s.ok[t0:end] & mask[t0:end]
        y = s.actual[t0:end]
        fcs = {"da": s.p50[t0:end], "nowcast": nowcast_fc(s, t0, end, beta),
               "persistence": np.full(end - t0, pers)}
        for f, x in fcs.items():
            err = np.abs(x - y)
            for bname, (lo, hi) in buckets.items():
                mm = m & (k >= lo) & (k <= hi) & np.isfinite(err)
                acc[f"{f}_{bname}"][0] += float(err[mm].sum())
                acc[f"{f}_{bname}"][1] += int(mm.sum())
    return {k: (v[0] / v[1] if v[1] else float("nan")) for k, v in acc.items()} | {
        "n": {k: v[1] for k, v in acc.items()}}


# ================================================================================================ data
def series_from_positions(df, policy: str = "sentinel") -> list[Series]:
    """W3 positions table (polars) -> one Series per park."""
    import polars as pl

    d = df.filter(pl.col("policy") == policy).sort(["park_id", "ts"])
    out = []
    for pid in d["park_id"].unique(maintain_order=True).to_list():
        p = d.filter(pl.col("park_id") == pid)
        ts = p["ts"].dt.replace_time_zone(None).to_numpy().astype("datetime64[m]")
        day = ts.astype("datetime64[D]")
        sid = ((ts - day).astype(int) // 15).astype(np.int64)
        dnum = day.astype(np.int64)
        new = np.r_[True, np.diff(dnum) > 1]  # W3 writes whole days; a missing day starts a new block
        f = lambda c: np.nan_to_num(p[c].to_numpy().astype(np.float64))  # noqa: E731
        sold_da, sold_id = f("sold_da_mwh"), f("sold_id_mwh")
        out.append(Series(park_id=str(pid), cap_mwh_step=float(p["cap_mw"][0]) * 0.25, ts=ts,
                          sold=sold_da + sold_id, sold_da=sold_da, sold_id=sold_id, actual=f("actual_mwh"),
                          p50=f("fc_p50_mwh"), pzu=f("pzu_lei_mwh"), pi=f("pi_lei_mwh"),
                          ok=p["score_ok"].to_numpy().astype(bool), test=(p["split"] == "test").to_numpy(),
                          holdout=bool(p["holdout"][0]), sid=sid, block=np.cumsum(new) - 1))
    return out


def synth_positions(days: int = 20, parks: Sequence[str] = ("WP1", "WP2"), cap_mw: float = 80.0, seed: int = 0,
                    rho: float = 0.97, err_sd: float = 0.06):
    """Stand-in W3 table for tests / local profiling (debug only, not a result): AR(1) forecast error on a smooth
    capacity factor, PZU with an evening peak, sells P33 of production like W3. Half the days val, half test."""
    import polars as pl

    rng = np.random.default_rng(seed)
    T = days * H
    ts = np.datetime64("2025-01-01T00:00") + np.arange(T) * np.timedelta64(15, "m")
    hour = (np.arange(T) % H) / 4
    pzu = 400 + 300 * np.exp(-((hour - 19) ** 2) / 6) - 200 * np.exp(-((hour - 12) ** 2) / 8) + rng.normal(0, 40, T)
    rows = []
    for j, pid in enumerate(parks):
        cf = np.clip(0.4 + 0.3 * np.sin(np.arange(T) / 150 + j) + 0.1 * rng.standard_normal(T).cumsum() / 30, 0, 1)
        err = np.zeros(T)
        for t in range(1, T):
            err[t] = rho * err[t - 1] + np.sqrt(1 - rho ** 2) * err_sd * rng.standard_normal()
        act = cf * cap_mw * 0.25
        p50 = np.clip(cf - err, 0, 1) * cap_mw * 0.25
        sold = np.clip(p50 - 0.43 * err_sd * cap_mw * 0.25, 0, None)
        ok = rng.random(T) > 0.02
        rows.append(pl.DataFrame({
            "policy": ["sentinel"] * T, "ts": ts.astype("datetime64[us]"), "park_id": [pid] * T,
            "cap_mw": np.full(T, cap_mw), "actual_mwh": act, "sold_da_mwh": sold, "sold_id_mwh": np.zeros(T),
            "fc_p10_mwh": p50 * 0.8, "fc_p50_mwh": p50, "fc_p90_mwh": p50 * 1.2, "pzu_lei_mwh": pzu,
            "pi_lei_mwh": pzu, "score_ok": ok,
            "split": np.where(np.arange(T) < (days // 2) * H, "val", "test").astype(object).tolist(),
            "holdout": np.full(T, j == 1), "stopped_day": np.zeros(T, bool)}))
    return pl.concat(rows).with_columns(pl.col("ts").dt.replace_time_zone("UTC"))


# ================================================================================================ evaluation
def _job(args):
    """Worker: one (variant, park). Returns (variant name, k, park index, sim)."""
    s, v, bat, beta, a, b = args
    t = time.perf_counter()
    sim = simulate(s, v, bat, beta, a, b)
    sim["t_wall"] = time.perf_counter() - t
    return v.name, v.k, s.park_id, sim


def run_jobs(jobs, workers: int):
    if workers <= 1:
        return [_job(j) for j in jobs]
    import multiprocessing as mp

    with mp.get_context("spawn").Pool(workers) as pool:
        return pool.map(_job, jobs, chunksize=1)


def output_frame(series: Sequence[Series], sims: dict, costs: dict, variants: Sequence[str]):
    import polars as pl

    frames = []
    for v in variants:
        for j, s in enumerate(series):
            sim, c = sims[v][j], costs[v][j]
            frames.append(pl.DataFrame({
                "variant": [v] * s.T, "ts": s.ts.astype("datetime64[us]"), "park_id": [s.park_id] * s.T,
                "split": np.where(s.test, "test", "val").astype(object).tolist(), "score_ok": s.ok,
                "holdout": np.full(s.T, s.holdout), "sold_mwh": s.sold, "actual_mwh": s.actual,
                "fc_p50_mwh": s.p50, "fc_used_mwh": sim["fc_used"], "charge_mwh": sim["C"],
                "discharge_mwh": sim["D"], "soc_mwh": sim["soc"], "delivered_mwh": c["delivered"],
                "dev_none_mwh": c["dev_none"], "dev_mwh": c["dev"], "pzu_lei_mwh": s.pzu,
                "avoidable_lei": c["avoidable"], "avoidable_none_lei": c["avoidable_none"],
                "energy_lei": c["energy"], "wear_lei": c["wear"], "net_cost_lei": c["net"]}))
    df = pl.concat(frames).with_columns(pl.col("ts").dt.replace_time_zone("UTC"))
    assert tuple(df.columns) == OUT_COLUMNS
    return df


def parse_args(argv):
    ap = argparse.ArgumentParser(prog="python -m sentinel.opt.park_battery", description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--positions", default=None,
                    help="W3 wind_positions.parquet (default $LAB_OUTPUT_DIR/wind_positions.parquet)")
    ap.add_argument("--policy", default="sentinel", help="W3 policy whose positions the battery corrects")
    ap.add_argument("--families", default=",".join(LP_FAMILIES), help="LP rows to run (lp is required)")
    ap.add_argument("--k-grid", default=",".join(str(k) for k in K_GRID))
    ap.add_argument("--c-deg", type=float, default=Battery.c_deg, help="wear cost lei/MWh moved (assumption)")
    ap.add_argument("--a", type=float, default=None)
    ap.add_argument("--b", type=float, default=None)
    ap.add_argument("--workers", type=int, default=0, help="process pool size (0 = cpu count)")
    ap.add_argument("--max-days", type=int, default=0, help="debug: first N val and N test days per park")
    ap.add_argument("--synth", action="store_true", help="stand-in positions (debug / profiling, not a result)")
    ap.add_argument("--synth-days", type=int, default=20)
    ap.add_argument("--out", default=None)
    return ap.parse_args(argv)


def main(argv=None) -> dict:
    import polars as pl

    from sentinel.eval.imbalance import load_ab
    from sentinel.eval.report import Report, emit
    from sentinel.eval.wind_portfolio import load_positions
    from sentinel.infra.timing import stage

    args = parse_args(argv if argv is not None else sys.argv[1:])
    t0 = time.time()
    a, b = load_ab() if args.a is None or args.b is None else (args.a, args.b)
    bat = replace(Battery(), c_deg=args.c_deg)
    rep = Report(vars(args), args.out)
    out_dir = rep.out_dir
    out_dir.mkdir(parents=True, exist_ok=True)
    label = LABEL if not args.synth else "stand-in synthetic positions (debug, not a result)"

    with stage("battery_data"):
        if args.synth:
            df = synth_positions(days=args.synth_days)
        else:
            path = Path(args.positions) if args.positions else out_dir / "wind_positions.parquet"
            df = load_positions(path)
        if args.max_days:
            day = pl.col("ts").dt.date()
            keep = (df.select(day.alias("d"), "split").unique().sort("d").group_by("split", maintain_order=True)
                    .head(args.max_days)["d"])
            df = df.filter(day.is_in(keep.to_list()))
        series = series_from_positions(df, args.policy)
    val_masks = [s.ok & ~s.test for s in series]
    test_masks = [s.ok & s.test for s in series]
    print(f"[w5] {label}: parks {[s.park_id for s in series]}, {series[0].T} steps each "
          f"(val {int((~series[0].test).sum() // H)} d, test {int(series[0].test.sum() // H)} d); a={a:.4f} b={b:.4f}; "
          f"battery {bat.p_mw} MW / {bat.e_mwh} MWh eta={bat.eta} c_deg={bat.c_deg} lei/MWh (assumption)", flush=True)

    with stage("battery_nowcast"):
        betas = [fit_nowcast(s.actual - s.p50, s.ok, s.block, ~s.test) for s in series]
        skill = {s.park_id: forecast_skill(s, beta, s.test) for s, beta in zip(series, betas)}
    print(f"[w5] nowcast beta (last step / last hour, at leads 1, 4, 24, 96 steps) per park: "
          + "; ".join(f"{s.park_id} " + " ".join(f"{bt[i, 0]:.2f}/{bt[i, 1]:.2f}" for i in (0, 3, 23, 95))
                    for s, bt in zip(series, betas)),
          flush=True)

    fams = [f for f in args.families.split(",") if f]
    assert "lp" in fams, "--families must include lp"
    ks = [float(k) for k in args.k_grid.split(",")]
    jobs = []
    for v in BASE.values():
        jobs += [(s, v, bat, beta, a, b) for s, beta in zip(series, betas)]
    for f in fams:
        for k in ks:
            jobs += [(s, replace(LP_FAMILIES[f], k=k), bat, beta, a, b) for s, beta in zip(series, betas)]
    workers = args.workers or min(len(jobs), os.cpu_count() or 1)
    with stage("battery_sim"):
        res = run_jobs(jobs, workers)
    by = {}
    for name, k, pid, sim in res:
        by.setdefault((name, k if name.startswith("lp") else None), {})[pid] = sim
    pids = [s.park_id for s in series]
    sims_of = lambda key: [by[key][p] for p in pids]  # noqa: E731

    # ---- costs, k chosen per family on validation (net cost, all parks)
    def evaluate(key):
        sims = sims_of(key)
        costs = [step_costs(s, sim, bat, a, b) for s, sim in zip(series, sims)]
        return sims, costs

    table, chosen, sims_final, costs_final, kval = {}, {}, {}, {}, {}
    for name in ("none", "rule"):
        sims_final[name], costs_final[name] = evaluate((name, None))
    for f in fams:
        vals = {}
        for k in ks:
            sims, costs = evaluate((f, k))
            vals[k] = summarize(costs, sims, series, val_masks, bat)["net"]
        kval[f] = vals
        chosen[f] = min(ks, key=lambda k: vals[k])
        sims_final[f], costs_final[f] = evaluate((f, chosen[f]))
    variants = ["none", "rule"] + fams
    groups = {"test": test_masks, "val": val_masks,
              "holdout": [m & s.holdout for s, m in zip(series, test_masks)]}
    for g, masks in groups.items():
        table[g] = {v: summarize(costs_final[v], sims_final[v], series, masks, bat) for v in variants}
        table[g + "_per_park"] = {v: {s.park_id: summarize([c], [sm], [s], [m], bat) for s, c, sm, m in
                                      zip(series, costs_final[v], sims_final[v], masks)} for v in variants}

    t = table["test"]
    for v in variants:
        emit(f"battery_cover_share_{v}", t[v]["cover_share"])
        emit(f"battery_cost_cut_lei_{v}", t[v]["cost_cut_lei"])
        emit(f"battery_cost_cut_pct_{v}", t[v]["cost_cut_pct"])
        emit(f"battery_avoidable_cut_pct_{v}", t[v]["avoidable_cut_pct"])
        emit(f"battery_cycles_per_day_{v}", t[v]["cycles_per_day"])
        emit(f"battery_net_cost_lei_{v}", t[v]["net"])
        emit(f"holdout_battery_cost_cut_pct_{v}", table["holdout"][v]["cost_cut_pct"])
    for f in fams:
        emit(f"battery_k_{f}", chosen[f])
    emit("battery_cover_share", t["lp"]["cover_share"])
    emit("lp_minus_rule_lei", t["lp"]["net"] - t["rule"]["net"])
    emit("lp_beats_rule", float(t["lp"]["net"] < t["rule"]["net"]))
    emit("val_lp_minus_rule_lei", table["val"]["lp"]["net"] - table["val"]["rule"]["net"])
    lp_stats = {f: {"n_solves": sum(by[(f, chosen[f])][p]["n_solves"] for p in pids),
                    "n_fail": sum(by[(f, chosen[f])][p]["n_fail"] for p in pids),
                    "t_solve_s": sum(by[(f, chosen[f])][p]["t_solve"] for p in pids),
                    "t_wall_s": sum(by[(f, chosen[f])][p]["t_wall"] for p in pids)} for f in fams}
    emit("lp_solve_fail", lp_stats["lp"]["n_fail"])
    emit("lp_ms_per_solve", 1000 * lp_stats["lp"]["t_solve_s"] / max(1, lp_stats["lp"]["n_solves"]))
    emit("lp_solve_share_of_sim", lp_stats["lp"]["t_solve_s"] / max(1e-9, lp_stats["lp"]["t_wall_s"]))
    sk = {k: float(np.nanmean([skill[p][k] for p in pids])) for k in skill[pids[0]] if k != "n"}
    for f in ("da", "nowcast", "persistence"):
        emit(f"fc_mae_{f}_mwh", sk[f"{f}_all"])
        emit(f"fc_mae_{f}_le1h_mwh", sk[f"{f}_le1h"])

    with stage("battery_write"):
        out = output_frame(series, sims_final, costs_final, variants)
        out.write_parquet(out_dir / "park_battery.parquet")
    cfg = {"battery": asdict(bat), "a": a, "b": b, "policy": args.policy, "k_grid": ks, "k_chosen": chosen,
           "val_net_by_k": kval, "deadband_mwh": 0.01 * bat.step_mwh, "label": label,
           "nowcast_beta": {s.park_id: bt.round(4).tolist() for s, bt in zip(series, betas)},
           "lp_stats": lp_stats, "c_deg_note": "c_deg is an assumption (spec W step 7), not a measured wear cost",
           "energy_term": "pzu x (C - D): energy the battery takes out of the sale (losses, timing)"}
    for name, x in {"label": label, "results": table, "forecast_skill": skill, "config": cfg}.items():
        rep.table(name, x)
    rep.write()
    (out_dir / "park_battery_summary.json").write_text(json.dumps(
        {"label": label, "test": t, "holdout": table["holdout"], "k_chosen": chosen, "forecast_skill_mean": sk},
        default=float, indent=1))
    emit("t_battery_total_s", time.time() - t0)
    print(f"[w5] {label}, test, {t['none']['park_days']} park-days:", flush=True)
    for v in variants:
        r = t[v]
        print(f"  {v:10s} net {r['net'] / 1e3:10,.0f}k lei (imb {r['avoidable'] / 1e3:,.0f}k, energy "
              f"{r['energy'] / 1e3:,.0f}k, wear {r['wear'] / 1e3:,.0f}k) cut {r['cost_cut_lei'] / 1e3:,.0f}k lei "
              f"{r['cost_cut_pct']:.1f}% | cover {r['cover_share']:.3f} | {r['cycles_per_day']:.2f} cycles/d"
              + (f" | k={chosen[v]}" if v in chosen else ""), flush=True)
    print(f"[w5] forecast MAE MWh/step (rest of day, test): " + ", ".join(
        f"{f} {sk[f + '_all']:.3f} (<=1h {sk[f + '_le1h']:.3f})" for f in ("da", "nowcast", "persistence")), flush=True)
    return {"results": table, "config": cfg, "skill": skill}


if __name__ == "__main__":
    main()
