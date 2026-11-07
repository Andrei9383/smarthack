"""Card #18: per-feeder LP, aggregate fallback and baseline policies on MockPredictor forecasts."""

from __future__ import annotations

import time

import numpy as np
import pytest

from sentinel.contracts import AssetStatus, BatterySpec, FeederPlan, FeederSpec, MockPredictor, self_consumption_step
from sentinel.opt.aggregate import plan_feeder_aggregate
from sentinel.opt.lp import LPParams, peak_mask_for, plan_feeder
from sentinel.opt.policies import (
    RestOfFeeder,
    Site,
    ValidationDay,
    evaluate_threshold,
    rest_of_feeder_from_forecasts,
    self_consumption_plan,
    simulate_rule,
    site_from_row,
    threshold_plan,
    tune_threshold,
)

SUMMER = "2025-06-21T00:00:00Z"
WINTER = "2025-01-15T00:00:00Z"
PV, LOAD = MockPredictor("pv"), MockPredictor("load")


def make_feeder(t0: str = SUMMER, n: int = 20, archetype: str = "base"):
    rows, fc = [], {}
    for k in range(n):
        sid = f"S{k:03d}"
        row = {
            "site_id": sid, "kwp_true": [6, 8, 10, 7][k % 4], "lat": 44.4, "archetype": archetype,
            "battery_kwh": [10.0, 13.5, 8.0, 10.0][k % 4], "battery_kw": [5.0, 5.0, 4.0, 3.6][k % 4],
            "comfort_reserve_kwh": 2.0, "soc": 0.3,
        }
        rows.append(row)
        fc[sid] = {"pv": PV.predict(row, t0=t0), "load": LOAD.predict(row, t0=t0)}
    return rows, fc


def worst_no_battery(fc) -> np.ndarray:
    return sum(np.asarray(f["pv"].q90) - np.asarray(f["load"].q10) for f in fc.values())


@pytest.fixture(scope="module")
def sunny():
    rows, fc = make_feeder()
    peak = float(worst_no_battery(fc).max())
    return rows, fc, peak


def assert_reserve_and_replay(plan, rows):
    """Comfort reserve never breached and the schedule replays exactly through the battery model (#27)."""
    for row in rows:
        sid = row["site_id"]
        if sid not in plan.schedule:
            continue
        site = site_from_row(row)
        b = site.battery
        soc = np.asarray(plan.soc_kwh[sid])
        assert soc.min() >= b.e_min - 1e-6, f"{sid} below reserve: {soc.min()} < {b.e_min}"
        assert soc.max() <= b.e_max + 1e-6
        p = np.asarray(plan.schedule[sid])
        assert np.abs(p).max() <= b.p_max_kw + 1e-6
        e = site.initial_energy()
        for t, pt in enumerate(p):
            applied, e = b.step(e, pt)
            assert applied == pytest.approx(pt, abs=1e-6)
            assert e == pytest.approx(soc[t + 1], abs=1e-6)


# --- LP ---------------------------------------------------------------------------------------------------
def test_lp_meets_limit_when_headroom_suffices(sunny):
    rows, fc, peak = sunny
    G = 0.7 * peak
    usable = sum(site_from_row(r).battery.e_max - site_from_row(r).battery.e_min for r in rows)
    excess = float(np.maximum(0, worst_no_battery(fc) - G).sum() * 0.25)
    assert excess < 0.7 * usable  # precondition: headroom suffices

    plan = plan_feeder("F1", rows, None, fc, G=G, prices=0.6)
    assert isinstance(plan, FeederPlan) and plan.policy == "sentinel" and plan.method == "lp"
    assert plan.solver_status in ("optimal", "optimal_inaccurate")
    assert plan.horizon == 96 and len(plan.export_curve) == 96 and set(plan.schedule) == {r["site_id"] for r in rows}
    assert plan.max_slack_kw == 0.0
    assert max(plan.worst_export_curve) <= G + 1e-4
    assert max(plan.export_curve) <= G + 1e-4  # P50 is below the worst case
    assert plan.simultaneous_steps == 0 and plan.extra["simultaneous_before_repair"] == 0
    assert plan.solve_time_s < 10
    assert_reserve_and_replay(plan, rows)
    plan.to_dict()  # serialisable


def test_baseline_exceeds_limit_same_day(sunny):
    rows, fc, peak = sunny
    G = 0.7 * peak
    base = self_consumption_plan("F1", rows, None, fc, G=G)
    lp = plan_feeder("F1", rows, None, fc, G=G, prices=0.6)
    assert base.policy == "self_consumption"
    assert base.max_slack_kw > 1.0 and base.over_limit_steps > 4
    assert lp.max_slack_kw < base.max_slack_kw
    assert_reserve_and_replay(lp, rows)


def test_slack_reported_when_headroom_insufficient(sunny):
    rows, fc, peak = sunny
    G = 0.3 * peak
    plan = plan_feeder("F1", rows, None, fc, G=G, prices=0.6)
    base = self_consumption_plan("F1", rows, None, fc, G=G)
    slack = np.asarray(plan.slack)
    assert slack.max() > 1.0
    np.testing.assert_allclose(slack, np.maximum(0, np.asarray(plan.worst_export_curve) - G), atol=1e-5)
    # Less energy and a lower peak above the limit than the baseline.
    assert slack.sum() < np.sum(base.slack) and slack.max() < base.max_slack_kw
    # The LP burns energy (charge + discharge at once) here; the repair removes it without binaries.
    assert plan.extra["simultaneous_before_repair"] > 0
    assert plan.simultaneous_steps == 0
    assert_reserve_and_replay(plan, rows)


def test_no_simultaneous_charge_discharge_in_lp_solution(sunny):
    rows, fc, peak = sunny
    for frac in (0.9, 0.6, 0.4):
        plan = plan_feeder("F1", rows, None, fc, G=frac * peak, prices=0.6)
        assert plan.simultaneous_steps == 0, frac


def test_evening_peak_shaved_in_winter():
    rows, fc = make_feeder(WINTER, archetype="heat_pump")
    lp = plan_feeder("F1", rows, None, fc, G=1e3, prices=0.6)
    base = self_consumption_plan("F1", rows, None, fc, G=1e3)
    assert lp.max_slack_kw == 0.0 and base.max_slack_kw == 0.0
    assert lp.peak_import_kw < 0.6 * base.peak_import_kw
    assert_reserve_and_replay(lp, rows)


def test_statuses_derate_fold_and_rest(sunny):
    rows, fc, peak = sunny
    G = 0.7 * peak
    statuses = {
        "S000": AssetStatus("S000", "derated", 0.5, "pv_fault:soiling", battery_ok=True),
        "S001": AssetStatus("S001", "excluded", 0.0, "stale_telemetry"),
        "S002": AssetStatus("S002", "derated", 0.7, "pv_fault:shading", battery_ok=False),
    }
    plan = plan_feeder(FeederSpec("F1", g_max_kw=G, g_conservative_kw=G), rows, None, fc, prices=0.6,
                       statuses=statuses)
    assert plan.g_kw == pytest.approx(G)
    assert "S000" in plan.schedule
    assert set(plan.folded_sites) == {"S001", "S002"} and "S001" not in plan.schedule
    assert plan.n_sites == len(rows) - 2
    assert plan.max_slack_kw == 0.0
    # The derated site's PV is cut: total planned export is lower than with no statuses.
    full = plan_feeder("F1", rows, None, fc, G=G, prices=0.6)
    assert sum(plan.worst_export_curve) < sum(full.worst_export_curve)


def test_rest_of_feeder_enters_the_limit(sunny):
    rows, fc, peak = sunny
    eligible, others = rows[:14], rows[14:]
    rest = rest_of_feeder_from_forecasts(others, fc)
    assert max(rest.worst_kw) > max(rest.net_kw) > 0
    G = 0.7 * peak
    plan = plan_feeder("F1", eligible, rest, fc, G=G, prices=0.6)
    assert set(plan.schedule) == {r["site_id"] for r in eligible}
    assert max(plan.worst_export_curve) <= G + 1e-4 + plan.max_slack_kw
    # Ignoring the rest would plan against a looser feeder: the plain-sequence form shifts the curve 1:1.
    shifted = plan_feeder("F1", eligible, [10.0] * 96, fc, G=G + 10.0, prices=0.6)
    no_rest = plan_feeder("F1", eligible, None, fc, G=G, prices=0.6)
    np.testing.assert_allclose(np.asarray(shifted.worst_export_curve) - 10.0, no_rest.worst_export_curve, atol=1e-3)


def test_no_controllable_sites_returns_rule_curve(sunny):
    rows, fc, peak = sunny
    st = {r["site_id"]: AssetStatus(r["site_id"], "human_review", 0.0, "ood") for r in rows}
    plan = plan_feeder("F1", rows, None, fc, G=0.7 * peak, prices=0.6, statuses=st)
    base = self_consumption_plan("F1", rows, None, fc, G=0.7 * peak)
    assert plan.solver_status == "no_controllable_sites" and plan.schedule == {}
    np.testing.assert_allclose(plan.export_curve, base.export_curve, atol=1e-9)


# --- aggregate fallback -----------------------------------------------------------------------------------
def test_aggregate_fallback_feasible(sunny):
    rows, fc, peak = sunny
    G = 0.7 * peak
    agg = plan_feeder_aggregate("F1", rows, None, fc, G=G, prices=0.6)
    assert agg.method == "aggregate" and agg.policy == "sentinel"
    assert agg.max_slack_kw <= 1e-6
    assert max(agg.worst_export_curve) <= G + 1e-4
    assert agg.extra["split_shortfall_kw"] < 1e-6
    assert_reserve_and_replay(agg, rows)
    tight = plan_feeder_aggregate("F1", rows, None, fc, G=0.3 * peak, prices=0.6)
    assert tight.max_slack_kw > 0
    assert_reserve_and_replay(tight, rows)


# --- policies ---------------------------------------------------------------------------------------------
def test_rule_simulation_matches_contract_step():
    b = BatterySpec(capacity_kwh=10, p_max_kw=4, comfort_reserve_kwh=2)
    pv = np.r_[np.zeros(30), np.full(30, 6.0), np.zeros(36)]
    load = np.full(96, 1.0)
    batt, e, grid = simulate_rule(b, 3.0, pv, load)
    ek = 3.0
    for t in range(96):
        bk, ek_next, gk = self_consumption_step(b, ek, pv[t], load[t])
        assert (batt[t], e[t + 1], grid[t]) == pytest.approx((bk, ek_next, gk))
        ek = ek_next


def test_threshold_rule_and_tuning(sunny):
    rows, fc, peak = sunny
    G = 0.7 * peak
    days = []
    for t0 in ("2025-06-20T00:00:00Z", SUMMER):
        r, f = make_feeder(t0)
        days.append(ValidationDay(
            sites=r, pv={k: v["pv"].q90 for k, v in f.items()}, load={k: v["load"].q10 for k, v in f.items()},
            g_kw=G, forecast_pv={k: v["pv"].q50 for k, v in f.items()},
            forecast_load={k: v["load"].q50 for k, v in f.items()},
        ))
    tuned = tune_threshold(days, alphas=[0.0, 0.2, 0.4, 0.6, 0.8])
    assert tuned.alpha > 0.0  # alpha = 0 is the baseline rule; a congested feeder wants a threshold
    assert tuned.scores[tuned.alpha]["over_kwh"] < tuned.scores[0.0]["over_kwh"]
    assert evaluate_threshold(days[0], 0.0)["over_kwh"] > 0
    th = threshold_plan("F1", rows, None, fc, G=G, alpha=tuned.alpha)
    base = self_consumption_plan("F1", rows, None, fc, G=G)
    assert th.policy == "threshold" and sum(th.slack) < sum(base.slack)
    assert_reserve_like_rule(th, rows)


def assert_reserve_like_rule(plan, rows):
    """Rule plans may use the battery down to soc_min (the rule serves load), never below it."""
    for r in rows:
        b = site_from_row(r).battery
        assert min(plan.soc_kwh[r["site_id"]]) >= b.soc_min * b.capacity_kwh - 1e-9


# --- helpers ----------------------------------------------------------------------------------------------
def test_peak_mask_local_time():
    summer = peak_mask_for(SUMMER, 96)  # UTC+3: 17:00 local = 14:00 UTC = step 56
    assert np.flatnonzero(summer)[[0, -1]].tolist() == [56, 75]
    winter = peak_mask_for(WINTER, 96)  # UTC+2
    assert np.flatnonzero(winter)[[0, -1]].tolist() == [60, 79]


def test_site_from_row_and_inputs():
    s = site_from_row({"site_id": "X", "battery_kwh": 10, "battery_kw": 5, "comfort_reserve_kwh": 3, "soc": 0.1})
    assert s.battery.e_min == 3 and s.initial_energy() == pytest.approx(1.0)
    assert not site_from_row({"site_id": "Y", "battery_kwh": 0}).has_battery
    with pytest.raises(ValueError):
        LPParams(big_m=1.0, lam=10.0)
    with pytest.raises(ValueError):
        RestOfFeeder([0.0] * 3, [0.0] * 4)


def test_battery_below_reserve_at_start_is_feasible(sunny):
    rows, fc, peak = sunny
    sites = [Site(r["site_id"], site_from_row(r).battery, e0_kwh=0.5) for r in rows]  # below the 2 kWh reserve
    plan = plan_feeder("F1", sites, None, fc, G=0.7 * peak, prices=0.6)
    for s in sites:
        assert min(plan.soc_kwh[s.site_id]) >= 0.5 - 1e-6
        assert plan.soc_kwh[s.site_id][-1] >= 0.5 - 1e-6


def test_lp_scales_to_60_sites():
    rows, fc = make_feeder(n=60)
    G = 0.7 * float(worst_no_battery(fc).max())
    t = time.perf_counter()
    plan = plan_feeder("F1", rows, None, fc, G=G, prices=0.6)
    assert time.perf_counter() - t < 20
    assert plan.max_slack_kw == 0.0 and plan.simultaneous_steps == 0


# --- negative prices (card #80) ---------------------------------------------------------------------------
# Real OPCOM PZU day-ahead prices (lei/MWh, hourly, data/raw/prices.parquet) for the local days 2025-06-07 and
# 2025-06-08 of the demo week: negative at midday. The merged LP was unbounded on both.
OPCOM_0607 = ("2025-06-06T21:00:00Z", [
    524.73, 502.85, 485.74, 486.39, 462.48, 455.89, 410.5, 407.25, 372.43, 300.6, 69.28, 47.73, 6.0, -0.03, -0.08,
    -0.09, 25.71, 450.83, 685.46, 990.68, 1138.9, 825.06, 623.22, 598.59])
OPCOM_0608 = ("2025-06-07T21:00:00Z", [
    484.1, 446.87, 435.59, 424.11, 398.14, 332.44, 229.1, 20.22, 0.01, -2.52, -10.08, -16.79, -86.42, -129.34,
    -163.67, -30.0, 72.86, 461.76, 795.67, 763.16, 895.58, 801.81, 628.82, 508.96])


def negative_day(day):
    t0, hourly = day
    rows, fc = make_feeder(t0)
    prices = np.repeat(np.asarray(hourly) / 1000.0, 4)  # lei/kWh per 15 min step, as the replay passes them
    return rows, fc, prices, t0


def lp_arrays(rows, fc, G, prices, t0):
    from sentinel.opt.policies import site_quantiles

    sites = [site_from_row(r) for r in rows]
    q = {s.site_id: site_quantiles(fc, s.site_id) for s in sites}
    bats = [s.battery for s in sites]
    e0 = np.array([s.initial_energy() for s in sites])
    return dict(
        net50=np.stack([q[s.site_id]["pv50"] - q[s.site_id]["l50"] for s in sites]),
        worst=np.stack([q[s.site_id]["pv90"] - q[s.site_id]["l10"] for s in sites]),
        p_max=np.array([b.p_max_kw for b in bats]), e_lo=np.minimum([b.e_min for b in bats], e0),
        e_hi=np.array([b.e_max for b in bats]), e0=e0, eta_c=np.array([b.eta_charge for b in bats]),
        eta_d=np.array([b.eta_discharge for b in bats]), rest_worst=np.zeros(96), rest_net=np.zeros(96),
        g_kw=G, prices=prices, peak_mask=peak_mask_for(t0, 96),
    )


@pytest.mark.parametrize("day", [OPCOM_0607, OPCOM_0608], ids=["2025-06-07", "2025-06-08"])
def test_negative_price_day_lp_optimal(day):
    rows, fc, prices, t0 = negative_day(day)
    assert prices.min() < 0
    G = 0.7 * float(worst_no_battery(fc).max())
    plan = plan_feeder("F1", rows, None, fc, G=G, prices=prices)
    assert plan.method == "lp" and plan.solver_status == "optimal"
    assert plan.max_slack_kw == 0.0 and max(plan.worst_export_curve) <= G + 1e-4  # headroom suffices
    assert plan.simultaneous_steps == 0
    assert np.isfinite(plan.objective) and plan.solve_time_s < 10
    assert_reserve_and_replay(plan, rows)
    for row in rows:  # end-of-day energy kept (e_T >= e0)
        assert plan.soc_kwh[row["site_id"]][-1] >= site_from_row(row).initial_energy() - 1e-6


def test_negative_price_lp_flows_are_physical():
    """One net flow per site: no simultaneous import + export, and the LP objective is the real cost of the plan
    (nothing earned from phantom import/export pairs)."""
    from sentinel.opt.lp import solve_lp

    rows, fc, prices, t0 = negative_day(OPCOM_0608)
    G = 0.7 * float(worst_no_battery(fc).max())
    args = lp_arrays(rows, fc, G, prices, t0)
    params = LPParams()
    sol = solve_lp(**args, params=params)
    assert sol.status == "optimal"
    imp, exp = sol.import_kw, sol.export_kw
    np.testing.assert_allclose(exp - imp, args["net50"] - sol.c + sol.d, atol=1e-9)
    assert (np.minimum(imp, exp) <= params.simultaneous_tol_kw).all()
    # objective = Σ π·imp·Δt − β_t·π·exp·Δt + c_deg·(c + d)·Δt + M·Σ s·Δt + M·w·max s + λ·z, β_t = 1 when π < 0
    beta_t = np.where(prices >= 0, params.beta, params.beta_negative)
    cost = 0.25 * float(np.sum(prices * imp - beta_t * prices * exp) + params.c_deg * np.sum(sol.c + sol.d)
                        + params.big_m * sol.s.sum()) + params.big_m * params.slack_peak_weight * sol.s.max() \
        + params.lam * sol.z
    assert sol.objective == pytest.approx(cost, rel=1e-6, abs=1e-4)
    # Charging during negative prices is the point: the batteries take energy while π < 0.
    neg = prices < 0
    assert sol.c[:, neg].sum() * 0.25 > 50.0


def test_negative_prices_same_as_before_when_prices_nonnegative(sunny):
    """π >= 0: the net-flow LP is the merged m/x LP after a change of variables (same optimum)."""
    from sentinel.opt.lp import solve_lp

    rows, fc, peak = sunny
    prices = np.r_[np.full(40, 0.3), np.zeros(30), np.full(26, 1.2)]
    args = lp_arrays(rows, fc, 0.7 * peak, prices, SUMMER)
    sol = solve_lp(**args)
    # Same optimum as the merged formulation on this fixture (recorded before the change).
    assert sol.objective == pytest.approx(6.681244, abs=1e-5)


def test_negative_price_day_aggregate_fallback():
    rows, fc, prices, t0 = negative_day(OPCOM_0608)
    G = 0.7 * float(worst_no_battery(fc).max())
    agg = plan_feeder_aggregate("F1", rows, None, fc, G=G, prices=prices)
    assert agg.method == "aggregate" and agg.solver_status == "optimal"
    assert agg.max_slack_kw <= 1e-6 and max(agg.worst_export_curve) <= G + 1e-4
    # The virtual battery pools energy and power, so the last full-power charge step at negative prices can't be
    # split exactly (sites with room are at their power cap); the applied schedule (and its slack) is what counts.
    assert agg.extra["split_shortfall_kw"] < 0.2 * sum(site_from_row(r).battery.p_max_kw for r in rows)
    assert_reserve_and_replay(agg, rows)


def test_grid_limit_and_beta_negative_params():
    rows, fc, prices, t0 = negative_day(OPCOM_0608)
    G = 0.7 * float(worst_no_battery(fc).max())
    lim = 2.0
    plan = plan_feeder("F1", rows, None, fc, G=G, prices=prices, params=LPParams(grid_limit_kw=lim))
    assert plan.solver_status == "optimal"
    for r in rows:  # the battery never pushes the connection beyond max(limit, |forecast net|)
        sid = r["site_id"]
        net50 = np.asarray(fc[sid]["pv"].q50) - np.asarray(fc[sid]["load"].q50)
        g = net50 - np.asarray(plan.schedule[sid])
        assert (np.abs(g) <= np.maximum(lim, np.abs(net50)) + 1e-5).all()
    with pytest.raises(ValueError):
        LPParams(beta_negative=0.9)
    with pytest.raises(ValueError):
        LPParams(grid_limit_kw=0.0)


# --- supplier objective (card G4 #162) ---------------------------------------------------------------------------
def rule_import_kw(rows, fc) -> np.ndarray:
    """Feeder P50 net import under the self-consumption rule: a supplier position bought without Sentinel."""
    return -np.asarray(rest_of_feeder_from_forecasts([site_from_row(r) for r in rows], fc, None, 96).net_kw)


def test_supplier_off_is_the_default_lp(sunny):
    """No supplier term -> the #18 LP exactly (replay#506 KPIs unchanged): same plan as an explicit supplier=None,
    no supplier diagnostics, and the recorded #18 optimum."""
    from sentinel.opt.lp import solve_lp

    rows, fc, peak = sunny
    a = plan_feeder("F", rows, None, fc, G=0.7 * peak, prices=0.6)
    b = plan_feeder("F", rows, None, fc, G=0.7 * peak, prices=0.6, supplier=None, params=LPParams())
    assert a.schedule == b.schedule and a.slack == b.slack and a.objective == b.objective
    assert not any(k.startswith(("supplier", "stage1")) for k in a.extra)
    prices = np.r_[np.full(40, 0.3), np.zeros(30), np.full(26, 1.2)]
    sol = solve_lp(**lp_arrays(rows, fc, 0.7 * peak, prices, SUMMER), supplier=None)
    assert sol.supplier_lei is None and sol.stage1_slack is None
    assert sol.objective == pytest.approx(6.681244, abs=1e-5)  # same recorded optimum as above


@pytest.mark.parametrize("frac", [0.7, 0.4], ids=["headroom", "congested"])
def test_supplier_term_keeps_limit_and_reserve_hard(sunny, frac):
    from sentinel.opt.lp import SupplierObjective

    rows, fc, peak = sunny
    G = frac * peak
    dso = plan_feeder("F", rows, None, fc, G=G, prices=0.6)
    sup = SupplierObjective(rule_import_kw(rows, fc), a=1.44, b=0.72)
    plan = plan_feeder("F", rows, None, fc, G=G, prices=0.6, supplier=sup)
    assert plan.solver_status == "optimal"
    # feeder limit hard: no step's slack grows (lexicographic stage 2), comfort reserve hard
    assert (np.asarray(plan.slack) <= np.asarray(dso.slack) + 1e-3).all()
    assert plan.max_slack_kw <= dso.max_slack_kw + 1e-3
    if frac == 0.4:
        assert dso.max_slack_kw > 0  # the limit really binds here
    assert_reserve_and_replay(plan, rows)
    # the supplier pays less for the plan it is scored on
    from sentinel.opt.lp import _supplier_extra

    w = sup.arrays(96, np.full(96, 0.6))
    assert plan.extra["supplier_planned_lei"] < _supplier_extra(dso, w, None)["supplier_planned_lei"] - 1.0
    assert plan.extra["supplier_planned_lei"] == pytest.approx(plan.extra["supplier_lp_lei"], rel=1e-3, abs=0.05)


def test_supplier_only_tracks_a_feasible_position(sunny):
    """bill off, position = the import curve of a feasible plan: the LP can match it, supplier cost ~ 0."""
    from sentinel.opt.lp import SupplierObjective

    rows, fc, peak = sunny
    G = 0.7 * peak
    dso = plan_feeder("F", rows, None, fc, G=G, prices=0.6)
    pos = -np.asarray(dso.export_curve)
    plan = plan_feeder("F", rows, None, fc, G=G, prices=0.6, params=LPParams(bill_weight=0.0),
                       supplier=SupplierObjective(pos, a=1.44, b=0.72, weight=1.0))
    # ~0: the per-site near-limit guard (#188) leaves stage 2 a little less room, so it trades 0.23 kWh of deviation
    # (0.1 lei) for less battery throughput (c_deg); without the guard the match is exact
    assert plan.extra["supplier_planned_lei"] < 0.15
    assert plan.extra["supplier_planned_abs_dev_kwh"] < 1e-3 * 0.25 * np.abs(pos).sum()
    assert plan.max_slack_kw <= 1e-6
    free = plan_feeder("F", rows, None, fc, G=G, prices=0.6,
                       params=LPParams(bill_weight=0.0, supplier_guard_margin=None),
                       supplier=SupplierObjective(pos, a=1.44, b=0.72, weight=1.0))
    assert free.extra["supplier_planned_lei"] < 1e-6


def test_supplier_single_solve_and_validation(sunny):
    from sentinel.opt.lp import SupplierObjective, supplier_cost_lei

    rows, fc, peak = sunny
    pos = rule_import_kw(rows, fc)
    plan = plan_feeder("F", rows, None, fc, G=0.4 * peak, prices=0.6,
                       supplier=SupplierObjective(pos, a=1.44, b=0.72),
                       params=LPParams(supplier_lexicographic=False))
    dso = plan_feeder("F", rows, None, fc, G=0.4 * peak, prices=0.6)
    assert "stage1_slack_max_kw" not in plan.extra
    assert plan.max_slack_kw <= dso.max_slack_kw + 1e-3  # M >> supplier weights does the same job here
    with pytest.raises(ValueError, match="horizon"):
        plan_feeder("F", rows, None, fc, G=peak, prices=0.6, supplier=SupplierObjective(pos[:95]))
    with pytest.raises(ValueError, match="big_m"):
        plan_feeder("F", rows, None, fc, G=peak, prices=0.6, params=LPParams(supplier_lexicographic=False),
                    supplier=SupplierObjective(pos, a=1e4, b=1.0))
    with pytest.raises(ValueError):
        LPParams(bill_weight=-1.0)
    # X1 units: MWh per 15-min step and lei/MWh -> kW and lei/kWh
    s = SupplierObjective.from_bought_da([0.001] * 96, pzu_lei_mwh=[500.0] * 96, a=2.0, b=1.0)
    target, ws, wl = s.arrays(96, np.full(96, 9.9))
    assert target == pytest.approx(np.full(96, 4.0)) and ws == pytest.approx(np.full(96, 1.0))
    assert wl == pytest.approx(np.full(96, 0.5))
    # hand case: 2 kW short for 1 step at 1 lei/kWh, 4 kW long for 1 step at 0.5 lei/kWh
    lei, dev = supplier_cost_lei([6.0, 0.0, 4.0], [4.0, 4.0, 4.0], [1.0] * 3, [0.5] * 3)
    assert lei == pytest.approx(0.25 * (2 * 1.0 + 4 * 0.5)) and dev == pytest.approx(0.25 * 6)


def test_supplier_replay_mock_one_day():
    """The 'replay with both objectives' runner on a 1-day mock world: minutes over limit do not increase."""
    from sentinel.opt.supplier_replay import run

    res = run("mock", days=1, seed=0, variants=("dso", "sup"))
    v = res["variants"]
    assert v["sup"]["minutes_over_limit"] <= v["dso"]["minutes_over_limit"]
    assert v["sup"]["comfort_breaches"] <= v["dso"]["comfort_breaches"]
    assert v["sup"]["supplier_planned_lei"] < v["dso"].get("supplier_planned_lei", np.inf)
    assert np.isfinite(v["sup"]["supplier_avoidable_lei"])


def test_supplier_guard_keeps_charge_at_near_limit_steps(sunny):
    """Stage 2 charges the feeder's batteries at least as much as the DSO-only LP wherever the no-battery worst
    export is within supplier_guard_margin of G, so realised export there cannot rise (replay#882 lesson)."""
    from sentinel.opt.lp import SupplierObjective

    rows, fc, peak = sunny
    G = 0.4 * peak
    dso = plan_feeder("F", rows, None, fc, G=G, prices=0.6)
    sup = SupplierObjective(rule_import_kw(rows, fc), a=1.44, b=0.72)
    plan = plan_feeder("F", rows, None, fc, G=G, prices=0.6, supplier=sup)
    risk = worst_no_battery(fc) >= 0.75 * G
    assert risk.any()
    p_dso = sum(np.asarray(v) for v in dso.schedule.values())
    p_sup = sum(np.asarray(v) for v in plan.schedule.values())
    assert (p_sup[risk] >= p_dso[risk] - 1e-3).all()
    assert (np.asarray(plan.worst_export_curve)[risk] <= np.asarray(dso.worst_export_curve)[risk] + 1e-3).all()
    with pytest.raises(ValueError):
        LPParams(supplier_guard_margin=1.5)


def _realised_export(plan, rows, fc, drop=None, k0=0):
    """Feeder export (kW) when the weather is the P50 and every pool site follows the plan, except `drop`, which
    leaves the pool at step k0 (e.g. a battery_fault exclusion) and runs the self-consumption rule from there."""
    x = np.zeros(plan.horizon)
    for r in rows:
        sid, b = r["site_id"], site_from_row(r).battery
        pv, ld = np.asarray(fc[sid]["pv"].q50), np.asarray(fc[sid]["load"].q50)
        p = np.asarray(plan.schedule[sid], float).copy()
        if sid == drop:
            e = plan.soc_kwh[sid][k0]
            for k in range(k0, plan.horizon):
                p[k], e, _ = self_consumption_step(b, e, pv[k], ld[k])
        x += pv - ld - p
    return x


@pytest.mark.parametrize("bill_weight", [1.0, 0.0], ids=["sup", "sup_only"])
def test_supplier_guard_survives_a_site_leaving_the_pool(sunny, bill_weight):
    """#188 (replay#892: sup_only +15 min on F7 day 2). The pooled guard (Σ_i charge >= DSO at near-limit steps) let
    the supplier plan spread the same feeder kW differently over the sites, fill them earlier, and end the day
    fuller. When one site then left the pool mid-day (battery_fault -> its own rule), the others charged less than
    under the DSO plan and realised export went over G. With a per-site guard (charge >= and SoC <= the DSO-only
    stage 1, also at the end of the day) realised export at near-limit steps can never exceed the DSO plan's,
    whichever single site drops out, also with the bill term off. (Old guard here: 39-40 of 40 dropout cases worse,
    up to +7.5 kW, and the day ended 85 vs 62 kWh in the batteries.)"""
    from sentinel.opt.lp import SupplierObjective

    rows, fc, peak = sunny
    G = 0.4 * peak
    dso = plan_feeder("F", rows, None, fc, G=G, prices=0.6)
    plan = plan_feeder("F", rows, None, fc, G=G, prices=0.6, params=LPParams(bill_weight=bill_weight),
                       supplier=SupplierObjective(rule_import_kw(rows, fc), a=1.44, b=0.72))
    risk = worst_no_battery(fc) >= 0.75 * G
    ri = np.flatnonzero(risk)
    last = ri[-1] + 1
    for r in rows:
        sid = r["site_id"]
        assert (np.asarray(plan.schedule[sid])[risk] >= np.asarray(dso.schedule[sid])[risk] - 1e-3).all(), sid
        assert (np.asarray(plan.soc_kwh[sid])[1:last + 1] <= np.asarray(dso.soc_kwh[sid])[1:last + 1] + 1e-3).all()
        assert plan.soc_kwh[sid][-1] <= dso.soc_kwh[sid][-1] + 1e-3  # tomorrow starts no fuller
    for k0 in (int(ri[0]), int(ri[len(ri) // 2])):
        for r in rows:
            x_dso = _realised_export(dso, rows, fc, r["site_id"], k0)
            x_sup = _realised_export(plan, rows, fc, r["site_id"], k0)
            assert (x_sup[risk] <= x_dso[risk] + 1e-3).all(), (r["site_id"], k0)
    from sentinel.opt.lp import _supplier_extra  # the supplier still pays less than for the DSO plan

    w = SupplierObjective(rule_import_kw(rows, fc), a=1.44, b=0.72).arrays(96, np.full(96, 0.6))
    assert plan.extra["supplier_planned_lei"] < _supplier_extra(dso, w, None)["supplier_planned_lei"]
