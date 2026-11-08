"""Card #168 W5 Wind D4: park battery (spec W step 7), none vs rule vs hourly-MPC LP."""

import json

import numpy as np
import polars as pl
import pytest

from wattsentry.opt.park_battery import (H, OUT_COLUMNS, Battery, MPCPlanner, Series, Variant, apply, fit_nowcast,
                                       horizon_end, limits, lp_u, main, origin_residuals, rule_u,
                                       series_from_positions, simulate, step_costs, summarize, synth_positions)

A, B = 1.44, 0.72
BAT = Battery()


def make_series(actual, sold, pzu=None, p50=None, ok=None, days=1, block=None, cap=200.0):
    T = len(actual)
    ts = np.datetime64("2025-03-01T00:00") + np.arange(T) * np.timedelta64(15, "m")
    pzu = np.full(T, 400.0) if pzu is None else np.asarray(pzu, float)
    return Series(park_id="WP1", cap_mwh_step=cap * 0.25, ts=ts, sold=np.asarray(sold, float),
                  sold_da=np.asarray(sold, float), sold_id=np.zeros(T), actual=np.asarray(actual, float),
                  p50=np.asarray(actual if p50 is None else p50, float), pzu=pzu, pi=pzu,
                  ok=np.ones(T, bool) if ok is None else ok, test=np.ones(T, bool), holdout=False,
                  sid=np.arange(T) % H, block=np.zeros(T, int) if block is None else block)


# ------------------------------------------------------------------------------------------------ physics
def test_limits_and_soc():
    e = BAT.e_hi - 1.0  # 1 MWh of room left
    lo, hi = limits(e, prod=50.0, bat=BAT)
    assert lo == pytest.approx(-1.0 / BAT.eta) and hi == pytest.approx(BAT.step_mwh)
    assert apply(e, lo, BAT) == pytest.approx(BAT.e_hi)
    lo, hi = limits(BAT.e_lo + 0.5, prod=2.0, bat=BAT)  # charge only from own output, discharge down to the floor
    assert lo == pytest.approx(-2.0) and hi == pytest.approx(0.5 * BAT.eta)
    assert apply(BAT.e_lo + 0.5, hi, BAT) == pytest.approx(BAT.e_lo)
    # round trip loses 1 - eta^2
    e1 = apply(apply(50.0, -10.0, BAT), 10.0 * BAT.eta ** 2, BAT)
    assert e1 == pytest.approx(50.0)


def test_rule_covers_gap_and_drifts_inside_deadband():
    e = BAT.soc0 * BAT.e_mwh
    assert rule_u(3.0, e, 20.0, BAT, 0.1) == pytest.approx(3.0)  # short 3: discharge 3
    assert rule_u(-4.0, e, 20.0, BAT, 0.1) == pytest.approx(-4.0)  # long 4: charge 4
    assert rule_u(50.0, e, 20.0, BAT, 0.1) == pytest.approx(BAT.step_mwh)  # power limit
    u = rule_u(0.02, BAT.e_hi, 20.0, BAT, 0.1)  # ~no gap, battery full: drift down, |dev| <= deadband
    assert u > 0.02 and abs(0.02 - u) <= 0.1 + 1e-12
    u = rule_u(-0.02, BAT.e_lo, 20.0, BAT, 0.1)  # empty: drift up
    assert u < -0.02 and abs(-0.02 - u) <= 0.1 + 1e-12


def test_lp_u_cover_only_never_widens():
    e = 50.0
    assert lp_u(2.0, 0.0, e, 10.0, BAT, True) == pytest.approx(2.0)  # plan covers all
    assert lp_u(2.0, 1.5, e, 10.0, BAT, True) == pytest.approx(0.5)  # plan leaves 1.5 short
    assert lp_u(-1.0, 3.0, e, 10.0, BAT, True) == pytest.approx(-1.0)  # expected short, actual long: absorb, no discharge
    assert lp_u(-1.0, 3.0, e, 10.0, BAT, False) == pytest.approx(-4.0)


# ------------------------------------------------------------------------------------------------ LP
def test_lp_saves_cheap_long_for_expensive_short():
    """4 steps: long 5 at PZU 100, nothing, short 4 at PZU 1000, nothing. Empty battery (at the floor): the LP charges
    just enough of the long (4 / eta^2) and covers the short; with a huge wear cost it does nothing."""
    bat = Battery(soc0=0.1)
    pl_ = MPCPlanner(bat, A, B, L=8)
    fc = np.array([10.0, 5.0, 1.0, 5.0])
    sold = np.array([5.0, 5.0, 5.0, 5.0])
    pzu = np.array([100.0, 300.0, 1000.0, 300.0])
    p = pl_.plan(fc, sold, pzu, bat.e_lo, k=0.0)
    assert p["C"][0] == pytest.approx(4.0 / bat.eta ** 2, abs=1e-6)
    assert p["D"][2] == pytest.approx(4.0, abs=1e-6)
    assert np.abs(p["dev"][2]) < 1e-6
    p = MPCPlanner(replace_cdeg(bat, 1e5), A, B, L=8).plan(fc, sold, pzu, bat.e_lo, k=0.0)
    assert np.allclose(p["C"], 0, atol=1e-6) and np.allclose(p["D"], 0, atol=1e-6)


def replace_cdeg(bat, c):
    from dataclasses import replace

    return replace(bat, c_deg=c)


def test_lp_cover_only_vs_free():
    """Long all day, low then high price: cover_only may only charge; free also discharges at the high price."""
    pl_ = MPCPlanner(BAT, A, B, L=8)
    fc = np.full(4, 10.0)
    sold = np.full(4, 8.0)
    pzu = np.array([50.0, 50.0, 2000.0, 2000.0])
    p = pl_.plan(fc, sold, pzu, 60.0, k=0.0, cover_only=True)
    assert np.allclose(p["D"], 0, atol=1e-6)
    p = pl_.plan(fc, sold, pzu, 60.0, k=0.0, cover_only=False)
    assert p["D"][2:].sum() > 1.0


def test_oracle_execution_equals_plan():
    """Perfect forecast: the real-time step is exactly the plan, and the LP is no worse than the rule."""
    rng = np.random.default_rng(1)
    T = 2 * H
    act = np.clip(20 + rng.normal(0, 4, T).cumsum() * 0.3, 0, 50)
    sold = act + rng.normal(0, 2, T)
    pzu = 300 + 400 * rng.random(T)
    s = make_series(act, sold, pzu)
    sims = {v.name: simulate(s, v, BAT, np.zeros((H, 2)), A, B) for v in
            (Variant("rule", "rule"), Variant("lp_oracle", "lp", forecast="oracle", k=1.0), Variant("none", "none"))}
    costs = {k: step_costs(s, v, BAT, A, B) for k, v in sims.items()}
    m = [np.ones(T, bool)]
    r = {k: summarize([costs[k]], [sims[k]], [s], m, BAT) for k in sims}
    assert r["none"]["cost_cut_lei"] == 0 and r["none"]["cover_share"] == 0
    assert r["lp_oracle"]["net"] <= r["rule"]["net"] + 1e-6
    assert 0 < r["rule"]["cover_share"] <= 1
    # accounting: net = avoidable + energy + wear
    c = costs["rule"]
    assert np.allclose(c["net"], c["avoidable"] + c["energy"] + c["wear"])
    # SoC within the window
    for v in sims.values():
        assert v["soc"].min() >= BAT.e_lo - 1e-9 and v["soc"].max() <= BAT.e_hi + 1e-9


def test_none_matches_w3_cost_and_idle_on_unscored():
    from wattsentry.eval.wind_portfolio import wind_cost_steps

    act = np.array([10.0, 12.0, 8.0, 9.0] * (H // 4))
    sold = act - 1.0
    ok = np.ones(H, bool)
    ok[5] = False
    s = make_series(act, sold, ok=ok)
    sim = simulate(s, Variant("none", "none"), BAT, np.zeros((H, 2)), A, B)
    c = step_costs(s, sim, BAT, A, B)
    assert np.allclose(c["avoidable"], wind_cost_steps(act, sold, np.zeros(H), s.pzu, s.pi, A, B)["avoidable"])
    sim = simulate(s, Variant("rule", "rule"), BAT, np.zeros((H, 2)), A, B)
    assert sim["u"][5] == 0 and sim["u"][4] < 0


def test_soc_restarts_after_gap_and_horizon():
    T = 2 * H
    s = make_series(np.full(T, 10.0), np.full(T, 5.0), block=np.r_[np.zeros(H, int), np.ones(H, int)])
    sim = simulate(s, Variant("rule", "rule"), BAT, np.zeros((H, 2)), A, B)
    first = BAT.soc0 * BAT.e_mwh + 5.0 * BAT.eta
    assert sim["soc"][0] == pytest.approx(first) and sim["soc"][H] == pytest.approx(first)
    assert horizon_end(s, 8, "day") == H
    s2 = make_series(np.full(T, 10.0), np.full(T, 5.0))
    assert horizon_end(s2, 4, "24h") == H  # before 12:00 UTC: tomorrow not known
    assert horizon_end(s2, 52, "24h") == 52 + H
    assert horizon_end(s, 52, "24h") == H  # next day is another block


# ------------------------------------------------------------------------------------------------ nowcast
def test_fit_nowcast_recovers_ar1():
    rng = np.random.default_rng(0)
    T, rho = 40 * H, 0.95
    r = np.zeros(T)
    for t in range(1, T):
        r[t] = rho * r[t - 1] + rng.normal()
    ok = np.ones(T, bool)
    block = np.zeros(T, int)
    beta = fit_nowcast(r, ok, block, ok)
    assert beta[0, 0] == pytest.approx(rho, abs=0.05) and abs(beta[0, 1]) < 0.1
    assert beta[9].sum() == pytest.approx(rho ** 10, abs=0.1)
    assert np.allclose(origin_residuals(r, ok, block, 0), 0)
    assert origin_residuals(r, ok, block, 8)[0] == r[7]


# ------------------------------------------------------------------------------------------------ data / end to end
def test_series_blocks_from_positions():
    df = synth_positions(days=4, parks=("WP1",))
    df = df.filter(pl.col("ts").dt.day() != 3)  # drop a day: the 4th starts a new block
    (s,) = series_from_positions(df)
    assert s.T == 3 * H and list(np.unique(s.block)) == [0, 1] and (s.block == 1).sum() == H
    assert s.sid[0] == 0 and s.sid[H - 1] == H - 1


def test_main_synth(tmp_path):
    r = main(["--synth", "--synth-days", "4", "--workers", "1", "--families", "lp,lp_oracle", "--k-grid", "0,1",
              "--out", str(tmp_path)])
    df = pl.read_parquet(tmp_path / "park_battery.parquet")
    assert tuple(df.columns) == OUT_COLUMNS
    assert set(df["variant"].unique()) == {"none", "rule", "lp", "lp_oracle"}
    t = r["results"]["test"]
    for v in ("rule", "lp", "lp_oracle"):
        assert 0 <= t[v]["cover_share"] <= 1 and t[v]["cycles_per_day"] >= 0
    assert t["none"]["net"] == pytest.approx(t["none"]["avoidable_none"])
    summ = json.loads((tmp_path / "park_battery_summary.json").read_text())
    assert set(summ["k_chosen"]) == {"lp", "lp_oracle"}
