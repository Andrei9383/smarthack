"""Card #166 W3 Wind D1: how much to sell tomorrow + avoidable cost (spec W step 4).

"Done when" (spec): the perfect forecast costs 0 lei, and the signs are checked on a small case computed by hand."""

import json

import numpy as np
import pytest

from sentinel.eval.wind_portfolio import (H, conformal_level, persistence_index, portfolio_metrics, positions_frame,
                                          promised_not_delivered, residual_offsets, sell_level, stopped_share,
                                          wind_cost_steps)

A, B = 0.5, 0.25
PZU = np.array([400.0, 500.0, 600.0, -100.0, 300.0])
PI = np.array([400.0, 500.0, 700.0, -100.0, 250.0])


def test_perfect_forecast_costs_zero():
    prod = np.array([10.0, 3.0, 0.0, 7.5, 1.0])
    r = portfolio_metrics(prod, prod, np.zeros(5), PZU, PI, a=A, b=B)
    assert r["wind_avoidable_cost_lei"] == 0 and r["abs_dev_mwh"] == 0 and r["lei_per_mwh_produced"] == 0
    # perfect after the PI correction (day-ahead off, PI fixes it) at pi = pzu: still 0
    r = portfolio_metrics(prod, prod + 2.0, np.full(5, -2.0), PZU, PZU, a=A, b=B)
    assert r["wind_avoidable_cost_lei"] == pytest.approx(0, abs=1e-9) and r["abs_dev_mwh"] == pytest.approx(0)


def test_signs_by_hand():
    """Five quarter-hours, producer side, a = 0.5, b = 0.25 (lei/MWh = factor x |PZU|).
    0: produced 10, sold 12 on PZU             -> short 2 (sold what it did not make): 0.5 x 400 x 2  = 400 lei
    1: produced 10, sold 8                     -> long 2 (made more than sold):        0.25 x 500 x 2 = 250 lei
    2: produced 10, sold 12, bought 2 back on PI at 700 (PZU 600) -> balanced; paid 2 x (700 - 600) = 200 lei
    3: produced 0 (turbines stopped), sold 3, PZU -100 -> short 3, |PZU| (X1):        0.5 x 100 x 3  = 150 lei
    4: produced 5, sold 4 on PZU + 1 more on PI at 250 (PZU 300) -> balanced; lost 1 x (300 - 250) = 50 lei
    """
    prod = np.array([10.0, 10.0, 10.0, 0.0, 5.0])
    da = np.array([12.0, 8.0, 12.0, 3.0, 4.0])
    idt = np.array([0.0, 0.0, -2.0, 0.0, 1.0])
    s = wind_cost_steps(prod, da, idt, PZU, PI, A, B)
    assert np.allclose(s["dev"], [2, -2, 0, 3, 0])  # sold - produced: > 0 short
    assert np.allclose(s["short"], [2, 0, 0, 3, 0]) and np.allclose(s["long"], [0, 2, 0, 0, 0])
    assert np.allclose(s["imbalance"], [400, 250, 0, 150, 0])
    assert np.allclose(s["intraday"], [0, 0, 200, 0, 50])
    assert np.allclose(s["avoidable"], [400, 250, 200, 150, 50])
    r = portfolio_metrics(prod, da, idt, PZU, PI, a=A, b=B)
    assert r["wind_avoidable_cost_lei"] == pytest.approx(1050)
    assert r["lei_per_mwh_produced"] == pytest.approx(1050 / 35)
    assert r["abs_dev_mwh"] == pytest.approx(7) and r["short_mwh"] == pytest.approx(5) and r["long_mwh"] == 2
    # same as revenue lost against selling everything produced on PZU, plus the imbalance penalty (hand cash flow)
    revenue = (da * PZU).sum() + (idt * PI).sum()  # what the producer is paid on PZU and PI
    s_dev = da + idt - prod
    settle = -(np.clip(s_dev, 0, None) * (PZU + A * np.abs(PZU))).sum() + (np.clip(-s_dev, 0, None) *
                                                                             (PZU - B * np.abs(PZU))).sum()
    assert (prod * PZU).sum() - (revenue + settle) == pytest.approx(1050)
    m = np.array([True, False, True, False, False])
    assert portfolio_metrics(prod, da, idt, PZU, PI, mask=m, a=A, b=B)["wind_avoidable_cost_lei"] == pytest.approx(600)


def test_sell_level_is_the_production_newsvendor():
    assert sell_level(1.0, 1.0) == pytest.approx(0.5)
    assert sell_level(2.0, 1.0) == pytest.approx(1 / 3)  # short costs twice: sell P33 of production, not P67
    rng = np.random.default_rng(0)
    prod = rng.gamma(2.0, 5.0, 20000)
    grid = np.linspace(0, 40, 801)
    cost = [wind_cost_steps(prod, s, None, 100.0, None, 2.0, 1.0)["avoidable"].mean() for s in grid]
    best = grid[int(np.argmin(cost))]
    assert best == pytest.approx(np.quantile(prod, sell_level(2.0, 1.0)), abs=0.1)
    assert abs(best - np.quantile(prod, 2 / 3)) > 2  # the consumption level would be the wrong side


def test_frozen_ab_level():
    from sentinel.eval.imbalance import load_ab, q_star

    a, b = load_ab()
    assert sell_level() == pytest.approx(b / (a + b)) == pytest.approx(1 - q_star())


def test_promised_but_not_delivered():
    w = np.array([[100.0, 100.0, 200.0], [0.0, 0.0, 0.0], [50.0, 0.0, 50.0]])
    st = np.array([[True, False, False], [True, True, True], [False, False, True]])
    sh = stopped_share(w, st)
    assert np.allclose(sh, [0.25, 0.0, 0.5])
    assert np.allclose(promised_not_delivered(np.array([8.0, 1.0, 2.0]), sh), [2.0, 0.0, 1.0])
    # buying back on PI what the stopped turbines were promised removes it; never below 0
    assert np.allclose(promised_not_delivered(np.array([8.0, 1, 2]), sh, np.array([-8.0, 0, -4]), np.array([.25, 0, .5])),
                       [0.0, 0.0, 0.0])
    # a policy whose expected production already excludes the stopped turbine promises nothing on it
    assert stopped_share(np.array([0.0, 100.0]), np.array([True, False])) == 0


def test_conformal_level_matches_calibrated_scenarios():
    from sentinel.model.windpower.scenarios import ParkConformal, scen_quantiles

    rng = np.random.default_rng(1)
    scen = np.clip(rng.normal(0.4, 0.1, (400, 96)), 0, 1)
    tau = 1 / 3
    raw = np.concatenate([scen_quantiles(scen), np.quantile(scen, tau, axis=0)[:, None]], -1)
    cal = ParkConformal(1.3, 0.8)
    q, lvl = conformal_level(cal, raw, "WP1")
    q2, sc2 = cal.apply(raw[:, :3], 1.0, "WP1", scen=scen)
    assert np.allclose(q, q2)
    assert np.allclose(lvl, np.quantile(sc2, tau, axis=0), atol=2e-3)
    assert np.all(lvl <= q[:, 1] + 1e-12) and np.all(lvl >= q[:, 0] - 1e-12)


def test_persistence_uses_only_measured_steps():
    d0 = 10 * H
    i0 = d0 - 56  # D-1 10:00
    lag = persistence_index(d0, i0)
    assert np.all(lag < i0)
    assert np.all(lag[:40] == d0 + np.arange(40) - H) and np.all(lag[40:] == d0 + np.arange(40, H) - 2 * H)


def test_residual_offsets():
    y = np.array([1.0, 2.0, 3.0, 4.0, np.nan])
    p = np.zeros(5)
    off = residual_offsets(p, y, np.array([True, True, True, True, True]), (0.0, 0.5, 1.0))
    assert np.allclose(off, [1.0, 2.5, 4.0])


def test_positions_frame_totals_match_metrics():
    pl = pytest.importorskip("polars")
    rng = np.random.default_rng(2)
    D, P = 2, 2
    ts = (np.datetime64("2025-03-01T00:00") + np.arange(D * H).reshape(D, H) * np.timedelta64(15, "m"))
    act = rng.uniform(0, 5, (D, H, P))
    pzu = rng.uniform(100, 500, (D, H))
    rows = {}
    for name, noise in (("windpower_only", 1.0), ("perfect", 0.0)):
        sd = np.clip(act + noise * rng.normal(0, 1, act.shape), 0, None)
        rows[name] = {"sold_da": sd, "sold_id": np.zeros_like(act), "fc": np.repeat(sd[..., None], 3, -1),
                      "pbnd": np.zeros_like(act)}
    extra = {"score_ok": np.ones((D, H, P), bool)}
    df = positions_frame(ts, ["WP1", "WP2"], np.array([50.0, 60.0]), rows, act, pzu, pzu, extra, A, B)
    assert len(df) == 2 * D * H * P and set(df["policy"].unique()) == {"windpower_only", "perfect"}
    got = df.group_by("policy").agg(pl.col("avoidable_lei").sum()).sort("policy")
    want = portfolio_metrics(act, rows["windpower_only"]["sold_da"], None, pzu[..., None], None, a=A, b=B)
    assert got.filter(pl.col("policy") == "windpower_only")["avoidable_lei"][0] == pytest.approx(
        want["wind_avoidable_cost_lei"])
    assert got.filter(pl.col("policy") == "perfect")["avoidable_lei"][0] == 0
    assert str(df["ts"].dtype).startswith("Datetime") and json.dumps(df.columns)
