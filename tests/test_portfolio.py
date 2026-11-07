"""Portfolio aggregation (card #175): shapes, perfect forecast, quantile monotonicity, conformal coverage."""

import numpy as np
import pandas as pd
import pytest

from sentinel.contracts import BatterySpec, self_consumption_step
from sentinel.eval import portfolio as P

T = P.H


def _fleet(n=12, regions=3, battery_every=2, seed=0):
    rng = np.random.default_rng(seed)
    rows = [dict(site_id=f"S{i:03d}", county=f"C{i % regions}", battery_kwh=(8.0 + 4 * rng.random()) if
                 i % battery_every == 0 else 0.0, battery_kw=4.0, eta_charge=0.95, eta_discharge=0.94,
                 soc_min=0.1, soc_max=1.0, inverter_kw=6.0) for i in range(n)]
    return P.Fleet.from_sites(rows), rows


def _profiles(n, seed=0):
    rng = np.random.default_rng(seed)
    h = np.arange(T)
    pv = np.clip(np.sin((h - 24) / 48 * np.pi), 0, None)[None] * rng.uniform(2, 5, (n, 1))
    load = 0.4 + 0.6 * rng.random((n, 1)) + 0.3 * (1 + np.cos((h - 76) / 96 * 2 * np.pi))[None]
    return pv, load


def _band(mid, lo=0.7, hi=1.3):
    return np.stack([lo * mid, mid, hi * mid], axis=-1)


# ------------------------------------------------------------------------------------------------- shapes
def test_shapes_and_levels():
    fl, _ = _fleet(12, regions=3)
    pv, load = _profiles(12)
    u_pv, u_load = P.draw_levels(50, fl.region, len(fl.regions), seed=1)
    assert u_pv.shape == (50, 3) and u_load.shape == (50, 12)
    for r in range(3):  # Latin hypercube: exactly one level per 1/K stratum, per region
        assert np.array_equal(np.sort(np.floor(u_pv[:, r] * 50).astype(int)), np.arange(50))
    total = P.portfolio_scenarios(_band(pv), _band(load), fl, n_scen=50, seed=1)
    assert total.shape == (50, T) and np.isfinite(total).all()
    b = P.scenario_bands(total)
    assert b.shape == (T, 3)
    q = P.bought_da(total, 0.6)
    assert q.shape == (T,)
    # chunking homes does not change the result (levels are drawn once for the fleet)
    t2 = P.portfolio_scenarios(_band(pv), _band(load), fl, n_scen=50, seed=1, chunk=5)
    np.testing.assert_allclose(total, t2, rtol=0, atol=1e-12)


def test_solar_level_shared_within_region():
    fl, _ = _fleet(6, regions=2)
    pv, _ = _profiles(6)
    u_pv, u_load = P.draw_levels(200, fl.region, 2, seed=3)
    from scipy.special import ndtri
    s = P.sample_homes(_band(pv, 0.5, 1.5), ndtri(u_pv)[:, fl.region])  # no inverter cap: pure levels
    noon = 48
    rel = s[:, :, noon] / pv[None, :, noon]
    same = [i for i in range(6) if fl.region[i] == fl.region[0]]
    other = [i for i in range(6) if fl.region[i] != fl.region[0]]
    np.testing.assert_allclose(rel[:, same[0]], rel[:, same[1]], rtol=1e-9)  # one level for the whole region
    assert abs(np.corrcoef(rel[:, same[0]], rel[:, other[0]])[0, 1]) < 0.3  # independent across regions


def test_sample_homes_matches_quantile_at():
    from sentinel.eval.uncertainty.conformal import quantile_at
    rng = np.random.default_rng(0)
    q = np.sort(rng.random((7, T, 3)) * 3, axis=-1)
    z = rng.normal(size=(20, 7))
    ref = np.maximum(quantile_at(q[None, ..., 0], q[None, ..., 1], q[None, ..., 2], z[..., None]), 0.0)
    np.testing.assert_allclose(P.sample_homes(q, z), ref, rtol=0, atol=1e-12)


# ------------------------------------------------------------------------------------------------- battery model
def test_self_consumption_matches_contract_rule():
    fl, rows = _fleet(8, battery_every=2)
    pv, load = _profiles(8, seed=2)
    soc0 = np.where(fl.battery.capacity_kwh > 0, 0.4, 0.0)
    net = P.behaviour(pv[None], load[None], fl.battery, soc0)[0]
    for i, r in enumerate(rows):
        b = (BatterySpec(r["battery_kwh"], r["battery_kw"], r["eta_charge"], r["eta_discharge"], r["soc_min"],
                         r["soc_max"]) if r["battery_kwh"] > 0 else None)
        e = soc0[i] * r["battery_kwh"]
        for t in range(T):
            _, e, grid = self_consumption_step(b, e, pv[i, t], load[i, t])
            assert net[i, t] == pytest.approx(-grid, abs=1e-9)


def test_set_behaviour_swaps_and_restores():
    fl, _ = _fleet(4)
    pv, load = _profiles(4)
    try:
        P.set_behaviour(lambda pv, load, battery, soc0=None: np.zeros(np.broadcast_shapes(pv.shape, load.shape)),
                        "zero")
        assert P.behaviour_name() == "zero"
        assert np.all(P.portfolio_scenarios(_band(pv), _band(load), fl, n_scen=10) == 0)
    finally:
        P.set_behaviour(None)
    assert P.behaviour_name().startswith("self_consumption")


# ------------------------------------------------------------------------------------------------- perfect forecast
def test_perfect_forecast_collapses_to_actual():
    fl, _ = _fleet(10)
    pv, load = _profiles(10, seed=4)
    soc0 = P.warm_soc0(pv, load, fl.battery)
    actual = P.behaviour(pv, load, fl.battery, soc0).sum(axis=0) * P.KW_TO_MWH
    perfect = lambda x: np.stack([x, x, x], axis=-1)  # noqa: E731
    d = P.forecast_day(perfect(pv), perfect(load), fl, q_star=0.67, n_scen=40, soc0=soc0)
    np.testing.assert_allclose(d.total, np.broadcast_to(actual, d.total.shape), atol=1e-12)
    np.testing.assert_allclose(d.bands, np.stack([actual] * 3, axis=-1), atol=1e-12)
    np.testing.assert_allclose(d.bought, actual, atol=1e-12)
    assert P.nmae(d.bands[:, 1], actual) == pytest.approx(0.0, abs=1e-12)
    assert P.coverage(d.bands, actual) == 1.0
    mask = np.zeros(T, bool)
    mask[64:] = True
    np.testing.assert_allclose(P.traded_id(d.bought, d.total, 0.67, mask), 0.0, atol=1e-12)
    # forecast_only on a perfect forecast = load - pv exactly
    fo = P.portfolio_scenarios(perfect(pv), perfect(load), fl, n_scen=5, battery=False)
    np.testing.assert_allclose(fo[0], (load - pv).sum(0) * P.KW_TO_MWH, atol=1e-12)


# ------------------------------------------------------------------------------------------------- monotonicity
def test_quantiles_monotone():
    fl, _ = _fleet(16)
    pv, load = _profiles(16, seed=5)
    total = P.portfolio_scenarios(_band(pv, 0.4, 1.4), _band(load, 0.5, 1.8), fl, n_scen=200, seed=2)
    b = P.scenario_bands(total)
    assert np.all(b[:, 0] <= b[:, 1] + 1e-12) and np.all(b[:, 1] <= b[:, 2] + 1e-12)
    qs = [P.bought_da(total, q) for q in (0.1, 0.3, 0.5, 2 / 3, 0.9)]
    for lo, hi in zip(qs, qs[1:]):
        assert np.all(lo <= hi + 1e-12)
    np.testing.assert_allclose(P.bought_da(total, 0.5), b[:, 1], atol=1e-12)
    conf = P.TotalConformal(scale=np.array([0.5, 1.0, 2.0, 3.0]))
    cb = conf.apply(b)
    assert np.all(cb[:, 0] <= cb[:, 1] + 1e-12) and np.all(cb[:, 1] <= cb[:, 2] + 1e-12)
    np.testing.assert_allclose(cb[:, 1], b[:, 1])
    # the calibrated purchase quantile = the stretched raw quantile (stretch is linear around P50)
    np.testing.assert_allclose(P.bought_da(total, 0.9, conf), conf.stretch(np.quantile(total, 0.9, 0), b[:, 1]),
                               atol=1e-9)
    # a home band that is wider can only widen the portfolio band (same levels)
    wide = P.portfolio_scenarios(_band(pv, 0.2, 1.8), _band(load, 0.5, 1.8), fl, n_scen=200, seed=2)
    wb = P.scenario_bands(wide)
    assert (wb[:, 2] - wb[:, 0]).sum() >= (b[:, 2] - b[:, 0]).sum()


# ------------------------------------------------------------------------------------------------- conformal
def test_conformal_coverage_synthetic():
    """Bands built as if errors were independent (too narrow), truth with a common factor: the raw band
    under-covers, the conformal scale fitted on validation days brings held-out coverage to ~80 %."""
    rng = np.random.default_rng(0)
    d_val, d_test = 300, 300
    mid = 5 + np.sin(np.arange(T) / T * 2 * np.pi)
    sig_model = 0.5 + 0.2 * np.cos(np.arange(T) / 24)
    sig_true = 1.7 * sig_model  # the model's spread is too small

    def days(n):
        y = mid + rng.normal(size=(n, T)) * sig_true
        z = 1.2815515655446004
        b = np.stack([mid - z * sig_model, mid, mid + z * sig_model], axis=-1)
        return np.broadcast_to(b, (n, T, 3)), y

    bv, yv = days(d_val)
    bt, yt = days(d_test)
    conf = P.TotalConformal.fit(bv, yv)
    assert conf.raw_coverage < 0.65
    assert np.all(conf.scale > 1.3)
    assert 0.79 <= P.coverage(conf.apply(bv), yv) <= 0.81
    assert 0.78 <= P.coverage(conf.apply(bt), yt) <= 0.82
    # too-wide bands are narrowed (the scale goes below 1)
    conf2 = P.TotalConformal.fit(bv, mid + rng.normal(size=(d_val, T)) * 0.5 * sig_model)
    assert np.all(conf2.scale < 1.0)


def test_conformal_end_to_end_with_scenarios():
    """Scenario bands from per-home draws vs a truth with a portfolio-wide load shock: calibrated coverage ~80 %."""
    fl, _ = _fleet(20, regions=2)
    pv, load = _profiles(20, seed=6)
    rng = np.random.default_rng(1)
    soc0 = P.warm_soc0(pv, load, fl.battery)
    n_days = 400
    bands, ys = [], []
    for d in range(n_days):
        total = P.portfolio_scenarios(_band(pv, 0.8, 1.2), _band(load, 0.8, 1.25), fl, n_scen=60, seed=d, soc0=soc0)
        # truth: a common load factor per step that independent home draws know nothing about, + regional PV error
        shock = 1 + 0.12 * rng.normal(size=T)
        pv_t = pv * (1 + 0.1 * rng.normal(size=(1, T)))
        ys.append(P.behaviour(pv_t, load * shock, fl.battery, soc0).sum(0) * P.KW_TO_MWH)
        bands.append(P.scenario_bands(total))
    bands, ys = np.stack(bands), np.stack(ys)
    h = n_days // 2
    conf = P.TotalConformal.fit(bands[:h], ys[:h])
    assert P.coverage(bands[h:], ys[h:]) < 0.7
    assert 0.77 <= P.coverage(conf.apply(bands[h:]), ys[h:]) <= 0.83


# ------------------------------------------------------------------------------------------------- purchase, 15:00
def test_purchase_quantile():
    assert P.purchase_quantile(1.0, 1.0) == pytest.approx(0.5)
    assert P.purchase_quantile(2.0, 1.0) == pytest.approx(2 / 3)
    q = P.purchase_quantile()
    assert 0.5 <= q < 1.0  # frozen a >= b (#158)


def test_correction_mask_local_time():
    winter = pd.date_range("2025-01-15 00:00", periods=T, freq="15min", tz="UTC")
    summer = pd.date_range("2025-07-15 00:00", periods=T, freq="15min", tz="UTC")
    mw, ms = P.correction_mask(winter), P.correction_mask(summer)
    assert mw.sum() == 32 and ms.sum() == 32  # 16:00-24:00 local = 8 h
    assert winter[mw][0].hour == 14 and summer[ms][0].hour == 13  # UTC+2 / UTC+3 (DST)


def test_traded_id_only_in_window():
    rng = np.random.default_rng(0)
    total_da = rng.normal(1.0, 0.2, (200, T))
    total_new = rng.normal(1.3, 0.1, (200, T))
    bought = P.bought_da(total_da, 2 / 3)
    mask = np.zeros(T, bool)
    mask[60:92] = True
    tr = P.traded_id(bought, total_new, 2 / 3, mask)
    assert np.all(tr[~mask] == 0)
    np.testing.assert_allclose(tr[mask], (np.quantile(total_new, 2 / 3, 0) - bought)[mask])
    assert tr[mask].mean() > 0.2  # the re-forecast says more is needed: buy on PI


# ------------------------------------------------------------------------------------------------- adapter
def test_validated_bands_passthrough_and_fallback():
    rng = np.random.default_rng(0)
    q = np.sort(rng.random((5, T, 3)), axis=-1)
    out, known, counts = P.validated_bands("pvnet", [f"S{i}" for i in range(5)], "2025-06-02", q)
    assert known.all() and counts["use"] == 5
    np.testing.assert_allclose(out, q)


def test_validated_bands_dont_use_takes_fallback():
    adapter = pytest.importorskip("sentinel.judge.adapter")
    rng = np.random.default_rng(0)
    q = np.sort(rng.random((3, T, 3)), axis=-1)
    fb = np.full((3, T, 3), 7.0)
    judge = adapter.MemoryVerdicts({("S1", "pvnet", "2025-06-02"): {"verdict": "dont_use"},
                                    ("S2", "pvnet", "2025-06-02"): {"verdict": "widen", "band_scale": 2.0}})
    out, known, _ = P.validated_bands("pvnet", ["S0", "S1", "S2"], "2025-06-02", q, fallback=fb, judge=judge)
    assert known.tolist() == [True, False, True]
    np.testing.assert_allclose(out[0], q[0])
    np.testing.assert_allclose(out[1], 7.0)
    assert np.all(out[2, :, 2] - out[2, :, 1] >= q[2, :, 2] - q[2, :, 1] - 1e-12)
