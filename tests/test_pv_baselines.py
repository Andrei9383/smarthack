"""Fast tests (< 10 s, no simulator run): CUSUM reference, pvlib model, predictors honour the contract."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from wattsentry.contracts.envelope import Predictor, Window
from wattsentry.eval.baselines.pv import lgbm
from wattsentry.eval.baselines.pv.cusum import cusum_two_sided
from wattsentry.eval.baselines.pv.faults import ThresholdCusumPredictor, cusum_detector, prf, threshold_rule
from wattsentry.eval.baselines.pv.features import LOAD_FEATURES, PV_FEATURES
from wattsentry.eval.baselines.pv.metrics import coverage, nmae, pinball
from wattsentry.eval.baselines.pv.persistence import PersistencePredictor, PvlibPredictor
from wattsentry.eval.baselines.pv.physical import pv_power_kw, recorded_meta, solar_geometry

SITE = {"site_id": "S0001", "lat": 44.43, "lon": 26.10, "kwp_recorded": 6.0, "tilt_recorded": 30.0,
        "azimuth_recorded": 180.0, "inverter_kw": 5.0, "kwp_true": 9.0, "tilt_true": 10.0, "azimuth_true": 90.0,
        "archetype": "heat_pump", "heating": "heat_pump", "has_ac": True, "has_ev": False,
        "has_electric_boiler": False, "works_from_home": True}
T0 = "2024-06-21T00:00:00Z"


def _clear_weather(t0=T0, H=96):
    ts = pd.date_range(pd.Timestamp(t0), periods=H, freq="15min")
    g = solar_geometry(ts, SITE["lat"], SITE["lon"])
    return {"ghi": g["cs_ghi"], "dni": g["cs_dni"], "dhi": g["cs_dhi"], "temp_c": np.full(H, 25.0),
            "wind10_ms": np.full(H, 2.0), "snow_cm": np.zeros(H)}, g


# ------------------------------------------------------------------------------------------------ CUSUM
def test_cusum_detects_step_and_resets_on_nan():
    x = np.r_[np.zeros(20), np.full(20, 0.5)]
    sp, sn, al = cusum_two_sided(x, k=0.1, h=1.0)
    assert al[:20].sum() == 0 and sn.max() == 0
    first = int(np.argmax(al == 1))
    assert 20 < first <= 23  # 0.4 per step past k -> 3 steps to reach h=1
    assert sp.max() == pytest.approx(1.0)  # saturates at h
    x2 = x.copy()
    x2[30] = np.nan
    sp2, _, al2 = cusum_two_sided(x2, 0.1, 1.0)
    assert sp2[30] == 0 and al2[30] == 0 and sp2[31] == pytest.approx(0.4)


def test_cusum_two_sided_and_2d_matches_1d():
    rng = np.random.default_rng(0)
    X = rng.normal(0, 0.2, (4, 50))
    X[2, 25:] -= 1.0
    sp, sn, al = cusum_two_sided(X, 0.1, 0.8)
    for i in range(4):
        a, b, c = cusum_two_sided(X[i], 0.1, 0.8)
        np.testing.assert_array_equal(sp[i], a)
        np.testing.assert_array_equal(sn[i], b)
        np.testing.assert_array_equal(al[i], c)
    assert (al[2, 30:] == -1).all()
    with pytest.raises(ValueError):
        cusum_two_sided(X, 0.1, 0.0)


# ------------------------------------------------------------------------------------------------ pvlib
def test_pvlib_uses_recorded_metadata_only():
    wx, g = _clear_weather()
    meta = recorded_meta(SITE)
    ac = pv_power_kw(meta, g, wx["ghi"], wx["dni"], wx["dhi"], wx["temp_c"], wx["wind10_ms"])["ac_kw"]
    assert ac.max() <= SITE["inverter_kw"] + 1e-9 and ac.max() > 3.5  # clear June noon in Bucharest, 6 kWp
    assert (ac[g["zenith"] >= 90] == 0).all()
    other = dict(SITE, kwp_true=1.0, tilt_true=80.0, azimuth_true=0.0)
    assert recorded_meta(other) == meta


# ------------------------------------------------------------------------------------------------ predictors
def _tiny_bundle(kind):
    rng = np.random.default_rng(1)
    names = PV_FEATURES if kind == "pv" else LOAD_FEATURES
    X = rng.normal(size=(600, len(names))).astype(np.float32)
    lead = names.index("pvlib_n" if kind == "pv" else "med7_n")
    X[:, lead] = rng.uniform(0, 1, 600)
    y = X[:, lead] + rng.normal(0, 0.05, 600)
    lgbm.PARAMS["min_data_in_leaf"] = 20
    try:
        return lgbm.fit_quantiles(X, y, names, kind, rounds=20, version=f"lgbm-{kind}-test")
    finally:
        lgbm.PARAMS["min_data_in_leaf"] = 200


def _window():
    wx, _ = _clear_weather()
    rng = np.random.default_rng(2)
    ref = np.tile(np.clip(np.sin(np.linspace(-np.pi / 2, 3 * np.pi / 2, 96)), 0, None) * 4000, 7)
    hist = rng.uniform(200, 900, 7 * 96)
    return Window(t0=T0, weather=wx, reference=ref, history=hist)


@pytest.mark.parametrize("kind", ["pv", "load"])
def test_lgbm_predictor_contract_and_roundtrip(kind, tmp_path):
    b = _tiny_bundle(kind)
    p = lgbm.LGBMPredictor(b)
    assert isinstance(p, Predictor)
    pred = p.predict(SITE, _window())
    pred.validate()
    assert pred.model == "lgbm" and pred.horizon == 96 and pred.unit == "kW"
    assert all(a <= b_ + 1e-9 <= c + 2e-9 for a, b_, c in zip(pred.q10, pred.q50, pred.q90))
    if kind == "pv":
        g = solar_geometry(pd.date_range(pd.Timestamp(T0), periods=96, freq="15min"), SITE["lat"], SITE["lon"])
        assert all(v == 0 for v, z in zip(pred.q90, g["zenith"]) if z >= 90)
    loaded = lgbm.LGBMPredictor(lgbm.QuantileBundle.load(b.save(tmp_path / "b.json")))
    assert loaded.predict(SITE, _window()).q50 == pred.q50


def test_lgbm_accepts_hourly_ghi_only_weather():
    b = _tiny_bundle("pv")
    wx, _ = _clear_weather()
    hourly = pd.DataFrame({"ghi_wm2": wx["ghi"][::4], "temp_c": wx["temp_c"][::4]})
    pred = lgbm.LGBMPredictor(b).predict(SITE, Window(t0=T0, weather=hourly, reference=_window().reference))
    assert max(pred.q50) > 0


def test_point_baselines_contract():
    w = _window()
    for k in ("pv_d1", "pv_ref7", "load_d1", "load_w1"):
        pr = PersistencePredictor(k).predict(SITE, w)
        assert pr.horizon == 96 and pr.q10 == pr.q50 == pr.q90
    np.testing.assert_allclose(PersistencePredictor("pv_d1").predict(SITE, w).q50, w.reference[-96:] / 1000)
    pv = PvlibPredictor(band=(0.8, 1.1)).predict(SITE, w)
    assert max(pv.q50) > 0 and pv.model == "pvlib"


def test_threshold_cusum_predictor_flags_drop():
    healthy = ThresholdCusumPredictor().predict(SITE, Window(t0=T0, weather=_window().weather,
                                                             residual_summary=np.ones((14, 1))))
    assert healthy.p_fault < 0.05
    drop = np.r_[np.ones(10), np.full(4, 0.6)][:, None]
    f = ThresholdCusumPredictor().predict(SITE, Window(t0=T0, residual_summary=drop))
    assert f.p_fault >= 0.5 and f.top_fault == "string_outage" and max(f.q50) == 0  # no weather -> no expectation
    assert abs(sum(f.fault_probs.values()) - 1) < 1e-9


# ------------------------------------------------------------------------------------------------ rules + metrics
def test_fault_rules_on_synthetic_ratio():
    q = np.ones((2, 40))
    q[1, 20:] = 0.6  # site 1 loses 40 % from day 20
    q[0, 5] = np.nan  # missing data day
    inf = np.ones_like(q, bool)
    inf[:, 10] = False  # uninformative day (snow): skipped
    al, _ = threshold_rule(q, inf)
    assert not al[0].any() and al[1, 22:].all() and not al[1, :22].any()
    cal, _ = cusum_detector(q, inf, k=0.1, h=0.5)
    assert not cal[0].any() and cal[1, 22:].all()
    truth = np.zeros_like(al)
    truth[1, 20:] = True
    assert prf(al, truth)["precision"] == 1.0


def test_metrics():
    y = np.array([1.0, 2.0, 3.0, 4.0])
    m = np.ones(4, bool)
    assert nmae(y + 1, y, m) == pytest.approx(0.4)
    q = np.stack([y - 1, y, y + 1])
    assert coverage(q, y, m) == 1.0 and pinball(q, y, m) < pinball(q + 2, y, m)


def test_sitehealth_day_by_day_with_model14_window_conventions():
    """Array weather with meta weather_channels, telemetry in W, ratios remembered across calls."""
    wx, _ = _clear_weather()
    chans = ["ghi", "dni", "dhi", "temp", "cloud", "wind", "snow"]
    arr = np.stack([wx["ghi"], wx["dni"], wx["dhi"], wx["temp_c"], np.zeros(96), wx["wind10_ms"], wx["snow_cm"]], 1)
    det = ThresholdCusumPredictor()
    expected = PvlibPredictor().expected_kw(SITE, Window(t0=T0, weather=arr, meta={"weather_channels": chans}))
    out = []
    for d in range(8):
        t0 = (pd.Timestamp(T0) + pd.Timedelta(days=d)).strftime("%Y-%m-%dT%H:%M:%SZ")
        factor = 1.0 if d < 4 else 0.5  # half the strings lost from day 4 (same weather every day)
        tel = {"pv_ac_w": expected * 1000 * factor, "inv_status": np.zeros(96), "load_w": np.full(96, 400.0)}
        out.append(det.predict(SITE, Window(t0=t0, weather=arr, meta={"weather_channels": chans, "telemetry": tel})))
    assert all(p.p_fault < 0.5 for p in out[:4])
    assert out[-1].p_fault >= 0.5 and out[-1].extra["threshold_alarm"]
    assert out[-1].extra["p_fault_threshold"] == 0.5


def test_predictors_registry_on_tiny_fleet(tmp_path, monkeypatch):
    monkeypatch.setenv("WATTSENTRY_CACHE", str(tmp_path))
    from wattsentry.eval.baselines.pv import predictors
    from wattsentry.sim.fleet import FleetConfig, generate

    fleet = generate(FleetConfig(n_sites=12, n_feeders=2, n_congested=1, start_year=2023, end_year=2023), plots=False)
    lgbm.PARAMS["min_data_in_leaf"] = 20
    try:
        P = predictors(fleet, max_sites=4, rounds=10, log=lambda *a: None)
    finally:
        lgbm.PARAMS["min_data_in_leaf"] = 200
    assert set(P) == {"pvlib", "persistence", "lgbm", "threshold_cusum"}
    assert set(P["lgbm"]) == {"pv", "load"} and set(P["threshold_cusum"]) == {"sitehealth"}
    w = _window()
    for name, kinds in P.items():
        for kind, pred in kinds.items():
            assert isinstance(pred, Predictor)
            assert pred.predict(SITE, w).horizon == 96, (name, kind)
    assert P["threshold_cusum"]["sitehealth"].predict(SITE, w).fault_probs is not None
