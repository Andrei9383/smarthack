"""J3 (#148): week sampling, drift / cold-start augmentation, grading, stand-in ledger, cache key, features."""

from __future__ import annotations

import json

import numpy as np
import pandas as pd
import pytest

from sentinel.contracts import Prediction
from sentinel.judge import features as F
from sentinel.judge import rows as R


def test_sample_weeks_shares_and_determinism():
    w = R.sample_weeks(2024, 12, seed=0)
    assert len(w) == 12 and [x.month for x in w] == list(range(1, 13))
    assert sum(x.drift is not None for x in w) == 6  # half with one witness degraded
    assert sum(x.cold_day is not None for x in w) == 3  # a quarter with the ledger wiped
    assert w == R.sample_weeks(2024, 12, seed=0)
    assert w != R.sample_weeks(2024, 12, seed=1)
    for x in w:
        assert x.start.startswith("2024-") and int(x.start[8:]) <= 22
        if x.drift:
            assert x.drift.witness in R.WITNESSES and x.drift.form in R.DRIFT_FORMS[x.drift.witness]
            if x.drift.form == "bias":
                assert 0.2 <= abs(x.drift.magnitude) <= 0.4


def _q(S=4, D=5, level=2.0):
    q = np.zeros((S, D, 96, 3))
    q[..., 1] = level
    q[..., 0], q[..., 2] = level * 0.8, level * 1.2
    q += np.arange(D)[None, :, None, None] * 0.1
    return q


def test_drift_forecast_forms_touch_only_scope_and_days():
    q = _q()
    days = np.array([False, False, True, True, True])
    assets = np.array([True, False, True, False])
    b = R.drift_forecast(q, R.DriftSpec("pvnet", "bias", 2, "battery", 0.3), days, assets)
    np.testing.assert_allclose(b[0, 2:], q[0, 2:] * 1.3)
    np.testing.assert_array_equal(b[1], q[1])
    np.testing.assert_array_equal(b[:, :2], q[:, :2])
    n = R.drift_forecast(q, R.DriftSpec("pvnet", "narrow", 2, "all", 0.5), days, assets)
    np.testing.assert_allclose(n[0, 3, :, 2] - n[0, 3, :, 1], 0.5 * (q[0, 3, :, 2] - q[0, 3, :, 1]))
    np.testing.assert_allclose(n[..., 1], q[..., 1])
    f = R.drift_forecast(q, R.DriftSpec("pvnet", "frozen", 2, "all", 0.5), days, assets)
    np.testing.assert_allclose(f[0, 4], q[0, 1])  # repeats the last healthy day's output


def test_drift_detector_keeps_probabilities():
    rng = np.random.default_rng(0)
    p = rng.dirichlet(np.ones(8), size=(3, 4))
    days, assets = np.array([False, True, True, True]), np.ones(3, bool)
    so = F.PV_CLASSES.index("string_outage")
    sf = F.PV_CLASSES.index("sensor_fault")
    for form in ("confusion", "false_alarm", "blind", "frozen"):
        d = R.drift_detector(p, R.DriftSpec("sitehealth", form, 1, "all", 0.5), days, assets)
        np.testing.assert_allclose(d.sum(-1), 1.0)
        np.testing.assert_array_equal(d[:, 0], p[:, 0])
    c = R.drift_detector(p, R.DriftSpec("sitehealth", "confusion", 1, "all", 0.5), days, assets)
    assert np.all(c[:, 1:, so] < 1e-12) and np.all(c[:, 1:, sf] > p[:, 1:, sf])
    b = R.drift_detector(p, R.DriftSpec("sitehealth", "blind", 1, "all", 0.5), days, assets)
    assert np.all(b[:, 1:, 0] > p[:, 1:, 0])


def test_grade_forecast_wide_bands_lose_and_valid_rule():
    rng = np.random.default_rng(1)
    y = rng.uniform(1, 3, (3, 96))
    mask = np.ones_like(y, bool)
    good = np.stack([y - 0.3 * np.abs(rng.normal(1, .3, y.shape)), y, y + 0.3 * np.abs(rng.normal(1, .3, y.shape))], -1)
    good[0, :19, 0] = good[0, :19, 1] = good[0, :19, 2] = y[0, :19] + 1  # ~20 % misses -> hit 0.8
    wide = np.stack([y - 50, y, y + 50], -1)
    ref = np.stack([y - 3, y, y + 3], -1)
    g = R.grade_forecast(good[:1], y[:1], mask[:1], ref[:1])
    w = R.grade_forecast(wide[:1], y[:1], mask[:1], ref[:1])
    assert g["hit"][0] == pytest.approx(77 / 96, abs=1e-9) and g["valid"][0] == 1.0
    assert w["hit"][0] == 1.0 and w["valid"][0] == 0.0  # hit > 0.95 and worse than the substitute
    assert w["sharpness"][0] == 0.0 and g["sharpness"][0] > 0
    assert g["e_q"].shape == (1, 20) and np.all(np.diff(g["e_q"][0]) >= 0)
    assert g["hit_m6"][0] >= g["hit_m1"][0] >= g["hit_m0.5"][0]
    few = R.grade_forecast(good[:1], y[:1], np.zeros_like(mask[:1]), ref[:1])
    assert not few["graded"][0]


def test_grade_detector_split_scores():
    alarm = np.array([1, 1, 1, 0, 0], bool)
    called = np.array(["string_outage", "soiling", "shading", "healthy", "healthy"], object)
    truth = np.array(["string_outage", "shading", "healthy", "healthy", "soiling"], object)
    g = R.grade_detector(alarm, called, truth)
    np.testing.assert_array_equal(g["alarm_precision"][:3], [1.0, 0.5, 0.0])
    np.testing.assert_array_equal(g["miss_score"][3:], [1.0, 0.0])
    np.testing.assert_array_equal(g["valid"], [1, 0, 0, 1, 0])
    tc = R.truth_class(np.array([[0] * 90 + [3] * 6, [0] * 94 + [3] * 2]))
    assert tc.tolist() == ["string_outage", "healthy"]


def test_mini_ledger_prior_decay_backoff_and_wipe():
    L = R.MiniLedger()
    key = R.bucket_key("site", True, True, 6, False, False)
    t0 = L.trust("pvnet", "calibration", key, 0)
    assert t0["trust"] == pytest.approx(R.DEFAULT_PRIOR["calibration"]) and t0["n_eff"] == 0  # no record -> prior
    for d in range(10):  # a degraded witness: scores 0.1, observed the next day
        L.add("pvnet", "calibration", [key] * 5, d, d + 1, np.full(5, 0.1))
    t = L.trust("pvnet", "calibration", key, 10)
    assert t["trust"] < 0.45 and t["n_eff"] > 5 and t["level"] == 6
    assert L.trust("pvnet", "calibration", key, 5)["n_eff"] < t["n_eff"]  # only outcomes observed by day 5
    other = R.bucket_key("site", False, False, 1, False, False)  # unseen bucket with no evidence of its own
    assert L.trust("pvnet", "calibration", other, 10)["level"] == 1  # backs off to "site"
    L.wipe("pvnet", 11)
    w = L.trust("pvnet", "calibration", key, 12)
    assert w["n_eff"] == 0 and w["trust"] == pytest.approx(R.DEFAULT_PRIOR["calibration"])


def test_mini_ledger_alert_hysteresis_and_band_scale():
    L = R.MiniLedger()
    seq = [-0.3, -0.3, -0.3, -0.15, -0.05]
    on = [L.alert_step("pvnet", "k", g, 10.0) for g in seq]
    assert on == [False, False, True, True, False]
    assert not L.alert_step("pvnet", "j", -0.5, 2.0)  # too little evidence
    key = R.bucket_key("site", True, False, 6, False, False)
    assert L.band_scale("pvnet", key, 5) == 1.0
    e = np.tile(np.linspace(0, 2, 20), (10, 1))  # 80 % quantile ~1.6
    for d in range(3):
        L.add_err("pvnet", [key] * 10, d, d + 1, e)
    assert 1.4 <= L.band_scale("pvnet", key, 5) <= 1.8


def test_cache_key_changes_with_config(tmp_path):
    win = tmp_path / "w"
    win.mkdir()
    (win / "index.json").write_text(json.dumps({"source_key": {"fleet": "a"}, "data_version": 4}))
    models = tmp_path / "m"
    models.mkdir()
    (models / "pvnet.s0.onnx").write_bytes(b"x")
    cfg = R.RowConfig(year=2024)
    weeks = R.sample_weeks(2024)
    k = R.cache_key(cfg, weeks, win, models, (0.8, 1.2))
    assert k == R.cache_key(R.RowConfig(year=2024), weeks, win, models, (0.8, 1.2))
    assert k != R.cache_key(R.RowConfig(year=2024, burn_in=14), weeks, win, models, (0.8, 1.2))
    assert k != R.cache_key(cfg, R.sample_weeks(2024, seed=3), win, models, (0.8, 1.2))
    (models / "pvnet.s0.onnx").write_bytes(b"y")  # new witness weights -> new rows
    assert k != R.cache_key(cfg, weeks, win, models, (0.8, 1.2))


def test_prediction_row_matches_bulk_summaries():
    q50 = np.r_[np.zeros(24), np.linspace(0, 4, 24), np.linspace(4, 0, 24), np.zeros(24)]
    p = Prediction(asset_id="S1", model="pvnet", model_version="v", t0="2024-06-01T21:00:00Z", horizon=96,
                   q10=(q50 * 0.8).tolist(), q50=q50.tolist(), q90=(q50 * 1.3).tolist(), ood_score=1.2,
                   member_std=0.1)
    sub = Prediction(asset_id="S1", model="pvlib", model_version="v", t0=p.t0, horizon=96, q10=(q50 * .7).tolist(),
                     q50=(q50 * 1.3).tolist(), q90=(q50 * 1.6).tolist())
    yprev = np.r_[np.zeros(30), np.full(8, 2.5), np.linspace(1, 0, 58)]
    row = F.prediction_row(p, day="2024-06-02", context={"battery": True, "seen": False, "month": 6, "sky": 0},
                           substitute=sub, ledger={"trust_main": 0.41, "n_eff": 3.0},
                           telemetry_prev={"y": yprev, "valid": np.ones(96), "energy_ref": 20.0},
                           thresholds={"ood": 3.0})
    bulk = F.summarize_forecast(q50[None] * 0.8, q50[None], q50[None] * 1.3, "pv")
    assert row["energy_q50"] == pytest.approx(bulk["energy_q50"][0])
    assert row["band_rel"] == pytest.approx(bulk["band_rel"][0])
    assert row["in_frozen"] == 1.0 and row["sub_model"] == "pvlib" and row["family"] == "forecast"
    X = F.build(F.rows_frame([row]), "forecast")
    assert list(X.columns) == F.FEATURES["forecast"]
    assert X["rel_vs_sub"].iloc[0] == pytest.approx((row["energy_q50"] - row["sub_energy"]) / row["sub_energy"])
    assert F.reason_text("trust_main", row) == "trust 0.41 for new battery homes"
    assert "disagrees with pvlib" in F.reason_text("rel_vs_sub", row)


def test_detector_row_and_features():
    p = Prediction(asset_id="S1", model="sitehealth", model_version="v", t0="2024-06-01T21:00:00Z", horizon=96,
                   q10=[1.0] * 96, q50=[1.0] * 96, q90=[1.0] * 96,
                   fault_probs={"healthy": 0.3, "string_outage": 0.6, "sensor_fault": 0.1},
                   extra={"p_fault_threshold": 0.4, "p_fault_std": 0.05})
    row = F.prediction_row(p, day="2024-06-02")
    assert row["alarm"] == 1.0 and row["top_class"] == "string_outage" and row["p_fault"] == pytest.approx(0.7)
    X = F.build(pd.DataFrame([row]), "detector")
    assert X["top_class"].iloc[0] == F.CLASS_CODE["string_outage"] and X["witness"].iloc[0] == 10
