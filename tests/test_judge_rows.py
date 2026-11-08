"""Judge training rows: week sampling, drift augmentation, grading, cache key, history features."""

from __future__ import annotations

import json

import numpy as np
import pytest

from wattsentry.judge import features as F
from wattsentry.judge import rows as R


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


def test_history_features_flag_jumps_and_repeats():
    D = 30
    band = np.full((2, D), 0.4)
    band[0, 25:] = 0.2  # witness 0 narrows its bands from day 25
    energy = np.tile(np.linspace(10, 12, D), (2, 1))
    h = F.history_features({"band_rel": band, "energy_q50": energy}, 27)
    assert h["hist_band_rel7"][0] == pytest.approx(0.4) and h["hist_band_rel28"][0] == pytest.approx(0.4)
    assert h["hist_energy7"].shape == (2,)
    assert np.isnan(F.history_features({"band_rel": band}, 0)["hist_band_rel7"]).all()  # no history yet
    pf = np.zeros((1, 10))
    pf[0, 5:] = 1.0
    assert F.history_features({"alarm": pf}, 8)["hist_alarm7"][0] == pytest.approx(3 / 7)
    q = np.random.default_rng(0).uniform(0, 3, (3, 96))
    prev = q.copy()
    prev[1] += 0.1
    assert F.repeats(q, prev).tolist() == [1.0, 0.0, 1.0]
    assert F.repeats(np.zeros((1, 96)), np.zeros((1, 96))).tolist() == [0.0]  # night / all-zero is not frozen
