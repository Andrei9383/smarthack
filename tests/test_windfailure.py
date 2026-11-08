"""Card #43 M7 WindFailure: feature determinism / causality, no leakage across splits, metrics, predictor shape,
CARE scorer. Needs requirements.txt only (lightgbm); the M6-GRU residual source is torch-gated."""

import os
import sys

if sys.platform == "darwin":  # torch + lightgbm OpenMP runtimes clash on macOS unless single-threaded
    os.environ.setdefault("OMP_NUM_THREADS", "1")

import json

import numpy as np
import pytest

from wattsentry.contracts.envelope import Prediction, Window
from wattsentry.model.wind.synth import SynthConfig, generate
from wattsentry.model.windfailure import COMPONENTS, thermal
from wattsentry.model.windfailure.features import FEATURES, daily_features, feature_index
from wattsentry.model.windfailure.labels import Failure, daily_targets, day_grid, failures_from_episodes
from wattsentry.model.windfailure.metrics import evaluate_alarms
from wattsentry.model.windfailure.train import day_folds

SMALL = SynthConfig(days=400, scale=0.3)


@pytest.fixture(scope="module")
def small():
    return generate(SMALL)


def _feats(d):
    rated = d.rated_kw.astype(np.float64)
    big = (rated >= 4000).astype(int)
    m = thermal.fit(d.sensors, d.err, rated, big, np.arange(len(rated)), np.ones(d.n_steps, bool))
    r = thermal.residuals(m, d.sensors, d.err, rated, big)
    return r, daily_features(r, d.sensors["act_pwt"], d.sensors["turbine_t"], rated, d.park_of, big)


def test_features_deterministic_and_causal(small):
    d = small.subset(np.arange(6), slice(0, 120 * 96))
    r1, X1 = _feats(d)
    r2, X2 = _feats(d)
    assert X1.shape == (120, 6, len(COMPONENTS), len(FEATURES))
    np.testing.assert_array_equal(np.nan_to_num(X1, nan=-999), np.nan_to_num(X2, nan=-999))
    # causality: corrupting the residuals after day 90 never changes features of days <= 90
    r3 = r1.copy()
    r3[91 * 96 :] += 25.0
    rated = d.rated_kw.astype(np.float64)
    big = (rated >= 4000).astype(int)
    X3 = daily_features(r3, d.sensors["act_pwt"], d.sensors["turbine_t"], rated, d.park_of, big)
    np.testing.assert_array_equal(np.nan_to_num(X1[:91], nan=-999), np.nan_to_num(X3[:91], nan=-999))
    assert np.nanmax(X3[95:, ..., feature_index("dev")]) > 10  # and the jump is visible afterwards


def test_drift_is_visible_in_dev():
    T, n = 100 * 96, 2
    rng = np.random.default_rng(0)
    r = rng.normal(0, 0.4, (T, n, len(COMPONENTS))).astype(np.float32)
    r[80 * 96 : 90 * 96, 0, 0] += np.linspace(0, 10, 10 * 96)  # gearbox drift on turbine 0
    p = np.full((T, n), 1000.0)
    X = daily_features(r, p, np.full((T, n), 10.0), np.array([2500.0, 2500.0]), np.zeros(n, int), np.zeros(n, int))
    dev = X[..., feature_index("dev")]
    assert dev[89, 0, 0] > 5 and abs(dev[89, 1, 0]) < 0.5 and abs(dev[70, 0, 0]) < 0.5
    assert X[89, 0, 0, feature_index("slope7")] > 0.5


def test_thermal_fit_ignores_heldout_turbines_and_other_time(small):
    d = small.subset(np.arange(8), slice(0, 200 * 96))
    rated = d.rated_kw.astype(np.float64)
    big = (rated >= 4000).astype(int)
    fit_t = np.zeros(d.n_steps, bool)
    fit_t[: 100 * 96] = True
    m1 = thermal.fit(d.sensors, d.err, rated, big, np.arange(6), fit_t)
    sensors = {k: v.copy() for k, v in d.sensors.items()}
    for c in COMPONENTS:
        sensors[c][:, 6:] += 30.0  # held-out turbines
        sensors[c][100 * 96 :, :6] += 30.0  # validation / test time
    m2 = thermal.fit(sensors, d.err, rated, big, np.arange(6), fit_t)
    assert m1["coef"] == m2["coef"]


def test_folds_never_leak_into_validation_or_test():
    days = (np.datetime64("2024-01-01") + np.arange(731)).astype("datetime64[m]")
    fold, test_d, train_mask = day_folds(days, 2025, 4, 14, 7)
    yr = days.astype("datetime64[Y]").astype(int) + 1970
    assert test_d.sum() == 365 and (yr[test_d] == 2025).all()
    assert (fold[test_d] == -1).all()
    last_pre = np.flatnonzero(fold >= 0).max()
    assert np.flatnonzero(test_d).min() - last_pre > 14  # a 14-day label window never reaches the test year
    for k in range(4):
        blk = np.flatnonzero(fold == k)
        tr = np.flatnonzero(train_mask(k))
        assert not np.isin(tr, blk).any()
        assert not ((tr >= blk[0] - 14) & (tr <= blk[-1] + 7)).any()
        assert not test_d[tr].any()


def test_targets_and_down_days():
    days = day_grid((np.datetime64("2025-01-01T00:00") + np.arange(40 * 96) * np.timedelta64(15, "m")))
    f = Failure(0, "WP1.TID01", "gen1_t", np.datetime64("2025-01-21T06:00"), np.datetime64("2025-01-23T00:00"),
                np.datetime64("2025-01-10T00:00"), 11.0, 12.0)
    y, down = daily_targets([f], days, 2)
    c = COMPONENTS.index("gen1_t")
    assert y[7][:, 0, c].sum() == 7 and y[14][:, 0, c].sum() == 14
    assert y[14][19, 0, c] == 1 and y[14][20, 0, c] == 0  # day 20 (Jan 21) holds the failure itself
    assert down[20:22, 0].all() and not down[19, 0] and not down[:, 1].any()
    assert y[14][:, 1].sum() == 0 and y[14][:, 0, COMPONENTS.index("gearbox_t")].sum() == 0


def test_event_metrics_episode_rules():
    days = (np.datetime64("2025-01-01") + np.arange(60)).astype("datetime64[m]")
    f = Failure(0, "T0", "gearbox_t", np.datetime64("2025-01-31T12:00"), None, None, None, None)
    alarm = np.zeros((60, 2, len(COMPONENTS)), bool)
    alarm[10:30, 0, 0] = True  # starts 21 days before the failure, runs into the window: one TRUE episode
    alarm[40:42, 1, 0] = True  # another turbine, no failure: one false episode
    m = evaluate_alarms(alarm, np.ones((60, 2), bool), [f], days, 14)
    assert m["n_detected"] == 1 and m["false_alarm_episodes"] == 1
    assert m["events"][0]["lead_h"] == pytest.approx(19 * 24 + 12)  # end of day 10 (Jan 12 00:00) → Jan 31 12:00


@pytest.fixture(scope="module")
def trained(tmp_path_factory, small):
    from wattsentry.model.windfailure import train

    out = tmp_path_factory.mktemp("wf")
    rep = train.main(["--data", "synth", "--synth-days", str(SMALL.days), "--synth-scale", str(SMALL.scale),
                      "--rounds", "60", "--out", str(out)])
    return out, rep


@pytest.mark.slow  # trains LightGBM on 400 synthetic days (#179: run in rustcheck)
def test_pipeline_artifacts_and_no_heldout_training(trained, small):
    out, rep = trained
    for f in ("windfailure_7d.txt", "windfailure_14d.txt", "windfailure_thresholds.json", "windfailure_report.json",
              "manifest.json"):
        assert (out / f).exists(), f
    assert "wf_ship" in rep["metrics"] and "lead_time_h" in rep["metrics"]
    sp = rep["splits"]
    assert sp["holdout_park"] == "WP2"
    # the residual model was fitted on training parks only: refitting without the held-out park reproduces it
    cal = json.loads((out / "windfailure_thresholds.json").read_text())
    from wattsentry.model.wind.data import make_splits
    from wattsentry.model.windfailure.train import day_folds as folds

    s = make_splits(small, "WP2")
    days = day_grid(small.ts)
    fold, test_d, _ = folds(days, s.test_year, 4, 14, 7)
    fit_t = np.repeat(fold >= 0, 96)[: small.n_steps]
    rated = small.rated_kw.astype(np.float64)
    m = thermal.fit(small.sensors, small.err, rated, (rated >= 4000).astype(int), s.train_turbines, fit_t)
    assert m["coef"] == cal["residual_model"]["coef"]
    # training rows end > 14 days before the test year, so no test failure is ever inside a training label window
    assert np.flatnonzero(test_d).min() > np.flatnonzero(fold >= 0).max() + 14


@pytest.mark.slow  # trains LightGBM on 400 synthetic days (#179: run in rustcheck)
def test_predictor_output_shape(trained, small):
    from wattsentry.model.windfailure.predictor import WindFailurePredictor

    out, _ = trained
    p = WindFailurePredictor(out)
    cols = np.flatnonzero(small.park_of == 1)
    sub = small.subset(cols, slice(0, 90 * 96))
    risks = p.risk_park(sub.turbines, sub.ts, sub.sensors, sub.err)
    assert set(risks) == set(sub.turbine_ids)
    r0 = risks[sub.turbine_ids[0]]
    assert 0 <= r0["risk"] <= 1 and set(r0["p_fail_14d"]) == set(COMPONENTS) and r0["risk_component"] in COMPONENTS
    assert isinstance(r0["shipped"], bool) and r0["day"] == "2024-03-30"
    pc = {"turbines": sub.turbines, "ts": sub.ts, "telemetry": sub.sensors, "err": sub.err}
    pred = p.predict(sub.turbines[0], Window(t0="2024-03-20T00:00:00Z", horizon=1, park_context=pc))
    assert isinstance(pred, Prediction) and pred.model == "windfailure" and pred.horizon == 1
    assert pred.extra["day"] == "2024-03-19" and pred.q50[0] == pytest.approx(pred.extra["risk"])


def test_care_score_and_skip(monkeypatch, tmp_path):
    from wattsentry.eval.care import download
    from wattsentry.eval.care.score import care_score, column_map, earliness_weights

    n = 200
    truth = np.zeros(n, bool)
    truth[100:] = True
    normal = np.ones(n, bool)
    perfect = [{"anomaly": True, "truth": truth, "pred": truth.copy(), "normal": normal},
               {"anomaly": False, "truth": np.zeros(n, bool), "pred": np.zeros(n, bool), "normal": normal}]
    s = care_score(perfect)
    assert s["care_score"] == pytest.approx(1.0) and s["reliability"] == pytest.approx(1.0)
    silent = [dict(perfect[0], pred=np.zeros(n, bool)), perfect[1]]
    assert care_score(silent)["care_score"] == 0.0  # nothing detected at all → 0
    noisy = [perfect[0], dict(perfect[1], pred=np.ones(n, bool))]
    assert care_score(noisy)["care_score"] == 0.0  # accuracy 0 < 0.5 → CARE = accuracy
    w = earliness_weights(11)
    assert w[0] == 1 and w[5] == 1 and w[-1] == 0
    fd = tmp_path / "fd.csv"
    fd.write_text("sensor_name;statistics_type;description;unit;is_angle;is_counter\n"
                  "sensor_0;average;Ambient temperature;C;False;False\n"
                  "sensor_12;average;Temperature oil in gearbox;C;False;False\n"
                  "power_30;average;Grid power;kW;False;False\n", encoding="latin-1")
    assert column_map(fd) == {"turbine_t": "sensor_0_avg", "gearbox_t": "sensor_12_avg", "act_pwt": "power_30_avg"}
    monkeypatch.setattr(download, "archive_url", lambda log=print: None)  # offline
    assert download.fetch(log=lambda *a: None) is None


def test_nbm_residual_source_smoke(tmp_path):
    pytest.importorskip("torch")
    import argparse

    from wattsentry.model.wind.data import make_splits
    from wattsentry.model.windfailure.residuals_nbm import nbm_residuals

    d = generate(SynthConfig(days=40, scale=0.08))
    sp = make_splits(d)
    args = argparse.Namespace(seed=0, device="cpu", nbm_hidden=16, nbm_lr=2e-3, nbm_epochs=1, nbm_steps=2,
                              nbm_max_minutes=1.0)
    r, meta = nbm_residuals(d, sp, np.ones(d.n_steps, bool), args, log=lambda *a: None, out=tmp_path)
    assert r.shape == (d.n_steps, len(d.turbines), len(COMPONENTS)) and np.isfinite(r).mean() > 0.8
    assert (tmp_path / meta["onnx"]).exists() and meta["turbine_index"]
