"""Card #24 wind baselines: binned curve, curve rules, WindMon shim, contract predictors (stand-in synth data)."""

import numpy as np
import pytest

from sentinel.contracts import Predictor
from sentinel.eval.baselines.wind import windmon_rules
from sentinel.eval.baselines.wind.curve import BinnedCurve, data_inputs, fit_curves
from sentinel.eval.baselines.wind.detector import detect_all
from sentinel.eval.baselines.wind.predictor import BinnedCurvePredictor, WindMonRulesPredictor
from sentinel.model.wind.data import CLASS_INDEX, POWER, make_splits
from sentinel.model.wind.metrics import event_metrics
from sentinel.model.wind.predictor import window_from_data
from sentinel.model.wind.synth import SynthConfig, generate


@pytest.fixture(scope="module")
def setup():
    d = generate(SynthConfig(days=60, scale=0.1))
    sp = make_splits(d)
    return d, sp, fit_curves(d, sp.train_t, sp.train_turbines)


def test_shim_reexports_rules():
    from sentinel.model.wind import windmon_rules as shim

    assert shim.windmon_flags is windmon_rules.windmon_flags and shim.RULES == windmon_rules.RULES


def test_curve_quantiles_ordered_and_fit_power(setup, tmp_path):
    d, sp, c = setup
    for s in c.curves:
        for q in c.curves[s].values():
            assert np.all(np.diff([q["q10"], q["q50"], q["q90"]], axis=0) >= 0)
    q = c.predict_block(d.turbines, data_inputs(d), "met")
    y = d.sensors[POWER] / d.rated_kw[None]
    ok = (d.labels == 0) & np.isfinite(y) & sp.val_t[:, None]
    assert np.nanmean(np.abs(y - q[..., 1])[ok]) < 0.08
    assert BinnedCurve.load(c.save(tmp_path / "c.json")).curves == c.curves


def test_curve_rules_separate_curtailment_from_stop(setup):
    d, sp, c = setup
    d = d.subset(t_slice=slice(None))
    d.labels[:] = 0
    cols = np.flatnonzero(d.park_of == 0)
    met = c.predict_block(d.turbines, data_inputs(d), "met")[..., 1]
    t = int(np.argmax((met[:, cols] > 0.5).all(1)[:-20] & (met[:, cols] > 0.5).all(1)[20:]))
    d.sensors[POWER][t : t + 8, cols] = 0.2 * d.rated_kw[cols]  # park-wide cap
    d.sensors[POWER][t + 12 : t + 20, cols[0]] = 0.0  # one turbine stops
    pred = detect_all(c, d)
    assert (pred[t : t + 8, cols] == CLASS_INDEX["curtailment"]).mean() > 0.8
    assert (pred[t + 12 : t + 20, cols[0]] == CLASS_INDEX["fault_stop"]).mean() >= 0.75  # 6-turbine park: noisy


def test_event_metrics_run_on_both_detectors(setup):
    d, sp, c = setup
    wm = windmon_rules.windmon_classes(windmon_rules.windmon_flags(d.sensors, d.err, d.rated_kw))
    for pred in (detect_all(c, d), wm):
        m = event_metrics(d.labels, pred, full=False)
        assert 0 <= m["wind_precision"] <= 1 and 0 <= m["wind_recall"] <= 1


def test_predictors_follow_contract(setup):
    d, sp, c = setup
    park, t0 = d.park_ids[0], str(d.ts[40 * 96]) + ":00Z"
    w = window_from_data(d, park, t0, horizon=96, history_days=28)
    asset = next(t for t in d.turbines if t["park_id"] == park)
    for p in (BinnedCurvePredictor(c), BinnedCurvePredictor(c, "forecast"), WindMonRulesPredictor(c)):
        assert isinstance(p, Predictor)
        pr = p.predict(asset, w)
        assert pr.horizon == 96 and pr.model == p.model and max(pr.q90) <= 1.2 * asset["rated_kw"]
        assert (pr.anomaly_class is None) == (p.mode == "forecast")


def test_anomalies_table(setup):
    d, _, _ = setup
    f = windmon_rules.windmon_flags(d.sensors, d.err, d.rated_kw)
    df = windmon_rules.anomalies_table(f, d.ts, d.turbine_ids)
    assert df.height == windmon_rules.raw_alert_count(f) and set(df["rule"].unique()) <= set(windmon_rules.RULES)
