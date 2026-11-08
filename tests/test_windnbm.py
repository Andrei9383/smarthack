"""Card #41 M6 WindNBM: stand-in data, contract conformance, normalisation, WindMon rules, splits, metrics.

Always-on tests need only requirements.txt; the end-to-end train → ONNX → predictor test needs torch.
"""

import os
import sys

if sys.platform == "darwin":  # torch + lightgbm OpenMP runtimes clash on macOS unless single-threaded
    os.environ.setdefault("OMP_NUM_THREADS", "1")

import numpy as np
import pytest

from wattsentry.contracts import WindAnomaly, is_error_code, parse_tag
from wattsentry.contracts.wind import SENSORS, TURBINE_META_COLUMNS
from wattsentry.model.wind.data import (
    ACTIONABLE,
    CLASS_INDEX,
    CLASSES,
    fault_labels_table,
    from_long,
    make_splits,
    meteo_table,
    to_long,
    turbines_table,
    weather_table,
)
from wattsentry.model.wind.features import clf_features, curtail_ok, decide, neighbours_for
from wattsentry.model.wind.metrics import event_metrics
from wattsentry.model.wind.physics import CLASS_2500, air_density, density_normalise, shear_to_hub, v_norm_from
from wattsentry.model.wind.synth import SynthConfig, generate
from wattsentry.model.wind.windmon_rules import windmon_classes, windmon_flags

TINY = SynthConfig(days=30, scale=0.08)


@pytest.fixture(scope="module")
def tiny():
    return generate(TINY)


def test_classes_match_contract():
    assert CLASSES == [a.value for a in WindAnomaly]


def test_synth_deterministic(tiny):
    again = generate(TINY)
    for k in tiny.sensors:
        np.testing.assert_array_equal(tiny.sensors[k], again.sensors[k])
    np.testing.assert_array_equal(tiny.labels, again.labels)
    np.testing.assert_array_equal(tiny.err, again.err)
    assert tiny.episodes == again.episodes
    other = generate(SynthConfig(days=30, scale=0.08, seed=7))
    assert not np.array_equal(np.nan_to_num(tiny.sensors["act_pwt"]), np.nan_to_num(other.sensors["act_pwt"]))


def test_synth_schema_and_plausibility(tiny):
    assert set(tiny.sensors) == set(SENSORS)
    assert {"WP1", "WP2", "WP3"} == set(tiny.park_ids)
    for t in tiny.turbines:
        assert set(TURBINE_META_COLUMNS) <= set(t)
    p = tiny.sensors["act_pwt"]
    assert np.nanmax(p / tiny.rated_kw[None]) <= 1.02
    assert 0.1 < np.nanmean(p / tiny.rated_kw[None]) < 0.6
    assert (tiny.labels != 0).any()


def test_tags_parse_and_codes_follow_contract(tiny):
    long = to_long(tiny.subset([0, 1], slice(0, 960)))
    assert list(long.columns) == ["ts", "tag", "value", "is_code"]
    tags = long["tag"].unique().to_list()
    sensors = set()
    for tag in tags:
        t = parse_tag(tag, strict=True)
        sensors.add(t.sensor)
        assert t.turbine_id in {tiny.turbines[0]["turbine_id"], tiny.turbines[1]["turbine_id"]}
    assert sensors == set(SENSORS)
    rows = long.drop_nulls("value").sample(min(3000, long.drop_nulls("value").height), seed=0).to_dicts()
    for r in rows:
        assert is_error_code(parse_tag(r["tag"]).sensor, r["value"]) == r["is_code"], r


def test_long_roundtrip_drops_in(tiny):
    """#22/#23 output in the contract tables loads into the same wide view M6 trains on."""
    sub = tiny.subset(np.arange(len(tiny.turbines)), slice(0, 2880))
    back = from_long(to_long(sub), turbines_table(sub), meteo=meteo_table(sub), weather=weather_table(sub),
                     fault_labels=fault_labels_table(sub))
    assert back.turbine_ids == sub.turbine_ids and len(back.ts) == len(sub.ts)
    for k in SENSORS:
        np.testing.assert_allclose(back.sensors[k], sub.sensors[k], atol=2e-3, equal_nan=True)
    np.testing.assert_array_equal(back.err, sub.err)
    np.testing.assert_allclose(back.meteo["wind_ms"], sub.meteo["wind_ms"], atol=1e-3)
    # episode labels cover whole episodes (synth labels curtailment / icing only where they bite)
    assert (back.labels == sub.labels).mean() > 0.97


def test_density_normalisation_numbers():
    assert air_density(1013.25, 15.0) == pytest.approx(1.22505, abs=1e-4)
    assert density_normalise(10.0, 1.225) == pytest.approx(10.0)
    rho = air_density(1000.0, -10.0)
    assert rho == pytest.approx(100000 / (287.05 * 263.15), rel=1e-9)
    assert v_norm_from(10.0, 1000.0, -10.0) == pytest.approx(10 * (rho / 1.225) ** (1 / 3), rel=1e-9)
    assert v_norm_from(10.0, 1000.0, -10.0) > 10.0  # cold dense air carries more power
    assert shear_to_hub(7.0, 120.0) == pytest.approx(7.0 * 1.2**0.14)
    assert CLASS_2500.power_frac(2.0) == 0 and CLASS_2500.power_frac(26.0) == 0
    assert CLASS_2500.power_frac(15.0) == pytest.approx(1.0, abs=1e-3)


def _park(T=12, n=4):
    s = {k: np.full((T, n), 30.0, np.float32) for k in SENSORS}
    s["act_pwt"][:] = 1500.0
    s["wd_spd"][:] = 40.0  # km/h
    s["turbine_t"][:] = 10.0
    s["act_pwt"][2:6, 0] = -4.2  # single-turbine stop while neighbours produce
    s["act_pwt"][8:12, :] = 0.3  # park-wide curtailment to zero
    noise = np.random.default_rng(0).normal(0, 0.05, (T, n)).astype(np.float32)  # real sensors are never frozen
    return {k: v + noise for k, v in s.items()}


def test_windmon_fires_on_stop_and_on_curtailment():
    s = _park()
    fl = windmon_flags(s, np.zeros((12, 4), np.int32), np.full(4, 2500.0))
    hw = fl["HIGH_WIND_ZERO_POWER"]
    assert hw[2:6, 0].all() and not hw[2:6, 1:].any()
    assert hw[8:12, :].all()  # the rule cannot tell curtailment from four faults: alert fatigue
    cls = windmon_classes(fl)
    assert (cls[8:12] == CLASS_INDEX["fault_stop"]).all()


def test_park_context_separates_stop_from_curtailment():
    s = _park()
    T, n = 12, 4
    pq = np.zeros((T, n, 3), np.float32)
    pq[..., 0], pq[..., 1], pq[..., 2] = 0.55, 0.6, 0.65
    tq = np.full((T, n, 5, 3), 30.0, np.float32)
    curve = {"edges": list(np.arange(0, 30.5, 0.5)), "values": [[0.6] * 60, [0.6] * 60]}
    lat = 44.5 + np.arange(n) * 0.003
    F, names = clf_features(s, np.zeros((T, n), np.int32), {"wind_ms": np.full((T, n), 11.0), "wdir_deg":
                            np.zeros((T, n))}, {}, np.full(n, 2500.0), np.zeros(n), neighbours_for(lat, lat * 0 + 28.5),
                            pq, np.zeros((T, n)), tq, curve)
    cap = F[..., names.index("park_cap_frac")]
    assert cap[3, 0] < 0.5 and (cap[9, :] > 0.9).all()


def test_holdout_park_never_in_train(tiny):
    sp = make_splits(tiny)
    park = np.array([t["park_id"] for t in tiny.turbines])
    assert sp.holdout_park == "WP2"
    assert set(park[sp.holdout_turbines]) == {"WP2"}
    assert "WP2" not in set(park[sp.train_turbines])
    assert not set(sp.train_turbines) & set(sp.holdout_turbines)
    assert not (sp.train_t & sp.val_t).any() and not (sp.train_t & sp.test_t).any() and not (sp.val_t & sp.test_t).any()
    full = make_splits(generate(SynthConfig(days=731, scale=0.05, parks=("WP1", "WP2"))))
    assert full.test_year == 2025


def test_event_metrics_and_decide():
    T = 200
    truth = np.zeros((T, 2), np.uint8)
    truth[20:40, 0] = CLASS_INDEX["fault_stop"]
    truth[100:120, :] = CLASS_INDEX["curtailment"]
    rules = np.zeros_like(truth)
    rules[20:40, 0] = CLASS_INDEX["fault_stop"]
    rules[100:120, :] = CLASS_INDEX["fault_stop"]
    m6 = truth.copy()
    r, m = event_metrics(truth, rules), event_metrics(truth, m6)
    assert r["wind_recall"] == m["wind_recall"] == 1.0
    assert r["n_alerts"] == 3 and m["n_alerts"] == 1
    assert m["wind_precision"] == 1.0 and r["wind_precision"] == pytest.approx(1 / 3)
    assert m["curtail_vs_fault_f1"] == 1.0 and r["curtail_vs_fault_f1"] < 0.5
    probs = np.zeros((T, 1, len(CLASSES)), np.float32)
    probs[..., 0] = 1.0
    probs[50:51, 0, 1], probs[50:51, 0, 0] = 0.9, 0.1  # one-step blip → removed by min_run
    probs[60:70, 0, 1], probs[60:70, 0, 0] = 0.9, 0.1
    th = {c: 0.5 for c in CLASSES[1:]}
    out = decide(probs, th)
    assert out[50, 0] == 0 and (out[60:70, 0] == 1).all()


def _park_ctx(T, n, cap_frac=0.0, missing_cols=()):
    """curtail_ok() on a minimal feature block: park_cap_frac_m4 + per-turbine missing power."""
    names = ["park_cap_frac_m4", "missing"]
    F = np.zeros((T, n, 2), np.float32)
    F[..., 0] = cap_frac
    F[:, list(missing_cols), 1] = 1.0
    return curtail_ok(F, names)


def _probs(T, n):
    p = np.zeros((T, n, len(CLASSES)), np.float32)
    p[..., 0] = 1.0
    return p


def test_park_capped_to_zero_is_curtailment_not_a_work_order():
    """#99: a park-wide cap to 0 % leaves noisy near-zero power, so no common ceiling (park_cap_frac_m4 = 0).
    The park consensus on p_curtailment still calls it curtailment: logged, no alert, no work order."""
    T, n = 160, 70
    ci = CLASS_INDEX["curtailment"]
    probs = _probs(T, n)
    probs[40:100, :, ci], probs[40:100, :, 0] = 0.999, 0.001
    probs[40:100, :10, ci], probs[40:100, :10, 0] = 0.2, 0.8  # a few turbines the classifier is unsure about
    ctx = _park_ctx(T, n, cap_frac=0.0, missing_cols=range(60, 70))  # 10 turbines not reporting
    th = {c: 0.5 for c in CLASSES[1:]}
    out = decide(probs, th, curtail_mask=ctx)
    assert (out[40:100, 10:60] == ci).all()
    assert (out[:, 60:] != ci).all()  # no power reading: no vote, not released (data_gap's job)
    assert (out[:40] == 0).all() and (out[100:] == 0).all()
    assert not np.isin(out, [CLASS_INDEX[c] for c in ACTIONABLE]).any()
    truth = np.zeros((T, n), np.uint8)
    truth[40:100] = ci
    m = event_metrics(truth, out)
    assert m["n_alerts"] == 0
    # below the consensus share (and no ceiling) the veto still holds
    weak = probs.copy()
    weak[40:100, :35, ci], weak[40:100, :35, 0] = 0.2, 0.8
    assert (decide(weak, th, curtail_mask=ctx) != ci).all()
    # with a low tuned θ_curtailment, a turbine the classifier sees as iced keeps icing: only confident
    # turbines vote, and only voting turbines are released from the veto
    ii = CLASS_INDEX["icing"]
    iced = probs.copy()
    iced[40:100, 10:20, ci], iced[40:100, 10:20, ii], iced[40:100, 10:20, 0] = 0.3, 0.9, 0.0
    out_i = decide(iced, dict(th, curtailment=0.15, icing=0.85), curtail_mask=ctx)
    assert (out_i[40:100, 10:20] == ii).all() and (out_i[40:100, 20:60] == ci).all()
    # a turbine overheating during the cap keeps its actionable class (the consensus never out-votes it)
    oi = CLASS_INDEX["overheating"]
    hot = probs.copy()
    hot[40:100, 30, oi] = 0.95
    out_h = decide(hot, th, curtail_mask=ctx)
    assert (out_h[40:100, 30] == oi).all() and (out_h[40:100, 31:60] == ci).all()
    # ... also when its run has short sub-threshold gaps the smoothing fills (the consensus never splits it)
    gap = hot.copy()
    gap[60:64, 30, oi] = 0.3
    out_g = decide(gap, th, curtail_mask=ctx)
    assert (out_g[40:100, 30] == oi).all()
    # a park-wide cap at a common ceiling is still allowed by the ceiling test alone
    assert (decide(weak, th, curtail_mask=_park_ctx(T, n, cap_frac=0.5))[40:100, 35:] == ci).all()


def test_lone_low_turbine_is_never_hidden_as_curtailment():
    """#99: one underperforming turbine in a normal park keeps its actionable class even when the classifier
    leans to curtailment, also when its park is tuned together with a capped park (consensus is per park) and
    in a 2-turbine park (one turbine is half the park, but never a consensus)."""
    T = 160
    ci, ui = CLASS_INDEX["curtailment"], CLASS_INDEX["underperformance"]
    th = {c: 0.5 for c in CLASSES[1:]}
    probs = _probs(T, 20)
    probs[30:100, 3, ci], probs[30:100, 3, ui], probs[30:100, 3, 0] = 0.9, 0.6, 0.0
    out = decide(probs, th, curtail_mask=_park_ctx(T, 20))
    assert (out[30:100, 3] == ui).all() and (out != ci).all()
    # concatenated with a park that IS capped at the same time (how train.py tunes on several parks)
    capped = _probs(T, 30)
    capped[30:100, :, ci], capped[30:100, :, 0] = 0.999, 0.001
    both = np.concatenate([probs, capped], axis=1)
    ctx = np.concatenate([_park_ctx(T, 20), _park_ctx(T, 30)], axis=1)
    out2 = decide(both, th, curtail_mask=ctx)
    assert (out2[30:100, 3] == ui).all() and (out2[:, :20] != ci).all()
    assert (out2[30:100, 20:] == ci).all()
    # tiny park: one of two turbines is 50 % but still a lone turbine
    small = _probs(T, 2)
    small[30:100, 0, ci], small[30:100, 0, ui], small[30:100, 0, 0] = 0.9, 0.6, 0.0
    assert (decide(small, th, curtail_mask=_park_ctx(T, 2))[30:100, 0] == ui).all()


def test_train_export_predict_end_to_end(tmp_path):
    import lightgbm  # noqa: F401  (load before torch: macOS OpenMP clash)

    pytest.importorskip("torch")
    pytest.importorskip("onnx")
    from wattsentry.model.wind.predictor import WindNBMPredictor, window_from_data
    from wattsentry.model.wind.train import main

    main(["--synth", "--days", "120", "--scale", "0.12", "--epochs", "1", "--steps-per-epoch", "10", "--batch", "32",
          "--clf-rounds", "15", "--max-minutes", "2", "--device", "cpu", "--out", str(tmp_path), "--no-eval-seen"])
    for f in ("windnbm.onnx", "windnbm_clf.txt", "windnbm_thresholds.json", "manifest.json", "windnbm_report.json"):
        assert (tmp_path / f).exists(), f
    d = generate(SynthConfig(days=120, scale=0.12))
    pr = WindNBMPredictor(tmp_path)
    w = window_from_data(d, "WP2", "2024-04-20T00:00:00Z", horizon=96)
    p = pr.predict(w.park_context["turbines"][0], w)
    p.validate()
    assert p.model == "windnbm" and p.horizon == 96 and p.anomaly_class in CLASSES
    assert p.ood_score >= 0 and p.member_std >= 0


def test_label_free_normal_selection_beats_base_rate():
    """#70: telemetry-only normal selection is cleaner than taking every step, and never peeks at the labels."""
    from wattsentry.model.wind.features import density_inputs
    from wattsentry.model.wind.normal_select import select_normal, selection_quality

    d = generate(SynthConfig(days=120, scale=0.12))
    vn, vm, _ = density_inputs(d.sensors, d.meteo, d.weather["pressure_hpa"][:, d.park_of])
    fit = np.zeros(d.n_steps, dtype=bool)
    fit[: int(0.7 * d.n_steps)] = True
    power_ok, temp_ok = select_normal(d.sensors, d.err, d.rated_kw, vn, vm, fit, d.meteo["wdir_deg"])
    assert power_ok.shape == temp_ok.shape == d.labels.shape
    q = selection_quality(power_ok, d.labels)
    assert q["precision"] > (d.labels == 0).mean() + 0.05
    assert q["recall"] > 0.7
    assert q["kept_data_gap"] == 0.0
    assert q["kept_curtailment"] < 0.15
    # same telemetry, scrambled labels -> same selection (labels are diagnostics only)
    d2 = generate(SynthConfig(days=120, scale=0.12))
    d2.labels[:] = 0
    p2, _ = select_normal(d2.sensors, d2.err, d2.rated_kw, vn, vm, fit, d2.meteo["wdir_deg"])
    np.testing.assert_array_equal(power_ok, p2)
