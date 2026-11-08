"""M5 WindPower: features, binned baseline, park aggregation, train + ONNX predict API (stand-in synth data).
The training test needs torch, so it is skipped where torch is missing."""

import numpy as np
import pytest

from wattsentry.eval.baselines.wind.curve import data_inputs
from wattsentry.model.wind.data import POWER, make_splits
from wattsentry.model.wind.synth import SynthConfig, generate
from wattsentry.model.windpower import WindPowerModel, park_aggregate
from wattsentry.model.windpower.binned import fit_binned
from wattsentry.model.windpower.features import FEATS, build, build_data, gather, weather_per_turbine


@pytest.fixture(scope="module")
def setup():
    d = generate(SynthConfig(days=45, scale=0.1))
    return d, make_splits(d)


def test_features_shapes_and_finite(setup):
    d, _ = setup
    base = build_data(d)
    T, N = d.n_steps, len(d.turbines)
    assert set(base) == set(FEATS)
    for k, x in base.items():
        assert x.dtype == np.float32 and x.shape[0] in (1, T) and x.shape[1] in (1, N), k
        assert np.isfinite(x).all(), k
    X = gather(base, np.array([0, T - 1]), np.array([0, N - 1]))
    assert X.shape == (2, len(FEATS))


def test_hub_wind_from_100m_when_meteo_missing(setup):
    d, _ = setup
    w = weather_per_turbine(d.weather, d.park_of)
    w["wind100_ms"] = np.full((d.n_steps, len(d.turbines)), 8.0)
    a = build(d.turbines, d.ts, {}, w)  # NWP-only input: 100 m wind sheared to hub height
    hub = np.array([t["hub_height_m"] for t in d.turbines])
    assert np.allclose(a["v"][0] * 15, 8.0 * (hub / 100) ** 0.14 * (a["rho"][0] * 0.05 / 1.225 + 1) ** (1 / 3), atol=0.05)


def test_binned_fallback_ordered_and_accurate(setup):
    d, sp = setup
    v = np.clip(np.nan_to_num(data_inputs(d)["met"]), 0, 35)
    y = d.sensors[POWER] / d.rated_kw[None]
    m = (d.labels == 0) & sp.train_t[:, None]
    bp = fit_binned(d.turbines, v, y, m)
    q = bp.predict_frac(d.turbines, v)
    assert np.all(np.diff(q, axis=-1) >= 0)
    ok = (d.labels == 0) & np.isfinite(y) & sp.val_t[:, None]
    assert np.abs(y - q[..., 1])[ok].mean() < 0.08


def test_park_aggregate_availability_and_spread():
    q = np.tile(np.array([80.0, 100.0, 130.0]), (4, 3, 1))  # [T=4, n=3, 3] kW
    out = park_aggregate(q, None, np.ones(4), (1.0, 1.0))
    assert np.allclose(out[0], [240, 300, 390])
    av = np.ones((4, 3))
    av[:, 0] = 0
    out = park_aggregate(q, av, np.full(4, 1.1), (2.0, 0.5))
    assert np.allclose(out[0], [220 - 80, 220, 220 + 30])


def test_train_end_to_end_and_onnx_api(tmp_path):
    pytest.importorskip("torch")
    pytest.importorskip("onnx")
    from wattsentry.model.windpower.train import main

    r = main(["--synth", "--synth-days", "45", "--synth-scale", "0.1", "--epochs", "1", "--max-rows", "20000",
              "--out", str(tmp_path)])
    assert {"model", "manufacturer", "persist", "binned"} <= set(r["park"][24])
    m = WindPowerModel.load(tmp_path / "windpower.s0.onnx")
    assert m.sessions and m.model_version.startswith("windpower-r")
    d = generate(SynthConfig(days=45, scale=0.1))
    sub = d.subset(t_slice=slice(0, 96))
    q = m.predict_frac(sub.turbines, sub.ts, sub.meteo, weather_per_turbine(sub.weather, sub.park_of))
    assert q.shape == (96, len(d.turbines), 3) and np.all(np.diff(q, axis=-1) >= 0) and np.isfinite(q).all()
    m2 = WindPowerModel.load([tmp_path / "windpower.s0.onnx"] * 2)  # seed ensemble path
    assert np.allclose(m2.predict_frac(sub.turbines, sub.ts, sub.meteo, weather_per_turbine(sub.weather, sub.park_of)), q,
                       atol=1e-5)
