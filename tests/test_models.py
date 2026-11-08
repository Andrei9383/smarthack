"""Card #14: synthetic data determinism, split leakage, features, metrics, predictor envelope, train.py registry.

Runs under the lab check (requirements.txt only, no torch); torch-only tests use pytest.importorskip.
"""

import importlib.util
import json
import sys
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from wattsentry.contracts import Prediction, PvFault, Window  # noqa: E402
from wattsentry.model import features as F  # noqa: E402
from wattsentry.model import metrics as M  # noqa: E402
from wattsentry.model import physics as ph  # noqa: E402
from wattsentry.model.data import ArraySource, H, make_splits, window_labels  # noqa: E402
from wattsentry.model.synth import FAULTS, SynthConfig, generate  # noqa: E402

SMALL = SynthConfig(n_sites=12, days=90, seed=3)


@pytest.fixture(scope="module")
def small_src(tmp_path_factory):
    out = tmp_path_factory.mktemp("synth") / "a"
    return ArraySource(generate(SMALL, out, verbose=False))


def test_synth_deterministic(small_src, tmp_path):
    b = ArraySource(generate(SMALL, tmp_path / "b", verbose=False))
    for name in ("pv_ac", "load", "soc", "inv_v", "fault", "weather_fc", "pvlib_rec"):
        assert np.array_equal(np.asarray(small_src.array(name)), np.asarray(b.array(name))), name
    assert small_src.sites == b.sites
    assert small_src.episodes == b.episodes


def test_synth_physical_ranges(small_src):
    pv = np.asarray(small_src.array("pv_healthy"), np.float32)
    inv = small_src.site_meta("inverter_kw")
    assert (pv >= 0).all() and (pv <= inv[:, None] + 1e-2).all()  # inverter clipping
    geom = np.asarray(small_src.array("geom"))[small_src.loc_idx()]
    night = geom[..., 0] > 95
    assert pv[night].max() < 1e-3
    soc = np.asarray(small_src.array("soc"), np.float32)
    assert soc.min() >= 0 and soc.max() <= 1.0 + 1e-3
    assert (np.asarray(small_src.array("load"), np.float32) > 0).all()
    assert set(np.unique(np.asarray(small_src.array("fault")))) <= set(range(len(FAULTS)))
    assert FAULTS == [f.value for f in PvFault]


def test_fault_onsets_exact(small_src):
    f = np.asarray(small_src.array("fault"))
    for e in small_src.episodes:
        k = FAULTS.index(e["fault"])
        assert f[e["site"], e["onset"]] == k
        if e["onset"] > 0:
            assert f[e["site"], e["onset"] - 1] != k or e["onset"] == 0


def test_wrong_metadata_fraction():
    src_sites = __import__("wattsentry.model.synth", fromlist=["_sites"])._sites(SynthConfig(n_sites=400),
                                                                                np.random.default_rng(0))
    frac = np.mean([s["meta_wrong"] for s in src_sites])
    assert 0.15 < frac < 0.25
    unseen = np.mean([s["split"] == "unseen" for s in src_sites])
    assert abs(unseen - 0.30) < 0.01


def test_splits_no_leakage():
    from wattsentry.model.data import Splits  # noqa: F401

    class Fake:
        sites = [{"split": "unseen" if i % 10 < 3 else "train"} for i in range(50)]

        def day_dates(self):
            import pandas as pd

            return pd.date_range("2022-01-01", periods=1461, freq="D", tz="UTC")

    sp = make_splits(Fake())
    assert not set(sp.train_sites) & set(sp.unseen_sites)
    days = {k: set(v.tolist()) for k, v in sp.days.items()}
    for a in days:
        for b in days:
            if a < b:
                assert not days[a] & days[b], (a, b)
    dates = Fake().day_dates()
    assert set(dates[sp.days["train"]].year) == {2022, 2023}
    assert set(dates[sp.days["val"]].year) == {2024} and set(dates[sp.days["cal"]].year) == {2024}
    assert set(dates[sp.days["test"]].year) == {2025}
    s_tr, _ = sp.site_days("train")
    s_va, _ = sp.site_days("val")
    s_te, _ = sp.site_days("test")
    assert not set(s_tr) & set(sp.unseen_sites) and not set(s_va) & set(sp.unseen_sites)
    assert set(sp.unseen_sites) <= set(s_te)
    assert min(sp.days["train"]) >= 14  # history never reaches before the data


def test_features_shapes_and_history_alignment(small_src):
    sites = np.array([0, 1, 2])
    starts = np.array([20 * H, 30 * H + 17, 40 * H])
    raw = F.gather(small_src, sites, starts, "pv")
    # history row k, step j is exactly (14 - k) days before target step j
    pv = np.asarray(small_src.array("pv_ac"))
    assert np.array_equal(raw["hist_pv"][1, -H:], pv[1, starts[1] - H: starts[1]].astype(np.float32))
    x = F.pv_inputs(raw)
    assert x["seq"].shape == (3, H, len(F.PV_SEQ))
    assert x["summ"].shape == (3, 14, len(F.PV_SUM)) and x["stat"].shape == (3, len(F.PV_STATIC))
    assert x["ref"].shape == (3, 7 * 24)
    xl = F.load_inputs(F.gather(small_src, sites, starts, "load"))
    assert xl["seq"].shape == (3, H, len(F.LOAD_SEQ)) and xl["summ"].shape == (3, 14, len(F.LOAD_SUM))
    m1 = np.zeros((len(small_src.sites), small_src.n_steps, 3), np.float16)
    m2 = np.zeros((len(small_src.sites), small_src.n_steps), np.float16)
    xs = F.site_inputs(F.gather(small_src, sites, starts, "site", m1=m1, m2=m2))
    assert xs["seq"].shape == (3, H, len(F.SITE_SEQ)) and xs["summ"].shape == (3, 14, len(F.SITE_SUM))
    for d in (x, xl, xs):
        for k in ("seq", "summ", "stat"):
            assert np.isfinite(d[k]).all(), k


def test_window_labels():
    fault = np.zeros((3, H), int)
    day = np.zeros((3, H), bool)
    day[:, 30:70] = True
    fault[0, 50:60] = 3  # string outage in daylight -> labelled
    fault[1, 0:20] = 2  # shading at night only -> invisible -> healthy
    fault[2, 0:20] = FAULTS.index("battery_fade")  # battery fade visible at night
    assert window_labels(fault, day).tolist() == [3, 0, FAULTS.index("battery_fade")]


def test_metrics():
    y = np.array([[0.0, 1.0, 2.0, 3.0]])
    q = np.stack([y - 0.5, y, y + 0.5], -1)
    assert M.nmae(q[..., 1], y) == 0
    assert M.coverage(q, y) == 1.0
    assert M.nmae(y * 0, y) == 1.0
    r = np.array([[0.0, 1.0, 1.0, 0.0, 2.0]])
    assert np.allclose(M.cusum(r, 0.5), [[0, 0.5, 1.0, 0.5, 2.0]])
    probs = np.eye(8)[[0, 1, 2, 1]]
    mac, per = M.prauc_macro(probs, np.array([0, 1, 2, 1]))
    assert mac == pytest.approx(1.0)


def test_train_registry_errors_cleanly(capsys):
    spec = importlib.util.spec_from_file_location("train_mod", ROOT / "train.py")
    tm = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(tm)
    assert {"pvnet", "loadnet", "sitehealth", "windnbm", "demandnet", "windpower", "windfailure",
            "gridtwin"} <= set(tm.REGISTRY)
    with pytest.raises(SystemExit, match="unknown model"):
        tm.main(["--model", "nope"])
    with pytest.raises(SystemExit):
        tm.main([])
    # a registered model whose card has not landed: clear error, not a traceback
    tm.REGISTRY["ghost"] = "wattsentry.model.ghost_card_xyz.train:main"
    with pytest.raises(SystemExit, match="does not exist yet"):
        tm.main(["--model", "ghost", "--lr", "1e-3"])
    assert tm.split_argv(["--lr", "3e-4", "--model", "pvnet", "--seed", "1"]) == ("pvnet", ["--lr", "3e-4", "--seed", "1"])
    assert tm.split_argv(["--model=loadnet"]) == ("loadnet", [])


def test_train_module_targets_resolve_for_14():
    """pvnet / loadnet / sitehealth modules import without torch (torch is imported lazily inside run())."""
    from wattsentry.model import forecast, sitehealth

    assert callable(forecast.main_pvnet) and callable(forecast.main_loadnet) and callable(sitehealth.main)


# --- torch-dependent ---------------------------------------------------------------------------------------
def _tiny_window(src, site, start, kind="pv"):
    raw = F.gather(src, np.array([site]), np.array([start]), "site",
                   m1=np.zeros((len(src.sites), src.n_steps, 3), np.float16),
                   m2=np.zeros((len(src.sites), src.n_steps), np.float16))
    s = src.sites[site]
    asset = {k: s[k] for k in ("site_id", "lat", "lon", "kwp_recorded", "tilt_recorded", "azimuth_recorded",
                               "inverter_kw", "battery_kwh", "battery_kw", "archetype")}
    t0 = (src.t_start + np.timedelta64(int(start) * 15, "m")).strftime("%Y-%m-%dT%H:%M:%SZ")
    w = Window(t0=t0, weather=raw["wx"][0], solar_geometry=raw["geom"][0], reference=raw["hist_pv"][0] * 1000,
               history=raw["hist_load"][0] * 1000, battery=np.stack([raw["soc"][0], raw["batt"][0] * 1000], 1),
               meta={"telemetry": {"pv_ac_w": raw["pv"][0] * 1000, "inv_voltage_v": raw["volt"][0],
                                   "inv_status": raw["status"][0], "load_w": raw["load"][0] * 1000},
                     "telemetry_history": {"inv_voltage_v": raw["hist_volt"][0], "inv_status": raw["hist_status"][0],
                                           "soc": raw["hist_soc"][0], "batt_w": raw["hist_batt"][0] * 1000}})
    return asset, w


def test_predictor_envelope_with_tiny_onnx(small_src, tmp_path):
    torch = pytest.importorskip("torch")
    pytest.importorskip("onnx")
    from wattsentry.model.forecast import Wrapped, fit_ood
    from wattsentry.model.harness import export_onnx
    from wattsentry.model.nets import QuantileEncoder, SiteHealthNet
    from wattsentry.model.predictor import load_predictor
    from wattsentry.model.sitehealth import _wrap

    torch.manual_seed(0)
    sites, starts = np.array([0, 1, 2, 3]), np.array([20, 25, 30, 35]) * H
    for model, kind in (("pvnet", "pv"), ("loadnet", "load")):
        x = F.build(kind, F.gather(small_src, sites, starts, kind))
        stats = F.fit_stats([x])
        dims = {k: x[k].shape[-1] for k in ("seq", "ref", "summ", "stat")}
        for tag in ("", ".s1"):  # two members -> ensemble path; .s1 uses the pointwise branch + residual base
            extra = {} if not tag else {"pointwise": 8, "base": (2, stats["seq"]["mean"][2], stats["seq"]["std"][2])}
            net = QuantileEncoder(dims["seq"], dims["ref"], dims["summ"], dims["stat"], d=16, layers=1, heads=2, ff=32,
                                  **extra)
            feeds = {k: x[k] for k in ("seq", "ref", "summ", "stat")}
            feeds["mask"] = (~x["night"]).astype(np.float32)
            info = export_onnx(Wrapped.make(net, stats), feeds, tmp_path / f"{model}{tag}.onnx", ["q", "emb"])
            assert info["parity_ok"]
            with torch.no_grad():
                emb = Wrapped.make(net, stats)(*[torch.from_numpy(feeds[k]) for k in feeds])[1].numpy()
            emb = np.concatenate([emb + np.random.default_rng(0).normal(0, 0.1, emb.shape) for _ in range(20)])
            cal = {"model": model, "model_version": f"{model}-test", "ood": fit_ood(emb),
                   "quantile_offsets": {"buckets": [[0, 24], [24, 48], [48, 72], [72, 96]], "q10": [0.01] * 4,
                                        "q90": [0.02] * 4}, "thresholds": {"ood": 3.0}}
            (tmp_path / f"{model}{tag}.calibration.json").write_text(json.dumps(cal))
    m1 = np.zeros((len(small_src.sites), small_src.n_steps, 3), np.float16)
    m2 = np.zeros((len(small_src.sites), small_src.n_steps), np.float16)
    xs = F.site_inputs(F.gather(small_src, sites, starts, "site", m1=m1, m2=m2))
    st = F.fit_stats([xs])
    net = SiteHealthNet(xs["seq"].shape[-1], xs["summ"].shape[-1], xs["stat"].shape[-1], width=8)
    net.eval()
    export_onnx(_wrap(net, st, np.zeros(8, np.float32)), {k: xs[k] for k in ("seq", "summ", "stat")},
                tmp_path / "sitehealth.onnx", ["probs", "emb"])
    (tmp_path / "sitehealth.calibration.json").write_text(json.dumps({"model_version": "sitehealth-test",
                                                                      "thresholds": {"p_fault": 0.5}}))

    asset, w = _tiny_window(small_src, 1, 40 * H)
    for model in ("pvnet", "loadnet", "sitehealth"):
        p = load_predictor(model, tmp_path)
        pred = p.predict(asset, w)
        assert isinstance(pred, Prediction)
        pred.validate()
        assert pred.horizon == 96 and len(pred.q50) == 96 and pred.model == model
        assert pred.ood_score >= 0 and pred.member_std >= 0
        assert len(pred.extra["members"]) >= 1
        if model != "sitehealth":
            assert len(pred.extra["members"]) == 2 and pred.model_version.endswith("x2")
        else:
            assert set(pred.fault_probs) == set(FAULTS) and abs(sum(pred.fault_probs.values()) - 1) < 1e-3
            assert "p_fault_std" in pred.extra
    # eval.py-style inputs: a weather channel that is all NaN (PVDAQ snow_cm), an ERA5 cloud column, NaN telemetry
    # gaps and raw inverter codes (37 = overvoltage) must still give a valid envelope
    wx = np.asarray(w.weather, np.float64).copy()
    wx[:, 6] = np.nan
    tel = {k: np.asarray(v, np.float64).copy() for k, v in w.meta["telemetry"].items()}
    tel["pv_ac_w"][10:14] = np.nan
    tel["inv_status"] = np.where(np.arange(96) == 50, 37.0, 0.0)
    w2 = Window(t0=w.t0, weather=wx, reference=np.where(np.arange(len(w.reference)) % 97 == 0, np.nan, w.reference),
                history=w.history, battery=w.battery,
                meta={"weather_channels": list(ph.WEATHER_CHANNELS), "power_unit": "W", "telemetry": tel,
                      "telemetry_history": dict(w.meta["telemetry_history"], inv_status=np.zeros(14 * 96))})
    for model in ("pvnet", "loadnet", "sitehealth"):
        load_predictor(model, tmp_path).predict(asset, w2).validate()
    # PV: night steps are exactly zero
    pv = load_predictor("pvnet", tmp_path).predict(asset, w)
    zen = np.asarray(w.solar_geometry)[:, 0]
    assert max(abs(v) for v, z in zip(pv.q90, zen) if z > 100) == 0.0


def test_quantile_encoder_monotone_and_params():
    torch = pytest.importorskip("torch")
    from wattsentry.model.nets import QuantileEncoder, SiteHealthNet, n_params, pinball

    net = QuantileEncoder(len(F.PV_SEQ), 168, len(F.PV_SUM), len(F.PV_STATIC), d=96, layers=3, heads=4, ff=192)
    assert 0.2e6 < n_params(net) < 0.4e6
    q, emb = net(torch.randn(4, 96, len(F.PV_SEQ)), torch.randn(4, 168), torch.randn(4, 14, len(F.PV_SUM)),
                 torch.randn(4, len(F.PV_STATIC)), torch.ones(4, 96))
    assert q.shape == (4, 96, 3) and (q[..., 0] <= q[..., 1] + 1e-6).all() and (q[..., 1] <= q[..., 2] + 1e-6).all()
    assert float(pinball(q, torch.zeros(4, 96)).detach()) >= 0
    sh = SiteHealthNet(len(F.SITE_SEQ), len(F.SITE_SUM), len(F.SITE_STATIC))
    assert 0.05e6 < n_params(sh) < 0.15e6


def test_train_presets_expand():
    spec = importlib.util.spec_from_file_location("train_mod2", ROOT / "train.py")
    tm = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(tm)
    for name, argv in tm.PRESETS.items():
        model, _ = tm.split_argv(argv)
        assert model in tm.REGISTRY, name
    with pytest.raises(SystemExit, match="--preset"):
        tm.main(["--preset", "nope"])
    with pytest.raises(SystemExit, match="--preset"):
        tm.main(["--preset", "pv_now@sx"])


def test_predictor_status_mapping_matches_windows():
    pytest.importorskip("onnxruntime")
    from wattsentry.model.predictor import _status

    pv = np.array([0.0, 0.002, 1.0, 4.99, 0.0])
    raw = np.array([0, 0, 0, 0, 37])  # sim codes: 0 ok, 37 overvoltage
    assert _status(raw, pv, 5.0).tolist() == [0, 0, 1, 2, 3]  # #10 encoding: off, off, producing, limiting, trip
    enc = np.array([0, 1, 2, 3, 1])
    assert _status(enc, pv, 5.0).tolist() == enc.tolist()  # already encoded -> unchanged
