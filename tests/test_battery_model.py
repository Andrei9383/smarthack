"""Battery behaviour model: no hidden truth in features, day-ahead information rule, K8 simulation,
portfolio behaviour shape, classifier round trip, and a tiny end-to-end run on generated data-v2 (slow)."""

from __future__ import annotations

import inspect
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

pytest.importorskip("wattsentry.kernels.strategies")

from wattsentry.model.battery import classifier as C
from wattsentry.model.battery import data as D
from wattsentry.model.battery import features as F
from wattsentry.model.battery import net_model as N

HIDDEN = {"pv_kw", "load_kw", "batt_kw", "soc_kwh", "strategy", "has_battery", "cap_kwh", "p_kw", "reserve_frac",
          "k_top", "eta", "pv_ac_w", "load_w"}


def _world(n_days: int = 40, homes: pd.DataFrame | None = None, seed: int = 0) -> D.World:
    """A small public World (no files): March-April 2024 local days (no DST change before Mar 31 handled by cal)."""
    from wattsentry.kernels.strategies import day_rank_median
    from wattsentry.sim.fleet.ro_calendar import local_features

    ts = pd.date_range(pd.Timestamp("2024-05-01", tz=D.TZ).tz_convert("UTC"), periods=n_days * 96, freq="15min")
    cal = local_features(ts)
    cal["dow"] = ts.tz_convert(D.TZ).day_of_week.to_numpy()
    day = cal["day_index"] - cal["day_index"][0]
    cal["day_index"] = day
    rng = np.random.default_rng(seed)
    hour = cal["hour"]
    price = 400 + 300 * np.exp(-((hour - 19.5) ** 2) / 2) - 200 * np.exp(-((hour - 13) ** 2) / 6) + rng.normal(0, 5, len(ts))
    rank, med = day_rank_median(price, day)
    homes = homes if homes is not None else pd.DataFrame({
        "home_id": ["H1", "H2"], "locality_id": ["x", "x"], "county": ["c", "c"], "lat": [44.4, 44.4],
        "lon": [26.1, 26.1], "kwp": [5.0, 5.0], "tilt_recorded": [30.0, 30.0], "azimuth_recorded": [180.0, 180.0],
        "inverter_kw": [5.0, 5.0], "contract": ["maxcontrol", "standard"], "meter": ["smart", "smart"],
        "battery_declared": [True, False], "enrolled": [False, False], "split": ["train", "train"]})
    n = int(day[-1]) + 1
    ds = np.searchsorted(day, np.arange(n))
    return D.World(out=Path("."), ts=ts, cal=cal, year=ts.tz_convert(D.TZ).year.to_numpy(), pzu=price, price=price,
                   rank=rank, med=med, load_coef=1.0 + 0.5 * ((hour >= 18) & (hour < 22)), homes=homes,
                   weather=pd.DataFrame(), n_days=n, day_start=ds,
                   day_month=(ts.tz_convert(D.TZ).month.to_numpy()[ds] - 5), month_keys=["2024-05", "2024-06"])


def _pv(world: D.World, n: int, kwp: float = 5.0) -> np.ndarray:
    h = world.cal["hour"]
    return np.tile(np.clip(np.sin((h - 6) / 14 * np.pi), 0, None) * 0.7 * kwp, (n, 1)).astype(np.float32)


# ------------------------------------------------------------------------------------------- leakage guards
def test_feature_names_never_hidden():
    assert not HIDDEN & set(F.CLS_FEATURES)
    assert not HIDDEN & set(F.ROW_FEATURES)
    try:
        from wattsentry.sim.prosumers_v2 import FORBIDDEN_COLUMNS
    except ImportError:
        return
    assert not set(FORBIDDEN_COLUMNS) & (set(F.CLS_FEATURES) | set(F.ROW_FEATURES))


def test_hidden_files_read_only_by_labels_and_truth():
    src = {m.__name__: inspect.getsource(m) for m in (D, F, C, N)}
    from wattsentry.model.battery import api, train

    src.update({m.__name__: inspect.getsource(m) for m in (api, train)})
    for name, s in src.items():
        for hidden in ('"homes.parquet"', "truth/home_15min", "load_truth"):
            if name.endswith(".data") and hidden in ('"homes.parquet"', "truth/home_15min"):
                continue
            assert hidden not in s, f"{name} reads hidden {hidden}"
    assert "labels(" in inspect.getsource(D.labels) and '"homes.parquet"' in inspect.getsource(D.labels)
    assert "truth/home_15min" in inspect.getsource(D.truth_chunk)


def test_strategy_features_use_only_days_up_to_d_minus_2():
    w = _world(40)
    n = 2
    rng = np.random.default_rng(1)
    imp = rng.random((n, w.T)).astype(np.float32)
    exp = rng.random((n, w.T)).astype(np.float32)
    pv, load = _pv(w, n), np.ones((n, w.T), np.float32)
    kwp = np.array([5.0, 5.0])
    m = np.full((n, 2), 100.0)
    d = 35
    base = F.strategy_features(w, F.daily_stats(w, imp, exp, pv, load, kwp), w.homes, np.array([d]), m, m)
    imp2, exp2 = imp.copy(), exp.copy()
    for dd in (d - 1, d):  # meter data PPC does not have yet when forecasting day d
        a, b = w.day_start[dd], w.day_start[dd + 1] if dd + 1 < w.n_days else w.T
        imp2[:, a:b] *= 7
        exp2[:, a:b] *= 7
    again = F.strategy_features(w, F.daily_stats(w, imp2, exp2, pv, load, kwp), w.homes, np.array([d]), m, m)
    for k in F.CLS_FEATURES:
        np.testing.assert_allclose(base[k], again[k], equal_nan=True, err_msg=k)


def test_price_driven_signature():
    """A home that exports in the expensive evening quarter-hours (no sun) gets high top-price / no-sun export."""
    w = _world(40)
    pv, load = _pv(w, 2), np.full((2, w.T), 0.5, np.float32)
    exp = np.clip(pv - load, 0, None) * 0.25
    exp[0] += np.where(w.rank < 12, 1.0, 0.0)  # home 0 discharges into the grid at the top prices
    imp = np.zeros_like(exp)
    m = np.full((2, 2), 100.0)
    f = F.strategy_features(w, F.daily_stats(w, imp, exp, pv, load, np.array([5.0, 5.0])), w.homes, np.array([35]), m, m)
    assert f["share_exp_top_price"][0, 0] > 0.3 > 0.05 > f["share_exp_top_price"][1, 0]
    assert f["exp_nosun_per_kwp"][0, 0] > 0.5 and f["exp_nosun_per_kwp"][1, 0] < 0.05
    assert f["share_exp_18_21"][0, 0] > f["share_exp_18_21"][1, 0]


def test_load_scale_reproduces_monthly_import():
    w = _world(40)
    pv = _pv(w, 2)
    true_c = np.array([0.8, 1.6])
    load = true_c[:, None] * w.load_coef[None]
    sm = w.day_month[w.cal["day_index"]]
    imp_m = np.stack([(np.clip(load - pv, 0, None)[:, sm == k] * 0.25).sum(1) for k in range(2)], 1)
    c = F.load_scale(w, pv, imp_m)
    np.testing.assert_allclose(c, np.repeat(true_c[:, None], 2, 1), rtol=1e-4)


# ------------------------------------------------------------------------------------------- simulation
def _params(n, cap=10.0):
    return {"cap_kwh": np.full(n, cap), "p_kw": np.full(n, 4.0), "reserve_frac": np.full(n, 0.2),
            "k_top": np.full(n, 12, np.int32)}


def test_simulate_strategies():
    w = _world(5)
    pv, load = _pv(w, 3), np.full((3, w.T), 0.6, np.float32)
    code = np.array([0, 1, 2], np.int8)
    net, soc = N.simulate(pv, load, code, _params(3), w.price, w.rank, w.med, w.cal["day_index"])
    np.testing.assert_allclose(net[0], load[0] - pv[0], atol=1e-5)
    export_cap = np.clip(pv - load, 0, None)
    assert (-net[1] <= export_cap[1] + 1e-5).all(), "self-consumption never exports from the battery"
    nosun_top = (pv[2] == 0) & (w.rank < 12)
    assert (net[2][nosun_top] < 0).any(), "price-driven exports at the top prices without sun"
    assert soc.shape == (3,)


def test_mix_soft_and_hard():
    var = {0: np.zeros((1, 4), np.float32), 1: np.ones((1, 4), np.float32), 2: np.full((1, 4), 3, np.float32)}
    probs = np.array([[[0.1, 0.2, 0.6, 0.1], [0.7, 0.1, 0.1, 0.1]]])  # 2 days, enrolled counts as self
    step_day = np.array([0, 0, 1, 1])
    soft = N.mix(var, probs, step_day)
    np.testing.assert_allclose(soft[0], [0.3 + 1.8, 0.3 + 1.8, 0.2 + 0.3, 0.2 + 0.3], rtol=1e-5)
    hard = N.mix(var, probs, step_day, hard=True)
    np.testing.assert_allclose(hard[0], [3, 3, 0, 0])


# ------------------------------------------------------------------------------------------- models
def test_classifier_and_quantile_roundtrip(tmp_path):
    rng = np.random.default_rng(0)
    n = 1200
    y = rng.integers(0, 3, n)
    X = rng.normal(size=(n, len(F.CLS_FEATURES))).astype(np.float32)
    X[:, F.CLS_FEATURES.index("share_exp_top_price")] += 3 * (y == 2)
    X[:, F.CLS_FEATURES.index("m_exp_ratio")] -= 3 * (y == 1)
    P = {k: np.where(y > 0, C.PRIOR[k] + rng.normal(0, 0.1, n), np.nan) for k in C.PARAMS}
    clf = C.StrategyClassifier.fit(X, y, P, rounds=60, min_leaf=10, threads=1, log=lambda *a: None)
    g = clf.guess(X)
    assert C.macro_f1(y, g["code"])[0] > 0.9
    assert g["probs"].shape == (n, 4) and ((g["confidence"] > 0) & (g["confidence"] <= 1)).all()
    assert (g["cap_kwh"] >= C.BOUNDS["cap_kwh"][0]).all() and g["k_top"].dtype == np.int32
    clf.save(tmp_path)
    np.testing.assert_allclose(C.StrategyClassifier.load(tmp_path).predict_proba(X), clf.predict_proba(X))

    Xr = rng.normal(size=(2000, len(F.ROW_FEATURES))).astype(np.float32)
    yr = Xr[:, 0] + rng.normal(0, 0.3, 2000)
    q = N.NetQuantileModel.fit(Xr, yr, rounds=40, min_leaf=20, threads=1, log=lambda *a: None)
    p = q.predict(Xr)
    assert p.shape == (2000, 3) and (np.diff(p, axis=1) >= 0).all()
    inside = ((yr >= p[:, 0]) & (yr <= p[:, 2])).mean()
    assert 0.6 < inside < 0.95
    q.save(tmp_path)
    np.testing.assert_allclose(N.NetQuantileModel.load(tmp_path).predict(Xr), p, rtol=1e-6)


def test_registered_in_train_py():
    import train

    assert train.REGISTRY["battery"] == "wattsentry.model.battery.train:main"


# ------------------------------------------------------------------------------------------- end to end (slow)
@pytest.mark.slow
def test_end_to_end_tiny(tmp_path):
    gen = pytest.importorskip("wattsentry.sim.prosumers_v2")
    out = gen.generate(gen.ProsumerConfig(n_homes=80, n_localities=2, chunk_homes=80, start_year=2022,
                                          end_year=2025), out_dir=tmp_path / "pv2", log=lambda *a: None)
    from wattsentry.model.battery import BatteryModel
    from wattsentry.model.battery.train import run

    res = run(["--data-dir", str(out), "--chunk", "80", "--rounds", "30", "--rows-per-chunk", "5000",
               "--val-homes", "20", "--out", str(tmp_path / "run")])
    for k in ("strategy_f1", "net_nmae", "net_nmae_17_21", "fo_net_nmae", "lgbm_net_nmae", "t_total_s"):
        assert np.isfinite(res[k]), k
    bm = BatteryModel.load(tmp_path / "run" / "battery")
    assert bm.clf is not None and bm.mode


def test_residual_bias_uses_only_past_meter_days():
    w = _world(40)
    steps = np.arange(w.T)
    rng = np.random.default_rng(3)
    resid = rng.normal(0.2, 0.05, (2, w.T)).astype(np.float32)
    resid[1] = np.nan  # profile meter: no correction
    b = N.residual_bias(w, steps, resid)
    d = 30
    a, e = w.day_start[d], w.day_start[d + 1]
    assert np.allclose(b[0, a:e].mean(), 0.2, atol=0.03) and (b[1] == 0).all()
    r2 = resid.copy()
    r2[0, w.day_start[d - 1]:] += 5  # days D-1 and later are not known when forecasting day D
    np.testing.assert_allclose(N.residual_bias(w, steps, r2)[0, a:e], b[0, a:e], rtol=1e-5)
    assert (b[0, :w.day_start[3]] == 0).all()  # no history yet
