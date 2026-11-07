"""K4 fused feature build (#94): the Python reference is features.pv_arrays / load_arrays, and Rust matches it
bit for bit except the trig channels (cos/sin of zenith/azimuth: <= 1 ulp)."""

import numpy as np
import pytest

from sentinel import kernels
from sentinel.kernels.features import load_features_py, pv_features_py
from sentinel.model import features as F

H, HH = 96, 14 * 96
TRIG = [7, 8, 9]  # pv seq channels cos_zen, sin_az, cos_az


def make_raw(B=64, seed=0, extra_hist=0):
    rng = np.random.default_rng(seed)
    kwp = rng.uniform(2, 12, B)
    tod = np.arange(H) / H * 2 * np.pi
    prof = np.clip(np.sin(tod - np.pi / 2), 0, None)
    zen = rng.uniform(0, 180, (B, H))
    zen[:, :8] = 90.0  # night boundary: cos(90 deg) ~ -4e-8
    geom = np.stack([zen, rng.uniform(0, 360, (B, H)), rng.uniform(0, 1000, (B, H))], -1)
    geom[:, :4, 2] = 0.0
    hist = prof[None, None] * kwp[:, None, None] * rng.uniform(0, 1.2, (B, 14 + extra_hist, 1))
    hist = hist.reshape(B, -1) + rng.normal(0, 0.05, (B, (14 + extra_hist) * H)).clip(0)
    hist[:, 10::7] = hist[:, 9::7][:, :hist[:, 10::7].shape[1]]  # stuck readings -> frozen_frac
    hist[: B // 4, -3 * H:] = 0.0  # outage days -> zero_frac
    hist[1, -40] = np.nan  # a NaN must propagate like numpy (max, clip, nan_to_num)
    h15 = np.arange(H) * 0.25  # exact hours 17.0 and 23.0 hit the evening-window edges
    cal = np.tile(np.stack([np.sin(2 * np.pi * h15 / 24), np.cos(2 * np.pi * h15 / 24)], -1), (B, 1, 4))
    return {"wx": rng.random((B, H, 7)) * 600, "geom": geom.astype(np.float32), "cal": cal.astype(np.float32),
            "pvclear": (prof[None] * kwp[:, None]).astype(np.float32), "pvlib": (prof * 0.8)[None] * kwp[:, None],
            "hist": hist.astype(np.float32), "kwp": kwp}


def pv_args(r):
    return r["wx"], r["geom"], r["pvclear"], r["pvlib"], r["hist"], r["kwp"]


def load_args(r):
    return r["wx"], r["cal"], np.abs(r["hist"][:, -HH:])


def test_registered_and_shapes():
    assert {"pv_features", "load_features"} <= set(kernels.names())
    assert kernels.get("pv_features", "python") is pv_features_py
    r = make_raw(8)
    seq, ref, summ, night = pv_features_py(*pv_args(r))
    assert (seq.shape, ref.shape, summ.shape, night.shape) == ((8, H, 17), (8, 168), (8, 14, 5), (8, H))
    assert seq.dtype == ref.dtype == summ.dtype == np.float32 and night.dtype == bool
    seq, ref, summ, scale = load_features_py(*load_args(r))
    assert (seq.shape, ref.shape, summ.shape, scale.shape) == ((8, H, 14), (8, 168), (8, 14, 4), (8,))


def test_build_uses_dispatch(monkeypatch):
    """features.build gives the same inputs on either backend (reference = the pre-#94 numpy code path)."""
    r = make_raw(16)
    raw = {"wx": r["wx"], "geom": r["geom"], "cal": r["cal"], "pvclear": r["pvclear"], "pvlib": r["pvlib"],
           "hist_pv": r["hist"], "hist_load": load_args(r)[2], "kwp_recorded": r["kwp"], "azimuth_recorded": r["kwp"],
           "tilt_recorded": r["kwp"], "inverter_kw": r["kwp"], "lat": r["kwp"], "lon": r["kwp"],
           "has_batt": np.zeros(16), "doy_sin": np.zeros(16), "doy_cos": np.ones(16), "arch_idx": np.arange(16) % 6 - 1}
    monkeypatch.setenv("SENTINEL_KERNELS", "python")
    ref = {k: F.build(k, raw) for k in ("pv", "load")}
    assert np.array_equal(ref["pv"]["seq"], pv_features_py(*pv_args(r))[0], equal_nan=True)
    if not kernels.rust_available():
        return
    monkeypatch.setenv("SENTINEL_KERNELS", "rust")
    for k in ("pv", "load"):
        got = F.build(k, raw)
        for key, a in ref[k].items():
            np.testing.assert_allclose(got[key], a, rtol=0, atol=1e-6, err_msg=f"{k}.{key}")


rust = pytest.mark.skipif(not kernels.rust_available(), reason="sentinel._core not built")


@rust
@pytest.mark.parametrize("extra_hist", [0, 2])
def test_rust_pv_parity(extra_hist):
    r = make_raw(300, seed=extra_hist, extra_hist=extra_hist)
    py = kernels.get("pv_features", "python")(*pv_args(r))
    rs = kernels.get("pv_features", "rust")(*pv_args(r))
    for a, b in zip(py, rs, strict=True):
        assert a.shape == b.shape and a.dtype == b.dtype
    keep = [c for c in range(17) if c not in TRIG]
    assert py[0][..., keep].tobytes() == rs[0][..., keep].tobytes()  # bit-exact incl. NaN
    np.testing.assert_allclose(rs[0][..., TRIG], py[0][..., TRIG], rtol=0, atol=2e-7)
    assert py[1].tobytes() == rs[1].tobytes() and py[2].tobytes() == rs[2].tobytes()
    assert np.array_equal(py[3], rs[3])
    assert np.isnan(rs[0][1]).any() and not np.isnan(rs[2]).any()  # NaN kept in seq, nan_to_num in summ
    assert rs[2][..., 3].max() > 0 and rs[2][..., 4].max() > 0  # zero_frac / frozen_frac exercised


@rust
def test_rust_load_parity():
    r = make_raw(300, seed=3)
    py = kernels.get("load_features", "python")(*load_args(r))
    rs = kernels.get("load_features", "rust")(*load_args(r))
    for a, b in zip(py, rs, strict=True):
        assert a.shape == b.shape and a.dtype == b.dtype and a.tobytes() == b.tobytes()


@rust
def test_rust_size_errors():
    core = kernels._load_core()
    with pytest.raises(ValueError):  # sizes are checked in Rust too
        core.load_features(np.zeros((2, H * 7), np.float32), np.zeros((2, H * 8), np.float32),
                           np.zeros((2, HH), np.float32), np.zeros((2, H * 14), np.float32),
                           np.zeros((2, 100), np.float32), np.zeros((2, 56), np.float32), np.zeros(2, np.float32))


def site_raw(B=64, seed=0):
    """M4 site window (site_inputs) with NaN / inf / no-battery / frozen edge cases."""
    from sentinel.kernels.bench import site_raw as bench_site_raw

    raw = bench_site_raw(B / 256, np.random.default_rng(seed))[0]
    raw["pv"][2, 50], raw["m1"][3, 20, 1], raw["hist_soc"][4, -30], raw["hist_volt"][5, -9] = np.nan, np.nan, np.nan, np.inf
    raw["pv"][:, 61] = raw["pv"][:, 60]  # frozen flags
    return raw


@rust
@pytest.mark.parametrize("float_codes", [False, True])
def test_rust_site_parity(float_codes):
    raw = site_raw(300, seed=int(float_codes))
    if float_codes:  # predictor path: float status codes / valid
        raw["status"], raw["hist_status"], raw["valid"] = (raw[k] * 1.0 for k in ("status", "hist_status", "valid"))
    py, rs = kernels.get("site_features", "python")(raw), kernels.get("site_features", "rust")(raw)
    assert [(a.shape, a.dtype) for a in rs] == [((300, H, 18), np.float32), ((300, 14, 8), np.float32)]
    keep = [c for c in range(18) if c != 7]  # cos_zen: <= 1 ulp
    assert py[0][..., keep].tobytes() == rs[0][..., keep].tobytes() and py[1].tobytes() == rs[1].tobytes()
    np.testing.assert_allclose(rs[0][..., 7], py[0][..., 7], rtol=0, atol=2e-7)
    raw["hist_volt"] = None  # degraded mode -> reference
    assert np.array_equal(kernels.get("site_features", "rust")(raw)[1], F.site_arrays(raw)[1])
