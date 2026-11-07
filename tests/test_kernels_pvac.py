"""K8 pv_ac_batch (#133): the reference equals per-site physics.pvlib_expected_kw / pvlib_clear_kw bit for bit, and the
Rust copy matches the reference (bit-exact where numpy's float64 cos/arccos are libm's, within a few ulp otherwise)."""

import math

import numpy as np
import pandas as pd
import pytest

from sentinel import kernels
from sentinel.kernels.pvac import pv_ac_batch_py
from sentinel.model import physics as ph

T = 4 * 96


def make(rng, n=9):
    times = pd.date_range("2025-06-20", periods=T, freq="15min", tz="UTC")
    geom = ph.solar_geometry(times, 45.0 + rng.uniform(-2, 2), 25.0 + rng.uniform(-3, 3)).astype(np.float32)
    g64 = geom.astype(np.float64)
    ghi = np.clip(g64[:, 2] * rng.uniform(0.2, 1.05, T), 0, None)
    dni, dhi = ph.decompose(ghi, g64[:, 0], times.day_of_year.to_numpy())
    wx = np.stack([ghi, dni, dhi, rng.normal(18, 6, T), rng.uniform(0, 100, T), rng.gamma(2, 2, T), np.zeros(T)], 1)
    wx = wx.astype(np.float32).astype(np.float64)
    meta = dict(kwp_recorded=rng.uniform(2, 12, n), tilt_recorded=rng.uniform(0, 60, n),
                azimuth_recorded=rng.uniform(90, 270, n), inverter_kw=rng.uniform(2, 10, n))
    meta["tilt_recorded"][0] = 0.0  # flat roof
    meta["inverter_kw"][1] = meta["kwp_recorded"][1] * 0.5  # heavy clipping
    return times, g64, wx, meta


def site_args(meta):
    return meta["kwp_recorded"], meta["tilt_recorded"], meta["azimuth_recorded"], meta["inverter_kw"]


def expected_args(meta, wx, g64):
    return (*site_args(meta), wx[:, 0], wx[:, 1], wx[:, 2], wx[:, 3], wx[:, 5], g64[:, 0], g64[:, 1])


def clear_args(meta, times, g64):
    cs = ph.clearsky(times + pd.Timedelta(minutes=ph.STEP_MIN / 2), g64[:, 0])
    return (*site_args(meta), cs["ghi"], cs["dni"], cs["dhi"], 20.0, None, g64[:, 0], g64[:, 1])


def per_site(meta, i):
    return {k: float(v[i]) for k, v in meta.items()}


def test_reference_matches_per_site_physics():
    assert kernels.get("pv_ac_batch", "python") is pv_ac_batch_py
    times, g64, wx, meta = make(np.random.default_rng(0))
    got = pv_ac_batch_py(*expected_args(meta, wx, g64))
    clear = pv_ac_batch_py(*clear_args(meta, times, g64))
    assert got.shape == clear.shape == (len(meta["kwp_recorded"]), T)
    for i in range(got.shape[0]):
        m = per_site(meta, i)
        assert np.array_equal(got[i], ph.pvlib_expected_kw(m, wx, g64))
        assert np.array_equal(clear[i], ph.pvlib_clear_kw(m, times, g64))
    assert np.all(got >= 0) and np.all(got <= meta["inverter_kw"][:, None])


def _numpy_trig_is_libm() -> bool:
    x = np.random.default_rng(3).uniform(-1, 1, 4096)
    return (np.array_equal(np.arccos(x), [math.acos(v) for v in x])
            and np.array_equal(np.cos(x * 3), [math.cos(v * 3) for v in x]))


rust = pytest.mark.skipif(not kernels.rust_available(), reason="sentinel._core not built")


@rust
@pytest.mark.parametrize("seed", [0, 1, 2])
def test_rust_matches_reference(seed):
    times, g64, wx, meta = make(np.random.default_rng(seed), n=17)
    rs = kernels.get("pv_ac_batch", "rust")
    for args in (expected_args(meta, wx, g64), clear_args(meta, times, g64)):
        a, b = pv_ac_batch_py(*args), rs(*args)
        assert a.shape == b.shape and a.dtype == b.dtype == np.float64
        if _numpy_trig_is_libm():
            assert np.array_equal(a, b)
        else:  # numpy uses its own SIMD float64 trig on this CPU: allow a few ulp
            np.testing.assert_allclose(b, a, rtol=1e-12, atol=1e-12)
        assert np.array_equal(a.astype(np.float16), b.astype(np.float16))


@rust
def test_rust_edges():
    rs = kernels.get("pv_ac_batch", "rust")
    z = np.array([0.0, 45.0, 89.9, 90.0, 120.0, 180.0])
    saz = np.array([180.0, 90.0, 270.0, 0.0, 360.0, 180.0])
    irr = np.array([0.0, 500.0, 1000.0, 50.0, 0.0, 1200.0])
    temp = np.array([-20.0, 0.0, 25.0, 45.0, 10.0, 60.0])
    wind = np.array([0.0, -1.0, 3.0, 20.0, 5.0, 15.0])  # negative / above-cap wind is clipped
    site = (np.array([0.0, 5.0, 3.0]), np.array([0.0, 35.0, 90.0]), np.array([180.0, 0.0, 359.0]),
            np.array([0.0, 2.0, 4.0]))  # includes a zero kWp / zero inverter site
    for t_, w_ in ((temp, wind), (20.0, None)):
        args = (*site, irr, irr * 0.8, irr * 0.2, t_, w_, z, saz)
        np.testing.assert_allclose(rs(*args), pv_ac_batch_py(*args), rtol=1e-12, atol=1e-12)
    empty = rs(np.zeros(0), np.zeros(0), np.zeros(0), np.zeros(0), irr, irr, irr, temp, None, z, saz)
    assert empty.shape == (0, len(z))
    with pytest.raises(ValueError):
        rs(*site, irr, irr[:3], irr, temp, None, z, saz)
