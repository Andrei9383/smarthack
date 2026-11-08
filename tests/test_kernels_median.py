"""#136 exact medians: references vs scalar loops, Rust == Python bit for bit, select_normal / _frozen unchanged."""

from __future__ import annotations

import hashlib

import numpy as np
import pandas as pd
import pytest

from wattsentry import kernels
from wattsentry.kernels import binned_median_mad_py, rolling_median_py

needs_rust = pytest.mark.skipif(not kernels.rust_available(), reason="wattsentry._core not built")


def _data(shape, seed=0, nan=0.15, decimals=1):
    rng = np.random.default_rng(seed)
    x = np.round(rng.standard_normal(shape) * 3, decimals)  # rounded -> many ties
    x[rng.random(shape) < nan] = np.nan
    if nan and len(shape) > 1 and shape[0] > 50:
        x[:50, 0] = np.nan  # a long gap at the start
        x[-30:, -1] = np.nan  # and at the end
    return x


def _roll_loop(x, w, minp):
    off = (w - 1) // 2
    out = np.full(len(x), np.nan)
    for i in range(len(x)):
        e = min(max(i + 1 + off, 0), len(x))
        s = min(max(i + 1 + off - w, 0), len(x))
        v = x[s:e]
        v = v[~np.isnan(v)]
        if len(v) >= max(minp, 1):
            out[i] = np.median(v)
    return out


@pytest.mark.parametrize("w,minp", [(1, 1), (2, 1), (5, 0), (8, 3), (96, 12), (301, 300)])
def test_rolling_reference_matches_loop(w, minp):
    x = _data((400, 3))
    r = rolling_median_py(x, w, minp)
    for j in range(3):
        np.testing.assert_array_equal(r[:, j], _roll_loop(x[:, j], w, minp))


def test_rolling_reference_is_pandas_per_column():
    x = _data((500, 4), 1)
    r = rolling_median_py(x, 96, 12)
    for j in range(4):
        np.testing.assert_array_equal(r[:, j], pd.Series(x[:, j]).rolling(96, center=True, min_periods=12).median())


def test_binned_reference_matches_loop():
    rng = np.random.default_rng(2)
    b, x, ok = rng.integers(0, 7, (300, 4)), _data((300, 4), 2, nan=0.0), rng.random((300, 4)) < 0.6
    med, mad = binned_median_mad_py(b, x, ok, 8, 5)
    assert med.shape == mad.shape == (8, 4)
    for j in range(4):
        for k in range(8):
            v = x[ok[:, j] & (b[:, j] == k), j]
            if len(v) >= 5:
                assert med[k, j] == np.median(v) and mad[k, j] == np.median(np.abs(v - np.median(v)))
            else:
                assert np.isnan(med[k, j]) and np.isnan(mad[k, j])


def test_edges_and_validation():
    assert rolling_median_py(np.zeros((0, 3)), 4, 1).shape == (0, 3)
    assert np.isnan(rolling_median_py(np.full((5, 2), np.nan), 3, 0)).all()
    for bad in [(0, 0), (3, 4), (3, -1)]:
        with pytest.raises(ValueError):
            rolling_median_py(np.zeros(5), *bad)
    with pytest.raises(ValueError):
        binned_median_mad_py(np.full((4, 1), 3), np.zeros((4, 1)), np.ones((4, 1), bool), 3, 1)
    med, _ = binned_median_mad_py(np.full((4, 1), 3), np.zeros((4, 1)), np.zeros((4, 1), bool), 3, 1)  # bin unused
    assert np.isnan(med).all()


@needs_rust
@pytest.mark.parametrize("shape", [(0, 3), (1,), (700,), (3000, 37), (300, 1), (200, 3, 5)])
@pytest.mark.parametrize("w,minp", [(1, 1), (2, 0), (4, 4), (96, 12), (192, 24), (1344, 96), (5000, 1)])
def test_rolling_parity_rust_vs_python(shape, w, minp):
    x = _data(shape, 3)
    if x.size:
        x.reshape(-1)[::97] = np.inf  # infinities sort like pandas' skiplist
        x.reshape(-1)[::89] = -np.inf
    a, b = rolling_median_py(x, w, minp), kernels.get("rolling_median", "rust")(x, w, minp)
    assert a.shape == b.shape and a.dtype == b.dtype
    np.testing.assert_array_equal(a, b)


@needs_rust
def test_rolling_parity_non_contiguous_and_float32():
    x = _data((900, 10), 4).astype(np.float32)[:, ::2]
    np.testing.assert_array_equal(rolling_median_py(x, 16, 2), kernels.get("rolling_median", "rust")(x, 16, 2))
    with pytest.raises(ValueError):
        kernels._load_core().rolling_median(np.zeros((3, 2)), 0, 0)


@needs_rust
@pytest.mark.parametrize("shape,n_bins,min_n", [((0, 3), 4, 1), ((500,), 6, 1), ((2000, 37), 60, 30),
                                                ((300, 9), 1, 30), ((50, 2), 3, 0), ((400, 2, 3), 11, 5)])
def test_binned_parity_rust_vs_python(shape, n_bins, min_n):
    rng = np.random.default_rng(5)
    b, x, ok = rng.integers(0, n_bins, shape), _data(shape, 5, nan=0.0), rng.random(shape) < 0.5
    x.reshape(-1)[::53] = np.inf
    py, rs = binned_median_mad_py(b, x, ok, n_bins, min_n), kernels.get("binned_median_mad", "rust")(b, x, ok,
                                                                                                    n_bins, min_n)
    for a, c in zip(py, rs, strict=True):
        assert a.shape == c.shape == (n_bins, *shape[1:]) and a.dtype == c.dtype
        np.testing.assert_array_equal(a, c)
    with pytest.raises(ValueError):
        kernels.get("binned_median_mad", "rust")(np.full((4, 1), n_bins), x[:4].reshape(4, -1)[:, :1],
                                                 np.ones((4, 1), bool), n_bins, 1)


def test_frozen_matches_per_step_loop():
    from wattsentry.eval.baselines.wind.windmon_rules import _frozen

    def loop(x, n):
        same = np.zeros_like(x, dtype=bool)
        same[1:] = (x[1:] == x[:-1]) & np.isfinite(x[1:])
        run = np.zeros(x.shape, dtype=np.int32)
        for t in range(1, x.shape[0]):
            run[t] = np.where(same[t], run[t - 1] + 1, 0)
        return run >= n - 1

    rng = np.random.default_rng(6)
    for T in (0, 1, 3, 8, 9, 400):
        x = rng.integers(0, 2, (T, 5)).astype(float)
        x[rng.random(x.shape) < 0.05] = np.nan
        for n in (0, 1, 2, 3, 8, 20):
            np.testing.assert_array_equal(_frozen(x, n), loop(x, n))


def test_select_normal_same_mask_both_backends(monkeypatch):
    """The vectorised select_normal gives the same masks with the Python reference and (if built) Rust."""
    from wattsentry.model.wind.features import density_inputs
    from wattsentry.model.wind.normal_select import select_normal
    from wattsentry.model.wind.synth import SynthConfig, generate

    d = generate(SynthConfig(days=60, scale=0.08))
    vn, vm, _ = density_inputs(d.sensors, d.meteo, d.weather["pressure_hpa"][:, d.park_of])
    fit = np.zeros(d.n_steps, dtype=bool)
    fit[: int(0.7 * d.n_steps)] = True
    out = {}
    for be in ("python", "rust") if kernels.rust_available() else ("python",):
        monkeypatch.setenv("WATTSENTRY_KERNELS", be)
        pw, tp = select_normal(d.sensors, d.err, d.rated_kw, vn, vm, fit, d.meteo["wdir_deg"])
        out[be] = hashlib.sha256(np.packbits(pw).tobytes() + np.packbits(tp).tobytes()).hexdigest()
    assert len(set(out.values())) == 1, out
