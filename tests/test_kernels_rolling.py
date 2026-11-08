"""K4 rolling mean: reference vs a scalar loop, Rust == Python bit for bit, M6 rmean uses the kernel."""

from __future__ import annotations

import numpy as np
import pytest

from wattsentry import kernels
from wattsentry.kernels import rolling_mean, rolling_mean_py as ref
from wattsentry.model.wind.features import rmean

needs_rust = pytest.mark.skipif(not kernels.rust_available(), reason="wattsentry._core not built")


def _loop(x, w):
    out = np.zeros(len(x))
    for t in range(len(x)):
        v = x[max(0, t + 1 - w) : t + 1]
        v = v[np.isfinite(v)]
        out[t] = v.mean() if len(v) else 0.0
    return out


def _data(shape, seed=0):
    rng = np.random.default_rng(seed)
    x = rng.standard_normal(shape) * 3 + 1
    x[rng.random(shape) < 0.1] = np.nan
    x[rng.random(shape) < 0.01] = np.inf
    if len(shape) > 1:
        x[:40, 0] = np.nan  # a long gap at the start
    return x


@pytest.mark.parametrize("w", [1, 4, 16, 1000])
def test_reference_matches_loop(w):
    x = _data((300, 3))
    r = ref(x, w)
    for j in range(3):
        np.testing.assert_allclose(r[:, j], _loop(x[:, j], w), rtol=1e-9, atol=1e-12)


def test_edges():
    assert ref(np.zeros((0, 3)), 4).shape == (0, 3)
    np.testing.assert_array_equal(ref(np.array([1.0, 2.0, 3.0]), 0), 0.0)
    np.testing.assert_array_equal(ref(np.full((5, 2), np.nan), 3), 0.0)
    with pytest.raises(ValueError):
        ref(np.zeros(3), -1)


def test_rmean_is_dispatched_kernel():
    x = _data((500, 4), 2)
    np.testing.assert_array_equal(rmean(x, 96), ref(x, 96))
    np.testing.assert_array_equal(rolling_mean(x, 16), ref(x, 16))


@needs_rust
@pytest.mark.parametrize("shape", [(0, 3), (1,), (700,), (2000, 70), (300, 1), (200, 3, 5)])
@pytest.mark.parametrize("w", [0, 1, 4, 96, 672, 5000])
def test_parity_rust_vs_python(shape, w):
    x = _data(shape, 1)
    a, b = ref(x, w), kernels.get("rolling_mean", "rust")(x, w)
    assert a.shape == b.shape and a.dtype == b.dtype
    np.testing.assert_array_equal(a, b)


@needs_rust
def test_parity_non_contiguous_and_float32():
    x = _data((800, 6), 3).astype(np.float32)[:, ::2]
    np.testing.assert_array_equal(ref(x, 16), kernels.get("rolling_mean", "rust")(x, 16))
    with pytest.raises(ValueError):
        kernels._load_core().rolling_mean(np.zeros((3, 2)), -1)
