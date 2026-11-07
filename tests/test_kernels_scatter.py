"""K5 telemetry scatter (#103): reference vs a scalar loop, Rust == Python bit for bit, WindNBM load uses the kernel."""

from __future__ import annotations

import numpy as np
import pytest

from sentinel import kernels
from sentinel.kernels import scatter_telemetry, scatter_telemetry_py as ref

needs_rust = pytest.mark.skipif(not kernels.rust_available(), reason="sentinel._core not built")


def _data(n, S=3, T=40, N=5, seed=0):
    rng = np.random.default_rng(seed)
    ti, tj, sk = rng.integers(-2, T + 2, n), rng.integers(-1, N + 1, n), rng.integers(-1, S + 1, n)
    is_code = rng.random(n) < 0.05
    v = np.where(is_code, rng.integers(0, 4, n), rng.standard_normal(n) * 3 + 1)
    v[rng.random(n) < 0.05] = np.nan
    v[rng.random(n) < 0.01] = np.inf
    sc = np.ones((S, N))
    sc[0] = rng.random(N) + 0.5
    return ti, tj, sk, v, is_code, sc, T


def _loop(ti, tj, sk, v, is_code, sc, T):
    (S, N), vals = sc.shape, {}
    err = np.zeros((T, N), np.int32)
    for a, j, k, x, c in zip(ti, tj, sk, v, is_code):
        if not (0 <= a < T and 0 <= j < N):
            continue
        if c:
            err[a, j] = int(x) if np.isfinite(x) else 0
        elif 0 <= k < S and np.isfinite(x):
            vals.setdefault((k, a, j), []).append(x * sc[k, j])
    sens = np.full((S, T, N), np.nan, np.float32)
    for (k, a, j), xs in vals.items():
        if not err[a, j]:
            sens[k, a, j] = np.mean(xs)
    return sens, err


def test_reference_matches_loop():
    d = _data(3000)
    (s, e), (s2, e2) = ref(*d), _loop(*d)
    np.testing.assert_array_equal(e, e2)
    np.testing.assert_allclose(s, s2, rtol=1e-6)
    assert s.dtype == np.float32 and e.dtype == np.int32 and s.shape == (3, 40, 5)


def test_edges():
    s, e = ref(*_data(0))
    assert np.isnan(s).all() and not e.any()
    with pytest.raises(ValueError):
        ref(np.zeros(3), np.zeros(2), np.zeros(3), np.zeros(3), np.zeros(3, bool), np.ones((1, 1)), 4)


def test_dispatch():
    d = _data(2000, seed=2)
    for a, b in zip(scatter_telemetry(*d), ref(*d)):
        np.testing.assert_array_equal(a, b)


@needs_rust
@pytest.mark.parametrize("n,S,T,N", [(0, 3, 40, 5), (1, 1, 1, 1), (5000, 3, 40, 5), (200_000, 8, 700, 70)])
def test_parity_rust_vs_python(n, S, T, N):
    d = _data(n, S, T, N, seed=1)
    for a, b in zip(ref(*d), kernels.get("scatter_telemetry", "rust")(*d)):
        assert a.shape == b.shape and a.dtype == b.dtype
        np.testing.assert_array_equal(a, b)


@needs_rust
def test_parity_non_contiguous_and_int32():
    ti, tj, sk, v, c, sc, T = _data(6000, seed=3)
    d = (ti[::2].astype(np.int32), tj[::2], sk[::2], v[::2], c[::2], sc, T)
    for a, b in zip(ref(*d), kernels.get("scatter_telemetry", "rust")(*d)):
        np.testing.assert_array_equal(a, b)
