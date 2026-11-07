"""K9 park_scenarios (#187): Rust == Python reference bit for bit; the W2 call site dispatches to the kernel."""

from __future__ import annotations

import numpy as np
import pytest

from sentinel import kernels
from sentinel.kernels import park_scenarios_py as ref
from sentinel.model.windpower import scenarios

needs_rust = pytest.mark.skipif(not kernels.rust_available(), reason="sentinel._core not built")


def _data(n_all=40, T=96, S=50, dtype=np.float32, seed=0):
    rng = np.random.default_rng(seed)
    rated = rng.choice([2000.0, 3000.0, 3600.0], n_all)
    p50 = rng.uniform(-50, 1.1, (T, n_all)) * rated
    q = np.stack([p50 * 0.7 - 30, p50, p50 * 1.3 + 40], -1).astype(dtype)
    q[rng.random((T, n_all)) < 0.02] = np.nan
    u = scenarios.common_levels(S, T, 1, seed=seed)[..., 0]
    u[0, :3] = [0.5, 0.0, 1.0]
    pu = rng.uniform(0, 1, (T, n_all))
    pu[rng.random((T, n_all)) < 0.3] = 1.0
    der = rng.uniform(0.5, 1, (T, n_all))
    return q, rated, rng.uniform(0.85, 1.0, T), u, pu, der


@needs_rust
@pytest.mark.parametrize("dtype", [np.float32, np.float64])
@pytest.mark.parametrize("n_cols", [1, 5, 8, 39, 130])
@pytest.mark.parametrize("mode", ["plain", "pu", "derate_pu", "bern", "lab"])
def test_rust_matches_reference(dtype, n_cols, mode):
    q, rated, ratio, u, pu, der = _data(n_all=max(n_cols + 3, 40), dtype=dtype)
    cols = np.sort(np.random.default_rng(1).choice(q.shape[1], n_cols, replace=False))
    if mode == "lab":
        pu = np.broadcast_to((np.arange(q.shape[1]) % 3 != 0).astype(float)[None], pu.shape)
    kw = {"plain": {}, "pu": dict(p_up=pu), "derate_pu": dict(p_up=pu, derate=der), "lab": dict(p_up=pu),
          "bern": dict(p_up=pu, derate=der, bernoulli=True)}[mode]
    a = ref(q, rated, cols, ratio, u, **kw, rng=np.random.default_rng(7))
    b = kernels.get("park_scenarios", "rust")(q, rated, cols, ratio, u, **kw, rng=np.random.default_rng(7))
    assert a.shape == b.shape == u.shape and b.dtype == np.float64
    np.testing.assert_array_equal(a, b)


def test_call_site_dispatches(monkeypatch):
    q, rated, ratio, u, pu, der = _data()
    cols = np.flatnonzero(np.arange(q.shape[1]) % 3 == 1)
    monkeypatch.setenv("SENTINEL_KERNELS", "python")
    a = scenarios.park_scenarios(q, rated, cols, ratio, u, pu, der)
    np.testing.assert_array_equal(a, ref(q, rated, cols, ratio, u, pu, der))
    if kernels.rust_available():
        monkeypatch.setenv("SENTINEL_KERNELS", "rust")
        np.testing.assert_array_equal(scenarios.park_scenarios(q, rated, cols, ratio, u, pu, der), a)
