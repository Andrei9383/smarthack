"""Kernel dispatch + parity tests (team/infra.md §1-4).

Python-only tests always run (check has no Rust toolchain). Parity tests compare the Rust extension with the
Python reference bit for bit and are skipped when ``sentinel._core`` is not built.
"""

from __future__ import annotations

import numpy as np
import pytest

from sentinel import kernels
from sentinel.kernels import cusum_two_sided_py

needs_rust = pytest.mark.skipif(not kernels.rust_available(), reason="sentinel._core not built")


def _cusum_loop(x, k, h):
    """Scalar textbook loop, independent of the vectorised reference."""
    sp, sn, al = [], [], []
    cp = cn = 0.0
    for v in x:
        if not np.isfinite(v):
            cp = cn = 0.0
        else:
            cp = max(cp + v - k, 0.0)
            cn = max(cn - v - k, 0.0)
        sp.append(cp)
        sn.append(cn)
        al.append(cp > h or cn > h)
    return np.array(sp), np.array(sn), np.array(al, dtype=bool)


def _assert_same(a, b):
    for x, y in zip(a, b, strict=True):
        assert x.shape == y.shape and x.dtype == y.dtype
        np.testing.assert_array_equal(x, y)


# ------------------------------------------------------------------ python-only (always run)


def test_registry_and_backend(monkeypatch):
    assert "cusum_two_sided" in kernels.names()
    monkeypatch.setenv("SENTINEL_KERNELS", "python")
    assert kernels.backend() == "python"
    assert kernels.get("cusum_two_sided") is cusum_two_sided_py
    monkeypatch.setenv("SENTINEL_KERNELS", "auto")
    assert kernels.backend() == ("rust" if kernels.rust_available() else "python")
    monkeypatch.setenv("SENTINEL_KERNELS", "bogus")
    with pytest.raises(ValueError):
        kernels.backend()
    with pytest.raises(KeyError):
        kernels.get("no_such_kernel")


def test_forced_rust_without_extension(monkeypatch):
    if kernels.rust_available():
        pytest.skip("extension is built")
    monkeypatch.setenv("SENTINEL_KERNELS", "rust")
    with pytest.raises(ImportError):
        kernels.backend()
    monkeypatch.setenv("SENTINEL_KERNELS", "auto")
    assert kernels.backend() == "python"


def test_cusum_hand_example():
    x = np.array([0.0, 2.0, 2.0, np.nan, -3.0, -3.0, 0.5])
    sp, sn, al = cusum_two_sided_py(x, k=0.5, h=2.5)
    np.testing.assert_array_equal(sp, [0.0, 1.5, 3.0, 0.0, 0.0, 0.0, 0.0])
    np.testing.assert_array_equal(sn, [0.0, 0.0, 0.0, 0.0, 2.5, 5.0, 4.0])
    np.testing.assert_array_equal(al, [False, False, True, False, False, True, True])


@pytest.mark.parametrize("seed", [0, 1, 2])
def test_cusum_matches_scalar_loop(seed):
    rng = np.random.default_rng(seed)
    x = rng.standard_normal((5, 400)) + rng.normal(0, 0.5, (5, 1))
    x[rng.random(x.shape) < 0.05] = np.nan
    x[0, 10] = np.inf
    x[1, 20] = -np.inf
    sp, sn, al = cusum_two_sided_py(x, 0.25, 4.0)
    assert sp.shape == x.shape and sp.dtype == np.float64 and al.dtype == bool
    for i in range(x.shape[0]):
        _assert_same((sp[i], sn[i], al[i]), _cusum_loop(x[i], 0.25, 4.0))


def test_cusum_shapes_and_validation():
    sp, sn, al = cusum_two_sided_py(np.zeros(0), 0.5, 1.0)
    assert sp.shape == sn.shape == al.shape == (0,)
    sp, sn, al = cusum_two_sided_py(np.zeros((3, 0)), 0.5, 1.0)
    assert sp.shape == (3, 0)
    sp, sn, al = cusum_two_sided_py(np.full(5, np.nan), 0.5, 1.0)
    assert not sp.any() and not sn.any() and not al.any()
    sp, _, _ = cusum_two_sided_py([1, 2, 3], 0, 10)  # lists and ints are accepted
    np.testing.assert_array_equal(sp, [1.0, 3.0, 6.0])
    with pytest.raises(ValueError):
        cusum_two_sided_py(np.zeros((2, 2, 2)), 0.5, 1.0)
    with pytest.raises(ValueError):
        cusum_two_sided_py(np.zeros(3), np.nan, 1.0)


def test_dispatched_cusum_runs_on_current_backend():
    x = np.random.default_rng(3).standard_normal((4, 100))
    _assert_same(kernels.cusum_two_sided(x, 0.5, 3.0), _cusum_loop_2d(x, 0.5, 3.0))


def _cusum_loop_2d(x, k, h):
    rows = [_cusum_loop(r, k, h) for r in x]
    return tuple(np.stack([r[j] for r in rows]) for j in range(3))


# ------------------------------------------------------------------ Rust vs Python parity (needs the extension)


def _parity_inputs():
    rng = np.random.default_rng(42)
    yield "random", rng.standard_normal((16, 1000)), 0.5, 5.0
    x = rng.standard_normal((8, 3000)) * 2 + 0.2
    x[rng.random(x.shape) < 0.03] = np.nan
    yield "nan_gaps", x, 0.3, 4.0
    x = rng.standard_normal((3, 500))
    x[:, ::7] = np.inf
    x[:, 3::11] = -np.inf
    yield "inf_gaps", x, 0.5, 2.0
    yield "1d", rng.standard_normal(777), 0.1, 3.0
    yield "all_nan", np.full((2, 50), np.nan), 0.5, 1.0
    yield "empty_1d", np.zeros(0), 0.5, 1.0
    yield "empty_2d", np.zeros((4, 0)), 0.5, 1.0
    yield "zero_rows", np.zeros((0, 10)), 0.5, 1.0
    yield "single", np.array([3.0]), 0.0, 2.0
    yield "k0_h0", rng.standard_normal((2, 200)), 0.0, 0.0
    yield "negative_k", rng.standard_normal((2, 200)), -0.2, 3.0
    yield "huge", rng.standard_normal((2, 200)) * 1e300, 1e299, 1e300
    yield "float32_input", rng.standard_normal((3, 300)).astype(np.float32), 0.5, 2.0
    yield "non_contiguous", rng.standard_normal((300, 6)).T, 0.5, 2.0
    yield "many_threads", rng.standard_normal((257, 300)), 0.5, 4.0


@needs_rust
@pytest.mark.parametrize("case", list(_parity_inputs()), ids=lambda c: c[0])
def test_cusum_parity_rust_vs_python(case):
    _, x, k, h = case
    py = kernels.get("cusum_two_sided", "python")(x, k, h)
    rs = kernels.get("cusum_two_sided", "rust")(x, k, h)
    _assert_same(py, rs)


@needs_rust
def test_rust_backend_selected(monkeypatch):
    monkeypatch.setenv("SENTINEL_KERNELS", "auto")
    assert kernels.backend() == "rust"
    monkeypatch.setenv("SENTINEL_KERNELS", "rust")
    assert kernels.get("cusum_two_sided") is not cusum_two_sided_py


@needs_rust
def test_rust_rejects_bad_scalars():
    with pytest.raises(ValueError):
        kernels.get("cusum_two_sided", "rust")(np.zeros(3), 0.5, np.inf)
