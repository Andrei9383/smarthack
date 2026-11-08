"""K3 CUSUM + kernel dispatch (team/infra.md §1-4). Rust parity tests skip when ``wattsentry._core`` is not built."""

from __future__ import annotations

import numpy as np
import pytest

from wattsentry import kernels
from wattsentry.eval.baselines.pv.cusum import cusum_two_sided as pv_cusum
from wattsentry.kernels import cusum_two_sided, cusum_two_sided_py
from wattsentry.kernels.cusum import cusum_two_sided_py as ref

needs_rust = pytest.mark.skipif(not kernels.rust_available(), reason="wattsentry._core not built")


def _loop(x, k, h, cap=np.inf):
    """Scalar textbook loop, independent of the vectorised reference."""
    sp, sn = [], []
    cp = cn = 0.0
    for v in x:
        cp, cn = (min(cap, max(cp + v - k, 0.0)), min(cap, max(cn - v - k, 0.0))) if np.isfinite(v) else (0.0, 0.0)
        sp.append(cp)
        sn.append(cn)
    sp, sn = np.array(sp), np.array(sn)
    return sp, sn, (sp > h) | (sn > h)


def _same(a, b):
    for x, y in zip(a, b, strict=True):
        assert x.shape == y.shape and x.dtype == y.dtype
        np.testing.assert_array_equal(x, y)


def _loop_2d(x, k, h, cap=np.inf):
    rows = [_loop(r, k, h, cap) for r in x]
    return tuple(np.stack([r[j] for r in rows]) for j in range(3))


# ------------------------------------------------------------------ python-only (always run)

def test_registry_and_backend(monkeypatch):
    assert "cusum_two_sided" in kernels.names()
    assert cusum_two_sided_py is ref and kernels.get("cusum_two_sided", "python") is ref
    monkeypatch.setenv("WATTSENTRY_KERNELS", "python")
    assert kernels.backend() == "python"
    assert kernels.get("cusum_two_sided") is cusum_two_sided_py
    monkeypatch.setenv("WATTSENTRY_KERNELS", "auto")
    assert kernels.backend() == ("rust" if kernels.rust_available() else "python")
    monkeypatch.setenv("WATTSENTRY_KERNELS", "bogus")
    with pytest.raises(ValueError):
        kernels.backend()
    with pytest.raises(KeyError):
        kernels.get("no_such_kernel")


def test_forced_rust_without_extension(monkeypatch):
    if kernels.rust_available():
        pytest.skip("extension is built")
    monkeypatch.setenv("WATTSENTRY_KERNELS", "rust")
    with pytest.raises(ImportError):
        kernels.backend()
    monkeypatch.setenv("WATTSENTRY_KERNELS", "auto")
    assert kernels.backend() == "python"


def test_hand_examples():
    x = np.array([0.0, 2.0, 2.0, np.nan, -3.0, -3.0, 0.5])
    sp, sn, al = ref(x, k=0.5, h=2.5)
    np.testing.assert_array_equal(sp, [0.0, 1.5, 3.0, 0.0, 0.0, 0.0, 0.0])
    np.testing.assert_array_equal(sn, [0.0, 0.0, 0.0, 0.0, 2.5, 5.0, 4.0])
    np.testing.assert_array_equal(al, [False, False, True, False, False, True, True])
    x = np.array([0.0, 2.0, 2.0, 2.0, np.nan, -3.0, -3.0, 0.5])
    sp, sn, al = ref(x, k=0.5, h=2.5, cap=2.5)
    np.testing.assert_array_equal(sp, [0.0, 1.5, 2.5, 2.5, 0.0, 0.0, 0.0, 0.0])
    np.testing.assert_array_equal(sn, [0.0, 0.0, 0.0, 0.0, 0.0, 2.5, 2.5, 1.5])
    assert not al.any()  # strict > never fires when capped at h


@pytest.mark.parametrize("seed", [0, 1, 2])
def test_matches_scalar_loop(seed):
    rng = np.random.default_rng(seed)
    x = rng.standard_normal((5, 400)) + rng.normal(0, 0.5, (5, 1))
    x[rng.random(x.shape) < 0.05] = np.nan
    x[0, 10], x[1, 20] = np.inf, -np.inf
    sp, sn, al = ref(x, 0.25, 4.0)
    assert sp.shape == x.shape and sp.dtype == np.float64 and al.dtype == bool
    _same((sp, sn, al), _loop_2d(x, 0.25, 4.0))
    for cap in (0.0, 0.7, 3.0):
        _same(ref(x, 0.2, 1.0, cap=cap), _loop_2d(x, 0.2, 1.0, cap))


def test_shapes_and_validation():
    for shape in ((0,), (3, 0)):
        assert all(a.shape == shape for a in ref(np.zeros(shape), 0.5, 1.0))
    assert not any(a.any() for a in ref(np.full(5, np.nan), 0.5, 1.0))
    np.testing.assert_array_equal(ref([1, 2, 3], 0, 10)[0], [1.0, 3.0, 6.0])  # lists and ints are accepted
    x = np.random.default_rng(5).standard_normal((3, 300))
    _same(ref(x, 0.3, 2.0), ref(x, 0.3, 2.0, cap=None))
    _same(ref(x, 0.3, 2.0), ref(x, 0.3, 2.0, cap=1e300))
    for args, kw in [((np.zeros((2, 2, 2)), 0.5, 1.0), {}), ((np.zeros(3), np.nan, 1.0), {})] + \
            [((np.zeros(3), 0.5, 1.0), {"cap": c}) for c in (np.nan, np.inf, -1.0)]:
        with pytest.raises(ValueError):
            ref(*args, **kw)


def test_dispatched_runs_on_current_backend():
    x = np.random.default_rng(3).standard_normal((4, 100))
    _same(cusum_two_sided(x, 0.5, 3.0), _loop_2d(x, 0.5, 3.0))
    x = np.random.default_rng(6).standard_normal((5, 200))
    _same(cusum_two_sided(x, 0.1, 0.5, cap=0.5), ref(x, 0.1, 0.5, cap=0.5))


def _old_pv_cusum(x, k, h):
    """Verbatim copy of the pre-#49 numpy baseline in wattsentry/eval/baselines/pv/cusum.py."""
    x = np.asarray(x, np.float64)
    one_d = x.ndim == 1
    x2 = np.atleast_2d(x)
    n, T = x2.shape
    s_pos, s_neg = np.zeros((n, T)), np.zeros((n, T))
    sp, sn = np.zeros(n), np.zeros(n)
    for t in range(T):
        xt = x2[:, t]
        gap = np.isnan(xt)
        xt = np.where(gap, 0.0, xt)
        sp = np.where(gap, 0.0, np.minimum(h, np.maximum(0.0, sp + xt - k)))
        sn = np.where(gap, 0.0, np.minimum(h, np.maximum(0.0, sn - xt - k)))
        s_pos[:, t], s_neg[:, t] = sp, sn
    alarm = np.where(s_pos >= h, 1, np.where(s_neg >= h, -1, 0)).astype(np.int8)
    return (s_pos[0], s_neg[0], alarm[0]) if one_d else (s_pos, s_neg, alarm)


@pytest.mark.parametrize("backend", ["python", pytest.param("rust", marks=needs_rust)])
@pytest.mark.parametrize("shape,k,h", [((30, 2000), 0.1, 0.1), ((1500,), 0.5, 4.0), ((6, 400), 0.0, 2.0)])
def test_pv_wrapper_unchanged(monkeypatch, backend, shape, k, h):
    monkeypatch.setenv("WATTSENTRY_KERNELS", backend)
    rng = np.random.default_rng(11)
    x = rng.standard_normal(shape) * 0.7 + 0.2
    x[rng.random(shape) < 0.05] = np.nan
    _same(pv_cusum(x, k, h), _old_pv_cusum(x, k, h))
    with pytest.raises(ValueError):
        pv_cusum(x, k, 0.0)


# ------------------------------------------------------------------ Rust vs Python parity (needs the extension)

def _parity_inputs():
    rng = np.random.default_rng(42)
    yield "random", rng.standard_normal((16, 1000)), 0.5, 5.0
    x = rng.standard_normal((8, 3000)) * 2 + 0.2
    x[rng.random(x.shape) < 0.03] = np.nan
    yield "nan_gaps", x, 0.3, 4.0
    x = rng.standard_normal((3, 500))
    x[:, ::7], x[:, 3::11] = np.inf, -np.inf
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
def test_parity_rust_vs_python(case):
    _, x, k, h = case
    _same(kernels.get("cusum_two_sided", "python")(x, k, h), kernels.get("cusum_two_sided", "rust")(x, k, h))


@needs_rust
@pytest.mark.parametrize("cap", [0.0, 0.5, 2.0, 1e300])
def test_cap_parity_rust_vs_python(cap):
    rng = np.random.default_rng(3)
    x = rng.standard_normal((40, 1500)) * 1.5 + 0.1
    x[rng.random(x.shape) < 0.03] = np.nan
    x[:, ::97] = np.inf
    rs = kernels.get("cusum_two_sided", "rust")
    _same(ref(x, 0.25, 2.0, cap=cap), rs(x, 0.25, 2.0, cap=cap))
    _same(ref(x[0], 0.25, 2.0, cap=cap), rs(x[0], 0.25, 2.0, cap=cap))


@needs_rust
def test_rust_backend_and_bad_scalars(monkeypatch):
    monkeypatch.setenv("WATTSENTRY_KERNELS", "auto")
    assert kernels.backend() == "rust"
    monkeypatch.setenv("WATTSENTRY_KERNELS", "rust")
    assert kernels.get("cusum_two_sided") is not cusum_two_sided_py
    with pytest.raises(ValueError):
        kernels.get("cusum_two_sided", "rust")(np.zeros(3), 0.5, np.inf)
    for bad in (np.nan, np.inf, -1.0):
        with pytest.raises(ValueError):
            kernels._load_core().cusum_two_sided(np.zeros((1, 3)), 0.5, 1.0, bad)
