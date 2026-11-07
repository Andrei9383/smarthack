"""K3 CUSUM (#49): the cap option, Rust == Python bit for bit with a cap, and the PV baseline wrapper.

Uncapped behaviour and the dispatcher are covered in tests/test_kernels.py.
"""

from __future__ import annotations

import numpy as np
import pytest

from sentinel import kernels
from sentinel.eval.baselines.pv.cusum import cusum_two_sided as pv_cusum
from sentinel.kernels import cusum_two_sided, cusum_two_sided_py
from sentinel.kernels.cusum import cusum_two_sided_py as ref

needs_rust = pytest.mark.skipif(not kernels.rust_available(), reason="sentinel._core not built")


def _capped_loop(x, k, h, cap):
    """Scalar textbook loop of the capped CUSUM (the old numpy baseline, written per element)."""
    sp, sn = [], []
    cp = cn = 0.0
    for v in x:
        if not np.isfinite(v):
            cp = cn = 0.0
        else:
            cp = min(cap, max(0.0, cp + v - k))
            cn = min(cap, max(0.0, cn - v - k))
        sp.append(cp)
        sn.append(cn)
    sp, sn = np.array(sp), np.array(sn)
    return sp, sn, (sp > h) | (sn > h)


def _same(a, b):
    for x, y in zip(a, b, strict=True):
        assert x.shape == y.shape and x.dtype == y.dtype
        np.testing.assert_array_equal(x, y)


def test_reexports():
    assert cusum_two_sided_py is ref
    assert kernels.get("cusum_two_sided", "python") is ref


def test_cap_hand_example():
    x = np.array([0.0, 2.0, 2.0, 2.0, np.nan, -3.0, -3.0, 0.5])
    sp, sn, al = ref(x, k=0.5, h=2.5, cap=2.5)
    np.testing.assert_array_equal(sp, [0.0, 1.5, 2.5, 2.5, 0.0, 0.0, 0.0, 0.0])
    np.testing.assert_array_equal(sn, [0.0, 0.0, 0.0, 0.0, 0.0, 2.5, 2.5, 1.5])
    assert not al.any()  # strict > never fires when capped at h


@pytest.mark.parametrize("seed", [0, 1])
def test_cap_matches_scalar_loop(seed):
    rng = np.random.default_rng(seed)
    x = rng.standard_normal((4, 500)) * 0.8 + rng.normal(0, 0.4, (4, 1))
    x[rng.random(x.shape) < 0.05] = np.nan
    x[2, 7] = np.inf
    for cap in (0.0, 0.7, 3.0):
        sp, sn, al = ref(x, 0.2, 1.0, cap=cap)
        for i in range(x.shape[0]):
            _same((sp[i], sn[i], al[i]), _capped_loop(x[i], 0.2, 1.0, cap))


def test_cap_none_is_default_and_huge_cap_is_noop():
    x = np.random.default_rng(5).standard_normal((3, 300))
    _same(ref(x, 0.3, 2.0), ref(x, 0.3, 2.0, cap=None))
    _same(ref(x, 0.3, 2.0), ref(x, 0.3, 2.0, cap=1e300))


@pytest.mark.parametrize("cap", [np.nan, np.inf, -1.0])
def test_cap_validation(cap):
    with pytest.raises(ValueError):
        ref(np.zeros(3), 0.5, 1.0, cap=cap)


def test_dispatched_cap_runs_on_current_backend():
    x = np.random.default_rng(6).standard_normal((5, 200))
    _same(cusum_two_sided(x, 0.1, 0.5, cap=0.5), ref(x, 0.1, 0.5, cap=0.5))


def _old_pv_cusum(x, k, h):
    """Verbatim copy of the pre-#49 numpy baseline in sentinel/eval/baselines/pv/cusum.py."""
    x = np.asarray(x, np.float64)
    one_d = x.ndim == 1
    x2 = np.atleast_2d(x)
    n, T = x2.shape
    s_pos = np.zeros((n, T))
    s_neg = np.zeros((n, T))
    sp = np.zeros(n)
    sn = np.zeros(n)
    for t in range(T):
        xt = x2[:, t]
        gap = np.isnan(xt)
        xt = np.where(gap, 0.0, xt)
        sp = np.where(gap, 0.0, np.minimum(h, np.maximum(0.0, sp + xt - k)))
        sn = np.where(gap, 0.0, np.minimum(h, np.maximum(0.0, sn - xt - k)))
        s_pos[:, t] = sp
        s_neg[:, t] = sn
    alarm = np.where(s_pos >= h, 1, np.where(s_neg >= h, -1, 0)).astype(np.int8)
    if one_d:
        return s_pos[0], s_neg[0], alarm[0]
    return s_pos, s_neg, alarm


@pytest.mark.parametrize("backend", ["python", pytest.param("rust", marks=needs_rust)])
@pytest.mark.parametrize("shape,k,h", [((30, 2000), 0.1, 0.1), ((1500,), 0.5, 4.0), ((6, 400), 0.0, 2.0)])
def test_pv_wrapper_unchanged(monkeypatch, backend, shape, k, h):
    monkeypatch.setenv("SENTINEL_KERNELS", backend)
    rng = np.random.default_rng(11)
    x = rng.standard_normal(shape) * 0.7 + 0.2
    x[rng.random(shape) < 0.05] = np.nan
    _same(pv_cusum(x, k, h), _old_pv_cusum(x, k, h))
    with pytest.raises(ValueError):
        pv_cusum(x, k, 0.0)


@needs_rust
@pytest.mark.parametrize("cap", [0.0, 0.5, 2.0, 1e300])
def test_cap_parity_rust_vs_python(cap):
    rng = np.random.default_rng(3)
    x = rng.standard_normal((40, 1500)) * 1.5 + 0.1
    x[rng.random(x.shape) < 0.03] = np.nan
    x[:, ::97] = np.inf
    _same(ref(x, 0.25, 2.0, cap=cap), kernels.get("cusum_two_sided", "rust")(x, 0.25, 2.0, cap=cap))
    _same(ref(x[0], 0.25, 2.0, cap=cap), kernels.get("cusum_two_sided", "rust")(x[0], 0.25, 2.0, cap=cap))


@needs_rust
def test_rust_rejects_bad_cap():
    core = kernels._load_core()
    for bad in (np.nan, np.inf, -1.0):
        with pytest.raises(ValueError):
            core.cusum_two_sided(np.zeros((1, 3)), 0.5, 1.0, bad)
