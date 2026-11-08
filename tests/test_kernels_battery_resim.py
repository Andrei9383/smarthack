"""K7 fault-injector battery resim: Rust == Python reference bit for bit, incl. NaN and onset edge cases."""

import numpy as np
import pytest

from wattsentry import kernels
from wattsentry.kernels.battery_resim import battery_resim_py


def make(rng, m=6, T=960, nan=True):
    cap = rng.uniform(5, 15, m)
    cap[0] = 0.0 if m > 2 else cap[0]  # a row whose battery is gone: power and energy stay 0
    pmax = rng.uniform(2, 6, m)
    eta_c, eta_d = rng.uniform(0.9, 0.98, m), rng.uniform(0.9, 0.98, m)
    soc_min, soc_max = rng.uniform(0.0, 0.15, m), rng.uniform(0.85, 1.0, m)
    # PV-like surplus: large swings so power clipping and full / empty limits bind often
    tod = np.sin(np.arange(T) / 96 * 2 * np.pi)
    surplus = 4000 * tod[None] + rng.normal(0, 3000, (m, T))
    soc0 = rng.uniform(0, 1, (m, T))
    batt0 = rng.normal(0, 2000, (m, T))
    cf = np.ones((m, T))
    pf = np.ones((m, T))
    for r in range(m):  # fade ramps and underuse steps, as apply_battery_fade sets them
        i0 = int(rng.integers(0, T))
        cf[r, i0:] = np.linspace(1, rng.uniform(0.45, 0.8), T - i0)
        j0 = int(rng.integers(0, T))
        pf[r, j0:j0 + T // 4] = rng.uniform(0.05, 0.3)
    if nan:
        surplus[rng.random((m, T)) < 0.01] = np.nan
        soc0[rng.random((m, T)) < 0.01] = np.nan
        batt0[rng.random((m, T)) < 0.01] = np.nan
    start = rng.integers(0, T, m)
    t0 = int(start.min())
    e0 = (np.nan_to_num(soc0[:, t0 - 1]) if t0 > 0 else np.full(m, 0.5)) * cap
    return surplus, soc0, batt0, cf, pf, start, e0, cap, pmax, eta_c, eta_d, soc_min, soc_max, t0


def test_reference_keeps_series_before_onset():
    args = make(np.random.default_rng(0), nan=False)
    batt, soc = battery_resim_py(*args)
    _, soc0, batt0, *_, start, _e0, cap, pmax, _ec, _ed, _smin, smax, _t0 = (*args[:5], *args[5:])
    for r in range(len(start)):
        s = start[r]
        assert np.array_equal(batt[r, :s], batt0[r, :s]) and np.array_equal(soc[r, :s], soc0[r, :s])
        assert np.all(np.abs(batt[r, s:]) <= pmax[r] * 1000 + 1e-9)
        if cap[r] > 0:
            assert np.all(soc[r, s:] <= smax[r] + 1e-9)
        else:
            assert np.all(batt[r, s:] == 0)


rust = pytest.mark.skipif(not kernels.rust_available(), reason="wattsentry._core not built")


def _bits(a):
    """Bit patterns with every NaN made the same NaN: numpy's SIMD max/min sometimes return a NaN with the sign bit
    set; the injector writes NaN as a Parquet null, so only the NaN mask matters. Signed zeros are compared."""
    return np.where(np.isnan(a), np.nan, a).view(np.uint64)


@rust
@pytest.mark.parametrize("seed,m,T,nan", [(0, 6, 960, True), (1, 1, 1, False), (2, 40, 35040, True),
                                          (3, 3, 97, False), (4, 0, 50, False), (5, 8, 2000, False)])
def test_rust_parity(seed, m, T, nan):
    args = make(np.random.default_rng(seed), m, T, nan) if m else (
        *(np.zeros((0, T)) for _ in range(5)), np.zeros(0, np.int64), *(np.zeros(0) for _ in range(7)), 0)
    for x, y in zip(battery_resim_py(*args), kernels.get("battery_resim", "rust")(*args)):
        assert x.shape == y.shape and x.dtype == y.dtype
        assert np.array_equal(_bits(x), _bits(y))  # bit for bit, incl. signed zeros (NaN payloads normalised)


@rust
def test_rust_parity_same_onset_everywhere():
    """All rows start at t0 (prev_cap from cf[t0]) vs rows that start later (prev_cap from cf[start-1])."""
    rng = np.random.default_rng(7)
    args = list(make(rng, 5, 500))
    args[5] = np.full(5, args[13], np.int64)
    for x, y in zip(battery_resim_py(*args), kernels.get("battery_resim", "rust")(*args)):
        assert np.array_equal(_bits(x), _bits(y))
