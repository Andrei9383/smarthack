"""K1 battery rollout: Python reference == sentinel.opt.commit.rollout, Rust == Python bit for bit (#50)."""

import numpy as np
import pytest

from sentinel import kernels
from sentinel.kernels.rollout import battery_rollout_py
from sentinel.opt.commit import BatteryParams, rollout


def make(rng, n=7, s=5, t=96):
    has = np.arange(s) % 3 != 0  # every third site has no battery
    cap = np.where(has, rng.uniform(5, 15, s), 0.0)
    pmax = np.where(has, rng.uniform(3, 5, s), 0.0)
    eta_c = np.where(has, rng.uniform(0.9, 0.98, s), 1.0)
    eta_d = np.where(has, rng.uniform(0.9, 0.98, s), 1.0)
    e_lo = 0.1 * cap + np.where(has, 1.0, 0.0)
    e_hi = 0.95 * cap
    req = rng.normal(0, 4, (n, s, t))
    soc0 = rng.uniform(0.0, 1.0, (n, s))  # some start below the floor: may charge, not discharge
    return req, soc0, cap, pmax, eta_c, eta_d, e_lo, e_hi


def test_reference_matches_commit_rollout():
    assert kernels.get("battery_rollout", "python") is battery_rollout_py
    req, soc0, *bp = make(np.random.default_rng(0))
    soc, batt = battery_rollout_py(req, soc0, *bp)
    soc_r, batt_r, _ = rollout(req, soc0, BatteryParams(*bp), np.zeros(1), np.zeros(1))
    assert np.array_equal(soc, soc_r) and np.array_equal(batt, batt_r)


def test_reference_2d_and_limits():
    req, soc0, cap, pmax, eta_c, eta_d, e_lo, e_hi = make(np.random.default_rng(1), n=1)
    soc, batt = battery_rollout_py(req[0], soc0[0], cap, pmax, eta_c, eta_d, e_lo, e_hi)
    assert soc.shape == batt.shape == req[0].shape
    assert np.all(np.abs(batt) <= pmax[:, None] + 1e-12) and np.all(soc * cap[:, None] <= e_hi[:, None] + 1e-9)
    assert np.all(batt[cap == 0] == 0) and np.all(soc[cap == 0] == 0)


rust = pytest.mark.skipif(not kernels.rust_available(), reason="sentinel._core not built")


@rust
@pytest.mark.parametrize("seed,n,s,t", [(0, 7, 5, 96), (1, 1, 1, 1), (2, 200, 50, 96), (3, 3, 4, 0), (4, 2, 9, 500)])
def test_rust_parity(seed, n, s, t):
    args = make(np.random.default_rng(seed), n, s, t)
    for x, y in zip(battery_rollout_py(*args), kernels.get("battery_rollout", "rust")(*args)):
        assert x.shape == y.shape and np.array_equal(x, y)


@rust
def test_rust_parity_2d_broadcast_soc():
    req, soc0, *bp = make(np.random.default_rng(5), n=1)
    a = battery_rollout_py(req[0], 0.5, *bp)
    b = kernels.get("battery_rollout", "rust")(req[0], 0.5, *bp)
    assert all(np.array_equal(x, y) for x, y in zip(a, b))
