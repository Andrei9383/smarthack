"""K8 battery strategies: reference behaviour on hand cases, physics invariants, Rust == Python bit for bit."""

import numpy as np
import pytest

from wattsentry import kernels
from wattsentry.kernels.strategies import STRATEGY_CODES, battery_strategies_py, day_rank_median

DT = 0.25


def make(rng, n=12, days=3, steps=96):
    t = days * steps
    hod = np.tile(np.arange(steps) / steps * 24, days)
    pv = np.clip(np.sin((hod - 6) / 12 * np.pi), 0, None)[None] * rng.uniform(2, 8, (n, 1)) * rng.uniform(0.3, 1, (n, t))
    load = rng.uniform(0.2, 2.5, (n, t))
    strategy = (np.arange(n) % 4).astype(np.int8)
    has = strategy != 0
    cap = np.where(has, rng.uniform(5, 15, n), 0.0)
    pmax = np.where(has, rng.uniform(3, 5, n), 0.0)
    reserve = rng.uniform(0.1, 0.4, n)
    eta = np.full(n, 0.95)
    k_top = rng.integers(8, 21, n).astype(np.int32)
    soc0 = cap * rng.uniform(0, 1, n)  # some start below the reserve: may charge, not discharge
    day = np.repeat(np.arange(days), steps).astype(np.int32)
    price = 400 + 300 * np.sin((hod - 13) / 24 * 2 * np.pi) + rng.normal(0, 40, t)
    tie = np.arange(1, t, 7)
    price[tie - 1] = price[tie]  # ties
    rank, med = day_rank_median(price, day)
    comply = rng.random((n, days)) > 0.2
    return pv, load, strategy, cap, pmax, reserve, eta, k_top, soc0, price, rank, med, day, comply


def test_rank_median():
    price = np.array([3.0, 1.0, 3.0, 2.0, 5.0, 5.0, 0.0])
    day = np.array([0, 0, 0, 0, 1, 1, 1])
    rank, med = day_rank_median(price, day)
    assert rank.tolist() == [0, 3, 1, 2, 0, 1, 2]  # ties: earlier step ranks first
    assert med.tolist() == [2.5] * 4 + [5.0] * 3


def test_physics_invariants():
    args = make(np.random.default_rng(0))
    pv, load, strategy, cap, pmax, reserve, eta, k_top, soc0, *_ = args
    batt, soc = battery_strategies_py(*args)
    assert batt.shape == soc.shape == pv.shape
    assert np.all(np.abs(batt) <= pmax[:, None] + 1e-12)
    assert np.all(soc <= cap[:, None] + 1e-9)
    floor = np.minimum(reserve * cap, soc0)[:, None]
    assert np.all(soc >= floor - 1e-9)
    none = strategy == STRATEGY_CODES["none"]
    assert np.all(batt[none] == 0) and np.all(soc[none] == soc0[none, None])
    # energy balance: soc' - soc = eta * c * dt - d * dt / eta
    prev = np.concatenate([soc0[:, None], soc[:, :-1]], 1)
    c, d = np.clip(batt, 0, None), np.clip(-batt, 0, None)
    assert np.allclose(soc - prev, eta[:, None] * c * DT - d * DT / eta[:, None])
    # charging only from PV surplus, and self-consumption homes never export from the battery
    surplus = pv - load
    assert np.all(c <= np.clip(surplus, 0, None) + 1e-12)
    sc = np.isin(strategy, [STRATEGY_CODES["self_consumption"], STRATEGY_CODES["enrolled"]])
    assert np.all(d[sc] <= np.clip(-surplus[sc], 0, None) + 1e-12)


def test_price_driven_hand_case():
    """One night step, battery half full: top quarter -> full discharge (exports), above median -> cover the house,
    below median -> hold; non-compliant day -> self-consumption (cover the house)."""
    one = lambda **kw: battery_strategies_py(  # noqa: E731
        np.zeros((1, 1)), np.full((1, 1), 1.0), np.int8(2), 10.0, 4.0, 0.2, 1.0, 10, 5.0,
        kw.get("price", [100.0]), [kw.get("rank", 0)], [kw.get("med", 50.0)], [0], [[kw.get("comply", True)]], DT)
    assert one()[0][0, 0] == -4.0  # top: min(P=4, avail=(5-2)/0.25=12)
    assert one(rank=50)[0][0, 0] == -1.0  # above median: cover the 1 kW load
    assert one(rank=50, price=[10.0])[0][0, 0] == 0.0  # below median: keep it for the evening
    assert one(comply=False)[0][0, 0] == -1.0  # non-compliant day: self-consumption
    batt, soc = battery_strategies_py(np.full((1, 1), 3.0), np.full((1, 1), 1.0), np.int8(2), 10.0, 4.0, 0.2, 0.9,
                                      10, 9.8, [100.0], [0], [50.0], [0], [[True]], DT)
    assert batt[0, 0] == pytest.approx(0.2 / (0.9 * DT)) and soc[0, 0] == pytest.approx(10.0)  # PV fills it first


def test_dispatch_registered():
    assert "battery_strategies" in kernels.names()
    assert kernels.get("battery_strategies", "python") is battery_strategies_py


rust = pytest.mark.skipif(not kernels.rust_available(), reason="wattsentry._core not built")


@rust
@pytest.mark.parametrize("seed,n,days,steps", [(0, 12, 3, 96), (1, 1, 1, 1), (2, 400, 7, 96), (3, 4, 2, 0),
                                               (4, 9, 2, 100)])
def test_rust_parity(seed, n, days, steps):
    args = make(np.random.default_rng(seed), n, days, steps)
    for x, y in zip(battery_strategies_py(*args), kernels.get("battery_strategies", "rust")(*args)):
        assert x.shape == y.shape and x.dtype == y.dtype and np.array_equal(x, y)


@rust
def test_rust_rejects_bad_days():
    args = list(make(np.random.default_rng(5), 3, 2))
    args[12] = args[12] + 5
    with pytest.raises(ValueError):
        kernels.get("battery_strategies", "rust")(*args)
