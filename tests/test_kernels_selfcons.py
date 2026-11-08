"""K7 fleet self-consumption: Python reference == wattsentry.sim.fleet.battery.simulate_self_consumption, Rust == Python
bit for bit (compared as raw bits, so -0.0 vs +0.0 and NaN payloads count) (#135)."""

import numpy as np
import pytest

from wattsentry import kernels
from wattsentry.kernels.selfcons import self_consumption_py
from wattsentry.sim.fleet.battery import BatteryParams, simulate_self_consumption


def make(rng, n=7, t=96 * 3):
    """Fleet-like sites: every third has no battery; PV/load in W as float32, with exact zeros (night PV, and
    pv == load steps) so signed-zero handling is exercised; small and large batteries so both limits bind."""
    has = np.arange(n) % 3 != 0
    cap = np.where(has, rng.choice([2.0, 5.0, 10.0, 13.5], n), 0.0)
    p = BatteryParams(
        capacity_kwh=cap, p_max_kw=np.where(has, rng.uniform(1.0, 5.0, n), 0.0),
        eta_charge=np.where(has, rng.uniform(0.9, 0.98, n), 0.95),
        eta_discharge=np.where(has, rng.uniform(0.9, 0.98, n), 0.95),
        soc_min=np.where(has, 0.1, 0.0), soc_max=np.where(has, 0.95, 0.0), comfort_reserve_kwh=np.zeros(n))
    day = np.clip(np.sin(np.linspace(0, 2 * np.pi * t / 96, t) - np.pi / 2), 0, None)
    pv = (rng.uniform(2000, 8000, (n, 1)) * day[None] * rng.uniform(0.3, 1, (n, t))).astype(np.float32)
    load = rng.exponential(600, (n, t)).astype(np.float32)
    load[:, ::17] = pv[:, ::17]  # surplus exactly 0
    return p, pv, load


def bits(a):
    return a.view(np.uint32 if a.dtype == np.float32 else np.uint64)


def same(x, y):
    return all(a.shape == b.shape and a.dtype == b.dtype and np.array_equal(bits(a), bits(b)) for a, b in zip(x, y))


def test_reference_is_battery_module():
    assert kernels.get("self_consumption", "python") is self_consumption_py
    p, pv, load = make(np.random.default_rng(0))
    assert same(self_consumption_py(p, pv, load, 0.25), simulate_self_consumption(p, pv, load, 0.25))


rust = pytest.mark.skipif(not kernels.rust_available(), reason="wattsentry._core not built")


@rust
@pytest.mark.parametrize("seed,n,t", [(0, 7, 288), (1, 1, 1), (2, 50, 96 * 30), (3, 4, 0), (4, 0, 10), (5, 300, 500)])
def test_rust_parity(seed, n, t):
    p, pv, load = make(np.random.default_rng(seed), n, t)
    a = self_consumption_py(p, pv, load, 0.25)
    b = kernels.get("self_consumption", "rust")(p, pv, load, 0.25)
    assert same(a, b)


@rust
def test_rust_parity_carry_over_and_fallback():
    """Multi-year: e0 from the previous year; float64 input goes to the reference (same result)."""
    p, pv, load = make(np.random.default_rng(6), 20, 400)
    e0 = p.capacity_kwh * np.random.default_rng(7).uniform(0, 1, 20)
    rs = kernels.get("self_consumption", "rust")
    assert same(self_consumption_py(p, pv, load, 0.25, e0), rs(p, pv, load, 0.25, e0))
    assert same(self_consumption_py(p, pv.astype(np.float64), load, 0.25), rs(p, pv.astype(np.float64), load, 0.25))


@rust
def test_rust_parity_on_fleet():
    """The simulator's own inputs (tiny world, synthetic weather): same bits as the reference."""
    from wattsentry.sim.fleet.config import DT_H, FleetConfig
    from wattsentry.sim.fleet.load import simulate_load
    from wattsentry.sim.fleet.localities import BY_ID
    from wattsentry.sim.fleet.pv import simulate_pv
    from wattsentry.sim.fleet.ro_calendar import local_features, utc_index
    from wattsentry.sim.fleet.sites import build_world
    from wattsentry.sim.fleet.weather import load_weather

    cfg = FleetConfig(n_sites=40, n_feeders=3, n_congested=1, start_year=2024, end_year=2024,
                      weather_source="synthetic")
    feeders, sites = build_world(cfg)
    ts = utc_index(2024)[: 96 * 60]
    w = load_weather([BY_ID[i] for i in sorted(set(feeders.locality_id))], ts, cfg.seed, cfg.weather_source)
    pv, _ = simulate_pv(sites, w)
    load = simulate_load(sites, w, local_features(ts), np.random.default_rng(0))
    p = BatteryParams.from_sites(sites)
    assert same(self_consumption_py(p, pv, load, DT_H), kernels.get("self_consumption", "rust")(p, pv, load, DT_H))
