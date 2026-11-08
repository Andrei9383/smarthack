"""G2 voltage / inverter trips on the IEEE European LV feeder."""

from __future__ import annotations

import numpy as np
import pytest

from wattsentry.gridtwin import voltage as V


# ------------------------------------------------------------------------------------------ pure numpy parts
def test_map_homes_hero_first_and_cycles():
    by = {"F3": ["a", "b"], "F7": ["x", "y", "z"]}
    m = V.map_homes(by, "F7", n=7)
    assert m[:3] == ["x", "y", "z"] and m[3:5] == ["a", "b"]
    assert m[5:] == ["x", "y"]  # cycles deterministically when the world has fewer homes than loads
    assert len(V.map_homes(by, "nope", n=55)) == 55


def test_trips_counts_only_homes_above_limit():
    T = 4
    home_v = np.full((T, V.N_LOADS), 240.0)
    home_v[1, :3] = 254.0  # three homes trip in step 1
    v_max = home_v.max(1)
    v_max[2] = 253.5  # a non-home bus over the limit: a violation step but no home trips
    pv = np.full((T, V.N_LOADS), 4.0)
    price = np.array([0.5, 0.4, 0.3, 0.2])
    r = V.trips(pv, V.PolicyVoltage(v_max, home_v, 0), price)
    assert r["voltage_violations"] == 2
    assert r["home_steps_tripped"] == 3
    assert r["inverter_trip_kwh"] == pytest.approx(3 * 4.0 * V.DT_H)
    assert r["inverter_trip_lei"] == pytest.approx(3 * 4.0 * V.DT_H * 0.4)
    assert r["pv_kwh_in_violation_steps"] == pytest.approx(2 * V.N_LOADS * 4.0 * V.DT_H)


def test_pzu_prices_lei_kwh():
    ts = np.array([1748898000, 1748898900])  # 2025-06-02 21:00 / 21:15 UTC (demo replay week)
    p, src = V.pzu_lei_kwh(ts)
    assert "OPCOM" in src and p.shape == (2,) and np.all((p > -5) & (p < 10))


def test_pzu_falls_back_to_world_prices_when_missing():
    p, src = V.pzu_lei_kwh(np.array([4102444800]), fallback_lei_mwh=np.array([500.0]))  # 2100: no OPCOM row
    assert p[0] == pytest.approx(0.5) and src.startswith("world prices")


# ------------------------------------------------------------------------------------------ pandapower parts
pytest.importorskip("pandapower")


@pytest.fixture(scope="module")
def feeder_sens():
    fd = V.build_feeder(1.03)
    return fd, V.sensitivity(fd, p_ref_kw=np.full(V.N_LOADS, 2.0))


def test_ieee_feeder_shape_and_noload_voltage(feeder_sens):
    fd, _ = feeder_sens
    assert len(fd.load_bus) == V.N_LOADS and set(np.unique(fd.load_phase)) == {0, 1, 2}
    v0 = V.power_flow(fd, np.zeros(V.N_LOADS))
    assert np.allclose(v0, 1.03 * V.V_BASE, atol=0.5)  # no load: flat at the busbar setting


def test_export_raises_voltage_and_estimate_matches_full_pf(feeder_sens):
    fd, sens = feeder_sens
    assert (sens.s[fd.home_rows, np.arange(V.N_LOADS)] > 0).all()  # own export raises own voltage
    rng = np.random.default_rng(1)
    for _ in range(4):
        p = rng.uniform(0.0, 4.0, V.N_LOADS)
        v = V.power_flow(fd, p)
        assert np.abs(sens.estimate(p[None])[0] - v).max() < V.NEAR_LIMIT * V.V_LIMIT


def test_assess_uses_full_pf_near_limit(feeder_sens):
    fd, sens = feeder_sens
    # scale a uniform export until the estimated max sits right at the limit, then check the full PF took over
    p1 = np.ones(V.N_LOADS)
    v_hi = sens.estimate(p1[None])[0].max()
    v_lo = sens.estimate(np.zeros((1, V.N_LOADS)))[0].max()
    k = (V.V_LIMIT - v_lo) / (v_hi - v_lo)
    p = np.vstack([np.zeros(V.N_LOADS), k * p1])
    res = V.assess(fd, sens, p)
    assert res.n_full_pf == 1
    assert res.v_max[1] == pytest.approx(V.power_flow(fd, p[1]).max())
