"""Card I8 #178: imbalance level 3, enrolled batteries in the 15:00 LP (scipy HiGHS, no cvxpy needed)."""

from __future__ import annotations

import numpy as np
import pytest

pytest.importorskip("scipy")

from wattsentry.opt import level3 as L3  # noqa: E402

A, B = 1.4407, 0.7204


def homes(n=2, cap=10.0, p=5.0, reserve=2.0, feeder=None):
    rows = [{"site_id": f"S{i}", "battery_kwh": cap, "battery_kw": p, "eta_charge": 0.95, "eta_discharge": 0.95,
             "soc_min": 0.05, "soc_max": 1.0, "comfort_reserve_kwh": reserve,
             "feeder_id": (feeder[i] if feeder else "F0")} for i in range(n)]
    return rows, L3.enrolled_from_rows(rows, np.arange(n))


def test_enrolment_is_nested_and_deterministic():
    rows = [{"site_id": f"S{i}", "battery_kwh": (10.0 if i % 2 else 0.0)} for i in range(40)]
    small, big = L3.select_enrolled(rows, 0.25), L3.select_enrolled(rows, 0.5)
    assert len(small) == 5 and len(big) == 10
    assert set(small) <= set(big)
    assert all(rows[i]["battery_kwh"] > 0 for i in big)
    assert np.array_equal(small, L3.select_enrolled(rows, 0.25))


def test_commit_factor_is_the_p10_compliance_share():
    assert L3.commit_factor(0, 0.1) == 0.0
    assert L3.commit_factor(10, 0.0) == 1.0
    k = L3.commit_factor(27, 0.1)
    assert 0.8 < k < 0.9 - 1e-3
    assert L3.commit_factor(1000, 0.1) > k  # more homes, tighter P10


def test_quantile_nodes_split_normal():
    bands = np.array([[1.0, 2.0, 4.0], [0.0, 0.0, 0.0]])
    nd = L3.quantile_nodes(bands, 20)
    assert nd.shape == (2, 20)
    assert np.all(np.diff(nd[0]) >= 0)
    assert np.isclose(np.median(nd[0]), 2.0, atol=0.2)
    assert np.allclose(nd[1], 0.0)
    assert nd[0, -1] - 2.0 > 2.0 - nd[0, 0]  # right-skewed band -> right-skewed nodes


def test_no_battery_lp_is_the_newsvendor():
    """Without batteries and with PI = PZU, the LP trades to the a/(a+b) quantile of the nodes."""
    T = 4
    mask = np.array([False, True, True, True])
    nodes = np.tile(np.linspace(0.0, 1.0, 21)[None, :], (T, 1))
    plan = L3.plan_level3(nodes, np.full(T, 0.2), np.full(T, 500.0), np.full(T, 500.0), mask, a=A, b=B)
    assert plan.status == "optimal"
    assert plan.traded[0] == 0.0
    q = np.quantile(nodes[1], A / (A + B))
    assert np.allclose(plan.traded[1:] + 0.2, q, atol=0.06)


def test_batteries_shift_energy_to_the_expensive_pi_step_and_keep_limits():
    """Shortfall at both steps; PI is dear at step 1 and cheap at step 2: the LP discharges at 1, recharges at 2."""
    rows, en = homes(2)
    T = 3
    mask = np.array([False, True, True])
    nodes = np.full((T, 10), 0.01)  # 10 kWh short per step, no uncertainty
    pzu = np.full(T, 500.0)
    pi_hat = np.array([500.0, 900.0, 450.0])
    e0 = np.array([8.0, 8.0])
    bhat = np.zeros((2, 2))
    plan = L3.plan_level3(nodes, np.zeros(T), pzu, pi_hat, mask, en, e0, bhat, e_end_min=e0, a=A, b=B)
    assert plan.status == "optimal"
    assert np.all(plan.batt_kw[:, 0] < -1.0)  # discharge at the dear step
    assert np.all(plan.batt_kw[:, 1] > 1.0)  # recharge at the cheap step
    assert np.all(plan.e_kwh[:, -1] >= e0 - 1e-6)  # customer gets the energy back
    assert np.all(plan.e_kwh >= en.e_floor_kwh[:, None] - 1e-6)
    assert np.all(np.abs(plan.batt_kw) <= en.p_max_kw[:, None] + 1e-6)
    assert plan.flex_mwh[1] < 0 < plan.flex_mwh[2]
    nob = L3.plan_level3(nodes, np.zeros(T), pzu, pi_hat, mask, a=A, b=B)
    assert plan.objective < nob.objective


def test_battery_lp_never_worse_than_without_batteries():
    rng = np.random.default_rng(1)
    rows, en = homes(3)
    T = 8
    mask = np.zeros(T, bool)
    mask[2:] = True
    nodes = np.sort(rng.normal(0.02, 0.01, (T, 20)), axis=1)
    pzu = rng.uniform(300, 900, T)
    pi_hat = pzu + rng.normal(0, 80, T)
    e0 = np.array([3.0, 6.0, 9.5])
    bhat = rng.uniform(-2, 1, (3, 6))
    with_b = L3.plan_level3(nodes, np.full(T, 0.02), pzu, pi_hat, mask, en, e0, bhat, a=A, b=B)
    without = L3.plan_level3(nodes, np.full(T, 0.02), pzu, pi_hat, mask, a=A, b=B)
    assert with_b.status == "optimal" and without.status == "optimal"
    assert with_b.objective <= without.objective + 1e-6


def test_feeder_headroom_zero_blocks_extra_export():
    rows, en = homes(2)
    T = 2
    mask = np.array([True, True])
    nodes = np.full((T, 10), 0.01)
    pi_hat = np.array([2000.0, 100.0])  # strong reason to discharge at step 0
    e0 = np.array([9.0, 9.0])
    bhat = np.zeros((2, 2))
    plan = L3.plan_level3(nodes, np.zeros(T), np.full(T, 500.0), pi_hat, mask, en, e0, bhat, a=A, b=B,
                          headroom_kw=np.zeros((1, 2)))
    assert plan.status == "optimal"
    extra_export = (bhat - plan.batt_kw).sum(axis=0)
    assert np.all(extra_export <= 1e-6)
    free = L3.plan_level3(nodes, np.zeros(T), np.full(T, 500.0), pi_hat, mask, en, e0, bhat, a=A, b=B)
    assert (bhat - free.batt_kw).sum(axis=0)[0] > 1.0


def test_feeder_headroom_from_history():
    hist = np.array([[[1.0, -5.0], [3.0, np.nan]]])  # 1 feeder, 2 days, 2 steps
    h = L3.feeder_headroom(np.array([4.0]), hist)
    assert np.allclose(h, [[1.0, 9.0]])


def test_track_schedule_respects_physics():
    rows, en = homes(1)
    batt, e = L3.track_schedule(np.array([[-10.0, -10.0, -10.0, 10.0]]), np.array([2.5]), en,
                                floor=en.e_floor_kwh)
    assert batt[0, 0] >= -5.0 - 1e-9
    assert np.all(e >= en.e_floor_kwh[0] - 1e-9)
    assert batt[0, 3] > 0


def test_run_day_noncompliant_homes_keep_the_rule():
    rows, en = homes(2)
    T = 12
    mask = np.zeros(T, bool)
    mask[6:10] = True
    rng = np.random.default_rng(0)
    pv = np.zeros((2, T))
    load = rng.uniform(0.5, 1.5, (2, T))
    batt, e = L3.self_consumption_rollout(pv, load, np.array([9.0, 9.0]), en)
    nodes = np.full((T, 10), 0.003)
    pi_hat = np.full(T, 500.0)
    pi_hat[7] = 1500.0
    res = L3.run_day(nodes=nodes, bought=np.zeros(T), pzu=np.full(T, 500.0), pi_hat=pi_hat, mask=mask, en=en,
                     pv=pv, load=load, batt=batt, soc_kwh=e, pv_prev=pv, load_prev=load,
                     compliant=np.array([True, False]), a=A, b=B, kappa=L3.commit_factor(2, 0.1))
    assert res.plan.status == "optimal"
    assert res.diag["compliant"] == 1
    assert np.all(res.delta_mwh[~mask] == 0.0)
    assert res.delta_mwh[7] < 0  # extra discharge at the dear step reduces the portfolio's import
    # the non-compliant home contributes nothing: delta = compliant home's change only
    only = L3.run_day(nodes=nodes, bought=np.zeros(T), pzu=np.full(T, 500.0), pi_hat=pi_hat, mask=mask, en=en,
                      pv=pv, load=load, batt=batt, soc_kwh=e, pv_prev=pv, load_prev=load,
                      compliant=np.array([False, False]), a=A, b=B, kappa=0.5)
    assert np.allclose(only.delta_mwh, 0.0)


class _FakeSource:
    """Minimal #10 window source: sites rows + (S, steps) arrays, days from 2025-06-01 UTC."""

    def __init__(self, n_sites=12, n_days=5, seed=0):
        rng = np.random.default_rng(seed)
        self.sites = [{"site_id": f"S{i}", "battery_kwh": (10.0 if i % 2 == 0 else 0.0), "battery_kw": 5.0,
                       "eta_charge": 0.95, "eta_discharge": 0.95, "soc_min": 0.05, "soc_max": 1.0,
                       "comfort_reserve_kwh": 2.0, "feeder_id": f"F{i % 3}"} for i in range(n_sites)]
        steps = n_days * 96
        t = np.arange(steps) % 96
        pv = np.clip(np.sin((t - 20) / 56 * np.pi), 0, None)[None] * rng.uniform(3, 6, (n_sites, 1))
        load = rng.uniform(0.3, 1.5, (n_sites, steps))
        cap = np.array([s["battery_kwh"] for s in self.sites])
        en = L3.enrolled_from_rows(self.sites, np.arange(n_sites))
        batt, e = L3.self_consumption_rollout(pv, load, np.where(cap > 0, 3.0, 0.0), en)
        self._a = {"pv_ac": pv, "load": load, "batt": batt,
                   "soc": np.where(cap[:, None] > 0, e / np.where(cap > 0, cap, 1.0)[:, None], 0.0)}

    def array(self, name):
        return self._a[name]


def test_replay_level3_glue_on_a_fake_source():
    import pandas as pd

    from wattsentry.data.prices import prices_at

    src = _FakeSource()
    days = np.array([2, 3, 4])
    t0 = pd.Timestamp("2025-06-01")
    ts = np.stack([(t0 + pd.Timedelta(days=int(d)) + pd.to_timedelta(np.arange(96) * 15, "min")).to_numpy()
                   for d in days])
    pzu, pi = prices_at(ts.ravel())
    pzu, pi = pzu.reshape(3, 96), pi.reshape(3, 96)
    hrs = L3._local_hours(ts)
    mask = (hrs >= 16) & (hrs < 24)
    bought = np.full((3, 96), 0.002)
    band = np.stack([np.full((3, 96), 0.001), np.full((3, 96), 0.003), np.full((3, 96), 0.006)], axis=-1)
    out = L3.replay_level3(src, np.arange(12), days, ts=ts, bought=bought, id_cal=band, mask=mask, pzu=pzu, pi=pi,
                           policies=list(L3.L3_POLICIES), a=A, b=B, share=0.5, feeders=False,
                           log=lambda *a, **k: None)
    d = out["diag"]
    assert d["battery_homes"] == 6
    assert d["per_policy"]["sentinel_l3"]["enrolled"] == 3 and d["per_policy"]["sentinel_l3_all"]["enrolled"] == 6
    for p in L3.L3_POLICIES:
        assert out[p]["traded"].shape == (3, 96)
        assert d["per_policy"][p]["failed_days"] == 0
        assert np.all(out[p]["traded"][~mask] == 0.0) and np.all(out[p]["delta"][~mask] == 0.0)
    assert np.allclose(out["sentinel_id_lp"]["delta"], 0.0)
