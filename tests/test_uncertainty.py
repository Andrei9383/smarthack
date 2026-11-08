"""Conformal calibration, thresholds.json, P10 scenario rollout + commitment."""

import json
import time

import numpy as np
import pytest

from wattsentry.contracts import DAY_STEPS, BatterySpec, Commitment, FeederPlan, MockPredictor, self_consumption_step
from wattsentry.eval.uncertainty import (
    DEFAULT_THRESHOLDS,
    THRESHOLDS_PATH,
    ConformalCalibration,
    coverage,
    mean_pinball,
    rel_width,
    save_thresholds,
    tune_thresholds,
    validate_thresholds,
)
from wattsentry.eval.uncertainty.conformal import Z90, quantile_at
from wattsentry.eval.uncertainty.demo import make_feeder, mock_members, rule_schedule, run
import wattsentry.opt.commit as commit_mod
from wattsentry.opt.commit import (
    battery_params,
    commit,
    hit_stats,
    make_scenarios,
    realise,
    realised_deliverable,
    rollout,
    rollout_baseline,
)

T0 = "2025-06-21T00:00:00Z"


# --- conformal -----------------------------------------------------------------------------------------------
def _synthetic(n, h, rng, bias=0.0, claim=0.6):
    """y ~ N(mu, sigma); members claim sigma·claim and carry a bias -> raw bands under-cover."""
    mu = 5 + 3 * np.sin(np.linspace(0, np.pi, h))[None, :] * rng.uniform(0.5, 1.5, (n, 1))
    sigma = 0.2 * mu
    y = mu + sigma * rng.standard_normal((n, h))
    s_c = sigma * claim
    m = mu + bias * sigma
    return y, (m - Z90 * s_c, m, m + Z90 * s_c)


def test_conformal_coverage_per_member():
    rng = np.random.default_rng(0)
    h = 24
    y_cal, a_cal = _synthetic(400, h, np.random.default_rng(1))
    members_cal = {"a": a_cal, "b": tuple(q + 0.3 * (a_cal[1] * 0.2) for q in a_cal)}  # b is biased up
    cal = ConformalCalibration.fit(y_cal, members_cal, n_buckets=2, model="pvnet")
    y, a = _synthetic(400, h, rng)
    members = {"a": a, "b": tuple(q + 0.3 * (a[1] * 0.2) for q in a)}
    for mid, trip in members.items():
        raw_cov = coverage(y, trip[0], trip[2])
        q10, q50, q90 = cal.apply(*trip, member=mid)
        cov = coverage(y, q10, q90)
        assert raw_cov < 0.7
        assert 0.75 <= cov <= 0.85, (mid, cov)
        assert mean_pinball(y, q10, q50, q90) < mean_pinball(y, *trip)
        assert np.all(q10 <= q50 + 1e-12) and np.all(q50 <= q90 + 1e-12)
    rep = cal.report(y, members)
    assert 0.75 <= rep["a"]["calibrated"]["coverage_10_90"] <= 0.85


def test_conformal_json_roundtrip(tmp_path):
    rng = np.random.default_rng(2)
    y, trip = _synthetic(200, DAY_STEPS, rng)
    cal = ConformalCalibration.fit(y, trip, model="pvnet", model_version="mock-1")
    path = tmp_path / "cal.json"
    cal.save(path)
    cal2 = ConformalCalibration.load(path)
    assert json.loads(path.read_text())["members"].keys() == {"0"}
    np.testing.assert_allclose(cal.apply(*[q[0] for q in trip])[2], cal2.apply(*[q[0] for q in trip])[2])


# --- thresholds ----------------------------------------------------------------------------------------------
def test_thresholds_file_and_roundtrip(tmp_path):
    assert json.loads(THRESHOLDS_PATH.read_text()) == DEFAULT_THRESHOLDS
    rng = np.random.default_rng(0)
    th = tune_thresholds({"rel_width": rng.gamma(2, 0.2, 500), "ood": rng.exponential(1, 500),
                          "pfault_std": rng.uniform(0, 0.3, 500)},
                         {"ood": rng.exponential(1, 300), "anomaly_score": rng.random(300),
                          "is_anomaly": rng.random(300) < 0.3}, source_run="run-123")
    p = save_thresholds(th, tmp_path / "t.json")
    back = json.loads(p.read_text())
    assert back == th and back["source_run"] == "run-123"
    assert set(back["site"]) == set(DEFAULT_THRESHOLDS["site"])
    assert set(back["turbine"]) == set(DEFAULT_THRESHOLDS["turbine"])
    for bad in ({**DEFAULT_THRESHOLDS, "extra": 1},
                {**DEFAULT_THRESHOLDS, "site": {**DEFAULT_THRESHOLDS["site"], "ood_max": -1}},
                {**DEFAULT_THRESHOLDS, "turbine": {"ood_max": 3.0}}):
        with pytest.raises(ValueError):
            validate_thresholds(bad)


def test_rel_width_daylight_only():
    q50 = np.array([0, 0, 1, 2, 1, 0], float)
    assert rel_width(0.5 * q50, q50, 1.5 * q50) == pytest.approx(1.0)
    p = MockPredictor("pv").predict({"site_id": "S1", "kwp_true": 6, "lat": 44.4}, t0=T0)
    assert 0 < rel_width(p) < 2


# --- scenarios -----------------------------------------------------------------------------------------------
@pytest.fixture(scope="module")
def feeder_day():
    f = make_feeder("F01", n_sites=12, seed=3)
    pv = {s["site_id"]: mock_members(s, T0, "pv") for s in f.sites}
    ld = {s["site_id"]: mock_members(s, T0, "load") for s in f.sites}
    return f, pv, ld


def test_scenario_shapes_and_structure(feeder_day):
    f, pv, ld = feeder_day
    cap = {s["site_id"]: s["kwp_true"] for s in f.sites}
    sc = make_scenarios(pv, ld, n=200, seed=0, pv_cap=cap, batteries=f.batteries, soc0=f.soc0)
    n, s, t = 200, len(f.sites), DAY_STEPS
    assert sc.pv.shape == sc.load.shape == (n, s, t)
    assert sc.pv_level.shape == (n,) and sc.load_level.shape == (n, s) and sc.member.shape == (n,)
    assert sc.pv_p50.shape == sc.load_p50.shape == (s, t)
    assert set(np.unique(sc.member)) <= {0, 1, 2} and len(np.unique(sc.member)) == 3
    assert (sc.pv >= 0).all() and (sc.load >= 0).all()
    assert (sc.pv <= np.array(list(cap.values()))[None, :, None] + 1e-9).all()
    # one shared PV level per scenario: Latin hypercube -> exactly one level per 1/n stratum
    assert np.array_equal(np.sort(np.floor(sc.pv_level * n)), np.arange(n))
    # PV at a shared level: the higher the level, the more PV on every site at midday
    mid = sc.pv[:, :, 44]
    lo, hi = np.argmin(sc.pv_level), np.argmax(sc.pv_level)
    if sc.member[lo] == sc.member[hi]:
        assert (mid[hi] >= mid[lo]).all()
    # empirical PV quantiles bracket the forecast band (q10 below q50 below q90 on aggregate)
    tot = sc.pv.sum(axis=1)[:, 44]
    assert np.quantile(tot, 0.1) < sc.pv_p50.sum(axis=0)[44] < np.quantile(tot, 0.9)


def test_quantile_at_hits_the_three_points():
    q10, q50, q90 = np.array([1.0]), np.array([2.0]), np.array([4.0])
    assert quantile_at(q10, q50, q90, np.array([-Z90]))[0] == pytest.approx(1.0)
    assert quantile_at(q10, q50, q90, np.array([0.0]))[0] == pytest.approx(2.0)
    assert quantile_at(q10, q50, q90, np.array([Z90]))[0] == pytest.approx(4.0)
    assert quantile_at(q10, q50, q90, np.array([2 * Z90]))[0] == pytest.approx(6.0)  # linear tail


# --- rollout -------------------------------------------------------------------------------------------------
def test_rollout_respects_limits():
    rng = np.random.default_rng(0)
    specs = [BatterySpec(capacity_kwh=c, p_max_kw=p, comfort_reserve_kwh=r)
             for c, p, r in zip(rng.uniform(5, 15, 8), rng.uniform(3, 5, 8), rng.uniform(0.5, 2, 8))] + [None]
    s = len(specs)
    bp = battery_params(specs, respect_reserve=True)
    soc0 = np.where(bp.capacity_kwh > 0, rng.uniform(0.3, 0.9, s), 0.0)
    sched = rng.uniform(-8, 8, (50, s, DAY_STEPS))  # wildly infeasible requests
    pv = rng.uniform(0, 8, (50, s, DAY_STEPS))
    load = rng.uniform(0, 3, (50, s, DAY_STEPS))
    soc, batt, net = rollout(sched, soc0, bp, pv, load)
    assert soc.shape == batt.shape == net.shape == (50, s, DAY_STEPS)
    e = soc * bp.capacity_kwh[None, :, None]
    assert (np.abs(batt) <= bp.p_max_kw[None, :, None] + 1e-9).all()
    assert (e >= bp.e_min_kwh[None, :, None] - 1e-9).all()  # SoC floor incl. comfort reserve
    assert (e <= bp.e_max_kwh[None, :, None] + 1e-9).all()
    assert (batt[:, -1] == 0).all()  # no battery -> no power
    np.testing.assert_allclose(net, pv - load - batt)
    # energy bookkeeping: each step moves e by eta_c·p·dt or p·dt/eta_d
    de = np.diff(np.concatenate([np.broadcast_to(soc0 * bp.capacity_kwh, (50, s))[..., None], e], -1), axis=-1)
    expect = np.where(batt >= 0, bp.eta_charge[None, :, None] * batt * 0.25, batt * 0.25 / bp.eta_discharge[None, :, None])
    np.testing.assert_allclose(de, expect, atol=1e-9)
    # SpecS accepted directly (same result)
    np.testing.assert_allclose(rollout(sched, soc0, specs, pv, load)[1], batt)


def test_rollout_baseline_matches_self_consumption_step():
    rng = np.random.default_rng(1)
    b = BatterySpec(capacity_kwh=10, p_max_kw=4, comfort_reserve_kwh=2)
    pv, load = rng.uniform(0, 7, DAY_STEPS), rng.uniform(0, 3, DAY_STEPS)
    e = 0.4 * 10
    ref_b, ref_x = [], []
    for t in range(DAY_STEPS):
        p, e, x = self_consumption_step(b, e, pv[t], load[t])
        ref_b.append(p)
        ref_x.append(x)
    _, batt, net = rollout_baseline(np.array([0.4]), [b], pv[None], load[None])
    np.testing.assert_allclose(batt[0], ref_b, atol=1e-12)
    np.testing.assert_allclose(net[0], ref_x, atol=1e-12)


# --- commitment ----------------------------------------------------------------------------------------------
def _commit_setup(feeder_day, seed=0, haircut=0.05):
    f, pv, ld = feeder_day
    cap = {s["site_id"]: s["kwp_true"] for s in f.sites}
    sc = make_scenarios(pv, ld, n=200, seed=seed, pv_cap=cap, batteries=f.batteries, soc0=f.soc0,
                        feeder_id=f.feeder_id, t0=T0)
    sched = rule_schedule(sc.site_ids, sc.pv_p50, sc.load_p50, f.batteries, f.soc0, T0)
    return f, sc, sched, commit(f.feeder_id, sched, sc, haircut=haircut)


def test_commit_ordering_and_haircut(feeder_day):
    f, sc, sched, c = _commit_setup(feeder_day)
    n, a, k = map(np.asarray, (c.nominal, c.available, c.committed))
    assert isinstance(c, Commitment) and c.horizon == DAY_STEPS and c.t0 == T0
    assert (n + 1e-9 >= a).all() and (a + 1e-9 >= k).all() and (k >= 0).all()
    assert k.sum() > 0 and n.sum() > k.sum()
    assert c.extra["n_scenarios"] == 200
    c0 = commit(f.feeder_id, sched, sc, haircut=0.0)
    c5 = commit(f.feeder_id, sched, sc, haircut=0.5)
    assert sum(c5.committed) < sum(c.committed) <= sum(c0.committed)
    # FeederPlan input + JSON round-trip
    plan = FeederPlan(feeder_id=f.feeder_id, t0=T0, policy="sentinel", g_kw=100.0, schedule=sched,
                      export_curve=[0.0] * DAY_STEPS, slack=[0.0] * DAY_STEPS)
    cp = commit(plan, None, sc, haircut=0.05)
    np.testing.assert_allclose(cp.committed, c.committed)
    assert Commitment.from_dict(json.loads(json.dumps(cp.to_dict()))).committed == cp.committed
    # empty pool -> zero commitment
    assert sum(commit(f.feeder_id, {}, sc).committed) == 0


def test_p10_hit_rate_on_same_distribution(feeder_day):
    f, sc, sched, c = _commit_setup(feeder_day, seed=0)
    _, pv, ld = feeder_day
    actual = make_scenarios(pv, ld, n=400, seed=12345, pv_cap={s["site_id"]: s["kwp_true"] for s in f.sites})
    pv_a = {s: actual.pv[:, j] for j, s in enumerate(actual.site_ids)}
    ld_a = {s: actual.load[:, j] for j, s in enumerate(actual.site_ids)}
    r = realised_deliverable(sched, pv_a, ld_a, f.batteries, f.soc0)
    assert r.shape == (400, DAY_STEPS)
    st = hit_stats(committed=np.asarray(c.committed), realised=r)
    assert st["p10_hit_rate"] >= 0.9, st
    assert st["n_steps"] > 0 and st["p10_miss_kw"] >= 0
    # realise() on one actual day fills `realised`, and Commitment.hit_rate agrees with hit_stats
    c1 = realise(c, sched, {s: v[0] for s, v in pv_a.items()}, {s: v[0] for s, v in ld_a.items()},
                 f.batteries, f.soc0)
    assert len(c1.realised) == DAY_STEPS
    assert c1.hit_rate() == pytest.approx(hit_stats(c1)["p10_hit_rate"])


def _deliverable_ref(sched, soc0, specs, pv, load):
    """The pre-#77 _deliverable: numpy rollout / rollout_baseline directly."""
    _, _, x_plan = rollout(sched, soc0, battery_params(specs, respect_reserve=True), pv, load)
    _, _, x_base = rollout_baseline(soc0, battery_params(specs, respect_reserve=False), pv, load)
    return (x_base - x_plan).sum(axis=-2)


@pytest.mark.parametrize("mode", ["python", "auto"])
def test_deliverable_kernel_matches_numpy_rollout(feeder_day, monkeypatch, mode):
    monkeypatch.setenv("WATTSENTRY_KERNELS", mode)
    f, pv, ld = feeder_day
    cap = {s["site_id"]: s["kwp_true"] for s in f.sites}
    sc = make_scenarios(pv, ld, n=64, seed=7, pv_cap=cap, batteries=f.batteries, soc0=f.soc0)
    sched_map = rule_schedule(sc.site_ids, sc.pv_p50, sc.load_p50, f.batteries, f.soc0, T0)
    ids = list(sc.site_ids)
    sched = np.array([sched_map[s] for s in ids], dtype=float)
    specs = [f.batteries.get(s) for s in ids]
    specs[0] = None  # a site without a battery
    specs[1] = BatterySpec(capacity_kwh=0.0, p_max_kw=0.0)  # and a zero-capacity one
    s0 = np.array([float(f.soc0.get(s, 0.5)) for s in ids])
    rng = np.random.default_rng(3)
    wild = rng.uniform(-8, 8, sched.shape)  # infeasible requests exercise every clip
    for req in (sched, wild):
        for pv_x, ld_x in ((sc.pv, sc.load), (sc.pv_p50, sc.load_p50), (sc.pv[:5], sc.load_p50)):
            got = commit_mod._deliverable(req, s0, specs, pv_x, ld_x)
            ref = _deliverable_ref(req, s0, specs, pv_x, ld_x)
            assert got.shape == ref.shape == pv_x.shape[:-2] + (DAY_STEPS,)
            assert np.array_equal(got, ref)
    # per-scenario requests and soc0 (the general (N, S, T) path)
    req3 = rng.uniform(-6, 6, sc.pv[:5].shape)
    s03 = rng.uniform(0.1, 0.9, (5, len(ids)))
    assert np.array_equal(commit_mod._deliverable(req3, s03, specs, sc.pv[:5], sc.load[:5]),
                          _deliverable_ref(req3, s03, specs, sc.pv[:5], sc.load[:5]))


# --- end to end ----------------------------------------------------------------------------------------------
def test_cli_end_to_end(capsys):
    from wattsentry.eval.uncertainty.__main__ import main

    t = time.perf_counter()
    assert main(["--feeders", "2", "--sites", "10", "--cal-days", "12", "--test-days", "3",
                 "--actual-draws", "20"]) == 0
    out = capsys.readouterr().out
    metrics = dict(line.split(" ", 1)[1].split("=") for line in out.splitlines() if line.startswith("LAB_METRIC"))
    for k in ("coverage_10_90", "p10_hit_rate", "p10_miss_kw"):
        assert k in metrics
    assert 0.7 <= float(metrics["coverage_10_90"]) <= 0.9
    assert float(metrics["p10_hit_rate"]) >= 0.85
    assert time.perf_counter() - t < 30


def test_run_metrics_default_scale():
    m = run()["metrics"]  # the CLI's default scale (8 feeders x 10 sites, 30 cal + 10 test days), seed 0
    assert 0.75 <= m["coverage_10_90"] <= 0.85
    assert m["coverage_10_90_raw"] < 0.7
    assert m["p10_hit_rate"] >= 0.9
