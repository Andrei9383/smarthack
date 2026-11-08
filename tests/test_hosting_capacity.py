"""G1 hosting capacity (card #159): search, synthetic homes, replay physics vs the loop's rules, end to end."""

from __future__ import annotations

import json

import numpy as np
import pytest

from wattsentry.contracts import BatterySpec, self_consumption_step
from wattsentry.eval import hosting_capacity as hc
from wattsentry.opt.policies import threshold_step

T = 96


# ------------------------------------------------------------------------------------------- search
def test_search_finds_largest_feasible_n_per_tol_and_shares_evaluations():
    calls = []

    def minutes(n):
        calls.append(n)
        return 0.0 if n <= 37 else 15.0 * (n - 37)  # 30 min/week reached at n = 39

    c = hc.capacity_search(minutes, (0, 30), n_max=300)
    assert c.capacity == {0: 37, 30: 39}
    assert c.censored == {0: False, 30: False}
    assert c.monotone
    assert len(calls) == len(set(calls))  # cached: no n evaluated twice
    assert len(calls) <= 16


def test_search_zero_when_already_over_and_censored_when_never_over():
    assert hc.capacity_search(lambda n: 45.0, (0, 30)).capacity == {0: 0, 30: 0}
    c = hc.capacity_search(lambda n: 0.0, (0, 30), n_max=50)
    assert c.capacity == {0: 50, 30: 50} and c.censored == {0: True, 30: True}


def test_search_reports_non_monotone_curves():
    vals = {4: 20.0, 5: 10.0}  # the tol-30 search lands on 5 after seeing 4 > 5

    c = hc.capacity_search(lambda n: 0.0 if n <= 3 else vals.get(n, 100.0), (0, 30), n_max=20)
    assert c.capacity == {0: 3, 30: 5}
    assert not c.monotone


# ------------------------------------------------------------------------------------------- toy feeder
def _toy(days=2, n=4, g=9.0, seed=1):
    rng = np.random.default_rng(seed)
    t = np.arange(T)
    shape = np.clip(np.sin((t - 24) / 48 * np.pi), 0, None)  # 06:00-18:00
    kwp = np.array([4.0, 6.0, 5.0, 3.0])[:n]
    pv = np.stack([[k * shape * (0.8 + 0.2 * rng.random()) for _ in range(days)] for k in kwp])
    load = np.stack([[0.4 + 0.3 * rng.random(T) for _ in range(days)] for _ in kwp])
    q = lambda a: np.stack([0.8 * a, a, 1.2 * a], axis=2)  # noqa: E731 (S, D, 3, T)
    bats = [BatterySpec(10.0, 4.0, comfort_reserve_kwh=2.0), None, BatterySpec(5.0, 2.5, comfort_reserve_kwh=1.0),
            None][:n]
    ids = [f"S{i}" for i in range(n)]
    st = [{s: hc._eligible(s) for s in ids} for _ in range(days)]
    return hc.FeederWeek("FX", g, [f"2025-06-0{2 + d}T21:00:00Z" for d in range(days)], np.full((days, T), 0.6),
                         ids, bats, kwp, pv, load, q(pv), q(load), st, np.ones(n, bool),
                         meta={"src_feeder_id": "FX", "g_max_kw": g, "g_conservative_kw": g, "tau": 1.0})


def _reference(fw, step):
    """Per-site per-step loop physics: batteries start at e_min and carry energy across days."""
    S, D, _ = fw.pv.shape
    net = np.zeros(D * T)
    for i, b in enumerate(fw.batteries):
        e = b.e_min if b else 0.0
        for d in range(D):
            for k in range(T):
                pv, ld = fw.pv[i, d, k], fw.load[i, d, k]
                if b is None:
                    g = pv - ld
                else:
                    _, e, g = step(b, e, pv, ld, i, d)
                net[d * T + k] += g
    return net


def test_rule_replays_match_the_loop_rules_step_by_step():
    fw = _toy()
    r = hc.replay_week(fw, "self_consumption", keep_net=True)
    ref = _reference(fw, lambda b, e, pv, ld, i, d: self_consumption_step(b, e, pv, ld))
    np.testing.assert_allclose(r.net_kw, ref, atol=1e-9)
    assert r.minutes_over == 15.0 * np.sum(ref > fw.g_kw + 1e-6)

    th = hc.thresholds(fw, 0.5)
    r = hc.replay_week(fw, "threshold", alpha=0.5, keep_net=True)
    ref = _reference(fw, lambda b, e, pv, ld, i, d: threshold_step(b, e, pv, ld, th[i, d]))
    np.testing.assert_allclose(r.net_kw, ref, atol=1e-9)


def test_synthetic_homes_are_nested_scaled_and_follow_the_variant():
    fw = _toy()
    fleet = hc.Fleet(np.array([3.0, 5.0, 8.0]), [BatterySpec(7.5, 3.0, comfort_reserve_kwh=1.5)], 0.5)
    plan = hc.plan_synthetic(fw, fleet, n_max=20, seed=0)
    a, b = hc.with_homes(fw, plan, 5), hc.with_homes(fw, plan, 9)
    assert a.site_ids == b.site_ids[:a.n_sites]
    np.testing.assert_array_equal(a.pv, b.pv[:a.n_sites])
    assert a.n_existing == fw.n_sites and int(a.synthetic.sum()) == 5
    k = 0  # new home 0: donor PV scaled to its kWp, load = its load donor's
    np.testing.assert_allclose(a.pv[4 + k], fw.pv[plan.pv_donor[k]] * plan.kwp[k] / fw.kwp[plan.pv_donor[k]])
    np.testing.assert_array_equal(a.load[4 + k], fw.load[plan.load_donor[k]])
    np.testing.assert_allclose(a.pv_q[4 + k, :, 1], a.pv[4 + k])  # toy q50 = realised
    assert set(a.kwp[4:]) <= {3.0, 5.0, 8.0}
    nb = hc.with_homes(fw, plan, 9, batteries=False)
    assert all(x is None for x in nb.batteries[4:]) and nb.batteries[:4] == fw.batteries
    assert any(x is not None for x in b.batteries[4:])
    assert all(a.statuses[d][s].in_pool for d in range(fw.n_days) for s in a.site_ids[4:])
    assert hc.with_homes(fw, plan, 0) is fw


def test_kernel_backends_agree_on_the_week(monkeypatch):
    from wattsentry import kernels

    fw = _toy()
    monkeypatch.setenv("WATTSENTRY_KERNELS", "python")
    py = hc.replay_week(fw, "self_consumption", keep_net=True).net_kw
    if not kernels.rust_available():
        pytest.skip("wattsentry._core not built")
    monkeypatch.setenv("WATTSENTRY_KERNELS", "rust")
    np.testing.assert_allclose(hc.replay_week(fw, "self_consumption", keep_net=True).net_kw, py, atol=1e-9)


def test_sentinel_lp_cuts_minutes_over_on_the_toy_feeder():
    fw = _toy(g=12.0)  # the batteries can hold this limit; far below it the LP spreads the overshoot (fewer kWh)
    base = hc.replay_week(fw, "self_consumption")
    sent = hc.replay_week(fw, "sentinel")
    assert base.minutes_over > 0
    assert sent.minutes_over < base.minutes_over
    assert sent.kwh_over < base.kwh_over and sent.peak_export_kw < base.peak_export_kw
    assert sum(sent.plan_status.values()) == fw.n_days


def test_sentinel_folds_broken_homes_out_of_the_pool():
    from wattsentry.contracts import AssetStatus

    fw = _toy()
    fw.statuses[0]["S0"] = AssetStatus("S0", "excluded", 0.0, "battery_fault", battery_ok=False)
    seen = {}

    def planner(spec, elig, rest, fc, g, prices, *, statuses, t0):
        seen[t0] = [s.site_id for s in elig]
        from wattsentry.opt.policies import self_consumption_plan

        return self_consumption_plan(spec, elig, rest, fc, g, statuses=statuses, t0=t0)

    hc.replay_week(fw, "sentinel", planner=planner)
    assert seen[fw.t0s[0]] == ["S2"] and seen[fw.t0s[1]] == ["S0", "S2"]


def test_run_job_and_assemble_schema(tmp_path):
    fw = _toy(g=13.0)
    fleet = hc.Fleet(np.array([4.0, 6.0]), [BatterySpec(10.0, 4.0, comfort_reserve_kwh=2.0)], 0.5)
    plan = hc.plan_synthetic(fw, fleet, n_max=12)
    res = [hc.run_job(hc.Job(fw, plan, p, v, (0, 30), 12)) for p in hc.POLICIES for v in hc.VARIANTS]
    doc = hc.assemble([fw], res, run_id="t", n_max=12)
    f = doc["feeders"][0]
    assert doc["schema"] == "hosting_capacity/v1" and doc["label"] == "synthetic RO"
    for v in hc.VARIANTS:
        for t in ("0", "30"):
            caps = f["capacity"][v][t]
            assert set(caps) == set(hc.POLICIES)
            assert f["gain"][v][t]["vs_self_consumption"] == caps["sentinel"] - caps["self_consumption"]
            assert caps["sentinel"] >= caps["self_consumption"]
            assert f["sentinel_ge_baseline"][v][t]
            assert caps["self_consumption"] <= f["capacity"][v]["30"]["self_consumption"]
    assert doc["total"]["capacity"]["batteries"]["30"]["sentinel"] == f["capacity"]["batteries"]["30"]["sentinel"]
    jp, cp = hc.write_outputs(doc, tmp_path)
    assert json.loads(jp.read_text())["feeders"][0]["feeder_id"] == "FX"
    assert len(cp.read_text().strip().splitlines()) == 1 + 2 * 2 * 3
    assert "total" in hc.summary_table(doc)


# ------------------------------------------------------------------------------------------- end to end (mock world)
def test_end_to_end_on_the_mock_world(tmp_path, monkeypatch):
    from wattsentry.agent.loop import default_components
    from wattsentry.agent.world import MockWorld

    w = MockWorld(days=1, scale=0.4, with_wind=False)
    comps = default_components(w)
    fws = hc.extract_feeders(w, comps, ["F7"])
    fw = fws[0]
    assert fw.n_sites == len(w.feeders["F7"].site_ids) and fw.pv.shape == (fw.n_sites, 1, T)
    assert not fw.healthy[fw.site_ids.index(w.story_site)]  # faulty homes are never donors
    assert fw.meta["verdicts"]["pv"] == {"use": fw.n_sites}  # every read went through `validated`
    base0 = hc.replay_week(fw, "self_consumption").minutes_over
    plan = hc.plan_synthetic(fw, hc.Fleet.from_world(w), n_max=16)
    res = [hc.run_job(hc.Job(fw, plan, p, "batteries", (0, 30), 16)) for p in ("self_consumption", "sentinel")]
    doc = hc.assemble(fws, res, run_id="mock", policies=("self_consumption", "sentinel"), variants=("batteries",),
                      n_max=16)
    f = doc["feeders"][0]
    assert f["minutes_over_now"]["batteries"]["self_consumption"] == base0
    assert doc["sentinel_below_baseline"] == 0


def test_judge_verdicts_flow_through_the_c1_adapter():
    adapter = pytest.importorskip("wattsentry.judge.adapter")  # C1 #157; until it merges reads are passthrough
    from wattsentry.agent.loop import default_components
    from wattsentry.agent.world import MockWorld

    w = MockWorld(days=1, scale=0.4, with_wind=False)
    comps = default_components(w)
    base = hc.extract_feeders(w, comps, ["F7"])[0]
    s0, s1 = base.site_ids[:2]
    day = adapter.day_key(hc._local_day(w.day_t0(0)))
    pv = str(getattr(comps.predictors["pv"], "model", "pvnet"))
    adapter.set_default_judge(adapter.MemoryVerdicts({
        (s0, pv, day): {"verdict": "dont_use"}, (s1, pv, day): {"verdict": "widen", "band_scale": 2.0},
        ("F7", "gridtwin", day): {"verdict": "dont_use"}}))
    try:
        fw = hc.extract_feeders(w, comps, ["F7"])[0]
    finally:
        adapter.set_default_judge(None)
    assert fw.statuses[0][s0].status == "human_review" and not fw.statuses[0][s0].in_pool
    np.testing.assert_allclose(fw.pv_q[1, 0, 2] - fw.pv_q[1, 0, 1], 2 * (base.pv_q[1, 0, 2] - base.pv_q[1, 0, 1]))
    assert fw.g_kw == pytest.approx(w.feeders["F7"].spec.g_conservative_kw)
    assert fw.meta["verdicts"]["pv"]["dont_use"] == 1 and fw.meta["verdicts"]["pv"]["unknown"] == 1


def test_process_pool_matches_serial_and_does_not_hang():
    # workers are spawned (not forked) so a parent that already runs Rust/OpenMP threads cannot deadlock them
    fw = _toy(g=13.0)
    fleet = hc.Fleet(np.array([4.0, 6.0]), [BatterySpec(10.0, 4.0, comfort_reserve_kwh=2.0)], 0.5)
    plan = hc.plan_synthetic(fw, fleet, n_max=12)
    jobs = [hc.Job(fw, plan, p, v, (0, 30), 12) for p in ("self_consumption", "threshold") for v in hc.VARIANTS]
    key = lambda r: (r["policy"], r["variant"])  # noqa: E731
    serial = sorted(hc.run_jobs(jobs, workers=1), key=key)
    pooled = sorted(hc.run_jobs(jobs, workers=2), key=key)
    assert [(key(r), r["capacity"]) for r in pooled] == [(key(r), r["capacity"]) for r in serial]
