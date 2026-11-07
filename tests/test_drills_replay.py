"""#78: replay drills corrupt only what the agent observes (never the physics), never crash the agent, are
deterministic, and report KPI deltas vs the clean run (sentinel.agent.drills)."""

from __future__ import annotations

import contextlib
import io
import json

import pytest

from sentinel.agent import drills as D
from sentinel.agent.loop import LoopConfig, default_components, run_policy
from sentinel.agent.world import MockWorld

DAYS, SCALE = 2, 0.5
ARGS = ["--world", "mock", "--days", str(DAYS), "--scale", str(SCALE), "--fallback", "all", "--seed", "0"]


@pytest.fixture(scope="module")
def world():
    return MockWorld(days=DAYS, scale=SCALE)


@pytest.fixture(scope="module")
def cli(tmp_path_factory):
    """`python -m sentinel.agent.drills --drill all` on a small mock world, built-in components (deterministic)."""
    d = tmp_path_factory.mktemp("drills")
    buf = io.StringIO()
    with contextlib.redirect_stdout(buf):
        code = D.main(["--drill", "all", *ARGS, "--report", str(d / "drills_report.json")])
    out = buf.getvalue()
    metrics = dict(line[len("LAB_METRIC "):].split("=", 1) for line in out.splitlines()
                   if line.startswith("LAB_METRIC "))
    return code, metrics, json.loads((d / "drills_report.json").read_text())


def _implemented():
    return [n for n in D.DRILL_NAMES if D.drill_available(n)[0]]


# --- the batch: never crashes, skips what waits on the hook, prints metrics + deltas ---------------------------
def test_cli_runs_every_drill(cli):
    code, metrics, rep = cli
    assert code == 0 and rep["clean"]["status"] == "ok"
    assert set(rep["drills"]) == set(D.DRILL_NAMES)
    for name, r in rep["drills"].items():
        if D.drill_available(name)[0]:
            assert r["status"] == "ok", (name, r.get("error"), r.get("traceback"))
            assert metrics[f"drill_{name}_ok"] == "1"
            for k in ("human_review_share", "p10_hit_rate", "comfort_breaches", "minutes_over_limit", "clip_rate"):
                assert f"drill_{name}_{k}" in metrics and f"drill_{name}_delta_{k}" in metrics, (name, k)
                assert r["delta"][k] == pytest.approx(r["kpis"][k] - rep["clean"]["kpis"][k])
        else:
            assert r["status"] == "skipped" and "observed_site_day" in r["message"], name
            assert metrics[f"drill_{name}_skipped"] == "1" and f"drill_{name}_ok" not in metrics
    assert metrics["drills_failed"] == "0"
    assert int(metrics["drills_ok"]) == len(_implemented()) >= 4  # the presence drills need no hook


def test_no_increase_in_comfort_breaches(cli):
    _, _, rep = cli
    for name, r in rep["drills"].items():
        if r["status"] == "ok":
            assert r["kpis"]["comfort_breaches"] <= rep["clean"]["kpis"]["comfort_breaches"], name


def test_commitment_accounting(cli):
    """The loop's own check: P10 hit rate over committed steps + the committed vs realised energy it logs."""
    _, _, rep = cli
    c = rep["clean"]["kpis"]
    assert c["commit_steps"] > 0 and c["committed_kwh"] > 0  # levels depend on the components (built-ins here)
    for r in [rep["clean"], *(r for r in rep["drills"].values() if r["status"] == "ok")]:
        k = r["kpis"]
        assert 0.0 <= k["p10_hit_rate"] <= 1.0
        assert -1e-9 <= k["shortfall_kwh"] <= k["committed_kwh"] + 1e-9
        assert 0.0 <= k["human_review_share"] <= k["not_sure_share"] <= 1.0
    g = rep["drills"]["gaps_10"]["kpis"]  # isolated 15-min gaps never make a site stale (> 60 min)
    assert abs(g["p10_hit_rate"] - c["p10_hit_rate"]) <= 0.05


def test_p10_hit_rate_holds_under_every_drill(cli):
    _, _, rep = cli
    for name, r in rep["drills"].items():
        if r["status"] == "ok":
            assert r["kpis"]["p10_hit_rate"] >= rep["clean"]["kpis"]["p10_hit_rate"] - 0.1, name


def test_presence_drills_raise_not_sure_share(cli):
    _, _, rep = cli
    clean = rep["clean"]["kpis"]["not_sure_share"]
    for name in ("gaps_40", "missing_day", "telemetry_dropout"):
        r = rep["drills"][name]
        assert r["kpis"]["not_sure_share"] > clean, name
        assert r["observation"]["missing_share"] > r["observation"]["missing_share_clean"]


# --- observation vs truth ----------------------------------------------------------------------------------------
@pytest.mark.parametrize("name", D.DRILL_NAMES)
def test_drill_changes_observation_not_truth(world, name):
    dw = D.drilled(world, name, seed=0, via_hook=False)
    obs_changed = age_changed = False
    for s in world.sites:
        for d in range(world.days):
            t, x, o = world.site_day(s, d), dw.site_day(s, d), dw.observed_site_day(s, d)
            assert x.pv_kw == t.pv_kw and x.load_kw == t.load_kw  # physics inputs identical
            assert x.inv_status == t.inv_status and x.inv_voltage_v == t.inv_voltage_v
            assert x.telemetry_age_s == o.telemetry_age_s  # what the loop perceives today
            age_changed |= x.telemetry_age_s != t.telemetry_age_s
            got = [a == 0 for a in o.telemetry_age_s]
            obs_changed |= any(g and (p != q) for g, p, q in zip(got, o.pv_kw, t.pv_kw)) or age_changed
            assert all(a >= b for a, b in zip(o.telemetry_age_s, t.telemetry_age_s))  # a drill only removes
    assert obs_changed, name
    assert age_changed == (name in D.PRESENCE_DRILLS), name
    hooked = D.drilled(world, name, seed=0, via_hook=True)  # with the hook, site_day is the untouched truth
    s = next(iter(world.sites))
    assert vars(hooked.site_day(s, 0)) == vars(world.site_day(s, 0))


def test_value_drills_corrupt_values_as_documented(world):
    s = world.story_site
    for d in range(world.days):
        t = world.site_day(s, d)
        kw = D.drilled(world, "units_kw").observed_site_day(s, d)
        x1000 = D.drilled(world, "units_x1000").observed_site_day(s, d)
        for k, a in enumerate(t.telemetry_age_s):
            if a == 0:
                assert kw.pv_kw[k] == pytest.approx(t.pv_kw[k] * 1e-3)
                assert x1000.load_kw[k] == pytest.approx(t.load_kw[k] * 1e3)
    fr = D.drilled(world, "frozen_sensor")
    pv = [v for d in range(world.days) for v in fr.observed_site_day(s, d).pv_kw]
    stuck = max(set(pv) - {0.0}, key=pv.count)  # telemetry_drill: one daytime reading repeated for up to 3 days
    assert pv.count(stuck) >= D.DAY_STEPS // 2 and pv[-1] == stuck


def test_observation_deterministic_and_seeded(world):
    s = sorted(world.sites)[3]
    a = D.drilled(world, "gaps_40", seed=1).observed_site_day(s, 1)
    b = D.drilled(world, "gaps_40", seed=1).observed_site_day(s, 1)
    c = D.drilled(world, "gaps_40", seed=2).observed_site_day(s, 1)
    assert vars(a) == vars(b) and a.telemetry_age_s != c.telemetry_age_s


def test_noop_drill_reproduces_clean_world(world):
    dw = D.drilled(world, "forecast_noise")  # a weather drill: nothing to change in telemetry
    for s in world.sites:
        for d in range(world.days):
            assert vars(dw.site_day(s, d)) == vars(world.site_day(s, d))
    assert dw.drill_stats["missing_share"] == dw.drill_stats["missing_share_clean"]


def test_share_drills_a_subset(world):
    dw = D.drilled(world, "missing_day", share=0.25)
    hit = set(dw._drill.sites)
    assert 0 < len(hit) < len(world.sites)
    for s in set(world.sites) - hit:
        for d in range(world.days):
            assert dw.site_day(s, d).telemetry_age_s == world.site_day(s, d).telemetry_age_s


def test_physics_identical_under_drill(world):
    """The baseline rule never looks at telemetry: same true flows -> identical results. The sentinel run perceives
    the drill (different input hashes) on the same physics."""
    dw = D.drilled(world, "gaps_40", via_hook=False)
    res = {}
    for key, w in (("clean", world), ("drill", dw)):
        comps = default_components(w, force_fallback=True)
        res[key] = {p: run_policy(w, comps, LoopConfig(policy=p, run_id=f"{key}-{p}", write_inputs=False))
                    for p in ("baseline", "sentinel")}
    b0, b1 = res["clean"]["baseline"], res["drill"]["baseline"]
    assert b0.cost_lei == b1.cost_lei and b0.feeder_days == b1.feeder_days
    assert b0.comfort_breaches == b1.comfort_breaches
    s0, s1 = res["clean"]["sentinel"], res["drill"]["sentinel"]
    assert s0.input_hashes != s1.input_hashes


def test_drilled_run_deterministic(world, tmp_path):
    k = [D.run_once(D.drilled(world, "telemetry_dropout", seed=3), tmp_path / f"r{i}.sqlite", f"det{i}",
                    seed=3, force_fallback=True)["kpis"] for i in range(2)]
    assert k[0] == k[1]


def test_value_drills_wait_for_the_hook(world, tmp_path):
    assert D.drill_available("gaps_40", hook=False) == (True, "")
    ok, why = D.drill_available("units_kw", hook=False)
    assert not ok and "World.observed_site_day" in why and "loop.run_policy" in why
    assert D.drill_available("units_kw", hook=True)[0]
    if not D.observation_hook_available():
        pytest.skip(D.HOOK_MESSAGE)
    r = D.run_once(D.drilled(world, "units_kw"), tmp_path / "u.sqlite", "units", force_fallback=True)
    assert r["kpis"]["comfort_breaches"] == 0
