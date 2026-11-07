"""Card G3 #161: deferred capex in lei + GridTwin (M8) in the loop."""

from __future__ import annotations

import math
from types import SimpleNamespace

import pandas as pd
import pytest

from sentinel.agent.world import SiteDay
from sentinel.contracts.opt import BatterySpec, FeederSpec
from sentinel.eval import dso_value as dv
from sentinel.gridtwin import inloop as L


# ------------------------------------------------------------------------------------------- deferred capex
def test_deferred_value_formula():
    assert dv.deferred_value(1000.0, 0) == 0.0
    assert dv.deferred_value(1000.0, 1, r=0.07) == pytest.approx(1000 * (1 - 1 / 1.07))
    assert dv.deferred_value(1000.0, 10, r=0.07) == pytest.approx(1000 * (1 - 1.07 ** -10))
    assert dv.deferred_value(1000.0, 1e6) == pytest.approx(1000.0)  # deferred forever -> the whole cost
    assert dv.DISCOUNT_RATE == 0.07


def test_years_deferred():
    assert dv.years_deferred(40, 0, 0) == 0.0
    assert dv.years_deferred(40, 5, 3) == 0.0  # Sentinel below baseline never counts as value
    y = dv.years_deferred(40, 0, 10, growth=0.15)
    assert y == pytest.approx(math.log(50 / 40) / math.log(1.15))
    assert dv.years_deferred(40, 0, 20) > y  # more capacity -> longer deferral


def test_deferred_capex_labels_stub_and_placeholder():
    hc = dv.hosting_capacity()
    assert all("STUB" in h["source"] for h in hc.values())
    cap = dv.deferred_capex(hc)
    assert cap["assumptions"]["cost_is_placeholder"] and "PLACEHOLDER" in cap["assumptions"]["cost_note"]
    assert "STUB" in cap["label"] and cap["label"].startswith("proxy")
    assert cap["deferred_capex_lei"] == sum(r["deferred_capex_lei"] for r in cap["feeders"]) > 0
    lo, hi = cap["deferred_capex_lei_deo_basis"]
    assert 0 < lo < hi and "other-operator" in cap["label_deo"]
    # measured hosting capacity (G1) drops the STUB label
    real = dv.deferred_capex({"F7": {"n_homes": 30, "hc_baseline": 2, "hc_sentinel": 12, "source": "G1 #159"}})
    assert "STUB" not in real["label"] and real["hosting_capacity_gain"] == 10


def test_deo_order_of_magnitude():
    lo, hi = dv.deo_cost_per_prosumer_lei()
    assert lo == pytest.approx(400e6 / 150_000 * dv.EUR_RON) and hi == pytest.approx(400e6 / 100_000 * dv.EUR_RON)


# ------------------------------------------------------------------------------------------- recovery
def _twin_frames():
    truth = pd.DataFrame({"site_id": list("abcdef"), "feeder_id": ["F1", "F1", "F2", "F2", "F1", "F2"],
                          "phase": ["A", "B", "C", "A", "B", "C"]})
    res = pd.DataFrame({
        "site_id": list("abcdef"),
        "gis_feeder_id": ["F1", "F2", "F2", "F2", "F1", "F1"],  # b, f: wrong feeder
        "gis_phase": ["B", "B", "A", "A", "B", "C"],  # a, c: wrong phase (feeder right)
        "likely_feeder_id": ["F1", "F1", "F2", "F2", "F1", "F1"],
        "likely_phase": ["A", "B", "B", "A", "C", "C"],
        "feeder_flag": [False, True, False, False, False, False],  # b recovered, f missed
        "phase_flag": [True, False, True, False, True, False],  # a recovered, c wrong phase proposed, e false flag
    })
    return res, truth


def test_recovery_pools_feeder_and_phase_errors():
    res, truth = _twin_frames()
    m = L.recovery(res, truth)
    assert m["n_wrong_feeder"] == 2 and m["n_wrong_phase"] == 2 and m["n_corrupted"] == 4
    assert m["recovered_feeder"] == 1 and m["recovered_phase"] == 1
    assert m["gis_recovery_rate"] == 0.5
    assert m["n_flags"] == 4 and m["gis_flag_precision"] == 0.5 and m["false_flags_on_clean"] == 1


def test_recovery_matches_gridtwin_score_on_its_parts():
    from sentinel.gridtwin.run import score

    res, truth = _twin_frames()
    res = res.assign(tau=1.0)
    s = score(SimpleNamespace(sites=res, feeders=pd.DataFrame({"tau": [1.0]})), truth)
    m = L.recovery(res, truth)
    pooled = (s["feeder_recovery"] * s["n_wrong_feeder"] + s["phase_recovery"] * s["n_wrong_phase"]) / \
        (s["n_wrong_feeder"] + s["n_wrong_phase"])
    assert m["gis_recovery_rate"] == pytest.approx(pooled, abs=1e-3)


# ------------------------------------------------------------------------------------------- tau / over limit
def test_over_limit_counts_against_given_limits():
    rows = [{"feeder_id": "F1", "ts": 86400 * 3 + 900 * k, "export_kw": x, "import_kw": 0.0}
            for k, x in enumerate([5, 12, 15, 9])] + [{"feeder_id": "F2", "ts": 0, "export_kw": 0, "import_kw": 3}]
    o = L.over_limit(rows, {"F1": 10.0, "F2": 1.0})
    assert o.minutes == 30 and o.kwh == pytest.approx((2 + 5) * 0.25) and o.peak_overshoot_kw == 5
    assert o.feeder_days_over == 1 and o.per_feeder == {"F1": 30.0, "F2": 0.0}
    assert L.over_limit(rows, {"F1": 8.0}).minutes == 45  # tighter limit, more minutes


def test_validated_reads_pass_through_without_judge():
    w = SimpleNamespace(feeders={"F1": SimpleNamespace(spec=FeederSpec("F1", 100, 80, 0.6))})
    assert L.validated_tau(w, "2025-06-02") == {"F1": pytest.approx(0.6)}
    flags = [{"site_id": "1", "recorded_feeder": "F1", "suggested_feeder": "F2", "score": 0.9},
             {"site_id": "2", "recorded_feeder": "F1", "suggested_feeder": "F1", "score": 0.9}]  # phase-only
    mv = L.validated_moves(flags, "2025-06-02")
    assert [m["site_id"] for m in mv] == ["1"] and mv[0]["verdict"] == "use"


def test_dont_use_verdict_blocks_move_and_zeroes_tau():
    adapter = pytest.importorskip("sentinel.judge.adapter")
    j = adapter.MemoryVerdicts({("1", "gridtwin", "2025-06-02"): {"verdict": "dont_use"},
                                ("F1", "gridtwin", "2025-06-02"): {"verdict": "dont_use"}})
    adapter.set_default_judge(j)
    try:
        flags = [{"site_id": "1", "recorded_feeder": "F1", "suggested_feeder": "F2", "score": 0.9}]
        assert L.validated_moves(flags, "2025-06-02") == []
        w = SimpleNamespace(feeders={"F1": SimpleNamespace(spec=FeederSpec("F1", 100, 80, 0.6))})
        assert L.validated_tau(w, "2025-06-02") == {"F1": 0.0}
    finally:
        adapter.set_default_judge(None)


# ------------------------------------------------------------------------------------------- moved home
def test_members_moves_and_case_pick():
    sites = {"1": SimpleNamespace(row={"feeder_id": "F3", "gis_feeder_id": "F7"}),
             "2": SimpleNamespace(row={"feeder_id": "F7", "gis_feeder_id": "F7"}),
             "3": SimpleNamespace(row={"feeder_id": "F7", "gis_feeder_id": "F3"})}
    w = SimpleNamespace(feeders={"F7": None, "F3": None}, sites=sites)
    gis = L.gis_members(w)
    assert sorted(gis["F7"]) == ["1", "2"] and gis["F3"] == ["3"]
    flags = [{"site_id": "1", "recorded_feeder": "F7", "suggested_feeder": "F3", "score": 0.8},
             {"site_id": "3", "recorded_feeder": "F3", "suggested_feeder": "F7", "score": 0.99},
             {"site_id": "2", "recorded_feeder": "F7", "suggested_feeder": "F3", "score": 1.0}]  # wrong: not shown
    truth = {"1": "F3", "2": "F7", "3": "F7"}
    c = L.pick_case(flags, truth, "F7")
    assert c["site_id"] == "1" and c["case"] == "recorded_feeder=F7"  # prefers a home wrongly ON the hero feeder
    assert L.pick_case(flags[1:], truth, "F7")["site_id"] == "3"
    fixed = L.apply_moves(gis, [flags[0]])
    assert fixed["F7"] == ["2"] and sorted(fixed["F3"]) == ["1", "3"]


def test_play_feeder_schedule_vs_rule():
    b = BatterySpec(10.0, 5.0, 1.0, 1.0, 0.0, 1.0, 0.0)
    pv = [0.0] * 40 + [8.0] * 16 + [0.0] * 40
    sd = SiteDay(pv, [1.0] * 96, [0.0] * 96, [0] * 96, [230.0] * 96)
    w = SimpleNamespace(feeders={"F7": SimpleNamespace(site_ids=["1", "2"])},
                        sites={"1": SimpleNamespace(battery=b), "2": SimpleNamespace(battery=None)},
                        site_day=lambda s, d: sd)
    rule = L.play_feeder(w, "F7", 0, {}, limit_kw=12.0)  # self-consumption: battery full after 2 h, then 14 kW out
    spread = [0.0] * 40 + [2.5] * 16 + [0.0] * 40  # charge across the whole peak instead
    plan = L.play_feeder(w, "F7", 0, {"1": spread}, limit_kw=12.0)
    assert rule["minutes_over"] == 120 and rule["kwh_over"] == pytest.approx(4.0)
    assert plan["minutes_over"] == 0 and plan["peak_kw"] == pytest.approx(11.5)


def test_moved_home_case_on_mock_world():
    from sentinel.agent.loop import default_components
    from sentinel.agent.world import MockWorld

    w = MockWorld(days=1, scale=0.4, with_wind=False)
    hero, other = "F7", next(f for f in w.feeders if f != "F7")
    sid = next(s for s in w.feeders[other].site_ids if w.sites[s].battery is not None)
    w.sites[sid].row["gis_feeder_id"] = hero  # GIS puts it on F7; it is physically on `other`
    case = {"site_id": sid, "recorded_feeder": hero, "suggested_feeder": other, "score": 0.9, "case": "test"}
    r = L.moved_home_case(w, default_components(w, force_fallback=True), d=0, hero=hero, case=case)
    assert r["found"] and r["true_feeder"] == other and r["pool_gis"] == r["pool_m8"] + 1
    assert r["plan_diff"][0]["site_id"] == sid and r["plan_diff"][0]["moved"]
    assert r["plan_diff"][0]["midday_charge_kwh_m8"] is None  # off F7's plan once M8 moves it
    assert set(r["physical_gis"]) == {"minutes_over", "kwh_over", "peak_kw", "limit_kw"}
