"""Card #26: qualify (sites + turbines) and O&M triage, driven by the MockPredictor."""

import dataclasses
import json
from datetime import datetime, timezone
from types import SimpleNamespace

import numpy as np
import pytest

from sentinel.agent import qualify as Q
from sentinel.agent import triage as TR
from sentinel.agent.qualify import DEFAULT_THRESHOLDS, assess, load_thresholds, qualify, qualify_many, rel_width
from sentinel.agent.triage import (
    TriageConfig,
    WindInputs,
    lei_per_day,
    offline_windows,
    pzu_next24_mean,
    triage,
    triage_score,
    wind_triage_score,
)
from sentinel.contracts import (
    PAYLOAD_KEYS,
    Action,
    AssetStatus,
    CrewType,
    MockPredictor,
    PvFault,
    Status,
    WindAnomaly,
)

T0 = "2025-06-21T00:00:00Z"
TICK = "2025-06-21T15:00:00Z"
SITE_1042 = {"site_id": "S1042", "feeder_id": "F7", "kwp_true": 8.0, "lat": 44.2, "battery_kwh": 10.0,
             "battery_kw": 5.0}
PV = MockPredictor("sitehealth")
WIND = MockPredictor("wind")


@pytest.fixture(autouse=True)
def _builtin_thresholds(monkeypatch, tmp_path):
    """Tests assume the built-in defaults, whatever #27 commits to models/thresholds.json."""
    monkeypatch.setenv(Q.THRESHOLDS_ENV, str(tmp_path / "none.json"))


def turbine(i, park="WP1"):
    return {"turbine_id": f"{park}.TID{i:02d}", "park_id": park, "rated_kw": 2500.0}


def site_pred(site=SITE_1042, fault=None, p=0.9, **kw):
    """Mock M4 curves for the site, with the fault head forced to `fault` (None = healthy).

    p = P(fault) = 1 − P(healthy); the rest of the fault mass is spread over the other fault classes.
    """
    pred = PV.predict(site, t0=T0)
    classes = [f.value for f in PvFault]
    if fault is None:
        probs = {c: 0.05 / (len(classes) - 1) for c in classes}
        probs[PvFault.HEALTHY.value] = 0.95
    else:
        others = [c for c in classes if c not in (fault, PvFault.HEALTHY.value)]
        probs = {c: 0.02 * p / len(others) for c in others}
        probs[fault] = 0.98 * p
        probs[PvFault.HEALTHY.value] = 1 - p
    return dataclasses.replace(pred, fault_probs=probs, **kw)


def wind_pred(t, cls, score=0.9, day_t0=T0, **kw):
    pred = WIND.predict(t, t0=day_t0)
    return dataclasses.replace(pred, anomaly_class=cls, anomaly_score=score, **kw)


def wind_q50_nonzero(t):
    """A day where the mock park has wind, so turbine losses are non-trivial."""
    for d in range(1, 29):
        t0 = f"2025-06-{d:02d}T00:00:00Z"
        if sum(WIND.predict(t, t0=t0).q50) > 1000:
            return t0
    raise AssertionError("no windy mock day")


def check_action(a: Action):
    assert all(k in a.payload for k in PAYLOAD_KEYS[a.type])
    assert a.audit.model_version and len(a.audit.input_hash) == 16
    assert a.audit.status in {s.value for s in Status}
    Action.from_json(a.to_json())  # round-trips through the contract


# --- sites ----------------------------------------------------------------------------------------------------
def test_site_1042_string_outage_derated_with_work_order():
    pred = site_pred(fault=PvFault.STRING_OUTAGE.value, p=0.92)
    actual = np.array(pred.q50) * 0.7  # one of three strings out: ~30% below expected
    actual[64:] = np.nan  # observed only up to the tick (16:00 UTC)
    a = assess(SITE_1042, pred, telemetry_age=5, actual=actual)
    s = a.status
    assert s.status == "derated" and s.reason == "pv_fault:string_outage"
    assert s.in_pool and s.battery_ok  # battery telemetry healthy -> battery stays in the pool
    assert s.headroom_factor == pytest.approx(0.7, abs=0.01)
    assert a.measured and a.crew_type == CrewType.RESIDENTIAL.value
    assert "String outage" in s.detail

    res = triage([a], TICK, config=TriageConfig(price_lei_mwh=500.0))
    assert len(res.work_orders) == 1 and not res.alerts and not res.human_reviews
    wo = res.work_orders[0]
    check_action(wo)
    p = wo.payload
    exp_day = sum(pred.q50) * 0.25
    assert p["asset_id"] == "S1042" and p["fault"] == "string_outage" and p["crew_type"] == "residential_installer"
    assert p["lost_kwh_day"] == pytest.approx(0.3 * exp_day, rel=0.02)
    assert p["lost_lei_day"] == pytest.approx(p["lost_kwh_day"] / 1000 * 500.0, rel=1e-3)
    assert p["mw_at_risk"] == pytest.approx(0.3 * max(pred.q50) / 1000, rel=0.02)
    assert p["mw_at_risk"] > 0 and p["lost_lei_day"] > 0 and p["rank"] == 1 and p["feeder_id"] == "F7"
    assert wo.audit.status == "derated" and wo.audit.threshold == DEFAULT_THRESHOLDS["site"]["pfault_confident"]
    assert wo.audit.prediction["top_fault"] == "string_outage"
    # Same tick again: the open work order is not re-issued.
    again = triage([a], "2025-06-21T15:15:00Z", state=res.state)
    assert not again.work_orders


def test_site_without_actual_uses_class_default_loss():
    pred = site_pred(fault=PvFault.SOILING.value, p=0.8)
    a = assess(SITE_1042, pred, 0)
    assert a.status.status == "derated" and not a.measured
    assert a.status.headroom_factor == pytest.approx(1 - Q.PV_DEFAULT_LOSS["soiling"])


def test_soiling_needs_its_own_higher_probability():
    """#214: P(fault) 0.65 is enough for a string outage, not for soiling (P(soiling) 0.64 < soiling_confident)."""
    assert assess(SITE_1042, site_pred(fault=PvFault.STRING_OUTAGE.value, p=0.65), 0).status.status == "derated"
    a = assess(SITE_1042, site_pred(fault=PvFault.SOILING.value, p=0.65), 0)
    assert a.status.status == "eligible" and a.status.reason == "ok"
    assert assess(SITE_1042, site_pred(fault=PvFault.SOILING.value, p=0.75), 0).status.status == "derated"


def test_site_healthy_eligible():
    s = qualify(SITE_1042, site_pred(), 10)
    assert s.status == "eligible" and s.headroom_factor == 1.0 and s.reason == "ok"


@pytest.mark.parametrize("age", [61, 240.0])
def test_stale_telemetry_excluded(age):
    a = assess(SITE_1042, site_pred(), age)
    assert a.status.status == "excluded" and a.status.reason == "stale_telemetry" and a.offline
    assert a.status.headroom_factor == 0.0 and not a.needs_work_order


def test_data_gap_and_missing_prediction_excluded():
    assert qualify(SITE_1042, site_pred(), 0, data_gap=True).reason == "data_gap"
    assert qualify(SITE_1042, None, 0).reason == "no_prediction"


def test_sensor_fault_excluded_remote_check_never_truck():
    a = assess(SITE_1042, site_pred(fault=PvFault.SENSOR_FAULT.value, p=0.85), 0)
    assert a.status.status == "excluded" and a.status.reason == "sensor_fault"
    assert a.crew_type == CrewType.REMOTE.value
    res = triage([a], TICK)
    assert [w.payload["crew_type"] for w in res.work_orders] == ["remote_check"]
    # telemetry flag path too
    assert assess(SITE_1042, site_pred(), 0, sensor_fault=True).crew_type == CrewType.REMOTE.value


def test_battery_fault_excluded_battery_not_ok():
    a = assess(SITE_1042, site_pred(fault=PvFault.BATTERY_FADE.value, p=0.9), 0)
    assert a.status.status == "excluded" and a.status.reason == "battery_fault" and not a.status.battery_ok
    assert a.lost_flex_mwh_day == pytest.approx(0.010)
    s = qualify(SITE_1042, site_pred(), 0, battery_fault=True)
    assert s.status == "excluded" and not s.battery_ok


def test_pv_fault_with_unhealthy_battery_flag_is_battery_fault():
    s = qualify(SITE_1042, site_pred(fault=PvFault.SOILING.value), 0, battery_fault=True)
    assert s.reason == "battery_fault"


def test_ood_goes_to_human_review_no_work_order():
    pred = site_pred(fault=PvFault.STRING_OUTAGE.value, ood_score=5.0)
    a = assess(SITE_1042, pred, 0)
    assert a.status.status == "human_review" and a.status.reason == "ood" and not a.status.in_pool
    res = triage([a], TICK)
    assert not res.work_orders and len(res.human_reviews) == 1
    hr = res.human_reviews[0]
    check_action(hr)
    assert hr.payload["reason"] == "ood" and hr.audit.threshold == 3.0
    # still under review next tick: not re-queued; back to normal: review closes
    assert not triage([a], "2025-06-21T15:15:00Z", state=res.state).human_reviews
    ok = assess(SITE_1042, site_pred(), 0)
    assert "S1042" not in triage([ok], "2025-06-21T15:30:00Z", state=res.state).state.open_reviews


def test_wide_interval_goes_to_human_review():
    pred = site_pred()
    q50 = np.array(pred.q50)
    wide = dataclasses.replace(pred, q10=list(q50 * 0.4), q90=list(q50 * 1.6))  # rel width 1.2
    assert rel_width(wide) == pytest.approx(1.2, abs=1e-6)
    s = qualify(SITE_1042, wide, 0)
    assert s.status == "human_review" and s.reason == "wide_interval"


def test_ensemble_spread_on_p_fault_goes_to_human_review():
    pred = site_pred(fault=PvFault.SHADING.value, extra={"p_fault_std": 0.3})
    assert qualify(SITE_1042, pred, 0).reason == "ensemble_spread"


def test_rel_width_is_zero_at_night():
    pred = MockPredictor("pv").predict(SITE_1042, t0="2025-06-21T20:00:00Z", horizon=8)
    assert rel_width(pred) == 0.0


# --- turbines -------------------------------------------------------------------------------------------------
def test_fault_stop_excluded_with_wind_work_order():
    t = turbine(7)
    t0 = wind_q50_nonzero(t)
    pred = wind_pred(t, WindAnomaly.FAULT_STOP.value, day_t0=t0)
    actual = np.zeros(48)  # zero power for the observed half day
    a = assess(t, pred, 0, actual=actual)
    assert a.status.status == "excluded" and a.status.reason == "fault_stop"
    assert a.crew_type == CrewType.WIND.value and a.loss_frac == pytest.approx(1.0)
    res = triage([a], t0[:10] + "T12:00:00Z")
    wo = res.work_orders[0]
    check_action(wo)
    assert wo.payload["crew_type"] == "wind_technician" and wo.payload["park_id"] == "WP1"
    assert wo.payload["lost_kwh_day"] > 0 and wo.payload["mw_at_risk"] == pytest.approx(2.5)


def test_underperformance_derated():
    t = turbine(3)
    t0 = wind_q50_nonzero(t)
    pred = wind_pred(t, WindAnomaly.UNDERPERFORMANCE.value, day_t0=t0)
    a = assess(t, pred, 0, actual=np.array(pred.q50) * 0.8)
    assert a.status.status == "derated" and a.status.headroom_factor == pytest.approx(0.8, abs=0.01)
    assert a.crew_type == CrewType.WIND.value


def test_icing_excluded_with_safety_alert_no_truck():
    t = turbine(4)
    a = assess(t, wind_pred(t, WindAnomaly.ICING.value, ood_score=9.0), 0)  # safety wins over OOD
    assert a.status.status == "excluded" and a.status.reason == "icing" and a.crew_type is None
    res = triage([a], TICK)
    assert not res.work_orders and len(res.alerts) == 1
    al = res.alerts[0]
    check_action(al)
    assert al.payload["severity"] == "safety" and al.payload["code"] == "icing"
    # persists next tick: no second alert; clears, then returns: new alert
    assert not triage([a], "2025-06-21T15:15:00Z", state=res.state).alerts
    ok = assess(t, wind_pred(t, WindAnomaly.NORMAL.value, score=0.1), 0)
    st = triage([ok], "2025-06-21T15:30:00Z", state=res.state).state
    assert len(triage([a], "2025-06-21T15:45:00Z", state=st).alerts) == 1


def test_curtailment_eligible_logged_no_work_order():
    t = turbine(5)
    t0 = wind_q50_nonzero(t)
    pred = wind_pred(t, WindAnomaly.CURTAILMENT.value, day_t0=t0)
    cap = np.minimum(np.array(pred.q50), 1000.0)
    a = assess(t, pred, 0, actual=cap)
    assert a.status.status == "eligible" and a.status.reason == "curtailed" and a.crew_type is None
    res = triage([a], TICK)
    assert not res.work_orders and not res.alerts
    assert res.curtailment_log[0]["curtailed_kwh"] == pytest.approx(
        float(np.clip(np.array(pred.q50) - cap, 0, None).sum() * 0.25))


def test_park_wide_cap_is_curtailment_not_six_work_orders():
    park = [turbine(i, "WP2") for i in range(1, 7)]
    t0 = wind_q50_nonzero(park[0])
    preds, actuals = {}, {}
    for t in park:
        p = wind_pred(t, WindAnomaly.UNDERPERFORMANCE.value, day_t0=t0)  # model blames each turbine
        preds[t["turbine_id"]] = p
        actuals[t["turbine_id"]] = np.minimum(np.array(p.q50), 800.0)  # all capped at the same level
    ages = {t["turbine_id"]: 0 for t in park}
    out = qualify_many(park, preds, ages, actuals=actuals)
    assert all(a.status.status == "eligible" and a.status.reason == "curtailed_park_wide" for a in out)
    res = triage(out, TICK)
    assert not res.work_orders and len(res.curtailment_log) == 6
    # without the consensus guard the same tick would raise one work order per turbine (K permitting)
    raw = qualify_many(park, preds, ages, actuals=actuals, park_consensus=False)
    assert all(a.status.status == "derated" for a in raw)


def test_park_wide_curtailment_from_model_no_work_order():
    park = [turbine(i, "WP3") for i in range(1, 5)]
    preds = {t["turbine_id"]: wind_pred(t, WindAnomaly.CURTAILMENT.value) for t in park}
    out = qualify_many(park, preds, {t["turbine_id"]: 0 for t in park})
    assert {a.status.status for a in out} == {"eligible"}
    assert not triage(out, TICK).work_orders


def test_turbine_data_gap_is_one_offline_window():
    t = turbine(9)
    ticks = [f"2025-06-21T{h:02d}:00:00Z" for h in range(8, 16)]
    gap = {"2025-06-21T10:00:00Z", "2025-06-21T11:00:00Z", "2025-06-21T12:00:00Z", "2025-06-21T13:00:00Z"}
    state, alerts, opened, closed = None, [], [], []
    for tick in ticks:
        if tick in gap:
            a = assess(t, wind_pred(t, WindAnomaly.DATA_GAP.value), 20)
            assert a.status.status == "excluded" and a.status.reason == "data_gap" and a.offline
        else:
            a = assess(t, wind_pred(t, WindAnomaly.NORMAL.value, score=0.1), 5)
        res = triage([a], tick, state=state)
        state = res.state
        alerts += res.alerts
        opened += res.offline_opened
        closed += res.offline_closed
        assert not res.work_orders
    assert len(alerts) == 1 and alerts[0].payload["code"] == "offline"
    assert len(opened) == 1 and closed == [{"asset_id": "WP1.TID09", "asset_kind": "turbine",
                                           "start": "2025-06-21T10:00:00Z", "end": "2025-06-21T14:00:00Z"}]
    # stale SCADA counts as offline too
    s = assess(t, wind_pred(t, WindAnomaly.NORMAL.value, score=0.1), 120)
    assert s.status.reason == "stale_telemetry" and s.offline


def test_offline_windows_helper():
    recs = [("A", "t1", False), ("A", "t2", True), ("A", "t3", True), ("A", "t4", False), ("B", "t4", True)]
    assert offline_windows(recs) == [{"asset_id": "A", "start": "t2", "end": "t4"},
                                     {"asset_id": "B", "start": "t4", "end": None}]


def test_anemometer_fault_remote_check():
    t = turbine(11)
    a = assess(t, wind_pred(t, WindAnomaly.ANEMOMETER_FAULT.value), 0)
    assert a.status.status == "eligible" and a.crew_type == CrewType.REMOTE.value
    assert triage([a], TICK).work_orders[0].payload["crew_type"] == "remote_check"


def test_overheating_warning_and_weak_signal_eligible():
    t = turbine(12)
    a = assess(t, wind_pred(t, WindAnomaly.OVERHEATING.value), 0)
    assert a.status.status == "eligible" and a.alert[0] == "warning" and a.crew_type is None
    weak = qualify(t, wind_pred(t, WindAnomaly.FAULT_STOP.value, score=0.3), 0)
    assert weak.status == "eligible" and weak.reason == "weak:fault_stop"


def test_turbine_ood_human_review():
    t = turbine(13)
    s = qualify(t, wind_pred(t, WindAnomaly.UNDERPERFORMANCE.value, ood_score=4.0), 0)
    assert s.status == "human_review" and s.reason == "ood"


# --- triage ---------------------------------------------------------------------------------------------------
def test_score_formula():
    assert triage_score(0.5, 2000, 0.01, 500, 400, 2) == pytest.approx(0.5 * (2 * 500 + 0.01 * 400) * 2)


def test_top_k_per_crew_type():
    sites = [dict(SITE_1042, site_id=f"S{2000 + i}") for i in range(8)]
    items = []
    base = site_pred(fault=PvFault.STRING_OUTAGE.value)
    for i, s in enumerate(sites):
        pred = dataclasses.replace(base, asset_id=s["site_id"])  # same expected curve, different losses
        items.append(assess(s, pred, 0, actual=np.array(pred.q50) * (0.9 - 0.08 * i)))  # growing losses
    turbines = [turbine(i, "WP1") for i in range(20, 25)]
    t0 = wind_q50_nonzero(turbines[0])
    for t in turbines:
        items.append(assess(t, wind_pred(t, WindAnomaly.FAULT_STOP.value, day_t0=t0), 0))
    items.append(assess(turbines[0] | {"turbine_id": "WP1.TID99"}, wind_pred(turbine(99), WindAnomaly.ANEMOMETER_FAULT.value), 0))

    cfg = TriageConfig(k_per_crew={"residential_installer": 3, "wind_technician": 2, "remote_check": None})
    res = triage(items, TICK, config=cfg)
    by_crew = {}
    for w in res.work_orders:
        check_action(w)
        by_crew.setdefault(w.payload["crew_type"], []).append(w.payload)
    assert len(by_crew["residential_installer"]) == 3
    assert len(by_crew["wind_technician"]) == 2
    assert len(by_crew["remote_check"]) == 1
    res_ids = [p["asset_id"] for p in by_crew["residential_installer"]]
    assert res_ids == ["S2007", "S2006", "S2005"]  # biggest losses first
    scores = [p["score"] for p in by_crew["residential_installer"]]
    assert scores == sorted(scores, reverse=True) and [p["rank"] for p in by_crew["residential_installer"]] == [1, 2, 3]
    assert len([b for b in res.backlog if b["crew_type"] == "residential_installer"]) == 5
    assert len([b for b in res.backlog if b["crew_type"] == "wind_technician"]) == 3

    # Later the same day: K is a daily budget, so nothing more for residential; next day the backlog moves up.
    later = triage(items, "2025-06-21T18:00:00Z", state=res.state, config=cfg)
    assert not [w for w in later.work_orders if w.payload["crew_type"] == "residential_installer"]
    nxt = triage(items, "2025-06-22T06:00:00Z", state=later.state, config=cfg)
    nxt_res = [w.payload for w in nxt.work_orders if w.payload["crew_type"] == "residential_installer"]
    assert [p["asset_id"] for p in nxt_res] == ["S2004", "S2003", "S2002"]


def test_repaired_asset_closes_work_order_and_rejoins_pool():
    pred = site_pred(fault=PvFault.STRING_OUTAGE.value)
    res = triage([assess(SITE_1042, pred, 0)], TICK)
    assert "S1042" in res.state.open_work_orders
    healed = assess(SITE_1042, site_pred(), 0)
    assert healed.status.in_pool and healed.status.status == "eligible"
    assert "S1042" not in triage([healed], "2025-06-22T10:00:00Z", state=res.state).state.open_work_orders


# --- thresholds -----------------------------------------------------------------------------------------------
def test_thresholds_json_overrides_defaults(tmp_path, monkeypatch):
    f = tmp_path / "thresholds.json"
    f.write_text(json.dumps({"version": 1, "source_run": "unc-r42",
                             "site": {"stale_minutes": 30, "pfault_confident": 0.95, "pfault_std_max": 0.15,
                                      "rel_width_max": 0.8, "ood_max": 0.5},
                             "turbine": {"stale_minutes": 90, "anomaly_conf_min": 0.95, "ood_max": 3.0}}))
    th = load_thresholds(f)
    assert th["source_run"] == "unc-r42" and th["site"]["stale_minutes"] == 30
    assert load_thresholds(tmp_path / "missing.json")["site"] == DEFAULT_THRESHOLDS["site"]

    pred = site_pred(fault=PvFault.STRING_OUTAGE.value, p=0.9)
    assert qualify(SITE_1042, pred, 45).status == "derated"  # defaults: 45 min is fresh, P=0.9 is confident
    assert qualify(SITE_1042, pred, 45, thresholds=f).reason == "stale_telemetry"
    assert qualify(SITE_1042, pred, 10, thresholds=f).status == "eligible"  # P=0.9 < 0.95
    assert qualify(SITE_1042, dataclasses.replace(pred, ood_score=0.7), 10, thresholds=f).reason == "ood"
    # default path (models/thresholds.json) is picked up when present; here via the env override
    monkeypatch.setenv(Q.THRESHOLDS_ENV, str(f))
    assert qualify(SITE_1042, pred, 45).reason == "stale_telemetry"
    # partial dict merges over the defaults
    assert qualify(SITE_1042, pred, 45, thresholds={"site": {"stale_minutes": 120}}).status == "derated"


def test_thresholds_partial_file_keeps_defaults(tmp_path):
    f = tmp_path / "t.json"
    f.write_text(json.dumps({"version": 1, "source_run": "x", "turbine": {"anomaly_conf_min": 0.8}}))
    th = load_thresholds(f)
    assert th["turbine"]["anomaly_conf_min"] == 0.8 and th["turbine"]["stale_minutes"] == 60
    assert th["site"] == DEFAULT_THRESHOLDS["site"]


# --- whole tick with the mock ---------------------------------------------------------------------------------
def test_full_tick_with_mock_fleet():
    sites = [{"site_id": f"S{i:04d}", "feeder_id": f"F{i % 15}", "kwp_true": 3 + i % 8, "lat": 44.0 + (i % 5) / 4,
              "battery_kwh": 10.0 * (i % 2), "battery_kw": 5.0 * (i % 2)} for i in range(200)]
    turbines = [turbine(i, f"WP{1 + i % 3}") for i in range(1, 41)]
    preds = {s["site_id"]: PV.predict(s, t0=T0) for s in sites}
    preds |= {t["turbine_id"]: WIND.predict(t, t0=T0) for t in turbines}
    ages = {k: (90 if i % 50 == 0 else 5) for i, k in enumerate(preds)}
    out = qualify_many(sites + turbines, preds, ages)
    assert len(out) == 240 and all(isinstance(a.status, AssetStatus) for a in out)
    statuses = {a.status.status for a in out}
    assert {"eligible", "excluded"} <= statuses
    for a in out:
        s = a.status
        if a.asset_kind == "site" and s.status == "derated":
            assert 0 < s.headroom_factor < 1 and s.reason.startswith("pv_fault:")
        if a.crew_type == CrewType.REMOTE.value:
            assert a.fault in ("sensor_fault", "anemometer_fault")
    res = triage(out, TICK)
    for act in res.actions:
        check_action(act)
    k = TriageConfig().k_per_crew
    for crew, cap in k.items():
        assert len([w for w in res.work_orders if w.payload["crew_type"] == crew]) <= cap


def test_per_prediction_cache_follows_reassigned_fields():
    """#98: _assess_site caches per-prediction stats; reassigning q / fault_probs (as world mocks do) invalidates."""
    pred = site_pred()
    assert qualify(SITE_1042, pred, 0).status == "eligible"
    pred.fault_probs = {c.value: (0.9 if c == PvFault.STRING_OUTAGE else 0.1 / 7) for c in PvFault}
    assert qualify(SITE_1042, pred, 0).reason == f"pv_fault:{PvFault.STRING_OUTAGE.value}"
    q50 = np.array(pred.q50)
    pred.fault_probs, pred.q10, pred.q90 = None, list(q50 * 0.4), list(q50 * 1.6)
    assert qualify(SITE_1042, pred, 0).reason == "wide_interval"


# --- Wind D2 (card #167): turbine work orders ranked by lei per day ------------------------------------------
def _wind_fleet():
    """WP9 at noon on a windy mock day: T06 stopped, T03 10 % low, T11 anemometer, T05 curtailed, T09 data gap."""
    ts = {n: turbine(n, "WP9") for n in (6, 3, 11, 5, 9)}
    t0 = wind_q50_nonzero(ts[6])
    tick = t0[:10] + "T12:00:00Z"

    def half(x):  # observed until the tick (12:00 = step 48)
        return np.concatenate([np.asarray(x, float)[:48], np.full(48, np.nan)])

    p6 = wind_pred(ts[6], WindAnomaly.FAULT_STOP.value, day_t0=t0)
    p3 = wind_pred(ts[3], WindAnomaly.UNDERPERFORMANCE.value, day_t0=t0)
    p5 = wind_pred(ts[5], WindAnomaly.CURTAILMENT.value, day_t0=t0)
    items = [
        assess(ts[6], p6, 0, actual=half(np.zeros(96))),
        assess(ts[3], p3, 0, actual=half(np.array(p3.q50) * 0.9)),
        assess(ts[11], wind_pred(ts[11], WindAnomaly.ANEMOMETER_FAULT.value, day_t0=t0), 0),
        assess(ts[5], p5, 0, actual=half(np.minimum(np.array(p5.q50), 1000.0))),
        assess(ts[9], wind_pred(ts[9], WindAnomaly.DATA_GAP.value, day_t0=t0), 20),
    ]
    return items, tick, p6, p3


def test_wind_om_queue_ranked_by_triage_score_stopped_turbine_on_top():
    items, tick, p6, p3 = _wind_fleet()
    res = triage(items, tick, wind=WindInputs(pzu_next24=500.0))
    for act in res.actions:
        check_action(act)
    q = res.om_queue("turbine")
    assert [r["asset_id"] for r in q][:2] == ["WP9.TID06", "WP9.TID03"]
    assert [r["triage_score"] for r in q] == sorted([r["triage_score"] for r in q], reverse=True)
    top = q[0]
    # fault_stop: lost MWh = what it would produce over the next 24 h (M6 expected stands in for M5 until #164)
    lost6 = float(np.mean(p6.q50[48:])) * 24 / 1000
    assert top["lost_src"] == "windnbm_expected" and top["lost_mwh_24h"] == pytest.approx(lost6, rel=1e-3)
    assert top["lei_per_day"] == pytest.approx(lost6 * 500.0, rel=1e-3)
    assert top["lost_lei_day"] == top["lei_per_day"]
    assert top["triage_score"] == pytest.approx(0.9 * lost6 * 500.0 * 2.0, rel=1e-3)  # P × lei/day × 2 days
    assert top["crew_type"] == "wind_technician" and top["pzu_next24_lei_mwh"] == 500.0
    assert top["judge_verdict"] == "use" and top["judge_rule"] == "no_judge" and top["judge_reasons"]
    # underperformance: M6 Σ (expected − actual) over the observed last 12 h, scaled to 24 h
    r3 = next(r for r in q if r["asset_id"] == "WP9.TID03")
    lost3 = float(np.sum(0.1 * np.array(p3.q50[:48]))) * 0.25 / 1000 * 2
    assert r3["lost_src"] == "windnbm_measured" and r3["lost_obs_h"] == 12.0
    assert r3["lost_mwh_24h"] == pytest.approx(lost3, rel=1e-3)
    # routing: sensor fault -> remote check; curtailment -> log only; data gap -> offline alert routed remote
    crews = {r["asset_id"]: r["crew_type"] for r in q}
    assert crews["WP9.TID11"] == "remote_check" and "WP9.TID05" not in crews and "WP9.TID09" not in crews
    assert [c["asset_id"] for c in res.curtailment_log] == ["WP9.TID05"]
    off = [a for a in res.alerts if a.payload["code"] == "offline"]
    assert len(off) == 1 and off[0].payload["asset_id"] == "WP9.TID09" and off[0].payload["route"] == "remote_check"


def test_wind_top_k_per_crew_keeps_the_most_lei():
    items, tick, _, _ = _wind_fleet()
    cfg = TriageConfig(k_per_crew={"residential_installer": 5, "wind_technician": 1, "remote_check": 20})
    res = triage(items, tick, config=cfg, wind=WindInputs(pzu_next24=500.0))
    wind_wos = [w.payload for w in res.work_orders if w.payload["crew_type"] == "wind_technician"]
    assert [p["asset_id"] for p in wind_wos] == ["WP9.TID06"] and wind_wos[0]["rank"] == 1
    assert [b["asset_id"] for b in res.backlog] == ["WP9.TID03"]


def test_wind_m5_then_binned_curve_for_a_stopped_turbine():
    items, tick, p6, _ = _wind_fleet()
    m5 = dataclasses.replace(p6, model="windpower", t0=tick, q10=[1500.0] * 96, q50=[2000.0] * 96,
                             q90=[2400.0] * 96, anomaly_class=None, anomaly_score=None)
    res = triage(items, tick, wind=WindInputs(m5={"WP9.TID06": m5}, pzu_next24=400.0))
    top = res.om_queue("turbine")[0]
    assert top["lost_src"] == "windpower" and top["lost_mwh_24h"] == pytest.approx(48.0)
    assert top["lei_per_day"] == pytest.approx(48.0 * 400.0)
    binned = dataclasses.replace(m5, model="binned_curve", q50=[1000.0] * 96, q90=[1200.0] * 96, q10=[800.0] * 96)
    top = triage(items, tick, wind=WindInputs(m5_binned={"WP9.TID06": binned}, pzu_next24=400.0)).om_queue()[0]
    assert top["lost_src"] == "binned_curve" and top["lost_mwh_24h"] == pytest.approx(24.0)


def test_wind_judge_unknown_is_remote_check_after_scored_orders(monkeypatch):
    items, tick, _, _ = _wind_fleet()
    real = TR._adapter_validated()

    def judge(asset_id, model, day, *, raw, substitutes=None, judge=None):  # T06: dont_use, no substitute data
        v = real(asset_id, model, day, raw=raw, substitutes=substitutes, judge=judge)
        if asset_id != "WP9.TID06":
            return v
        return SimpleNamespace(**{**{k: getattr(v, k) for k in ("model", "day", "reasons", "judge_version")},
                                  "known": False, "verdict": "dont_use", "rule": "r6", "q50": None, "score": None,
                                  "source_model": None})

    monkeypatch.setattr(TR, "_adapter_validated", lambda: judge)
    res = triage(items, tick, wind=WindInputs(pzu_next24=500.0))
    q = res.om_queue("turbine")
    assert q[-1]["asset_id"] == "WP9.TID06" and q[-1]["judge_unknown"] and q[-1]["p_fault"] is None
    assert q[-1]["crew_type"] == "remote_check" and q[-1]["score"] == 0.0
    remote = [w.payload["asset_id"] for w in res.work_orders if w.payload["crew_type"] == "remote_check"]
    assert remote == ["WP9.TID11", "WP9.TID06"]  # ranked after the scored remote check
    assert [w.payload["asset_id"] for w in res.work_orders if w.payload["crew_type"] == "wind_technician"] \
        == ["WP9.TID03"]


def test_wind_reads_p_fault_through_the_c1_adapter():
    adapter = pytest.importorskip("sentinel.judge.adapter")  # card #157
    items, tick, _, _ = _wind_fleet()
    day = TR.local_day(tick)
    rules = dataclasses.replace(items[0].prediction, model="windmon_rules", anomaly_score=0.6)
    src = adapter.MemoryVerdicts({("WP9.TID06", "windnbm", day): {
        "verdict": "dont_use", "substitute": "windmon_rules", "reasons_json": '["disagrees with rules"]'}})
    res = triage(items, tick, wind=WindInputs(m6_rules={"WP9.TID06": rules}, judge=src, pzu_next24=500.0))
    r6 = next(r for r in res.om_queue() if r["asset_id"] == "WP9.TID06")
    assert r6["p_fault"] == 0.6 and r6["p_fault_src"] == "windmon_rules" and r6["judge_verdict"] == "dont_use"
    assert "disagrees with rules" in r6["judge_reasons"]
    res = triage(items, tick, wind=WindInputs(judge=src, pzu_next24=500.0))  # substitute not supplied
    r6 = next(r for r in res.om_queue() if r["asset_id"] == "WP9.TID06")
    assert r6["judge_unknown"] and r6["crew_type"] == "remote_check"


def test_pzu_next24_mean_real_opcom_and_fallback():
    pl = pytest.importorskip("polars")
    if not TR.PRICES_PATH.exists():
        pytest.skip("data/raw/prices.parquet not present")
    p = pl.read_parquet(TR.PRICES_PATH).filter(pl.col("market") == "opcom_pzu")
    lo, hi = datetime(2025, 6, 21, 12, tzinfo=timezone.utc), datetime(2025, 6, 22, 12, tzinfo=timezone.utc)
    want = p.filter((pl.col("ts") >= lo) & (pl.col("ts") < hi))["price_lei_mwh"].mean()  # 24 hourly prices
    price, src = pzu_next24_mean("2025-06-21T12:00:00Z", 450.0)
    assert src == "opcom_pzu" and price == pytest.approx(want, rel=1e-9)
    assert pzu_next24_mean("2031-01-01T00:00:00Z", 450.0) == (450.0, "config")
    # the default config prices turbines at the real PZU; wind_price="config" keeps price_lei_mwh
    items, tick, _, _ = _wind_fleet()
    row = triage(items, tick).om_queue("turbine")[0]
    assert row["price_src"] == "opcom_pzu"
    assert row["pzu_next24_lei_mwh"] == pytest.approx(pzu_next24_mean(tick, 450.0)[0], abs=0.01)
    row = triage(items, tick, config=TriageConfig(wind_price="config", price_lei_mwh=321.0)).om_queue("turbine")[0]
    assert row["price_src"] == "config" and row["pzu_next24_lei_mwh"] == 321.0


def test_wind_formulas_and_site_rows_unchanged():
    assert wind_triage_score(0.5, 1000.0, 2.0) == pytest.approx(1000.0)
    assert lei_per_day(2.0, 450.0) == pytest.approx(900.0) and lei_per_day(-1.0, 450.0) == 0.0
    wo = triage([assess(SITE_1042, site_pred(fault=PvFault.STRING_OUTAGE.value), 0)], TICK).work_orders[0]
    assert set(wo.payload) == {"asset_id", "asset_kind", "fault", "crew_type", "score", "lost_kwh_day",
                               "lost_lei_day", "lost_flex_mwh_day", "mw_at_risk", "p_fault", "visit_days", "status",
                               "reason", "headroom_factor", "detail", "feeder_id", "rank"}
