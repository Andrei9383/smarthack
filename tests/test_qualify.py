"""Card #26: qualify (sites + turbines) and O&M triage, driven by the MockPredictor."""

import dataclasses
import json

import numpy as np
import pytest

from sentinel.agent import qualify as Q
from sentinel.agent.qualify import DEFAULT_THRESHOLDS, assess, load_thresholds, qualify, qualify_many, rel_width
from sentinel.agent.triage import TriageConfig, offline_windows, triage, triage_score
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
