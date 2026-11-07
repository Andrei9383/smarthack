"""Card #7: contract schemas, mock output shapes, action JSON round-trip, wind tag parsing, battery rule."""

import json

import pytest

from sentinel.contracts import (
    DAY_STEPS,
    PAYLOAD_KEYS,
    Action,
    ActionType,
    AssetStatus,
    AuditRecord,
    BatterySpec,
    Commitment,
    FeederPlan,
    FeederSpec,
    MockPredictor,
    Prediction,
    Predictor,
    PvFault,
    TABLES,
    WindAnomaly,
    Window,
    check_columns,
    input_hash,
    is_error_code,
    kmh_to_ms,
    make_tag,
    parse_tag,
    self_consumption_step,
)

T0 = "2025-06-21T00:00:00Z"
SITE = {"site_id": "S0001", "kwp_true": 6.0, "lat": 44.2, "archetype": "ac"}
TURBINE = {"turbine_id": "WP1.TID07", "park_id": "WP1", "rated_kw": 2500.0}


# --- envelope ------------------------------------------------------------------------------------------------
def _pred(**kw):
    base = dict(asset_id="S1", model="pvnet", model_version="v", t0=T0, horizon=3,
                q10=[0, 1, 2], q50=[0, 2, 3], q90=[1, 3, 4])
    base.update(kw)
    return Prediction(**base)


def test_prediction_roundtrip_json():
    p = _pred(fault_probs={f.value: 1 / 8 for f in PvFault})
    d = json.loads(json.dumps(p.to_dict()))
    assert Prediction.from_dict(d) == p
    assert abs(p.p_fault - 7 / 8) < 1e-9


@pytest.mark.parametrize("bad", [
    dict(q50=[0, 2]),  # wrong length
    dict(q10=[0, 3, 2]),  # q10 > q50
    dict(t0="2025-06-21T00:00:00"),  # naive time
    dict(t0="2025-06-21T00:00:00+02:00"),  # not UTC
    dict(fault_probs={"healthy": 0.5}),  # does not sum to 1
    dict(fault_probs={"healthy": 0.5, "gremlins": 0.5}),  # unknown class
    dict(anomaly_class="aliens"),
    dict(ood_score=-1),
])
def test_prediction_rejects(bad):
    with pytest.raises(ValueError):
        _pred(**bad)


# --- mock ----------------------------------------------------------------------------------------------------
@pytest.mark.parametrize("kind,asset", [
    ("pv", SITE), ("load", SITE), ("sitehealth", SITE), ("wind", TURBINE),
    ("zone", {"zone_id": "CT", "n_homes": 5000}),
])
def test_mock_shapes_and_determinism(kind, asset):
    m = MockPredictor(kind)
    assert isinstance(m, Predictor)
    p = m.predict(asset, Window(t0=T0))
    assert p.horizon == DAY_STEPS and len(p.q50) == DAY_STEPS
    assert p.model_version == "mock-1"
    assert p == MockPredictor(kind).predict(asset, Window(t0=T0))  # deterministic
    assert all(v >= 0 for v in p.q10)


def test_mock_pv_is_plausible():
    p = MockPredictor("pv").predict(SITE, t0=T0)
    night, noon = p.q50[0], max(p.q50)  # 00:00 UTC is night in RO
    assert night == 0 and 0 < noon <= SITE["kwp_true"]
    assert max(p.q90) <= SITE["kwp_true"]


def test_mock_load_has_evening_peak():
    p = MockPredictor("load").predict(SITE, t0=T0)
    peak_step = max(range(DAY_STEPS), key=lambda k: p.q50[k])
    assert 15 * 4 <= peak_step <= 18 * 4  # 15-18 UTC = 18-21 Bucharest summer time


def test_mock_sitehealth_and_wind_classes():
    sh = [MockPredictor("sitehealth").predict({"site_id": f"S{i:04d}"}, t0=T0) for i in range(200)]
    assert all(set(p.fault_probs) == {f.value for f in PvFault} for p in sh)
    n_faulty = sum(p.top_fault != "healthy" for p in sh)
    assert 10 <= n_faulty <= 60  # ~15%
    w = [MockPredictor("wind").predict({"turbine_id": f"WP1.TID{i:02d}"}, t0=T0) for i in range(100)]
    assert all(p.anomaly_class in {a.value for a in WindAnomaly} for p in w)
    assert any(p.anomaly_class != "normal" for p in w)


# --- actions -------------------------------------------------------------------------------------------------
def _payload(t):
    return {k: [0.0] * DAY_STEPS if k in ("batt_kw", "nominal", "available", "committed") else "x"
            for k in PAYLOAD_KEYS[t]}


@pytest.mark.parametrize("t", [a.value for a in ActionType])
def test_action_json_roundtrip(t):
    audit = AuditRecord(model_version="mock-1", input_hash=input_hash({"a": 1}), prediction={"p_fault": 0.1},
                        threshold=0.5, status="eligible")
    a = Action(action_id="a1", type=t, t=T0, policy="sentinel", run_id="r1", payload=_payload(t), audit=audit)
    b = Action.from_json(a.to_json())
    assert b == a and isinstance(b.audit, AuditRecord)


def test_action_missing_payload_key():
    audit = AuditRecord("v", "h", {}, None, "eligible")
    with pytest.raises(ValueError):
        Action("a", "work_order", T0, "sentinel", "r", {"asset_id": "S1"}, audit)


def test_input_hash_is_order_independent():
    assert input_hash({"a": 1, "b": [1, 2]}) == input_hash({"b": [1, 2], "a": 1})
    assert input_hash({"a": 1}) != input_hash({"a": 2})


# --- status / opt --------------------------------------------------------------------------------------------
def test_asset_status_rules():
    assert AssetStatus("S1", "eligible", 1.0, "ok").in_pool
    assert AssetStatus("S1", "derated", 0.7, "pv_fault:soiling").in_pool
    assert not AssetStatus("S1", "human_review", 0.0, "ood").in_pool
    with pytest.raises(ValueError):
        AssetStatus("S1", "excluded", 0.5, "x")
    with pytest.raises(ValueError):
        AssetStatus("S1", "maybe", 1.0, "x")


def test_feeder_limit_only_tightens_with_low_trust():
    f = FeederSpec("F1", g_max_kw=200, g_conservative_kw=120, tau=0.25)
    assert f.g_kw == pytest.approx(140)
    assert FeederSpec("F1", 200, 120, tau=1.0).g_kw == 200


def test_commitment_ordering_and_hit_rate():
    n = [10.0] * 4
    c = Commitment("F1", T0, nominal=n, available=[8.0] * 4, committed=[5.0] * 4,
                   realised=[6.0, 4.0, None, 5.0], horizon=4)
    assert c.hit_rate() == pytest.approx(2 / 3)
    with pytest.raises(ValueError):
        Commitment("F1", T0, nominal=n, available=[11.0] * 4, committed=[5.0] * 4, horizon=4)


def test_feeder_plan_lengths():
    FeederPlan("F1", T0, "sentinel", 100.0, {"S1": [0.0] * 4}, [1.0] * 4, [0.0] * 4, horizon=4)
    with pytest.raises(ValueError):
        FeederPlan("F1", T0, "sentinel", 100.0, {"S1": [0.0] * 3}, [1.0] * 4, [0.0] * 4, horizon=4)


def test_battery_self_consumption_rule():
    b = BatterySpec(capacity_kwh=10, p_max_kw=4, soc_min=0.1, comfort_reserve_kwh=2)
    assert b.e_min == 2 and b.e_max == 10
    # PV surplus 6 kW, battery capped at 4 kW -> 2 kW exported
    batt, e, grid = self_consumption_step(b, 5.0, pv_kw=7.0, load_kw=1.0)
    assert batt == pytest.approx(4.0) and grid == pytest.approx(2.0)
    assert e == pytest.approx(5.0 + 0.95 * 4 * 0.25)
    # Deficit 1 kW at night -> discharge covers it, no import
    batt, e2, grid = self_consumption_step(b, 5.0, pv_kw=0.0, load_kw=1.0)
    assert batt == pytest.approx(-1.0) and grid == pytest.approx(0.0) and e2 < 5.0
    # Empty battery cannot go below the soc_min floor
    batt, e3, grid = self_consumption_step(b, 1.0, pv_kw=0.0, load_kw=3.0)
    assert batt == pytest.approx(0.0) and e3 == pytest.approx(1.0) and grid == pytest.approx(-3.0)
    # Full battery exports everything
    batt, _, grid = self_consumption_step(b, 10.0, pv_kw=5.0, load_kw=0.0)
    assert batt == pytest.approx(0.0) and grid == pytest.approx(5.0)
    # No battery
    assert self_consumption_step(None, 0.0, 3.0, 1.0) == (0.0, 0.0, 2.0)


# --- wind tags -----------------------------------------------------------------------------------------------
def test_wind_tag_parsing():
    t = parse_tag("WP1.TID07.act_pwt")
    assert (t.park, t.turbine, t.sensor, t.turbine_id) == ("1", "07", "act_pwt", "WP1.TID07")
    assert str(t) == "WP1.TID07.act_pwt"
    assert make_tag(1, 7, "wd_spd") == "WP1.TID07.wd_spd"
    for bad in ["WP1.07.act_pwt", "TID07.act_pwt", "WP1.TID07", "wp1.tid07.act_pwt", "WP1.TID07.act pwt"]:
        with pytest.raises(ValueError):
            parse_tag(bad)
    parse_tag("WP1.TID07.something_new")  # lenient by default: PPC may add sensors
    with pytest.raises(ValueError):
        parse_tag("WP1.TID07.something_new", strict=True)


def test_wind_error_codes_and_units():
    assert is_error_code("act_pwt", 5021)
    assert not is_error_code("act_pwt", 1234.5)
    assert not is_error_code("act_pwt", None)
    assert not is_error_code("status", 3)
    assert kmh_to_ms(36.0) == pytest.approx(10.0)


# --- table schemas -------------------------------------------------------------------------------------------
def test_table_schemas():
    for name, cols in TABLES.items():
        assert cols, name
        assert all(isinstance(c, str) and isinstance(d, str) for c, d in cols.items())
    tel = ["ts", "site_id", "pv_ac_w", "inv_voltage_v", "inv_status", "soc", "batt_w", "load_w"]
    assert check_columns("telemetry", tel) == []
    assert check_columns("telemetry", tel[:-1]) == ["telemetry: missing column 'load_w'"]
