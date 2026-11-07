"""Card #17: agent loop, replay into SQLite, audit log (plan §3.1, §3.6; team/contracts/replay.md, actions.md)."""

import json
import sqlite3
import time
from pathlib import Path

import pytest

from sentinel.agent import loop as L
from sentinel.agent.audit import (ALLOWED_DROPS, FALLBACK_SQL, AuditDB, SchemaMismatch, Ts, allowed_drop,
                                  find_schema_files, format_time)
from sentinel.agent.replay import METRICS, replay
from sentinel.agent.world import MockWorld
from sentinel.contracts import PAYLOAD_KEYS, Action, input_hash

TABLES = ["asset_steps", "runs", "kpis", "feeders", "parks", "assets", "measurements", "predictions", "statuses", "feeder_curves",
          "commitments", "schedules", "work_orders", "alerts", "review_queue", "gis_flags", "park_forecasts",
          "zone_demand", "reports", "audit_log"]


@pytest.fixture(scope="module")
def small(tmp_path_factory):
    """2-day small mock world, built-in components (deterministic whatever has merged), fallback schema, strict."""
    d = tmp_path_factory.mktemp("replay")
    t = time.perf_counter()
    rep = replay(["baseline", "sentinel"], d / "s.sqlite", world=MockWorld(days=2, scale=0.5),
                 schema_dir=d / "no-drizzle", force_fallback=True, print_metrics=False, strict=True)
    rep.elapsed = time.perf_counter() - t
    conn = sqlite3.connect(rep.out)
    yield rep, conn
    conn.close()


def q(conn, sql, *a):
    return conn.execute(sql, a).fetchall()


# --- schema / audit store ---------------------------------------------------------------------------------------
def test_schema_applied_from_fallback(tmp_path):
    db = AuditDB(tmp_path / "a.sqlite", schema_dir=tmp_path / "missing")
    assert db.schema_files == [FALLBACK_SQL]
    assert "schema_fallback.sql" in db.schema_source
    for t in TABLES:
        assert db.has_table(t), t
    db.close()


def test_drizzle_migrations_preferred(tmp_path):
    mig = tmp_path / "drizzle" / "20260101000000_x"
    mig.mkdir(parents=True)
    (mig / "migration.sql").write_text("CREATE TABLE `runs` (`id` text PRIMARY KEY);\n--> statement-breakpoint\n"
                                       "CREATE TABLE `kpis` (`run_id` text, `value` real);")
    assert find_schema_files(tmp_path / "drizzle") == [mig / "migration.sql"]
    db = AuditDB(tmp_path / "b.sqlite", schema_dir=tmp_path / "drizzle")
    assert set(db.tables) >= {"runs", "kpis"} and "assets" not in db.tables
    db.close()


def test_time_columns_follow_declared_type(tmp_path):
    sd = tmp_path / "schema"
    sd.mkdir()
    (sd / "0000.sql").write_text(
        "CREATE TABLE ti (ts integer NOT NULL, step integer NOT NULL, day integer NOT NULL, zone_type text);"
        "CREATE TABLE tt (ts text NOT NULL, step integer, date text, day text);")
    db = AuditDB(tmp_path / "t.sqlite", schema_dir=sd, warn=lambda m: None)
    t = 1750075200  # 2025-06-16T12:00:00Z = 15:00 Bucharest (EEST)
    for table in ("ti", "tt"):
        db.add(table, {"ts": Ts(t), "level": "county", "bogus": 1})
        db.add(table, {"ts": "2025-06-16T12:00:00Z", "level": "county"})
    db.flush()
    rows = q(db.conn, "SELECT ts, step, day, zone_type FROM ti")
    assert rows == [(t, 60, 1750021200, "county")] * 2  # unix seconds, local step, local midnight (21:00Z)
    rows = q(db.conn, "SELECT ts, step, date, day FROM tt")
    assert rows == [("2025-06-16T12:00:00Z", 60, "2025-06-16", "2025-06-16")] * 2
    assert "bogus" in db.dropped["ti"] and "bogus" in db.dropped["tt"]
    assert format_time(t, "INTEGER") == t and format_time(t, "TEXT") == "2025-06-16T12:00:00Z"
    db.close()


def test_strict_mode_raises_on_unmapped_fields(tmp_path):
    db = AuditDB(tmp_path / "s.sqlite", schema_dir=tmp_path / "none", strict=True)
    db.add("kpis", {"run_id": "r", "policy": "sentinel", "name": "x", "value": 1.0, "label": "synthetic RO",
                    "bogus_kw": 1.0})
    with pytest.raises(SchemaMismatch, match="kpis.bogus_kw"):
        db.flush()
    db = AuditDB(tmp_path / "s2.sqlite", schema_dir=tmp_path / "none", strict=True)
    db.add("kpis", {"run_id": "r", "policy": "sentinel", "name": "x", "value": 1.0})  # label is NOT NULL
    with pytest.raises(SchemaMismatch, match="kpis.label"):
        db.flush()
    db = AuditDB(tmp_path / "s3.sqlite", schema_dir=tmp_path / "none", strict=True)
    db.add("zone_demand", {"run_id": "r", "zone_id": "CT", "level": "county", "zone_name": "Constanta",
                           "ts": Ts(1750075200), "q50_mw": 1.0, "peak_stress": 0.5, "label": "synthetic RO",
                           "step": 60, "stress": 0.4})  # allow-listed drops pass
    db.close()
    assert all(reason for t in ALLOWED_DROPS.values() for reason in t.values())


def test_one_day_replay_against_38_schema_drops_nothing_unexpected(tmp_path):
    """Default schema (packages/db/drizzle if merged, else the d362a39 copy), real components, strict mode."""
    rep = replay(["baseline", "sentinel"], tmp_path / "x.sqlite", world=MockWorld(days=1, scale=0.4),
                 print_metrics=False, strict=True)
    s = rep.db
    assert s["unexpected_drops"] == {} and s["missing_tables"] == [] and s["filled_not_null"] == {}
    assert all(allowed_drop(t, f) for t, fs in s["dropped_fields"].items() for f in fs)


def test_value_map_and_not_null_fill(tmp_path):
    db = AuditDB(tmp_path / "v.sqlite", schema_dir=tmp_path / "none", warn=lambda m: None, strict=False)
    db.add("assets", {"id": "S1", "asset_kind": "site", "lat": 44.0, "lon": 28.0, "county": "Constanta"})
    db.flush()
    kind, name, rated = q(db.conn, "SELECT asset_kind, name, rated_kw FROM assets")[0]
    assert (kind, name, rated) == ("prosumer", "", 0)  # #38 enum; NOT NULL columns filled, never a crash
    db.close()


# --- replay -----------------------------------------------------------------------------------------------------
def test_replay_fast_and_fills_every_table(small):
    rep, conn = small
    assert rep.elapsed < 60
    for t in TABLES:
        assert q(conn, f"SELECT COUNT(*) FROM {t}")[0][0] > 0, f"{t} is empty"
    runs = q(conn, "SELECT id, policy, label, notes FROM runs ORDER BY policy")
    assert [r[1] for r in runs] == ["baseline", "sentinel"]
    assert all(r[2] == "synthetic RO" for r in runs)
    notes = json.loads(runs[1][3])
    assert notes["components"]["planner"].startswith("fallback") and "planner" in notes["fallbacks"]


def test_every_action_has_an_audit_record(small):
    rep, conn = small
    audit = {r[0]: r for r in q(conn, "SELECT id, action_type, model_version, input_hash, prediction_json, status, "
                                      "action_id FROM audit_log")}
    n_actions = sum(sum(r.n_actions.values()) for r in rep.results.values())
    assert len(audit) == n_actions == sum(len(r.action_ids) for r in rep.results.values())
    for r in audit.values():
        assert r[1] in PAYLOAD_KEYS and r[2] and len(r[3]) == 16 and r[4] is not None and r[5] is not None
    for t in ("work_orders", "alerts", "review_queue", "gis_flags", "reports"):
        ids = {x[0] for x in q(conn, f"SELECT action_id FROM {t}")}
        assert ids and ids <= set(audit), t
    types = {r[1] for r in audit.values()}
    assert types >= {"battery_schedule", "feeder_commitment", "work_order", "alert", "human_review", "gis_flag",
                     "park_report", "zone_demand"}


def test_both_policies_see_identical_inputs(small):
    rep, conn = small
    b, s = rep.results["baseline"], rep.results["sentinel"]
    assert b.input_hashes == s.input_hashes and b.day_hashes == s.day_hashes
    pairs = q(conn, "SELECT a.input_hash, b.input_hash FROM audit_log a JOIN audit_log b ON a.asset_id = b.asset_id "
                    "AND a.ts = b.ts AND a.action_type = b.action_type AND a.run_id < b.run_id "
                    "WHERE a.action_type = 'battery_schedule'")
    assert pairs and all(x == y for x, y in pairs)


def test_sentinel_never_breaches_comfort_reserve(small):
    rep, conn = small
    assert rep.results["sentinel"].comfort_breaches == 0
    sid = rep.run_ids["sentinel"]
    assert q(conn, "SELECT COUNT(*) FROM schedules WHERE run_id = ? AND comfort_breached = 1", sid)[0][0] == 0


def test_feasibility_clip_holds_against_an_adversarial_planner():
    w = MockWorld(days=1, scale=0.4, with_wind=False)
    comps = L.default_components(w, force_fallback=True)

    def drain(feeder, eligible_sites, rest, forecasts, G=None, prices=None, **kw):
        sched = {r["site_id"]: [-50.0] * 96 for r in eligible_sites}  # discharge flat out, all day
        return L.FeederPlan(feeder.feeder_id, kw.get("t0") or "2025-06-16T00:00:00Z", "sentinel", feeder.g_kw,
                            sched, [0.0] * 96, [0.0] * 96)

    comps.plan_feeder = drain
    res = L.run_policy(w, comps, L.LoopConfig(policy="sentinel", run_id="evil"))
    assert res.agent_steps > 0 and res.clipped_steps > 0
    assert res.comfort_breaches == 0


def test_sentinel_not_worse_on_congested_feeder(small):
    rep, _ = small
    b, s = rep.results["baseline"], rep.results["sentinel"]
    mb = sum(v["minutes_over"] for (f, _), v in b.feeder_days.items() if f == "F7")
    ms = sum(v["minutes_over"] for (f, _), v in s.feeder_days.items() if f == "F7")
    assert mb > 0, "F7 must congest under the baseline in the mock world"
    assert ms <= mb
    pb = max(v["peak_export"] for (f, _), v in b.feeder_days.items() if f == "F7")
    ps = max(v["peak_export"] for (f, _), v in s.feeder_days.items() if f == "F7")
    assert ps < pb


def test_kpis_and_commitment_invariants(small):
    rep, conn = small
    names = {r[0] for r in q(conn, "SELECT name FROM kpis WHERE policy = 'sentinel'")}
    assert names >= set(METRICS)
    bad = q(conn, "SELECT COUNT(*) FROM commitments WHERE NOT (nominal_kw + 1e-6 >= available_kw AND "
                  "available_kw + 1e-6 >= committed_kw AND committed_kw >= -1e-9)")[0][0]
    assert bad == 0
    # curtailment is logged, never a work order (plan §3.5, R12)
    assert q(conn, "SELECT COUNT(*) FROM alerts WHERE code = 'curtailment' AND creates_work_order = 1")[0][0] == 0


def test_mock_world_is_deterministic_and_has_the_demo_stories():
    a, b = MockWorld(days=7), MockWorld(days=7)
    sid = a.story_site
    assert sid == "1042" and a.sites[sid].feeder_id == "F7" and a.faults[sid]["fault"] == "string_outage"
    assert input_hash(a.site_day(sid, 3).pv_kw) == input_hash(b.site_day(sid, 3).pv_kw)
    assert a.feeders["F7"].congested and a.curtailment["park_id"] == "WP01"
    assert any(f["fault"] == "fault_stop" for f in a.faults.values())


def test_replay_with_default_components_prints_metrics(tmp_path, capsys):
    """Whatever real modules (#18/#26/#27/#44) are importable get used; the replay must still run end to end."""
    rep = replay(["baseline", "sentinel"], tmp_path / "d.sqlite", world=MockWorld(days=1, scale=0.4),
                 schema_dir=tmp_path / "none")
    out = capsys.readouterr().out
    for m in METRICS:
        assert f"LAB_METRIC {m}=" in out, m
    assert rep.seconds < 60
    assert all(isinstance(a, str) for a in rep.results["sentinel"].action_ids)


def test_actions_validate_against_contract():
    from sentinel.agent import actions as A

    rec = A.audit("v", {"x": 1}, {}, None, "ok")
    a = A.battery_schedule("2025-06-16T00:00:00Z", "sentinel", "r", "S1", "F1", [0.0] * 96, rec)
    assert Action.from_json(a.to_json()).payload["site_id"] == "S1"
    with pytest.raises(ValueError):
        A.make("work_order", "2025-06-16T00:00:00Z", "sentinel", "r", {"asset_id": "S1"}, rec)


# --- observation vs physics (#78 drills finding): blind sites, re-admission, commit-time blind haircut ----------
class _BlindMorning(MockWorld):
    """Every site reports nothing from step 20 to 44 of each day (05:00-11:00 UTC); the physics is untouched."""

    def observed_site_day(self, site_id, d):
        sd = self.site_day(site_id, d)
        age = list(sd.telemetry_age_s)
        for k in range(20, 45):
            age[k] = max(age[k], (k - 19) * 900.0)
        return type(sd)(sd.pv_kw, sd.load_kw, age, sd.inv_status, sd.inv_voltage_v)


def _hit(res):
    return res.hits / max(1, res.commit_steps)


def test_blind_sites_keep_their_schedule_and_commitment_holds():
    clean, blind = MockWorld(days=2, scale=0.5, with_wind=False), _BlindMorning(days=2, scale=0.5, with_wind=False)
    run = lambda w, **kw: L.run_policy(w, L.default_components(w, force_fallback=True),  # noqa: E731
                                       L.LoopConfig(policy="sentinel", run_id="b", **kw))
    c, b = run(clean), run(blind)
    old = run(blind, readmit=False, hold_blind=False, blind_haircut=False)  # pre-fix behaviour
    assert b.comfort_breaches == 0 and old.comfort_breaches == 0
    assert _hit(b) >= _hit(c) - 0.1
    assert _hit(old) < _hit(b)  # dropping blind sites to their local rule for the rest of the day loses the P10


def test_blind_haircut_removes_the_blind_share_by_step():
    import numpy as np

    c = L.Commitment("F1", "2025-06-16T00:00:00Z", [2.0] * 4, [2.0] * 4, [1.0] * 4, haircut=0.05, horizon=4)
    blind = {"a": np.array([True, True, False, False])}
    out = L._blind_haircut(c, {"a": np.array([1.0, 1.0, 1.0, 0.0]), "b": np.array([1.0, 3.0, 1.0, 1.0])}, blind)
    assert out.committed == pytest.approx([0.5, 0.75, 1.0, 1.0])
    assert out.extra["haircut_k"][0] == pytest.approx(1 - 0.95 * 0.5) and out.extra["haircut_k"][3] == 0.05
    prof = L._blind_profile({"a": type("SD", (), {"telemetry_age_s": [0.0] * 40 + [4000.0] + [0.0] * 55})()},
                            L.LoopConfig())
    assert prof["a"][36:45].all() and not prof["a"][35] and not prof["a"][45]


# --- stage 2: predictor versions, #24 WindMon baseline, energy over the limit, online conformal commitment ----
def test_predictor_versions_ignore_the_unused_zone_mock():
    from sentinel.contracts import MockPredictor

    w = MockWorld(days=1, scale=0.4, with_wind=False)
    real = type("P", (), {"model": "pvnet", "model_version": "pvnet-383-s0"})()
    preds = {**w.default_predictors(), "pv": real, "load": real, "sitehealth": real, "wind": real,
             "zone": MockPredictor("zone")}
    src = L.default_components(w, force_fallback=True, predictors=preds).sources
    assert src["predictors"] == "real:pvnet-383-s0" and "mock" not in src["predictors"]


def test_baseline_wind_alerts_use_the_24_windmon_rules():
    pytest.importorskip("sentinel.eval.baselines.wind.windmon_rules")
    w = MockWorld(days=2, scale=0.4)
    comps = L.default_components(w)
    assert comps.sources["wind_baseline"].startswith("real:") and comps.windmon_flags is not None
    ids, sens, err, rated = w.turbine_sensors()
    assert sens["act_pwt"].shape == (2 * 96, len(ids)) == err.shape and len(rated) == len(ids)
    b = L.run_policy(w, comps, L.LoopConfig(policy="baseline", run_id="b"))
    s = L.run_policy(w, comps, L.LoopConfig(policy="sentinel", run_id="s"))
    assert b.turbine_alerts > s.turbine_alerts > 0  # WindMon rules fire more than the Sentinel triage
    assert L.default_components(w, force_fallback=True).windmon_flags is None  # built-in stand-in stays the fallback


def test_energy_over_limit_and_overshoot_kpis(small):
    rep, conn = small
    rows = {(p, n): v for p, n, v in q(conn, "SELECT policy, name, value FROM kpis WHERE name LIKE 'energy_over%' "
                                             "OR name LIKE 'peak_overshoot%' OR name LIKE 'minutes_over_limit_F7%'")}
    for pol in ("baseline", "sentinel"):
        assert rows[(pol, "energy_over_limit")] == pytest.approx(rep.results[pol].energy_over_limit_kwh)
        assert sum(v for (p, n), v in rows.items() if p == pol and n.startswith("energy_over_limit_F7_d")) <= \
            rows[(pol, "energy_over_limit")] + 1e-6
        assert (pol, "peak_overshoot_F7_d0") in rows and (pol, "minutes_over_limit_F7_d1") in rows
    assert rows[("sentinel", "energy_over_limit")] <= rows[("baseline", "energy_over_limit")]
    unit = dict(q(conn, "SELECT name, unit FROM kpis WHERE policy = 'baseline' AND name IN "
                        "('energy_over_limit', 'peak_overshoot', 'energy_over_limit_F7_d0')"))
    assert unit == {"energy_over_limit": "kWh", "peak_overshoot": "kW", "energy_over_limit_F7_d0": "kWh"}


def test_online_conformal_uses_only_past_days_and_widens_too_narrow_bands():
    import numpy as np

    pytest.importorskip("sentinel.eval.uncertainty.conformal")
    cfg = L.LoopConfig()
    assert L._fit_conformal({"pv": [], "load": []}, cfg) == {}  # day 0: nothing replayed yet
    rng = np.random.default_rng(0)
    rows = []
    for _ in range(20):  # load forecast 1 kW +- 0.05, truth spread +- 0.5: the band is far too narrow
        q50 = np.ones(96)
        rows.append((q50 + rng.normal(0, 0.4, 96), q50 - 0.05, q50, q50 + 0.05, np.ones(96, bool)))
    cal = L._fit_conformal({"load": rows}, cfg)["load"]
    lo, _, hi = cal.apply(np.full(96, 0.95), np.ones(96), np.full(96, 1.05))
    assert (lo < 0.8).mean() > 0.9 and (hi > 1.2).mean() > 0.9
    w = MockWorld(days=2, scale=0.5, with_wind=False)
    res = L.run_policy(w, L.default_components(w, force_fallback=True), L.LoopConfig(policy="sentinel", run_id="c"))
    assert res.conformal[0] == {} and set(res.conformal[1]) == {"pv", "load"}
