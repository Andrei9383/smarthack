"""Card #149 (J4): Judge verdict rules (judge-design §7.5), output contract (§7.7), the loop slot, ledger-only mode,
and judge on/off giving identical decisions."""

import json
import math

import numpy as np
import pandas as pd
import pytest

from sentinel.agent import loop as L
from sentinel.agent.replay import replay
from sentinel.agent.world import MockWorld
from sentinel.judge import slot as S
from sentinel.judge.verdict import (DONT_USE, LEDGER_ONLY, OUTPUT_COLUMNS, RULE_BAD_INPUT, RULE_DONT_USE,
                                    RULE_NO_RECORD, RULE_SAFETY, RULE_USE, RULE_WIDEN, USE, WIDEN, Thresholds,
                                    apply_rules, verdict_rows)


def _row(**kw):
    base = {"family": "forecast", "input_bad": False, "safety_call": False, "p_valid": np.nan, "trust": 0.9,
            "n_eff": 10.0, "band_scale": 1.0, "substitute": "lgbm_pv", "sub_trust": np.nan, "reasons": None}
    base.update(kw)
    return base


def _judge(rows, th=None):
    df = pd.DataFrame(rows)
    return df, apply_rules(df, th)


# --------------------------------------------------------------------------------------------- §7.5 rule order
def test_rule_order_first_match_wins():
    rows = [
        _row(input_bad=True, safety_call=True, n_eff=0, trust=0.99),       # 1 beats 2 and 3
        _row(family="detector", safety_call=True, trust=0.05),              # 2: low trust, still used
        _row(n_eff=0, trust=0.05),                                          # 3: no record -> use despite low trust
        _row(trust=0.8),                                                    # 4: p >= tau_use
        _row(trust=0.6, band_scale=1.4),                                    # 5: forecast in [tau_widen, tau_use)
        _row(family="detector", trust=0.6),                                 # 6: detectors are never widened
        _row(trust=0.2, sub_trust=0.7),                                     # 6 + substitute better
        _row(trust=0.2, sub_trust=0.1),                                     # 6, substitute worse -> none
    ]
    _, out = _judge(rows)
    assert list(out["rule"]) == [RULE_BAD_INPUT, RULE_SAFETY, RULE_NO_RECORD, RULE_USE, RULE_WIDEN, RULE_DONT_USE,
                                 RULE_DONT_USE, RULE_DONT_USE]
    assert list(out["verdict"]) == [DONT_USE, USE, USE, USE, WIDEN, DONT_USE, DONT_USE, DONT_USE]
    assert out["reasons"][0][0].startswith("input unreliable")
    assert "low trust" in out["reasons"][1][0]
    assert out["band_scale"][4] == pytest.approx(1.4 * 1.25)  # bucket band scale x widen factor
    assert out["band_scale"][3] == pytest.approx(1.0)
    assert out["substitute"][6] == "lgbm_pv" and out["reasons"][6][0] == "use lgbm_pv instead"
    assert out["substitute"][7] is None and out["reasons"][7][0] == "no reliable source"
    assert out["substitute"][5] is None  # (sub_trust unknown)
    assert set(OUTPUT_COLUMNS) <= set(out.columns)
    assert all(len(r) <= 3 for r in out["reasons"])


def test_safety_pass_through_is_never_suppressed():
    """An icing / overheating call passes as use whatever P(valid), trust or substitute say (only a failed input
    gate, rule 1, comes first and says so)."""
    for p, trust, sub in ((0.0, 0.0, 0.99), (0.01, 0.3, 0.9), (np.nan, 0.05, 1.0), (0.95, 0.9, 0.1)):
        _, out = _judge([_row(family="detector", safety_call=True, p_valid=p, trust=trust, sub_trust=sub)])
        assert out["verdict"][0] == USE and out["rule"][0] == RULE_SAFETY and out["substitute"][0] is None


def test_no_record_means_use_even_if_the_model_disagrees():
    th = Thresholds(tau_use=0.9, tau_widen=0.8, judge_version="judge-test")
    _, out = _judge([_row(n_eff=0, trust=np.nan, p_valid=0.01), _row(n_eff=np.nan, trust=np.nan)], th)
    assert list(out["verdict"]) == [USE, USE] and list(out["rule"]) == [RULE_NO_RECORD, RULE_NO_RECORD]
    assert out["p_valid"][0] == pytest.approx(0.01)  # the model's opinion is shown, it does not override


def test_ledger_only_mode_uses_trust_as_p_valid():
    _, out = _judge([_row(trust=0.75), _row(trust=0.4, family="detector")])
    assert list(out["p_valid"]) == [0.75, 0.4]
    assert list(out["judge_version"]) == [LEDGER_ONLY, LEDGER_ONLY]
    assert out["p_valid_sd"].isna().all()
    assert list(out["verdict"]) == [USE, DONT_USE]


def test_judge_thresholds_per_family_and_model_rows_versioned():
    th = Thresholds(tau_use={"forecast": 0.5, "detector": 0.95}, tau_widen=0.3, widen_factor={"forecast": 2.0},
                    judge_version="judge-r1")
    rows = [_row(p_valid=0.6, p_valid_sd=0.02), _row(family="detector", p_valid=0.9), _row(p_valid=0.4),
            _row(p_valid=np.nan, trust=0.55)]
    _, out = _judge(rows, th)
    assert list(out["verdict"]) == [USE, DONT_USE, WIDEN, USE]
    assert out["band_scale"][2] == pytest.approx(2.0)
    assert list(out["judge_version"]) == ["judge-r1", "judge-r1", "judge-r1", LEDGER_ONLY]  # NaN -> ledger trust
    assert out["p_valid_sd"][0] == pytest.approx(0.02)


def test_verdict_rows_match_the_judge_verdicts_table():
    df = pd.DataFrame([_row(trust=0.8)]).assign(day="2025-06-02", t0="2025-06-01T21:00:00Z", asset_id="1042",
                                                witness="pvnet", model_version="v1", bucket="site")
    out = apply_rules(df)
    rows = verdict_rows(df, "r1", out)
    assert set(rows[0]) == {"run_id", "day", "t0", "asset_id", "model", "model_version", "bucket", "verdict",
                            "p_valid", "p_valid_sd", "trust", "n_eff", "band_scale", "substitute", "reasons_json",
                            "rule", "judge_version"}
    assert rows[0]["verdict"] == USE and isinstance(rows[0]["reasons_json"], list)


# --------------------------------------------------------------------------------------------- ledger / priors
def test_prior_ledger_reads_priors_with_backoff(tmp_path):
    (tmp_path / "pvnet.s0.calibration.json").write_text(json.dumps({"model": "pvnet", "priors": {
        "version": 1, "scores": {"calibration": {"*": {"prior": 0.7, "n": 100},
                                                 "site|unseen": {"prior": 0.4, "n": 30}}}}}))
    (tmp_path / "judge_priors.json").write_text(json.dumps({"models": {"lgbm_pv": {"scores": {
        "calibration": {"*": {"prior": 0.65, "n": 50}}}}}}))
    led = S.PriorLedger(tmp_path)
    full = "site|unseen|batt|warm|clean|ood_lo"
    assert led.lookup("pvnet", "x", "calibration", full)["trust"] == pytest.approx(0.4)
    assert led.lookup("pvnet", "x", "calibration", "site|seen|batt|warm|clean|ood_lo")["trust"] == pytest.approx(0.7)
    assert led.lookup("lgbm_pv", None, "calibration", full)["trust"] == pytest.approx(0.65)
    r = led.lookup("loadnet", "x", "calibration", full)
    assert math.isnan(r["trust"]) and r["n_eff"] == 0  # no prior known: no record, trust the model (rule 3)


# --------------------------------------------------------------------------------------------- input gate
def test_frozen_gate_ignores_clipping_and_night():
    x = np.zeros((3, 96))
    x[:, 30:70] = np.linspace(0.5, 3.0, 40)
    x[0, 40:50] = 1.7  # frozen mid-curve
    x[1, 50:66] = 3.0  # flat at the day's max = clipping
    x[1, 30:50] = np.linspace(0.5, 2.9, 20)
    ok = np.ones_like(x, bool)
    assert list(S._runs_frozen(x, ok)) == [True, False, False]


class _FakeJudge:
    tau_use, tau_widen, widen_factor, judge_version = 0.7, 0.5, 1.3, "judge-fake"

    def __init__(self):
        self.columns = None

    def predict(self, rows):
        self.columns = list(rows.columns)
        return pd.DataFrame({"p_valid": np.full(len(rows), 0.1), "p_valid_sd": 0.05,
                             "reasons": [["disagrees with pvlib by 58 %"]] * len(rows)}, index=rows.index)


class _LowLedger:
    """J2-shaped ledger: everything has evidence and low trust; records updates."""

    def __init__(self):
        self.updates = []

    def lookup(self, model, model_version, score_name, bucket, day=None):
        return {"trust": 0.2, "n_eff": 9.0, "prior": 0.8, "lo80": 0.1, "hi80": 0.3, "band_scale": 1.5,
                "health_alert": True}

    def update(self, outcomes):
        self.updates.append(len(outcomes))


def _small_world(days=2):
    return MockWorld(days=days, scale=0.3)


def _run(world, judge_slot=None, judge=True, sink=None):
    comps = L.default_components(world, force_fallback=True, judge=False)
    comps.judge = judge_slot
    sink = sink if sink is not None else L._MemSink()
    res = L.run_policy(world, comps, L.LoopConfig(policy="sentinel", run_id="t", judge=judge), sink=sink)
    return res, sink


def test_slot_with_a_judge_model_ledger_and_grader(monkeypatch):
    w = _small_world()
    fake = _FakeJudge()
    graded = []

    def grader(ctx):
        graded.append(ctx["d"])
        assert set(ctx["predictions"]) >= {"pvnet", "loadnet", "sitehealth", "windnbm"} and ctx["rows"] is not None
        sid = sorted(ctx["observed"])[0]
        return pd.DataFrame([{"model": "pvnet", "model_version": "mock-1", "asset_id": sid, "day": ctx["day"],
                              "score_name": "calibration", "score": 0.9, "weight": 1.0, "source": "telemetry",
                              "observed_at": "2025-06-03T21:00:00Z", "gated": 0, "bucket": "site", "kind": "site"}])

    slot = S.JudgeSlot(judge=fake, model_status="loaded", grader=grader)
    led = _LowLedger()
    monkeypatch.setattr(S, "_load_ledger", lambda d: S._LedgerAdapter(led, S.PriorLedger(d)))
    res, sink = _run(w, slot)
    v = pd.DataFrame(sink.rows["judge_verdicts"])
    assert set(v["model"]) >= {"pvnet", "loadnet", "sitehealth", "windnbm"}
    n_assets = len(w.sites) * 3 + len(w.turbines)
    assert len(v[v["day"] == v["day"].min()]) >= n_assets  # every witness prediction has a verdict
    model_rows = v[v["model"] != "gridtwin"]
    assert (model_rows["judge_version"] == "judge-fake").all()
    assert (v.loc[v["model"] == "gridtwin", "judge_version"] == LEDGER_ONLY).all()  # M8: ledger trust only
    fc = model_rows[model_rows["model"].isin(["pvnet", "loadnet"])]
    assert (fc["verdict"] == DONT_USE).all()  # p 0.1 < tau_widen; sub trust 0.2 > 0.1 -> substitute named
    assert set(fc["substitute"]) == {"lgbm_pv", "lgbm_load"}
    assert any("disagrees with pvlib" in r for rs in fc["reasons_json"] for r in rs)
    assert graded == [0, 1] and led.updates == [1, 1]
    assert len(sink.rows["outcomes"]) == 2 and "kind" not in sink.rows["outcomes"][0]
    assert sink.rows["trust"] and all(r["health_alert"] == 1 for r in sink.rows["trust"])
    assert {"ood_score", "trust", "n_eff", "input_bad", "park_consensus", "bucket"} <= set(fake.columns)
    assert res.judge["mode"] == "judge" and set(res.judge["t_judge_day_s"]) == {0, 1}


def test_slot_safety_call_passes_with_low_trust(monkeypatch):
    w = _small_world(days=1)
    slot = S.JudgeSlot()
    monkeypatch.setattr(S, "_load_ledger", lambda d: S._LedgerAdapter(_LowLedger(), S.PriorLedger(d)))
    slot.start_run(w, "t", L._MemSink())
    tid = sorted(w.turbines)[0]
    base = w.default_predictors()["wind"].predict(w.turbines[tid].row, L.Window(t0=w.day_t0(0), horizon=96))

    class P:
        anomaly_class, anomaly_score = "icing", 0.4

    for k in range(96):
        slot.observe_wind_tick(0, k, {tid: P() if k > 50 else base})
    tday = {tid: w.turbine_day(tid, 0)}
    judged = slot.at_end_of_day(0, w.day_t0(0), {}, tday, {tid: base}, {}, {}, {})
    r = judged.iloc[0]
    assert r["anomaly_class"] == "icing" and bool(r["safety_call"])
    if not r["input_bad"]:
        assert r["verdict"] == USE and r["rule"] == RULE_SAFETY and "low trust" in r["reasons"][0]


def test_stale_telemetry_is_rule_one(monkeypatch):
    w = _small_world(days=1)
    slot = S.JudgeSlot()
    slot.start_run(w, "t", L._MemSink())
    preds = w.default_predictors()
    t0 = w.day_t0(0)
    sid = sorted(w.sites)[0]
    pv = {sid: preds["pv"].predict(w.sites[sid].row, L.Window(t0=t0, horizon=96))}
    judged = slot.at_t0(0, t0, pv, {}, {}, {sid: 7200.0})
    r = judged.iloc[0]
    assert r["verdict"] == DONT_USE and r["rule"] == RULE_BAD_INPUT and "telemetry stale" in r["reasons"][0]


# --------------------------------------------------------------------------------------------- acceptance
def test_judge_on_and_off_give_identical_decisions():
    w = _small_world(days=2)
    on, s_on = _run(w, S.JudgeSlot.load(), judge=True)
    off, s_off = _run(w, S.JudgeSlot.load(), judge=False)
    assert "judge_verdicts" in s_on.rows and "judge_verdicts" not in s_off.rows
    for f in ("feeder_days", "comfort_breaches", "agent_steps", "clipped_steps", "cost_lei", "hits",
              "commit_steps", "turbine_alerts", "n_actions", "action_ids", "plan_status", "input_hashes"):
        assert getattr(on, f) == getattr(off, f), f
    for t in set(s_off.rows):
        assert s_on.rows[t] == s_off.rows[t], t  # every row the agent writes, not only the KPIs


def test_replay_judge_check_and_tables(tmp_path):
    rep = replay(["baseline", "sentinel"], tmp_path / "j.sqlite", world=MockWorld(days=2, scale=0.3),
                 force_fallback=True, print_metrics=False, strict=True, judge_check=True)
    assert rep.judge_check["identical"] and rep.judge_check["kpi_diffs"] == {}
    import sqlite3

    con = sqlite3.connect(rep.out)
    n = con.execute("select count(*) from judge_verdicts").fetchone()[0]
    assert n > 0 and con.execute("select count(*) from trust").fetchone()[0] > 0
    notes = json.loads(con.execute("select notes from runs where policy='sentinel'").fetchone()[0])
    assert notes["judge"]["mode"] == "ledger-only" and notes["judge"]["verdicts"]["use"] == n
    assert "judge" not in notes["fallbacks"]
    con.close()


def test_db_without_judge_tables_still_replays(tmp_path):
    """Old schema (the #38 copy, no judge tables), strict mode: the slot skips the tables, nothing raises."""
    rep = replay(["sentinel"], tmp_path / "old.sqlite", world=MockWorld(days=1, scale=0.3),
                 schema_dir=tmp_path / "no-drizzle", force_fallback=True, print_metrics=False, strict=True)
    s = rep.results["sentinel"].judge
    assert set(s["tables_missing"]) == {"judge_verdicts", "trust", "outcomes"} and s["verdicts"]["use"] > 0
