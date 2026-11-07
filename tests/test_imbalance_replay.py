"""Imbalance replay (card #176): settlement of every policy, 18-21 double cover, day alignment, output tables."""

import sqlite3

import numpy as np
import pandas as pd
import pytest

from sentinel.eval import imbalance_replay as R

A, B = 1.0, 0.5


def _day(date="2025-06-02"):
    """One UTC day of 15-min steps (Bucharest = UTC+3 in June: local 18-21 = UTC 15-18 = steps 60..71)."""
    return pd.date_range(f"{date} 00:00", periods=96, freq="15min", tz="UTC")


def test_perfect_forecast_costs_nothing():
    ts = _day()
    actual = np.sin(np.arange(96) / 10)
    exp = np.clip(-actual, 0, None)
    s, steps = R.evaluate(ts, actual, exp, {"p": (actual.copy(), None)}, np.full(96, 400.0), a=A, b=B)
    assert s["p"]["avoidable_cost_lei"] == pytest.approx(0.0)
    assert s["p"]["abs_dev_mwh"] == pytest.approx(0.0)
    assert s["p"]["double_cover_mwh_18_21"] == 0.0
    assert np.allclose(steps["p"]["dev"], 0.0)


def test_hand_case_short_long_and_double_cover():
    ts = _day()
    actual = np.ones(96)
    bought = np.ones(96)
    bought[10] = 0.0          # short 1 MWh at 03:30 local
    bought[62] = 3.0          # long 2 MWh at 18:30 local -> double cover
    pzu = np.full(96, 100.0)
    exp = np.zeros(96)
    exp[40:44] = 0.5          # 2 MWh injected in total
    s = R.evaluate(ts, actual, exp, {"p": (bought, None)}, pzu, a=A, b=B)[0]["p"]
    assert s["short_mwh"] == pytest.approx(1.0)
    assert s["long_mwh"] == pytest.approx(2.0)
    assert s["double_cover_mwh_18_21"] == pytest.approx(2.0)
    assert s["avoidable_cost_lei"] == pytest.approx(A * 100 * 1 + B * 100 * 2)
    assert s["lei_per_mwh_injected"] == pytest.approx(s["avoidable_cost_lei"] / 2.0)
    assert s["bias_mwh"] == pytest.approx(1.0)  # bought 1 MWh more than needed overall
    assert s["position_nmae"] == pytest.approx(3.0 / 96)


def test_intraday_trade_settles_at_pi():
    ts = _day()
    actual = np.ones(96)
    bought = np.zeros(96)
    traded = np.ones(96)      # the PI trade covers everything
    pzu, pi = np.full(96, 100.0), np.full(96, 130.0)
    s, _ = R.evaluate(ts, actual, np.zeros(96), {"id": (bought, traded)}, pzu, pi, a=A, b=B)
    assert s["id"]["abs_dev_mwh"] == pytest.approx(0.0)
    assert s["id"]["intraday_lei"] == pytest.approx(96 * 30.0)
    assert s["id"]["avoidable_cost_lei"] == pytest.approx(96 * 30.0)


def test_reduction_vs_profile():
    summ = {"profile": {"avoidable_cost_lei": 200.0}, "sentinel": {"avoidable_cost_lei": 50.0}}
    assert R.reduction_vs(summ, "sentinel") == pytest.approx(0.75)
    assert R.reduction_vs(summ, "profile") == pytest.approx(0.0)
    assert np.isnan(R.reduction_vs(summ, "missing"))


def test_align_days_drops_partial_days():
    year = pd.date_range("2025-01-01 00:00", "2025-01-03 11:45", freq="15min", tz="UTC")
    days = np.stack([_day("2025-01-01").tz_localize(None).to_numpy(), _day("2025-01-03").tz_localize(None).to_numpy()])
    keep, idx = R.align_days(days, year)
    assert keep.tolist() == [True, False]
    assert idx[0, 0] == 0 and idx[0, -1] == 95
    assert (idx[1, 48:] == -1).all()


def test_local_hour_dst():
    assert R.local_hour(np.array(["2025-01-15T16:00"], dtype="datetime64[ns]"))[0] == 18.0  # UTC+2
    assert R.local_hour(np.array(["2025-07-15T15:00"], dtype="datetime64[ns]"))[0] == 18.0  # UTC+3


def _tables(run_id="r1"):
    ts = _day()
    actual = np.ones(96)
    pos = {"profile": (np.full(96, 1.2), None), "sentinel": (np.ones(96), None),
           "sentinel_id": (np.ones(96), np.zeros(96))}
    pzu = np.full(96, 100.0)
    summ, steps = R.evaluate(ts, actual, np.zeros(96), pos, pzu, pzu, a=A, b=B)
    band = np.stack([0.9 * actual, actual, 1.1 * actual], axis=-1)
    mask = np.zeros(96, bool)
    mask[60:] = True
    bands = {("profile", "da"): (pos["profile"][0], None), ("sentinel", "da"): (band, None),
             ("sentinel", "id"): (band, mask)}
    extra = {p: {"avoidable_cost_reduction_vs_profile": R.reduction_vs(summ, p)} for p in summ}
    return R.build_tables(ts, actual, np.ones(96), np.zeros(96), pos, steps, pzu, pzu, bands, summ, extra,
                          {"default": "synthetic RO"}, run_id)


def test_tables_follow_contract():
    t = _tables()
    for name, cols in R.TABLES.items():
        assert list(t[name].columns) == list(cols), name
    assert len(t["positions"]) == 3 * 96
    fc = t["portfolio_forecast"]
    assert len(fc) == 96 + 96 + 36
    assert fc[fc.policy == "profile"].p10.isna().all() and fc[fc.policy == "sentinel"].p10.notna().all()
    da = fc[(fc.policy == "sentinel") & (fc.horizon == "da")]
    assert (da.issue_time < da.ts).all()
    k = t["imbalance_kpis"]
    assert set(k.policy) == {"profile", "sentinel", "sentinel_id"}
    assert k[(k.policy == "profile") & (k.name == "avoidable_cost_lei")].value.iloc[0] == pytest.approx(
        B * 100 * 0.2 * 96)


def test_write_sqlite_replaces_only_its_tables(tmp_path):
    db = tmp_path / "x.sqlite"
    con = sqlite3.connect(db)
    con.execute("CREATE TABLE runs (id TEXT)")
    con.execute("INSERT INTO runs VALUES ('keep')")
    con.commit()
    con.close()
    R.write_sqlite(db, _tables("r1"))
    rows = R.write_sqlite(db, _tables("r2"))
    assert rows["positions"] == 3 * 96
    con = sqlite3.connect(db)
    assert con.execute("SELECT id FROM runs").fetchall() == [("keep",)]
    assert con.execute("SELECT DISTINCT run_id FROM positions").fetchall() == [("r2",)]
    ts0, = con.execute("SELECT ts FROM positions ORDER BY ts LIMIT 1").fetchone()
    assert ts0 == "2025-06-02T00:00:00Z"
    assert con.execute("SELECT COUNT(*) FROM portfolio_forecast WHERE p10 IS NULL").fetchone()[0] == 96
    con.close()


def test_replay_mode_dispatch(monkeypatch):
    from sentinel.agent import replay

    seen = {}
    monkeypatch.setattr(R, "main", lambda argv: seen.setdefault("argv", list(argv)) or "ok")
    replay.main(["--mode", "imbalance", "--k", "5", "--policy", "profile,sentinel"])
    assert seen["argv"] == ["--k", "5", "--policy", "profile,sentinel"]
    with pytest.raises(SystemExit):
        replay.main(["--mode", "nope"])
