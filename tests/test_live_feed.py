"""Card #128: live 15-min loop on the mock world (team/contracts/live.md)."""

from __future__ import annotations

import dataclasses
import sqlite3

import pytest

from sentinel.agent.world import MockWorld
from sentinel.live import store
from sentinel.live.feed import LiveFeed, map_wall, parse, realtime_step, run_speed
from sentinel.live.forecast import cos_zenith, err_flag, flag, iso
from sentinel.live.source import SimulatedSource

START = parse("2025-06-04T08:00:00Z")  # hero day morning: PV ramping up
STEP = 900


def _world(cls=MockWorld, **kw):
    return cls(days=7, scale=0.25, with_wind=False, **kw)


def _feed(tmp_path, name="live.sqlite", world=None, run_id="t"):
    w = world or _world()
    src = SimulatedSource(w, ["F7", "F5"])
    con = store.connect(tmp_path / name)
    return LiveFeed(src, con, run_id, world="mock", clock="speed:0"), con


def _rows(con, sql, *args):
    return [dict(r) for r in con.execute(sql, args)]


def test_three_ticks_store_readings_predictions_and_errors(tmp_path):
    feed, con = _feed(tmp_path)
    assert run_speed(feed, START, 0, max_ticks=3, log=lambda s: None) == 3
    ticks = _rows(con, "SELECT ts, status, n_readings FROM live_ticks ORDER BY ts")
    assert [t["ts"] for t in ticks] == [iso(START + i * STEP) for i in range(3)]
    n_sites = len(feed.src.sites())
    assert all(t["status"] == "ok" and t["n_readings"] > 0 for t in ticks)
    for i in range(3):
        r = START + i * STEP
        assert con.execute("SELECT COUNT(*) FROM live_readings WHERE ts = ?", (iso(r),)).fetchone()[0] == n_sites
        # a prediction for the next 15 min per home and per feeder, each quantity and method
        nxt = iso(r + STEP)
        n_site_rows = con.execute("SELECT COUNT(*) FROM live_predictions WHERE ts = ? AND level = 'site'",
                                  (nxt,)).fetchone()[0]
        assert n_site_rows == n_sites * 3 * 3
        feeders = _rows(con, "SELECT * FROM live_predictions WHERE ts = ? AND level = 'feeder' AND quantity = 'net' "
                             "AND method = 'corrected'", nxt)
        assert {f["entity_id"] for f in feeders} == {"F7", "F5"}
        assert all(f["limit_kw"] > 0 and f["flag"] in ("none", "watch", "congested", "severe") for f in feeders)
        assert all(f["q10"] <= f["q50"] <= f["q90"] for f in feeders)
    # the predictions made at ticks 1 and 2 were scored when the next reading arrived; the last one is still open
    scored = _rows(con, "SELECT ts, COUNT(*) n, SUM(actual IS NOT NULL) a FROM live_predictions GROUP BY ts ORDER BY ts")
    assert [s["ts"] for s in scored] == [iso(START + i * STEP) for i in (1, 2, 3)]
    assert scored[0]["a"] > 0 and scored[1]["a"] > 0 and scored[2]["a"] == 0
    f7 = _rows(con, "SELECT * FROM live_predictions WHERE ts = ? AND level = 'feeder' AND entity_id = 'F7' "
                    "AND quantity = 'net' AND method = 'model'", iso(START + STEP))[0]
    sites = [r["net_kw"] for r in _rows(con, "SELECT net_kw FROM live_readings WHERE ts = ? AND feeder_id = 'F7'",
                                        iso(START + STEP))]
    if all(x is not None for x in sites):
        assert f7["actual"] == pytest.approx(sum(sites), abs=1e-3)
        assert f7["err"] == pytest.approx(f7["actual"] - f7["q50"], abs=1e-3)
    run = _rows(con, "SELECT * FROM live_runs")[0]
    assert run["source"] == "simulated" and "simulated" in run["label"]


class _FutureChanged(MockWorld):
    """Same world, but every reading at or after CUT is different. Predictions made before CUT must not move."""

    CUT = START + 3 * STEP

    def site_day(self, site_id, d):
        day = super().site_day(site_id, d)
        t0 = self.day_ts(d)
        pv = [x * 3 + 1 if t0 + k * STEP >= self.CUT else x for k, x in enumerate(day.pv_kw)]
        ld = [x * 2 + 1 if t0 + k * STEP >= self.CUT else x for k, x in enumerate(day.load_kw)]
        return dataclasses.replace(day, pv_kw=pv, load_kw=ld)


def test_no_prediction_reads_a_future_reading(tmp_path):
    a, con_a = _feed(tmp_path, "a.sqlite")
    b, con_b = _feed(tmp_path, "b.sqlite", world=_world(_FutureChanged))
    run_speed(a, START, 0, max_ticks=3, log=lambda s: None)
    run_speed(b, START, 0, max_ticks=3, log=lambda s: None)
    q = ("SELECT ts, t0, level, entity_id, quantity, method, q10, q50, q90, flag FROM live_predictions "
         "ORDER BY ts, level, entity_id, quantity, method")
    pa, pb = _rows(con_a, q), _rows(con_b, q)
    assert pa and pa == pb  # the last tick predicted ts = CUT with only readings before it
    assert max(p["ts"] for p in pa) == iso(_FutureChanged.CUT)
    # and the changed future does show up once it has happened
    run_speed(a, START, 0, max_ticks=1, log=lambda s: None)
    run_speed(b, START, 0, max_ticks=1, log=lambda s: None)
    ra = _rows(con_a, "SELECT pv_kw FROM live_readings WHERE ts = ? ORDER BY site_id", iso(_FutureChanged.CUT))
    rb = _rows(con_b, "SELECT pv_kw FROM live_readings WHERE ts = ? ORDER BY site_id", iso(_FutureChanged.CUT))
    assert ra != rb
    # every prediction's information cut-off is after every reading it could have used
    for p in pa:
        assert parse(p["t0"]) <= parse(p["ts"])
        used = con_a.execute("SELECT MAX(ts) FROM live_readings WHERE ts < ?", (p["t0"],)).fetchone()[0]
        assert parse(used) + STEP <= parse(p["t0"])


def test_restart_resumes_without_duplicates_and_keeps_battery_state(tmp_path):
    feed, con = _feed(tmp_path, "r.sqlite")
    run_speed(feed, START, 0, max_ticks=2, log=lambda s: None)
    con.close()
    feed2, con2 = _feed(tmp_path, "r.sqlite")  # new process: fresh world and source, same DB and run
    assert feed2.last == START + STEP
    assert feed2.tick(START)["status"] == "skipped"
    run_speed(feed2, START, 0, max_ticks=2, log=lambda s: None)
    ts = [r["ts"] for r in _rows(con2, "SELECT ts FROM live_ticks ORDER BY ts")]
    assert ts == [iso(START + i * STEP) for i in range(4)]
    assert con2.execute("SELECT COUNT(*) FROM live_runs").fetchone()[0] == 1
    # same readings as one uninterrupted run (battery soc carried over the restart)
    ref, con_ref = _feed(tmp_path, "ref.sqlite")
    run_speed(ref, START, 0, max_ticks=4, log=lambda s: None)
    q = "SELECT ts, site_id, pv_kw, load_kw, batt_kw, soc, net_kw FROM live_readings ORDER BY ts, site_id"
    for x, y in zip(_rows(con2, q), _rows(con_ref, q)):
        assert x.keys() == y.keys()
        for k in x:
            assert x[k] == pytest.approx(y[k], abs=1e-6) if isinstance(x[k], float) else x[k] == y[k]


def test_realtime_maps_wall_clock_and_backfills_gaps(tmp_path):
    feed, con = _feed(tmp_path, "rt.sqlite")
    span = feed.src.span()
    wall = parse("2026-11-11T10:07:00Z")  # a Wednesday
    r0 = map_wall(parse("2026-11-11T10:00:00Z"), span) - STEP
    assert span[0] <= r0 < span[1] and iso(r0).endswith("09:45:00Z")
    assert realtime_step(feed, wall, log=lambda s: None) == 1
    assert realtime_step(feed, wall + 60, log=lambda s: None) == 0  # same step again: nothing new
    assert realtime_step(feed, wall + 3 * STEP, log=lambda s: None) == 3  # down for two ticks
    ticks = _rows(con, "SELECT ts, status FROM live_ticks ORDER BY ts")
    assert [t["status"] for t in ticks] == ["ok", "missed", "missed", "ok"]
    assert [parse(t["ts"]) for t in ticks] == [r0 + i * STEP for i in range(4)]
    for t in ticks[1:3]:  # missed steps: readings backfilled, nobody predicted the step after them
        assert con.execute("SELECT COUNT(*) FROM live_readings WHERE ts = ?", (t["ts"],)).fetchone()[0] > 0
        nxt = iso(parse(t["ts"]) + STEP)
        assert con.execute("SELECT COUNT(*) FROM live_predictions WHERE t0 = ?", (nxt,)).fetchone()[0] == 0


def test_flags_and_helpers():
    assert flag(0, 50, 80, 100) == "none"
    assert flag(0, 92, 95, 100) == "watch"
    assert flag(0, 80, 120, 100) == "watch"
    assert flag(50, 110, 150, 100) == "congested"
    assert flag(105, 110, 150, 100) == "severe"
    assert flag(0, 135, 150, 100) == "severe"
    assert err_flag(10.0, 4, 5, 6) == 1 and err_flag(5.5, 4, 5, 6) == 0 and err_flag(6.5, 4, 5, 6) == 0
    lat, lon = 47.16, 27.59  # Iasi
    assert cos_zenith(parse("2025-06-21T10:00:00Z"), lat, lon) > 0.8  # ~13:00 local
    assert cos_zenith(parse("2025-06-21T22:00:00Z"), lat, lon) < 0


def test_wal_lets_a_reader_see_committed_ticks(tmp_path):
    feed, con = _feed(tmp_path, "w.sqlite")
    reader = sqlite3.connect(f"file:{tmp_path / 'w.sqlite'}?mode=ro", uri=True)
    assert con.execute("PRAGMA journal_mode").fetchone()[0] == "wal"
    run_speed(feed, START, 0, max_ticks=1, log=lambda s: None)
    assert reader.execute("SELECT COUNT(*) FROM live_ticks").fetchone()[0] == 1
