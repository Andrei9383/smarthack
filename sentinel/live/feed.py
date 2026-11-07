"""LiveFeed: one 15-min tick = one transaction (card #128, team/contracts/live.md).

At tick T the step r = T - 15 min has just ended: its readings arrive, the prediction made for r gets its actual and
error, and a prediction for T (the next 15 min) is stored with t0 = T. Ticks are keyed by (run_id, ts), so a restart
resumes after the last committed tick and never writes one twice. In real-time mode the wall clock is mapped onto the
world's week by whole weeks; steps missed while the process was down are backfilled as `missed` ticks (readings,
no prediction: nobody predicted them at the time).
"""

from __future__ import annotations

import json
import sqlite3
import time
from datetime import datetime, timezone
from typing import Any, Callable, Optional

from . import store
from .forecast import DayAhead, err_flag, feeder_rows, iso, site_rows
from .source import STEP_S, Reading

MAX_BACKFILL = 96  # steps (24 h)


def parse(ts: str) -> int:
    return int(datetime.fromisoformat(ts.replace("Z", "+00:00")).timestamp())


def map_wall(wall_ts: int, span: tuple[int, int]) -> int:
    """Wall clock -> world axis, shifted by whole periods (a 7-day world: same weekday and UTC time of day)."""
    lo, hi = span
    return lo + (wall_ts - lo) % (hi - lo)


class LiveFeed:
    def __init__(self, source: Any, con: sqlite3.Connection, run_id: str, *, world: str = "sim",
                 clock: str = "realtime", dayahead: Optional[DayAhead] = None, notes: str = ""):
        self.src, self.con, self.run_id = source, con, run_id
        self.da = dayahead or DayAhead(source.world, source.sites())
        feeders = [{"feeder_id": f, "limit_kw": round(float(x.spec.g_kw), 3),
                    "site_ids": [s for s in x.site_ids if s in source.sites()]} for f, x in source.feeders().items()]
        con.execute(
            "INSERT OR IGNORE INTO live_runs (run_id, source, label, world, tick_s, clock, feeders_json, "
            "model_versions_json, notes, created_at) VALUES (?,?,?,?,?,?,?,?,?,?)",
            (run_id, "simulated", source.label, world, STEP_S, clock, json.dumps(feeders),
             json.dumps(self.da.versions()), notes, iso(int(time.time()))))
        row = con.execute("SELECT MAX(ts) FROM live_ticks WHERE run_id = ?", (run_id,)).fetchone()
        self.last: Optional[int] = parse(row[0]) if row and row[0] else None
        self.prev: dict[str, Reading] = {}
        if self.last is not None:  # restart: latest readings feed the persistence / residual and the battery state
            for r in con.execute("SELECT * FROM live_readings WHERE run_id = ? AND ts = ?", (run_id, iso(self.last))):
                self.prev[r["site_id"]] = Reading(r["site_id"], r["feeder_id"], r["pv_kw"], r["load_kw"],
                                                  r["batt_kw"], r["soc"], r["net_kw"])
            if hasattr(source, "restore"):
                socs = {sid: soc for sid, soc in con.execute(
                    "SELECT site_id, soc FROM live_readings WHERE run_id = ? AND soc IS NOT NULL AND ts = ("
                    "SELECT MAX(ts) FROM live_readings r2 WHERE r2.run_id = live_readings.run_id AND "
                    "r2.site_id = live_readings.site_id AND r2.soc IS NOT NULL)", (run_id,))}
                source.restore(socs)

    # -- one tick ------------------------------------------------------------------------------------------------
    def tick(self, r: int, *, missed: bool = False) -> dict[str, Any]:
        """Handle the step starting at r (it has just ended). Returns a small summary."""
        if self.last is not None and r <= self.last:
            return {"ts": iso(r), "status": "skipped"}
        readings = self.src.readings(r)
        got = {s: x for s, x in readings.items() if x.pv_kw is not None}
        status = "missed" if missed else ("ok" if got else "empty")
        con = self.con
        con.execute("BEGIN IMMEDIATE")
        try:
            store.insert_many(con, "live_readings", store.READ_COLS,
                              [{**x.row(), "run_id": self.run_id, "ts": iso(r)} for x in readings.values()])
            self._score(r, readings)
            n_pred = 0
            if not missed:
                T = r + STEP_S
                rows: list[dict] = []
                for sid, site in self.src.sites().items():
                    last = got.get(sid)
                    e = self.src.e.get(sid, 0.0) if hasattr(self.src, "e") else 0.0
                    rows += site_rows(self.da, sid, site, T, last, e)
                rows += feeder_rows(rows, self.src.feeders(), T)
                n_pred = store.insert_many(con, "live_predictions", store.PRED_COLS,
                                           [{**x, "run_id": self.run_id} for x in rows])
            con.execute("INSERT OR IGNORE INTO live_ticks (run_id, ts, wall_at, status, n_readings) VALUES (?,?,?,?,?)",
                        (self.run_id, iso(r), iso(int(time.time())), status, len(got)))
            con.execute("COMMIT")
        except BaseException:
            con.execute("ROLLBACK")
            raise
        self.last, self.prev = r, got
        return {"ts": iso(r), "status": status, "n_readings": len(got), "n_predictions": n_pred}

    def _score(self, r: int, readings: dict[str, Reading]) -> None:
        """Fill actual / err / err_flag on the predictions made for step r."""
        ts = iso(r)
        site_actual = {s: {"pv": x.pv_kw, "load": x.load_kw, "net": x.net_kw} for s, x in readings.items()}
        feeder_actual: dict[str, dict[str, Optional[float]]] = {}
        for fid, f in self.src.feeders().items():
            members = [site_actual.get(s) for s in f.site_ids if s in self.src.sites()]
            feeder_actual[fid] = {
                q: (sum(m[q] for m in members) if members and all(m is not None and m[q] is not None for m in members)
                    else None) for q in ("pv", "load", "net")}
        upd = []
        for p in self.con.execute("SELECT level, entity_id, quantity, method, q10, q50, q90, n_sites FROM "
                                  "live_predictions WHERE run_id = ? AND ts = ?", (self.run_id, ts)):
            src = site_actual if p["level"] == "site" else feeder_actual
            a = (src.get(p["entity_id"]) or {}).get(p["quantity"])
            if a is None:
                continue
            upd.append((round(a, 4), round(a - p["q50"], 4), err_flag(a, p["q10"], p["q50"], p["q90"], p["n_sites"]),
                        self.run_id, ts, p["level"], p["entity_id"], p["quantity"], p["method"]))
        self.con.executemany("UPDATE live_predictions SET actual = ?, err = ?, err_flag = ? WHERE run_id = ? AND ts = ? "
                             "AND level = ? AND entity_id = ? AND quantity = ? AND method = ?", upd)


def summary(con: sqlite3.Connection, run_id: str) -> list[dict[str, Any]]:
    """Scored feeder rows per quantity / method: n, MAE kW, nMAE (MAE / mean |actual|), 10-90 coverage, large misses."""
    rows = con.execute(
        "SELECT quantity, method, COUNT(*) n, AVG(ABS(err)) mae, AVG(ABS(actual)) lvl, "
        "AVG(CASE WHEN actual BETWEEN q10 AND q90 THEN 1.0 ELSE 0.0 END) cov, SUM(err_flag) big "
        "FROM live_predictions WHERE run_id = ? AND level = 'feeder' AND actual IS NOT NULL "
        "GROUP BY quantity, method ORDER BY quantity, method", (run_id,)).fetchall()
    return [{"quantity": r["quantity"], "method": r["method"], "n": r["n"], "mae_kw": round(r["mae"], 3),
             "nmae": round(r["mae"] / r["lvl"], 3) if r["lvl"] else None, "coverage_10_90": round(r["cov"], 3),
             "large_misses": r["big"]} for r in rows]


# -- drivers ----------------------------------------------------------------------------------------------------
def run_speed(feed: LiveFeed, start: int, seconds: float, max_ticks: Optional[int] = None,
              sleep: Callable[[float], None] = time.sleep, log: Callable[[str], None] = print) -> int:
    """Virtual clock: one tick every `seconds` from `start` (or after the last stored tick); stops at the world's end."""
    lo, hi = feed.src.span()
    r = feed.last + STEP_S if feed.last is not None else max(lo, start - start % STEP_S)
    n = 0
    while r < hi and (max_ticks is None or n < max_ticks):
        log(json.dumps(feed.tick(r)))
        n += 1
        r += STEP_S
        if seconds > 0 and r < hi and (max_ticks is None or n < max_ticks):
            sleep(seconds)
    return n


def realtime_step(feed: LiveFeed, wall_ts: int, log: Callable[[str], None] = print) -> int:
    """One real-time wake-up: backfill missed steps, then tick the step that just ended. Returns ticks written."""
    span = feed.src.span()
    r = map_wall(wall_ts - wall_ts % STEP_S, span) - STEP_S
    if r < span[0]:
        r += span[1] - span[0]
    if feed.last is not None and r < feed.last:
        raise RuntimeError(f"clock wrapped to the start of the world ({iso(r)} < {iso(feed.last)}): "
                           "start a new --run-id")
    n = 0
    if feed.last is not None:
        missed = list(range(feed.last + STEP_S, r, STEP_S))[-MAX_BACKFILL:]
        for m in missed:
            log(json.dumps(feed.tick(m, missed=True)))
            n += 1
    if feed.last != r:
        log(json.dumps(feed.tick(r)))
        n += 1
    return n


def run_realtime(feed: LiveFeed, max_ticks: Optional[int] = None, now: Callable[[], float] = time.time,
                 sleep: Callable[[float], None] = time.sleep, log: Callable[[str], None] = print,
                 delay_s: float = 5.0) -> int:
    n = 0
    while max_ticks is None or n < max_ticks:
        n += realtime_step(feed, int(now()), log)
        t = now()
        sleep(max(1.0, STEP_S - t % STEP_S + delay_s))
    return n
