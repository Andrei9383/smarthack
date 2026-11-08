"""The agent decides each live tick (card #129, EXPERIMENT, merge=never).

Not a new policy: the replay's own `run_policy` (wattsentry/agent/loop.py, policy `sentinel`, the code behind
replay#506) driven one tick at a time. It runs in a worker thread and stops after every tick (LoopConfig.on_tick)
until the live feed has handled the same 15-min step, so its decisions appear as the readings arrive:
- at local midnight: qualify + day-ahead feeder LP + P10 commitment (from the day-ahead forecasts, like #506);
- every tick: qualify on telemetry age / measured loss, triage (work orders, alerts, reviews), each pool battery's
  planned set-point clipped to what the battery can do now, feeder export vs limit.
Because it is the same code, fed the same #506 week it gives the same KPIs (the gate), and decisions only use what
has arrived (tests/test_live_agent.py perturbs the future and checks nothing before it moves). What it is not: a
re-plan from each new reading (the plan stays day-ahead + clip, optionally hourly re-solve); that would be a
different policy with its own KPIs.

Tables (same SQLite as #128, team/contracts/live.md "Agent tables"): live_agent_steps (per battery home and tick),
live_agent_feeders (per feeder and tick), live_agent_actions (work orders / alerts / reviews / schedules issued).
"""

from __future__ import annotations

import json
import queue
import sqlite3
import threading
from typing import Any, Optional

from wattsentry.agent.loop import LoopConfig, PolicyResult, _MemSink, default_components, run_policy

from . import store
from .forecast import iso

SCHEMA = """
CREATE TABLE IF NOT EXISTS live_agent_steps (
  run_id TEXT NOT NULL,
  ts TEXT NOT NULL,
  site_id TEXT NOT NULL,
  feeder_id TEXT NOT NULL,
  status TEXT,
  reason TEXT,
  in_pool INTEGER NOT NULL,
  planned_kw REAL,
  batt_kw REAL NOT NULL,
  soc REAL NOT NULL,
  PRIMARY KEY (run_id, ts, site_id)
);
CREATE TABLE IF NOT EXISTS live_agent_feeders (
  run_id TEXT NOT NULL,
  ts TEXT NOT NULL,
  feeder_id TEXT NOT NULL,
  net_kw REAL NOT NULL,
  limit_kw REAL NOT NULL,
  over INTEGER NOT NULL,
  committed_kw REAL,
  n_pool INTEGER NOT NULL,
  PRIMARY KEY (run_id, ts, feeder_id)
);
CREATE TABLE IF NOT EXISTS live_agent_actions (
  run_id TEXT NOT NULL,
  ts TEXT NOT NULL,
  action_id TEXT NOT NULL,
  action_type TEXT NOT NULL,
  asset_id TEXT,
  feeder_id TEXT,
  payload_json TEXT,
  PRIMARY KEY (run_id, action_id)
);
"""

_DONE = object()


class _TickSink(_MemSink):
    """Keeps only what the current tick produced; LiveAgent drains it into SQLite at each tick."""

    def __init__(self):
        super().__init__()

    def has_table(self, table):
        return False  # no asset_steps rows

    def drain(self) -> list:
        acts, self.actions, self.rows = self.actions, [], {}
        return acts


class LiveAgent:
    """`advance()` lets run_policy do one more tick and returns that tick's state (None when the week is over)."""

    def __init__(self, world: Any, con: sqlite3.Connection, run_id: str, *, comps: Any = None,
                 resolve_hourly: bool = False, judge: bool = False, feeders: Optional[list[str]] = None):
        self.world, self.con, self.run_id = world, con, run_id
        con.executescript(SCHEMA)
        self.feeders = set(feeders or world.feeders)
        self.comps = comps or default_components(world, judge=judge)
        self.sink = _TickSink()
        self.cfg = LoopConfig(policy="sentinel", run_id=run_id, resolve_hourly=resolve_hourly, write_inputs=False,
                              step_predictions=False, write_schedules=False, judge=judge, on_tick=self._on_tick)
        self._out: queue.Queue = queue.Queue(maxsize=1)
        self._go = threading.Semaphore(0)
        self._thread: Optional[threading.Thread] = None
        self.result: Optional[PolicyResult] = None
        self.error: Optional[BaseException] = None
        self.last_ts: Optional[int] = None
        self.done = False

    # -- worker thread ----------------------------------------------------------------------------------------
    def _on_tick(self, st: dict) -> None:
        st["actions"] = self.sink.drain()
        self._out.put(st)
        self._go.acquire()  # wait until the feed has handled this step

    def _run(self) -> None:
        try:
            self.result = run_policy(self.world, self.comps, self.cfg, sink=self.sink)
        except BaseException as e:  # surfaced in advance()
            self.error = e
        finally:
            self._out.put(_DONE)

    # -- driver -----------------------------------------------------------------------------------------------
    def advance(self) -> Optional[dict]:
        if self.done:
            return None
        if self._thread is None:
            self._thread = threading.Thread(target=self._run, name=f"live-agent-{self.run_id}", daemon=True)
            self._thread.start()
        else:
            self._go.release()
        st = self._out.get()
        if st is _DONE:
            self.done = True
            if self.error is not None:
                raise RuntimeError(f"live agent stopped: {self.error!r}") from self.error
            return None
        self.last_ts = st["ts"]
        self._write(st)
        return st

    def until(self, ts: int) -> int:
        """Advance through the step starting at ts (fast-forwards from the week start after a (re)start)."""
        n = 0
        while not self.done and (self.last_ts is None or self.last_ts < ts):
            if self.advance() is None:
                break
            n += 1
        return n

    def finish(self) -> Optional[PolicyResult]:
        while self.advance() is not None:
            pass
        return self.result

    def _write(self, st: dict) -> None:
        w, ts = self.world, iso(st["ts"])
        pool = {s for f, ss in st["pool"].items() for s in ss}
        steps, feeders, acts = [], [], []
        for s, p in st["batt_kw"].items():
            fid = w.sites[s].feeder_id
            if fid not in self.feeders:
                continue
            status = st["status"].get(s) or (None, None)
            steps.append({"run_id": self.run_id, "ts": ts, "site_id": s, "feeder_id": fid, "status": status[0],
                          "reason": status[1], "in_pool": int(s in pool), "planned_kw": st["planned_kw"].get(s),
                          "batt_kw": round(p, 4), "soc": round(st["soc"][s], 4)})
        for fid, x in st["feeder_net_kw"].items():
            if fid not in self.feeders:
                continue
            g = float(w.feeders[fid].spec.g_kw)
            feeders.append({"run_id": self.run_id, "ts": ts, "feeder_id": fid, "net_kw": round(x, 4),
                            "limit_kw": round(g, 3), "over": int(x > g + 1e-6),
                            "committed_kw": st["committed_kw"].get(fid), "n_pool": len(st["pool"].get(fid, []))})
        for a in st["actions"]:
            pl = a.payload or {}
            fid = pl.get("feeder_id") or (w.sites[pl["asset_id"]].feeder_id if pl.get("asset_id") in w.sites else None)
            acts.append({"run_id": self.run_id, "ts": ts, "action_id": a.action_id, "action_type": str(a.type),
                         "asset_id": pl.get("asset_id") or pl.get("site_id"), "feeder_id": fid,
                         "payload_json": json.dumps(pl, default=str)})
        con = self.con
        con.execute("BEGIN IMMEDIATE")
        try:
            store.insert_many(con, "live_agent_steps", ("run_id", "ts", "site_id", "feeder_id", "status", "reason",
                                                        "in_pool", "planned_kw", "batt_kw", "soc"), steps)
            store.insert_many(con, "live_agent_feeders", ("run_id", "ts", "feeder_id", "net_kw", "limit_kw", "over",
                                                          "committed_kw", "n_pool"), feeders)
            store.insert_many(con, "live_agent_actions", ("run_id", "ts", "action_id", "action_type", "asset_id",
                                                          "feeder_id", "payload_json"), acts)
            con.execute("COMMIT")
        except BaseException:
            con.execute("ROLLBACK")
            raise


KPIS = ("minutes_over_limit", "energy_over_limit_kwh", "p10_hit_rate", "comfort_breaches")


def _kpis(r: PolicyResult) -> dict[str, Any]:
    out = {k: getattr(r, k) for k in KPIS}
    return {k: (round(v, 4) if isinstance(v, float) else v) for k, v in out.items()}


def gate(world: Any, db: Any = None, comps: Any = None, judge: bool = False) -> dict[str, Any]:
    """#129 gate: the whole week tick by tick through LiveAgent vs the batch loop on the same world and components.
    Also the baseline rule for context (#506: 6645 -> 1905 min, 1397 -> 94.5 kWh, P10 92.0%, 0 breaches)."""
    comps = comps or default_components(world, judge=judge)

    def cfg(policy: str, rid: str) -> LoopConfig:
        return LoopConfig(policy=policy, run_id=rid, write_inputs=False, step_predictions=False,
                          write_schedules=False, judge=judge)

    base = run_policy(world, comps, cfg("baseline", "gate:baseline"), sink=_MemSink())
    batch = run_policy(world, comps, cfg("sentinel", "gate:batch"), sink=_MemSink())
    con = store.connect(db) if db else sqlite3.connect(":memory:", isolation_level=None)
    agent = LiveAgent(world, con, "gate:batch", comps=comps, judge=judge)  # same run_id -> same action ids
    n_ticks = 0
    while agent.advance() is not None:
        n_ticks += 1
    tick = agent.result
    kb, kt, k0 = _kpis(batch), _kpis(tick), _kpis(base)
    out: dict[str, Any] = {"gate_ticks": n_ticks, "gate_homes": len(world.sites)}
    for k in KPIS:
        out[f"baseline_{k}"] = k0[k]
        out[f"batch_{k}"] = kb[k]
        out[f"tick_{k}"] = kt[k]
    out["gate_actions_identical"] = int(batch.action_ids == tick.action_ids)
    out["gate_identical"] = int(kb == kt and batch.action_ids == tick.action_ids)
    return out
