"""The replay agent decides each live tick (wattsentry/live/agent.py)."""

from __future__ import annotations

import dataclasses

from wattsentry.agent.world import MockWorld
from wattsentry.live import store
from wattsentry.live.agent import LiveAgent, gate
from wattsentry.live.feed import LiveFeed, parse, run_speed
from wattsentry.live.forecast import iso
from wattsentry.live.source import SimulatedSource

STEP = 900


def _world(cls=MockWorld, days=2):
    return cls(days=days, scale=0.25, with_wind=False)


def test_gate_tick_mode_matches_batch_loop():
    g = gate(_world(days=2))
    assert g["gate_ticks"] == 2 * 96
    for k in ("minutes_over_limit", "energy_over_limit_kwh", "p10_hit_rate", "comfort_breaches"):
        assert g[f"tick_{k}"] == g[f"batch_{k}"], k
    assert g["gate_actions_identical"] == 1 and g["gate_identical"] == 1


class _FutureChanged(MockWorld):
    CUT = None

    def site_day(self, site_id, d):
        day = super().site_day(site_id, d)
        t0 = self.day_ts(d)
        pv = [x * 3 + 1 if t0 + k * STEP >= self.CUT else x for k, x in enumerate(day.pv_kw)]
        ld = [x * 2 if t0 + k * STEP >= self.CUT else x for k, x in enumerate(day.load_kw)]
        return dataclasses.replace(day, pv_kw=pv, load_kw=ld)


def test_no_decision_reads_a_future_step(tmp_path):
    a = _world()
    _FutureChanged.CUT = a.day_ts(1) + 44 * STEP  # day 2, late morning (local)
    b = _world(_FutureChanged)
    ca, cb = store.connect(tmp_path / "a.sqlite"), store.connect(tmp_path / "b.sqlite")
    la, lb = LiveAgent(a, ca, "x"), LiveAgent(b, cb, "x")
    last = _FutureChanged.CUT - STEP
    la.until(last)
    lb.until(last)
    assert la.last_ts == lb.last_ts == last
    for table, cols in (("live_agent_steps", "ts, site_id, status, reason, in_pool, planned_kw, batt_kw, soc"),
                        ("live_agent_feeders", "ts, feeder_id, net_kw, over, committed_kw, n_pool"),
                        ("live_agent_actions", "ts, action_id, action_type, asset_id")):
        q = f"SELECT {cols} FROM {table} ORDER BY 1, 2"
        ra, rb = [tuple(r) for r in ca.execute(q)], [tuple(r) for r in cb.execute(q)]
        assert ra and ra == rb, table
        assert max(r[0] for r in ra) <= iso(last)  # nothing decided ahead of the feed
    # one more tick: the changed step has now happened and the decisions follow it
    la.advance()
    lb.advance()
    q = "SELECT net_kw FROM live_agent_feeders WHERE ts = ? ORDER BY feeder_id"
    assert [tuple(r) for r in ca.execute(q, (iso(_FutureChanged.CUT),))] != \
        [tuple(r) for r in cb.execute(q, (iso(_FutureChanged.CUT),))]


def test_feed_drives_agent_and_restart_does_not_duplicate(tmp_path):
    start = parse("2025-06-03T06:00:00Z")

    def make():
        w = _world()
        con = store.connect(tmp_path / "live.sqlite")
        src = SimulatedSource(w, ["F7"])
        agent = LiveAgent(w, con, "t:agent", feeders=["F7"])
        return LiveFeed(src, con, "t", world="mock", clock="speed:0", agent=agent), con

    feed, con = make()
    run_speed(feed, start, 0, max_ticks=2, log=lambda s: None)
    last = con.execute("SELECT MAX(ts) FROM live_agent_feeders").fetchone()[0]
    assert last == iso(start + STEP) == con.execute("SELECT MAX(ts) FROM live_ticks").fetchone()[0]
    n1 = con.execute("SELECT COUNT(*) FROM live_agent_steps").fetchone()[0]
    con.close()
    feed2, con2 = make()  # restart: the agent fast-forwards from the week start, rows already there are kept
    run_speed(feed2, start, 0, max_ticks=1, log=lambda s: None)
    assert con2.execute("SELECT MAX(ts) FROM live_agent_feeders").fetchone()[0] == iso(start + 2 * STEP)
    n_f7 = len([s for s in feed2.agent.world.feeders["F7"].site_ids if feed2.agent.world.sites[s].battery])
    assert con2.execute("SELECT COUNT(*) FROM live_agent_steps").fetchone()[0] == n1 + n_f7
    dup = con2.execute("SELECT COUNT(*) FROM (SELECT ts, feeder_id FROM live_agent_feeders GROUP BY 1, 2 "
                       "HAVING COUNT(*) > 1)").fetchone()[0]
    assert dup == 0
    assert con2.execute("SELECT COUNT(DISTINCT feeder_id) FROM live_agent_feeders").fetchone()[0] == 1
