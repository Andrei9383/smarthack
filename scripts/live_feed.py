#!/usr/bin/env python3
"""Live 15-min feed (card #128, EXPERIMENT): simulated readings in, next-15-min forecasts out, compared at the next
tick. Writes the SQLite file the app's live view reads (team/contracts/live.md). Readings are simulated.

  # demo: one tick every 10 s on the #506 week, F7 + F10, starting on the hero day morning
  python scripts/live_feed.py --world sim --feeders F7,F10 --speed 10 --start 2025-06-04T04:00:00Z
  # real time: wall clock mapped onto the #506 week by whole weeks, one tick per 15 min
  python scripts/live_feed.py --world sim --feeders F7,F10
  # no window cache (tests, laptops): the mock world
  python scripts/live_feed.py --world mock --speed 0 --ticks 8 --db /tmp/live.sqlite

Restart with the same --run-id and --db to resume after the last stored tick (no duplicates).
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from sentinel.live import store  # noqa: E402
from sentinel.live.feed import LiveFeed, parse, run_realtime, run_speed  # noqa: E402
from sentinel.live.source import SimulatedSource  # noqa: E402


def make_world(kind: str, days: int, seed: int, scale: float):
    from sentinel.agent.world import MockWorld, SimWorld

    if kind == "mock":
        return MockWorld(days=days, seed=seed, scale=scale, with_wind=False)
    return SimWorld(days=days, seed=seed, wind=False)


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--world", choices=("sim", "mock"), default="sim")
    ap.add_argument("--feeders", default="", help="comma-separated feeder ids (default: F7,F10 if present, else all)")
    ap.add_argument("--db", default=None, help="SQLite path (default $SENTINEL_LIVE_DB or demo/live.sqlite)")
    ap.add_argument("--run-id", default=None, help="default live-<world>")
    ap.add_argument("--speed", type=float, default=None,
                    help="virtual clock: seconds between ticks (0 = as fast as possible); omit for real time")
    ap.add_argument("--start", default=None, help="virtual clock start (UTC ISO), default the world's first step")
    ap.add_argument("--ticks", type=int, default=None, help="stop after this many ticks")
    ap.add_argument("--days", type=int, default=7)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--scale", type=float, default=0.25, help="mock world size")
    a = ap.parse_args(argv)

    world = make_world(a.world, a.days, a.seed, a.scale)
    feeders = [f for f in a.feeders.split(",") if f] or [f for f in ("F7", "F10") if f in world.feeders] or None
    src = SimulatedSource(world, feeders)
    con = store.connect(a.db)
    clock = "realtime" if a.speed is None else f"speed:{a.speed:g}"
    run_id = a.run_id or f"live-{a.world}"
    feed = LiveFeed(src, con, run_id, world=a.world, clock=clock,
                    notes=f"simulated readings ({world.label}); clock {clock}; feeders {sorted(src.feeders())}")
    print(f"[live] {run_id} -> {store.db_path(a.db)}  feeders={sorted(src.feeders())} homes={len(src.sites())} "
          f"models={feed.da.versions()}  resume_after={feed.last}", flush=True)
    log = lambda s: print(s, flush=True)  # noqa: E731
    if a.speed is None:
        run_realtime(feed, a.ticks, log=log)
    else:
        start = parse(a.start) if a.start else src.span()[0]
        run_speed(feed, start, a.speed, a.ticks, log=log)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
