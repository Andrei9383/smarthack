#!/usr/bin/env python3
"""Build a demo database for one fleet size, calibrated to public Romanian data (team/datasets.md).

Runs fleet -> faults -> windows -> wind -> grid-twin -> replay into its own directories, so the default
fleet, its window cache and demo/wattsentry.sqlite are untouched. Every feeder is planned and scored; only
some homes keep per-step rows, because 15-minute rows for every home are what fills a database.

  python scripts/build_dataset.py --homes 600 --out demo/   # -> demo/wattsentry-homes600.sqlite (gzip it, #211)

Ported from card #118 (Tudor) in #215 without the PPC service-area option (`--zones ppc`): its localities, weather
file and parks are only on card/118.
"""

import argparse
import os
import subprocess
import sys
import time
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent

from wattsentry.agent.replay import replay
from wattsentry.agent.world import SimWorld
from wattsentry.data.windows import build_windows
from wattsentry.sim.fleet import FleetConfig, generate
from wattsentry.sim.fleet.weather import weather_key
from wattsentry.sim.faults import FaultConfig, inject_dataset

SITES_PER_FEEDER = 25
CONGESTED_SHARE = 0.25  # our assumption: public data gives no share of congested LV feeders
BATTERY_SHARE = 0.20  # ANRE, Nov 2025: 58,012 of 287,985 prosumers
# ANRE per-county counts, Dec 2024, scaled so Ilfov = 3.0: Ilfov 13,405, Timiș 9,781, Bihor 7,966, Dolj 7,822,
# Iași 6,709. The rest have no public figure we found: below Iași (they are outside the top 6), else unchanged.
COUNTY_WEIGHTS = {
    "Ilfov": 3.0, "Timiș": 2.19, "Bihor": 1.78, "Dolj": 1.75, "Iași": 1.50, "Cluj": 1.4, "Brașov": 1.3,
    "Constanța": 1.2, "Prahova": 1.0, "Sibiu": 1.0, "Argeș": 0.8, "Galați": 0.5, "Bacău": 0.5,
}
ALL_PARKS = ("WP1", "WP2", "WP3")
DETAIL_HOMES_PER_FEEDER = {600: 4, 5000: 1}


def fleet_config(homes: int, battery_share: float = BATTERY_SHARE) -> FleetConfig:
    feeders = round(homes / SITES_PER_FEEDER)
    return FleetConfig(n_sites=homes, n_feeders=feeders, n_congested=round(feeders * CONGESTED_SHARE),
                       start_year=2025, end_year=2025, battery_share=battery_share, county_weights=COUNTY_WEIGHTS)


def detail_sites(world: SimWorld, per_feeder: int) -> frozenset[str]:
    """The hero feeder in full, plus a few homes elsewhere so every feeder has something to open."""
    hero = world.feeders["F7"].site_ids
    others = (sorted(f.site_ids)[:per_feeder] for fid, f in world.feeders.items() if fid != "F7")
    return frozenset(hero).union(*others)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--homes", type=int, required=True)
    ap.add_argument("--detail-per-feeder", type=int)
    ap.add_argument("--battery-share", type=float, default=BATTERY_SHARE)
    ap.add_argument("--days", type=int, default=7)
    ap.add_argument("--no-wind", action="store_true", help="skip parks (quick local check)")
    ap.add_argument("--out", type=Path, default=Path(os.environ.get("LAB_OUTPUT_DIR", ".")))
    a = ap.parse_args()
    t0 = time.perf_counter()
    suffix = "" if a.battery_share == BATTERY_SHARE else f"-batt{a.battery_share:g}"
    name = f"homes{a.homes}{suffix}"
    cfg = fleet_config(a.homes, a.battery_share)

    fleet = generate(cfg, out_dir=cfg.out_dir(weather_key=weather_key(cfg.weather_source)), plots=False)
    faults = inject_dataset(fleet, FaultConfig())
    windows = build_windows(fleet_dir=fleet, fault_dir=faults, out=fleet.parent / f"windows-{name}")
    twin = fleet.parent / f"gridtwin-{name}"
    subprocess.run([sys.executable, "-m", "wattsentry.gridtwin.run", "--fleet", str(fleet), "--out", str(twin),
                    "--start", "2025-01-01", "--days", "150"], check=True)  # nights before the June demo week only (#215)
    parks = ALL_PARKS
    if not a.no_wind:
        subprocess.run([sys.executable, "-m", "wattsentry.sim.wind", "--no-plots"], check=True)
    os.environ["WATTSENTRY_GRIDTWIN"] = str(twin)
    built = time.perf_counter() - t0
    print(f"LAB_METRIC data_build_seconds={built:.0f}", flush=True)

    feeder_ids = [f"F{i + 1:02d}" for i in range(cfg.n_feeders)]
    world = SimWorld(days=a.days, root=windows, feeders=feeder_ids, parks=parks, wind=not a.no_wind)
    per_feeder = a.detail_per_feeder if a.detail_per_feeder is not None else DETAIL_HOMES_PER_FEEDER.get(a.homes, 2)
    detail = detail_sites(world, per_feeder)
    a.out.mkdir(parents=True, exist_ok=True)
    db = a.out / f"wattsentry-{name}.sqlite"
    t1 = time.perf_counter()
    replay(("baseline", "sentinel"), db, world=world, days=a.days, detail_sites=detail)
    print(f"LAB_METRIC homes={len(world.sites)}", flush=True)
    print(f"LAB_METRIC feeders={len(world.feeders)}", flush=True)
    print(f"LAB_METRIC turbines={len(world.turbines)}", flush=True)
    print(f"LAB_METRIC detail_homes={len(detail)}", flush=True)
    print(f"LAB_METRIC replay_seconds={time.perf_counter() - t1:.0f}", flush=True)
    print(f"LAB_METRIC db_mb={db.stat().st_size / 1e6:.1f}", flush=True)
    print(f"wrote {db}", flush=True)


if __name__ == "__main__":
    main()
