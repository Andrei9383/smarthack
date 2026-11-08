"""Simulator configuration. Everything that changes the output lives here, so the manifest can hash it."""

from __future__ import annotations

import dataclasses
import hashlib
import json
import os
from dataclasses import dataclass, field
from pathlib import Path

# Bump when the generated data changes meaning or shape; the cache key is data-v{DATA_VERSION}.
DATA_VERSION = 4  # v3: DST-safe local day index; v4: 0.1 V voltage noise

DT_H = 0.25  # 15-min steps
STEPS_PER_DAY = 96


def default_cache_root() -> Path:
    return Path(os.environ.get("WATTSENTRY_CACHE", Path.home() / ".cache" / "sentinel"))


@dataclass(frozen=True)
class FleetConfig:
    seed: int = 7
    n_sites: int = 600
    n_feeders: int = 15
    # Local calendar years (Europe/Bucharest); stored in UTC. Splits: train 2022-23, val 2024, test 2025.
    start_year: int = 2022
    end_year: int = 2025  # inclusive
    # How many feeders are built to congest under the baseline self-consumption rule.
    n_congested: int = 5
    # County weights for where prosumers are (ANRE prosumer statistics are skewed to Ilfov, Cluj, Timiș, Bihor).
    county_weights: dict[str, float] = field(default_factory=lambda: {
        "Ilfov": 3.0, "Cluj": 2.0, "Timiș": 2.0, "Bihor": 1.5, "Brașov": 1.3, "Iași": 1.0,
        "Constanța": 1.2, "Prahova": 1.0, "Sibiu": 1.0, "Argeș": 0.8, "Dolj": 0.8, "Galați": 0.5, "Bacău": 0.5,
    })
    # Shares of hidden effects (pvlib only ever sees the recorded metadata).
    wrong_orientation_share: float = 0.20   # recorded tilt/azimuth off by 10-30 degrees
    wrong_kwp_share: float = 0.05           # recorded kWp off by 5-15 %
    shaded_share: float = 0.25              # a horizon obstruction in some azimuth sector
    battery_share: float = 0.45
    unseen_share: float = 0.30              # sites never seen in training (split = unseen)
    gis_wrong_phase_share: float = 0.10
    gis_wrong_feeder_share: float = 0.03
    weather_source: str = "auto"            # auto | raw | synthetic

    @property
    def years(self) -> list[int]:
        return list(range(self.start_year, self.end_year + 1))

    def to_json(self) -> dict:
        return dataclasses.asdict(self)

    def digest(self, weather_key: str = "") -> str:
        blob = json.dumps({"v": DATA_VERSION, "weather": weather_key, **self.to_json()}, sort_keys=True,
                          ensure_ascii=False)
        return hashlib.sha256(blob.encode()).hexdigest()[:12]

    def out_dir(self, root: Path | None = None, weather_key: str = "") -> Path:
        root = root or default_cache_root()
        return root / f"data-v{DATA_VERSION}" / f"fleet-{self.digest(weather_key)}"


QUICK = dict(n_sites=120, n_feeders=6, n_congested=3, start_year=2024, end_year=2024)
