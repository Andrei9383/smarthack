"""data-v2 configuration. Everything that changes the output lives here, so the digest keys the cache.

Shares are labelled: `real RO` = from a public RO source (cited), `synthetic` = our assumption, sweepable.
"""

from __future__ import annotations

import dataclasses
import hashlib
import json
from dataclasses import dataclass
from pathlib import Path

from wattsentry.infra.paths import cache_root as default_cache_root

# Bump when the generated data changes meaning or shape; the cache dir is prosumers-v2/v{PV2_VERSION}-<digest>.
PV2_VERSION = 1

DT_H = 0.25
STEPS_PER_DAY = 96


@dataclass(frozen=True)
class ProsumerConfig:
    seed: int = 11
    n_homes: int = 5000
    n_localities: int = 33                  # all RO localities in wattsentry.sim.fleet.localities (= solar regions)
    start_year: int = 2022                  # local calendar years; splits: train 2022-23, val 2024, test 2025
    end_year: int = 2025
    chunk_homes: int = 500                  # homes simulated together (part of the RNG streams -> in the digest)
    # --- who has what
    share_battery: float = 0.20             # real RO: ANRE ~66,000 batteries / 332,684 prosumers (Apr 2026)
    share_maxcontrol_of_battery: float = 0.25   # synthetic (unknown; sensitivity 0.10 / 0.25 / 0.50)
    share_maxcontrol_no_battery: float = 0.03   # synthetic: MaxControl prosumers without a battery
    share_price_driven_of_maxcontrol_battery: float = 0.90  # synthetic: the rest leave the inverter on self-use
    share_smart_meter: float = 0.60         # real RO: Retele Electrice smart-meter rollout 50-59 % (2024 targets)
    share_battery_declared: float = 0.70    # synthetic: not every battery owner declares it
    share_enrolled: float = 0.10            # synthetic: battery homes (standard contract, smart meter) in level 3
    # --- battery parameters, uniform ranges (synthetic, typical residential hybrid systems)
    battery_cap_kwh: tuple[float, float] = (5.0, 15.0)
    battery_p_kw: tuple[float, float] = (3.0, 5.0)
    reserve_frac: tuple[float, float] = (0.10, 0.40)
    k_top_intervals: tuple[int, int] = (8, 20)    # expensive quarter-hours a MaxControl home targets
    eta: float = 0.95                       # one-way efficiency
    noncompliance_prob: float = 0.10        # synthetic: days a price-driven home runs self-consumption instead
    # --- MaxControl customer price: PZU per 15 min, floor 0, cap 0.999 lei/kWh (real RO: PPC offer, Jul 2026)
    maxcontrol_floor_lei_mwh: float = 0.0
    maxcontrol_cap_lei_mwh: float = 999.0
    # --- metering
    meter_missing_day_prob: float = 0.003   # synthetic: smart-meter home-days that never arrive (nulls)
    standard_profile: str = "builtin"       # settlement profile: "builtin" (proxy) or "file:<parquet>"
    weather_source: str = "auto"            # auto | raw | synthetic (wattsentry.sim.fleet.weather)

    @property
    def years(self) -> list[int]:
        return list(range(self.start_year, self.end_year + 1))

    def to_json(self) -> dict:
        return dataclasses.asdict(self)

    def digest(self, weather_key: str = "", prices_key: str = "") -> str:
        blob = json.dumps({"v": PV2_VERSION, "weather": weather_key, "prices": prices_key, **self.to_json()},
                          sort_keys=True, ensure_ascii=False)
        return hashlib.sha256(blob.encode()).hexdigest()[:12]

    def out_dir(self, root: Path | None = None, weather_key: str = "", prices_key: str = "") -> Path:
        root = root or default_cache_root()
        return root / "prosumers-v2" / f"v{PV2_VERSION}-{self.digest(weather_key, prices_key)}"


QUICK = dict(n_homes=240, n_localities=8, chunk_homes=120, start_year=2024, end_year=2024)
