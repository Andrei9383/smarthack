"""Wind simulator configuration and cache location (light module: no numpy/polars at import time).

Everything that changes the output lives in `WindSimConfig` (+ `FaultConfig` in wattsentry.sim.wind_faults), so the
manifest can hash it. Output: `$WATTSENTRY_CACHE/data-v{WIND_DATA_VERSION}/wind/` (default root ~/.cache/sentinel); the
whole directory can be set with $WATTSENTRY_WIND_DIR, which wattsentry.model.wind.data.load_wind also reads.
"""

from __future__ import annotations

import dataclasses
import hashlib
import json
import os
from dataclasses import dataclass
from pathlib import Path

# Bump when the generated wind data changes meaning or shape. Starts at 4 so it never collides with the fleet's
# data-v1..v3 directories.
WIND_DATA_VERSION = 4

REPO = Path(__file__).resolve().parents[3]
RAW = REPO / "data" / "raw"


def default_cache_root() -> Path:
    return Path(os.environ.get("WATTSENTRY_CACHE", Path.home() / ".cache" / "sentinel"))


def default_out_dir() -> Path:
    env = os.environ.get("WATTSENTRY_WIND_DIR")
    if env:
        return Path(env)
    return default_cache_root() / f"data-v{WIND_DATA_VERSION}" / "wind"


@dataclass(frozen=True)
class WindSimConfig:
    seed: int = 22
    start: str = "2024-01-01"  # UTC, inclusive. M6 split: 2024 train/val (by day of month), 2025 test
    end: str = "2026-01-01"  # UTC, exclusive
    parks: tuple[str, ...] = ("WP1", "WP2", "WP3")
    # Turbines per park, picked from the real OSM positions (None = the park default in parks.py).
    max_turbines_per_park: int | None = None  # tests: cap every park at this many (nearest the park core)
    # Shear: "fit" = alpha per park and season from ERA5 10 m / 100 m wind on the first year; "fixed" = 0.14.
    shear: str = "fit"
    # Sub-hourly variability ERA5 does not resolve (multiplicative AR(1) noise on the interpolated hourly wind).
    turb_park_sd: float = 0.06  # park-common, 15-min, phi 0.85
    turb_turbine_sd: float = 0.035  # per turbine, phi 0.6
    # Scale ERA5 100 m wind by one factor so the clean fleet's first-year capacity factor matches the real RO national
    # wind capacity factor of that year (ENTSO-E generation / installed capacity). ERA5 is known to run low on land.
    cf_calibration: bool = True
    hot_by_design_frac: float = 0.10  # turbines whose components run hotter by design (not a fault)
    # Parks whose SCADA reports act_pwt in MW (exercises power_unit / power_scale in the contract). Default none.
    mw_parks: tuple[str, ...] = ()
    weather_path: str = ""  # default data/raw/weather.parquet ($WATTSENTRY_WEATHER overrides)

    def to_json(self) -> dict:
        return dataclasses.asdict(self)


def weather_file(cfg: WindSimConfig) -> Path:
    if cfg.weather_path:
        return Path(cfg.weather_path)
    env = os.environ.get("WATTSENTRY_WEATHER")
    return Path(env) if env else RAW / "weather.parquet"


def code_hash() -> str:
    """Hash of the simulator / injector / shared-physics sources: any code change invalidates the cache."""
    h = hashlib.sha256()
    pkg = REPO / "wattsentry"
    files = sorted((pkg / "sim" / "wind").glob("*.py")) + sorted((pkg / "sim" / "wind_faults").glob("*.py"))
    files += [pkg / "model" / "wind" / "physics.py", pkg / "model" / "wind" / "data.py"]
    for f in files:
        if f.exists():
            h.update(f.name.encode() + f.read_bytes())
    return h.hexdigest()[:12]


def digest(*parts: dict) -> str:
    blob = json.dumps({"v": WIND_DATA_VERSION, "code": code_hash(), "parts": parts}, sort_keys=True, default=str)
    return hashlib.sha256(blob.encode()).hexdigest()[:12]
