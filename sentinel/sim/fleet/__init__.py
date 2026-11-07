"""Synthetic Romanian prosumer fleet on LV feeders (card #11). Entry point: python -m sentinel.sim.fleet

    from sentinel.sim.fleet import FleetConfig, generate, load_dataset
    out = generate(FleetConfig())          # cached at ~/.cache/sentinel/data-v{N}/fleet-<digest>
    ds = load_dataset(out)                 # dict of DataFrames (telemetry is loaded lazily per year)
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd

from .battery import BatteryParams, apply_power, self_consumption_step, simulate_self_consumption
from .config import DATA_VERSION, FleetConfig
from .generate import generate

__all__ = ["DATA_VERSION", "FleetConfig", "generate", "load_dataset", "BatteryParams", "apply_power",
           "self_consumption_step", "simulate_self_consumption"]


def load_dataset(out: Path, years: list[int] | None = None, sites: list[str] | None = None) -> dict[str, pd.DataFrame]:
    out = Path(out)
    filt = [("site_id", "in", sites)] if sites else None
    tel = sorted((out / "telemetry").glob("*.parquet"))
    if years:
        tel = [p for p in tel if int(p.stem.split("=")[1]) in years]
    return {
        "sites": pd.read_parquet(out / "sites.parquet"),
        "feeders": pd.read_parquet(out / "feeders.parquet"),
        "telemetry": pd.concat([pd.read_parquet(p, filters=filt) for p in tel], ignore_index=True),
    }
