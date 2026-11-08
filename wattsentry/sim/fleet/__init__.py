"""Synthetic Romanian prosumer fleet on LV feeders: python -m wattsentry.sim.fleet [--quick] [--out DIR].

    from wattsentry.sim.fleet import FleetConfig, generate, load_dataset
    out = generate(FleetConfig())          # cached at ~/.cache/wattsentry/data-v{N}/fleet-<digest>
    ds = load_dataset(out)                 # dict of DataFrames (telemetry is loaded lazily per year)

Deterministic per config + seed. Default 600 sites on 15 feeders (5 built to congest), local years 2022-2025;
--quick = 120 sites, 6 feeders, 2024. Times are UTC interval starts (15 min); a `year=YYYY` file holds one
Europe/Bucharest year. Columns follow wattsentry.contracts.schemas; ground truth is `*_true` / `true_*` plus the
physical `feeder_id` / `phase` (models must use `*_recorded` and `gis_feeder_id` / `gis_phase`).

Files: sites.parquet (one row per prosumer: location, recorded vs true PV metadata, BatterySpec fields, split
train/unseen, hidden truths), feeders.parquet (g_max_kw, g_conservative_kw, tau, transformer, voltage sensitivity),
telemetry/year=YYYY.parquet (pv_ac_w, load_w, batt_w > 0 charging, soc at interval end or null without battery,
grid_w = pv - load - batt > 0 export, inv_voltage_v, inv_status), feeder_ts/year=YYYY.parquet (net export, passive
load, MV voltage, per-phase export), weather.parquet (the driving weather per locality), manifest.json,
feeder_stats.csv and plots/. History runs the baseline self-consumption rule (battery.py); faults are injected on
top by wattsentry.sim.faults.
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
