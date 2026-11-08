"""Parquet table schemas (column -> dtype). Spec + units: team/contracts/data.md and wind.md.

Stdlib only: #10 builds the Pandera checks from these dicts, #9/#11/#12/#22/#23 write tables that match them.
dtypes use Arrow names; `timestamp[UTC]` = timezone-aware UTC. Columns ending in `?` are nullable.
"""

from __future__ import annotations

from typing import Iterable, Mapping

from .wind import TELEMETRY_LONG_COLUMNS, TURBINE_META_COLUMNS

# --- prosumers --------------------------------------------------------------------------------------------
SITES = {
    "site_id": "string",  # "S0001"
    "feeder_id": "string",  # true feeder (simulator truth)
    "phase": "string",  # true phase: "A" | "B" | "C" | "ABC"
    "gis_feeder_id": "string",  # as the GIS records it (may be wrong, #13)
    "gis_phase": "string",
    "locality": "string",  # real RO locality name
    "county": "string",  # judet, e.g. "Constanta"
    "siruta": "int32?",  # RO locality code when known
    "lat": "float64",
    "lon": "float64",
    "kwp_true": "float64",
    "kwp_recorded": "float64",  # what pvlib is allowed to see
    "tilt_true": "float64",
    "tilt_recorded": "float64",
    "azimuth_true": "float64",  # deg, 180 = south
    "azimuth_recorded": "float64",
    "inverter_kw": "float64",
    "archetype": "string",  # household archetype (heat_pump, electric_heating, ac, base, ...)
    "battery_kwh": "float64",  # 0 = no battery
    "battery_kw": "float64",
    "eta_charge": "float64",
    "eta_discharge": "float64",
    "soc_min": "float64",
    "soc_max": "float64",
    "comfort_reserve_kwh": "float64",
    "split": "string",  # "train" | "unseen" (30% of sites never seen in training)
}

FEEDERS = {
    "feeder_id": "string",
    "locality": "string",
    "county": "string",
    "lat": "float64",
    "lon": "float64",
    "g_max_kw": "float64",
    "g_conservative_kw": "float64",
    "tau": "float64",  # GIS trust from #13; 1.0 until it lands
    "n_sites": "int32",
    "congested": "bool",  # simulator design flag, for reporting only
}

# 15-min, one row per site per step. Power + = produced / charging / consumed (see data.md sign table).
TELEMETRY = {
    "ts": "timestamp[UTC]",
    "site_id": "string",
    "pv_ac_w": "float32?",
    "inv_voltage_v": "float32?",
    "inv_status": "int16?",  # 0 = ok, else inverter code
    "soc": "float32?",  # 0..1
    "batt_w": "float32?",  # + charge, - discharge
    "load_w": "float32?",
}

# Hourly Open-Meteo / ERA5, per point (locality, park or grid cell).
WEATHER = {
    "ts": "timestamp[UTC]",
    "point_id": "string",
    "lat": "float64",
    "lon": "float64",
    "ghi_wm2": "float32",
    "dni_wm2": "float32",
    "dhi_wm2": "float32",
    "cloud_pct": "float32",
    "temp_c": "float32",
    "rh_pct": "float32",
    "precip_mm": "float32",
    "snow_cm": "float32?",
    "pressure_hpa": "float32",
    "wind10_ms": "float32",
    "wind100_ms": "float32",
    "wdir10_deg": "float32",
    "wdir100_deg": "float32",
    "gust_ms": "float32?",
    "source": "string",  # "era5" | "era5_land" | "forecast" (noisy, for planning)
}

PRICES = {
    "ts": "timestamp[UTC]",  # hour start
    "market": "string",  # "opcom_pzu" | "entsoe_da_ro"
    "price_lei_mwh": "float64",
    "price_eur_mwh": "float64?",
}

ENTSOE_RO = {
    "ts": "timestamp[UTC]",
    "series": "string",  # "load_actual" | "load_da" | "gen_wind" | "gen_solar" | ...
    "mw": "float64",
}

# One row per fault episode. onset is exact (injector truth); end null = until the end of the series.
FAULT_LABELS = {
    "asset_id": "string",  # site_id or turbine_id
    "asset_kind": "string",  # "site" | "turbine"
    "fault": "string",  # PvFault / WindAnomaly value
    "onset": "timestamp[UTC]",
    "end": "timestamp[UTC]?",
    "severity": "float32",  # 0..1, injector-specific scale (e.g. fraction of power lost)
    "params": "string?",  # JSON with injector parameters
}

TABLES: dict[str, Mapping[str, str]] = {
    "sites": SITES,
    "feeders": FEEDERS,
    "telemetry": TELEMETRY,
    "weather": WEATHER,
    "prices": PRICES,
    "entsoe_ro": ENTSOE_RO,
    "fault_labels": FAULT_LABELS,
    "wind_telemetry": TELEMETRY_LONG_COLUMNS,
    "turbines": TURBINE_META_COLUMNS,
}


def check_columns(table: str, columns: Iterable[str]) -> list[str]:
    """Return problems (missing required columns) for a table's column list; [] = ok. Extra columns are allowed."""
    have = set(columns)
    return [f"{table}: missing column {c!r}" for c in TABLES[table] if c not in have]
