# Data contract: prosumer tables (Parquet)

Column → dtype lists live in `sentinel/contracts/schemas.py` (`TABLES`); #10 builds the Pandera checks from them.
Writers: #9 (raw RO data), #11 (simulator), #12 (fault injector). Extra columns are allowed; missing ones are not.
Files: zstd Parquet. Raw inputs committed under `data/raw/`; generated data cached at `~/.cache/sentinel/data-v{N}`.

## sites (one row per prosumer site)
- Truth vs record: `feeder_id`/`phase` are simulator truth; `gis_feeder_id`/`gis_phase` are what the GIS records
  (corrupted on purpose for #13). `kwp_true`, `tilt_true`, `azimuth_true` vs `*_recorded`: pvlib only ever sees
  `*_recorded` (~20% of sites wrong by ±10–30°).
- Location: real RO `locality`, `county`, optional `siruta`, `lat`/`lon`.
- Battery: `battery_kwh` (0 = none), `battery_kw`, `eta_charge`, `eta_discharge`, `soc_min`, `soc_max`,
  `comfort_reserve_kwh` → maps 1:1 onto `BatterySpec` (opt.md).
- `archetype`: household type (`heat_pump`, `electric_heating`, `ac`, `base`, …). `split`: `train` | `unseen`.

## feeders
`feeder_id, locality, county, lat, lon, g_max_kw, g_conservative_kw, tau, n_sites, congested`.
`tau` = GIS trust from #13 (1.0 until it lands). Limit used by the LP: `G = tau·g_max + (1 − tau)·g_conservative`.

## telemetry (15 min, one row per site per step)

| column | unit | sign / meaning |
|---|---|---|
| `ts` | UTC | step start |
| `pv_ac_w` | W | ≥ 0, inverter AC output |
| `inv_voltage_v` | V | AC voltage at the inverter (overvoltage trips show here) |
| `inv_status` | code | 0 = ok, else inverter code |
| `soc` | 0..1 | battery state of charge (null if no battery) |
| `batt_w` | W | + charging, − discharging |
| `load_w` | W | ≥ 0, household consumption |

Grid exchange is derived, not stored: `grid_w = pv_ac_w − load_w − batt_w` (+ export). Missing readings are null,
never 0. History is generated under the self-consumption rule.

## weather (hourly, per point)
Open-Meteo / ERA5 at prosumer localities, wind parks and a coarse RO grid: irradiance (`ghi/dni/dhi_wm2`),
`cloud_pct`, `temp_c`, `rh_pct`, `precip_mm`, `snow_cm`, `pressure_hpa`, wind at 10 m and 100 m (m/s, deg),
`gust_ms`. `source`: `era5` | `era5_land` | `forecast`. Planning uses `forecast` (noisy), never ERA5 actuals.

## prices, entsoe_ro
`prices`: hourly `price_lei_mwh` (+ optional EUR), `market` = `opcom_pzu` (fallback `entsoe_da_ro`).
`entsoe_ro`: long format `ts, series, mw` with series `load_actual`, `load_da`, `gen_wind`, `gen_solar`, …

## fault_labels (one row per episode, shared by PV and wind)
`asset_id, asset_kind (site|turbine), fault, onset, end?, severity, params?`. `fault` is a `PvFault` or
`WindAnomaly` value (model.md). `onset` is exact injector truth; `end` null = lasts to the end.
`severity` 0..1 (e.g. fraction of power lost). `params` = injector parameters as JSON.

## Splits (plan §4)
Train 2022–2023, validate 2024, test 2025; 30% of sites (`split = unseen`) and one whole wind park never in
training; separate calibration slice; thresholds frozen on validation.
