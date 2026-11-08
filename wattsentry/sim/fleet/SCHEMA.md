# Fleet simulator output (card #11)

`python -m wattsentry.sim.fleet` writes to `~/.cache/wattsentry/data-v{DATA_VERSION}/fleet-<config digest>/`
(override with `--out`, or the cache root with `WATTSENTRY_CACHE`). Deterministic for a given config + seed.
Default: 600 sites, 15 feeders (5 built to congest), local years 2022-2025, ~30 s and ~630 MB.
Schema version: `DATA_VERSION = 3` (v2 = #7 contract names, v3 = DST-safe day index, v4 = 0.1 V voltage noise; real ERA5 weather from #9 when present).
`--quick`: 120 sites, 6 feeders, 2024 only, ~3 s.

Times are UTC (`timestamp[ns, UTC]`, interval start, 15 min). A "year" file holds one Europe/Bucharest calendar
year, so `year=2025` starts at `2024-12-31T22:00Z`. Splits (plan §4): train 2022-23, val 2024, test 2025.

Column names follow team/contracts/data.md and `wattsentry.contracts.schemas` (tests/test_fleet_sim.py checks
them). Extra columns are allowed. Ground truth is either `*_true` or prefixed `true_`, plus **`feeder_id` /
`phase`, which are the physical truth**: models, pvlib and the agent must use `*_recorded` and `gis_feeder_id` /
`gis_phase`.

## sites.parquet (one row per prosumer)

| column | meaning |
|---|---|
| site_id, locality_id, locality, county, siruta (null), lat, lon | identity and location (real RO locality, `localities.py`) |
| feeder_id, phase | **physical** connection (truth) |
| gis_feeder_id, gis_phase | as the GIS records it (~2 % wrong feeder, ~10 % wrong phase); what #13 corrects |
| kwp_recorded, tilt_recorded, azimuth_recorded | what pvlib / models see (azimuth 180 = south); wrong on ~20-25 % |
| kwp_true, tilt_true, azimuth_true | real PV metadata |
| inverter_kw, install_year | inverter rating (clips above it), commissioning year |
| archetype | electric_heating, heat_pump, ac, base (details: heating, has_ac, has_ev, has_electric_boiler, works_from_home) |
| battery_kwh, battery_kw, eta_charge, eta_discharge, soc_min, soc_max, comfort_reserve_kwh | `BatterySpec` fields; battery_kwh 0 = none (~45 % have one) |
| split | `train` or `unseen` (30 % of sites never seen in training) |
| true_metadata_wrong, true_feeder_position (0 = transformer, 1 = far end) | hidden |
| true_degradation_per_year, true_temp_coeff, true_system_losses | hidden PV effects |
| true_shaded, true_shade_az_deg, true_shade_width_deg, true_shade_elev_deg | hidden horizon obstruction |
| true_load_scale, true_ua_kw_per_k | hidden household size and heat-loss coefficient |

## feeders.parquet (one row per LV feeder)

feeder_id, locality_id, locality, county, lat, lon, n_sites (physical), congested (design flag, reporting only),
**g_max_kw**, **g_conservative_kw**, **tau** (1.0 here; the grid-twin #13 estimates it), transformer_kva,
n_passive_households (no PV), prosumer_ac_kw, v_base_pu, v_sens_v_per_kw.

## telemetry/year=YYYY.parquet (one row per site and 15 min; sorted by site, then time)

| column | unit | meaning |
|---|---|---|
| ts, site_id | | |
| pv_ac_w | W | inverter AC output (>= 0) |
| load_w | W | household consumption (>= 0) |
| batt_w | W | battery power, **> 0 charging, < 0 discharging** (0 without battery) |
| soc | 0..1 | state of charge at the end of the interval; **null without battery** |
| grid_w | W | extra column: meter at the connection, **> 0 export**; `grid_w = pv_ac_w - load_w - batt_w` |
| inv_voltage_v | V | inverter-side voltage (MV base + export-driven rise on the physical phase + noise) |
| inv_status | int16 | 0 = ok (always, in clean data; the fault injector #12 sets codes) |

History is generated under the baseline self-consumption rule (`battery.py`). Faults are not in this data: #12
injects them on top, with onset labels.

## feeder_ts/year=YYYY.parquet (one row per feeder and 15 min, physical connections)

ts, feeder_id, net_export_w (prosumers' export minus passive load; compare with g_max_kw), prosumer_export_w,
passive_load_w, v_mv_v, net_export_A_w / _B_w / _C_w (per phase).

## weather.parquet (one row per locality and 15 min, what the simulator was driven by)

ts, point_id (= locality_id), ghi_wm2, dni_wm2, dhi_wm2, temp_c, wind10_ms, snow_cm, source (`raw` = #9's
ERA5 file, `synthetic` = the stand-in). A subset of data.md's weather columns; the full table is #9's.
Input: `data/raw/weather_localities.parquet` (or `$WATTSENTRY_WEATHER`) in data.md's weather format with
point_id = `Locality.id`; rows with source `forecast` are ignored.

## Also written

`manifest.json` (config, digest, source, timing), `feeder_stats.csv` (days over the limit, peak ratio, share of
sunny summer days over the limit), `plots/` (peak day per feeder, export duration curves, one battery site's week).

## Battery model (shared with #18 and #27)

```python
from wattsentry.sim.fleet import BatteryParams, apply_power, simulate_self_consumption
p = BatteryParams.from_sites(sites)                 # arrays over sites
batt_w, e_kwh = apply_power(p, e_kwh, requested_batt_w, dt_h=0.25)   # clip any schedule to physics
```

Same semantics as `BatterySpec.step` / `self_consumption_step` in `wattsentry/contracts/opt.py` (tested), vectorised
over sites and in W: `e[t+1] = e[t] + eta_c * c * dt - d * dt / eta_d`, `c, d <= P`,
`soc_min * E <= e <= soc_max * E`. The comfort reserve binds the planner only: `p.e_min_planner_kwh`.
