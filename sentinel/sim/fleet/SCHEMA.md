# Fleet simulator output (card #11)

`python -m sentinel.sim.fleet` writes to `~/.cache/sentinel/data-v{DATA_VERSION}/fleet-<config digest>/`
(override with `--out`, or the cache root with `SENTINEL_CACHE`). Deterministic for a given config + seed.
Default: 600 sites, 15 feeders (5 built to congest), local years 2022-2025, ~30 s and ~630 MB.
`--quick`: 120 sites, 6 feeders, 2024 only, ~3 s.

Times are UTC (`timestamp[ns, UTC]`, interval start, 15 min). A "year" file holds one Europe/Bucharest calendar
year, so `year=2025` starts at `2024-12-31T22:00Z`. Splits (plan §4): train 2022-23, val 2024, test 2025.

Columns starting with `true_` are hidden ground truth: use them for labels and evaluation only, never as model
inputs (pvlib, models and the agent see the recorded metadata only).

## sites.parquet (one row per prosumer)

| column | meaning |
|---|---|
| site_id, locality_id, county, lat, lon | identity and location (real RO locality, see `localities.py`) |
| feeder_id, phase | feeder and phase **as the GIS records them** (~2 % wrong feeder, ~10 % wrong phase) |
| kwp, tilt_deg, azimuth_deg | **recorded** PV metadata (azimuth 180 = south); wrong on ~20-25 % of sites |
| inverter_ac_kw, install_year | inverter rating (clips above it), commissioning year |
| battery_kwh, battery_kw, battery_eta_charge, battery_eta_discharge, battery_soc_min | battery (0 = no battery, ~45 % have one) |
| comfort_reserve_kwh | energy the battery keeps for the household; never used for self-consumption or flexibility |
| heating, has_ac, has_ev, has_electric_boiler, works_from_home | household archetype |
| true_feeder_id, true_phase, true_feeder_position | physical connection; position 0 = transformer, 1 = far end |
| true_kwp, true_tilt_deg, true_azimuth_deg, true_metadata_wrong | real PV metadata |
| true_degradation_per_year, true_temp_coeff, true_system_losses | PV effects |
| true_shaded, true_shade_az_deg, true_shade_width_deg, true_shade_elev_deg | horizon obstruction |
| true_load_scale, true_ua_kw_per_k | household size and heat-loss coefficient |

## feeders.parquet (one row per LV feeder)

feeder_id, locality_id, locality, county, lat, lon, transformer_kva, n_prosumers (physical), n_passive_households
(no PV), prosumer_ac_kw, **g_max_kw**, **g_conservative_kw**, **tau** (1.0 here; the grid-twin #13 estimates it),
v_base_pu, v_sens_v_per_kw, true_designed_congested.

## telemetry/year=YYYY.parquet (one row per site and 15 min; sorted by site, then time)

| column | unit | meaning |
|---|---|---|
| ts, site_id | | |
| pv_ac_w | W | inverter AC output (>= 0) |
| load_w | W | household consumption (>= 0) |
| batt_w | W | battery power, **> 0 charging, < 0 discharging** (0 without battery) |
| soc | 0..1 | battery state of charge at the end of the interval (0 without battery) |
| grid_w | W | meter at the connection point, **> 0 export, < 0 import**; `grid_w = pv_ac_w - load_w - batt_w` |
| inv_voltage_v | V | inverter-side voltage (MV base + export-driven rise on the true phase + noise) |
| inv_status | int8 | 0 standby (no PV), 1 producing, 2 limiting (clipping at the inverter rating) |

History is generated under the baseline self-consumption rule (`battery.py`). Faults are not in this data: #12
injects them on top, with onset labels.

## feeder_ts/year=YYYY.parquet (one row per feeder and 15 min, physical connections)

ts, feeder_id, net_export_w (prosumers' export minus passive load; compare with g_max_kw), prosumer_export_w,
passive_load_w, v_mv_v, net_export_A_w / _B_w / _C_w (per phase).

## weather.parquet (one row per locality and 15 min)

ts, locality_id, ghi, dni, dhi (W/m2), temp_air (C), wind_speed (m/s, 10 m), snow_depth_cm. `manifest.json`
says whether it came from #9's raw ERA5 file (`weather_source: raw`) or the synthetic stand-in.

## Also written

`manifest.json` (config, digest, source, timing), `feeder_stats.csv` (days over the limit, peak ratio, share of
sunny summer days over the limit), `plots/` (peak day per feeder, export duration curves, one battery site's week).

## Battery model (shared with #18 and #27)

```python
from sentinel.sim.fleet import BatteryParams, apply_power, simulate_self_consumption
p = BatteryParams.from_sites(sites)                 # arrays over sites
batt_w, e_kwh = apply_power(p, e_kwh, requested_batt_w, dt_h=0.25)   # clip any schedule to physics
```

`e[t+1] = e[t] + eta_c * c * dt - d * dt / eta_d`, with `max(soc_min * E, comfort_reserve) <= e <= E` and
`c, d <= P`: the same constraints as the LP in plan §3.5.
