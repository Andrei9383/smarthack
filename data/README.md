# data/

`data/raw/` holds every raw input as zstd Parquet (or GeoJSON), committed so that no run needs the internet.
Rebuild with `python -m sentinel.data.download` (no keys or tokens needed; `--only <job>` for one source).
HTTP responses are cached in `data/.cache/` (gitignored), so an interrupted download resumes where it stopped.
`data/raw/MANIFEST.json` lists rows, bytes and sha256 for every file. Generated data (simulated prosumers, wind,
faults) is not stored here: it is rebuilt deterministically and cached per worker (plan §4).

Label: **RO** = real Romanian data (training and real-RO checks). **non-RO** = validation only, never used for
training, always reported with its label (plan R13, R14). All timestamps are UTC (`ts`, interval start).

| File | Contents | Source | Licence | Label |
|---|---|---|---|---|
| `raw/weather.parquet` | ERA5 hourly 2022–2025 at the 33 prosumer localities (read directly by the fleet simulator) (`loc:<id>`, from `sentinel/sim/fleet/localities.py`) and the 3 wind parks (`park:<id>`): GHI/DNI/DHI, cloud, temp, RH, precipitation, snow depth (from ERA5-Land 0.1°), pressure, wind 10 m + 100 m + direction, gusts. Contract `weather`. `era5_lat/lon` = the 0.25° cell used | [Open-Meteo Historical Weather API](https://open-meteo.com/en/docs/historical-weather-api) (Copernicus ERA5) | CC BY 4.0 (Open-Meteo); ERA5: Copernicus licence, attribution | RO |
| `raw/weather_points.parquet` | Every weather point: id, kind, name, county, lat/lon | localities list + park table | — | RO |
| `raw/weather_grid_ro.parquet` | ERA5 hourly, test year 2025, 0.5° grid of points inside Romania (`grid:<lat>_<lon>`): GHI, temp, 100 m wind speed + direction. For the Production-mode wind field | Open-Meteo (ERA5) | CC BY 4.0 / Copernicus | RO |
| `raw/entsoe_ro.parquet` | National load and generation by type, 15-min (hourly in older years), 2022–2025, long format `ts, series, mw`. Series: `load_actual`, `residual_load`, `gen_solar`, `gen_wind`, `gen_nuclear`, `gen_hydro_ror`, `gen_hydro_reservoir`, `gen_gas`, `gen_lignite`, `gen_hard_coal`, `gen_biomass`, `cross_border`. No `load_da` (day-ahead load forecast) | [Energy-Charts API](https://api.energy-charts.info) (Fraunhofer ISE), which republishes ENTSO-E Transparency data. Replaces the ENTSO-E API (token approval takes ~3 days, see decisions.md) | CC BY 4.0 (Energy-Charts public power data); credit ENTSO-E + Fraunhofer ISE | RO |
| `raw/ro_installed_capacity.parquet` | Installed capacity per type and year (GW) | Energy-Charts | CC BY 4.0 | RO |
| `raw/prices.parquet` | OPCOM PZU (day-ahead) clearing price for Romania, 2023–2025: hourly until 30 Sep 2025, 15-min from 1 Oct 2025 (`resolution`), plus traded volume. Contract `prices`, `market = opcom_pzu`; `price_eur_mwh` is null | [OPCOM PZU reports](https://www.opcom.ro) (public CSV export per delivery day) | public market data, attribution to OPCOM | RO |
| `raw/wind_parks.parquet` | The 3 parks used by the wind simulator: id, name, county, operator, turbine count, rated MW, lat/lon (OSM turbine centroid when mapped, else approximate), `coords_from` | public park data (to verify) + OSM | ODbL for OSM-derived coordinates | RO |
| `raw/osm_wind_turbines.parquet` | Wind turbines mapped in OSM in Romania, with the nearest park (≤ 12 km) when one matches. Present only if Overpass answered | [OpenStreetMap](https://www.openstreetmap.org) via Overpass | ODbL 1.0 | RO |
| `raw/geo/ro_counties.geojson` | 41 counties + Bucharest, simplified (4 decimals) | [geoBoundaries](https://www.geoboundaries.org) gbOpen ROU ADM1 | CC BY 4.0 | RO |
| `raw/geo/ro_country.geojson` | Country outline, simplified | geoBoundaries gbOpen ROU ADM0 | CC BY 4.0 | RO |
| `raw/validation/pvdaq_power.parquet` | 15-min mean AC power (W) of 4 real PV systems, one year each: NREL x-Si (#4) and CIS (#10), Golden CO, 2022; two ~50 kW Maryland rooftops (#1199, #1200), 2019. `n_samples` = raw readings in the 15 min | [NREL PVDAQ](https://data.openei.org/submissions/4568) (OEDI S3) | CC BY 4.0 | non-RO |
| `raw/validation/pvdaq_systems.parquet` | Their metadata: kWp, tilt, azimuth, tracking, location, climate, and the UTC offset used per month | PVDAQ | CC BY 4.0 | non-RO |
| `raw/validation/pvdaq_weather.parquet` | ERA5 hourly at the PVDAQ sites for the same years (contract `weather`) | Open-Meteo (ERA5) | CC BY 4.0 / Copernicus | non-RO |

## Notes

- **Units.** Weather follows the contract (W/m², °C, %, mm, cm, hPa, m/s, degrees). Prices in lei/MWh. Load and
  generation in MW. PVDAQ in W.
- **ERA5 is hourly.** Interpolating to 15-min (with the clear-sky index for irradiance) happens where the data is
  used (#10 windows, #11 simulator), not here, so the committed file stays small. ERA5 values are actuals:
  planning must use the noisy `forecast` weather, never these (plan §4).
- **PVDAQ time zone.** PVDAQ stores local wall-clock time without a zone. The downloader finds, for each month,
  the whole-hour shift that best lines AC power up with ERA5 GHI, and stores that shift in
  `utc_offset_h_by_month`.
- **OPCOM DST days** have 23 or 25 hourly intervals (92 or 100 at 15-min). Timestamps are built by stepping from
  local midnight in UTC, so they stay correct across the change.
- **Not committed:** CARE-to-Compare wind SCADA (downloaded in the eval run, #43), London Smart Meter (optional),
  the pandapower IEEE LV feeder (ships with pandapower).
