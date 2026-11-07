# Wind simulator (#22) + fault injector (#23)

```
python -m sentinel.sim.wind                 # ~30 s, ~450 MB -> ~/.cache/sentinel/data-v4/wind/
python -m sentinel.sim.wind --out DIR       # or $SENTINEL_WIND_DIR; --force rebuilds; --no-faults = clean only
python -m sentinel.sim.wind_faults          # label summary of the cached dataset
```

Label: **synthetic RO, ERA5-driven**. Deterministic (seeds 22 / 23). Cached by a digest of both configs, the input
files and the simulator source, so it rebuilds only when something changed. Bump `WIND_DATA_VERSION` (config.py)
when the output changes meaning or shape.

## Inputs (all committed by #9, no internet)
- `data/raw/weather.parquet`: ERA5 hourly 100 m + 10 m wind, direction, temperature, RH, precipitation, surface
  pressure at each park point.
- `data/raw/osm_wind_turbines.parquet`: real turbine positions (OSM, ODbL).
- `data/raw/entsoe_ro.parquet` + `ro_installed_capacity.parquet`: real RO wind capacity factor (calibration only).

## Parks (parks.py)
| park | turbines | class | ERA5 point |
|---|---|---|---|
| WP1 Fântânele-Cogealac area, Constanța | 70 OSM 2.5 MW positions nearest the park core | generic-2.5MW | `park:fantanele_cogealac` |
| WP2 Peștera, Constanța (held out of M6 training) | all 36 OSM 3 MW positions | generic-2.5MW | `park:pestera` |
| WP3 Moldova 6 MW cluster, Iași (DTEK-tagged in OSM) | 15 OSM positions | generic-6.1MW | `loc:valea_lupului` (nearest, ~54 km) |

WP3 stands in for plan §4's 6.1 MW Moldova park. The public Deleni (Vaslui) coordinates are unverified and OSM maps
no turbines there, so WP3 sits where 6 MW turbines really are mapped, and is labelled with that county.

## Physics (generate.py)
ERA5 hourly is interpolated to 15 min, then: park AR(1) turbulence → shear to hub height (alpha per park and
season, fitted on 2024 from ERA5 10 m / 100 m wind: 0.21–0.28) → Jensen wakes from the real layout by direction →
turbine AR(1) turbulence → rho = p / (R_d·T) → v_norm = v·(rho/1.225)^(1/3) → generic class curve → power (just
under rated at full load). The model also covers the nacelle anemometer (behind the rotor, k≈0.93, noise), the
first-order thermal model (ambient + base + load rise + lag) for five component temperatures, and 5-min free-stream
meteo wind per turbine.

ERA5 runs low over land. One speed factor (1.149) scales the 100 m wind so that the clean fleet's 2024 capacity
factor matches the real RO national wind capacity factor for 2024 (ENTSO-E generation / installed capacity: 23.8 %).
Simulated 2025 comes out at 21.75 % against the real 21.9 %, and 2025 was never used to calibrate.

## Faults (sentinel/sim/wind_faults)
Rates (FaultConfig, per turbine-year unless noted): trips 5 (2 h–3 d); curtailment 30 per **park**-year
(1–8 h, caps 0/30/50/70 %, onsets weighted to windy hours); icing on real ERA5 spells (T < 1 °C and RH ≥ 90 % or
precipitation, ≥ 3 h; 40 % of spells ice the park, 25 % of iced turbines are shut down by the ice detector); yaw
0.5 and power-curve degradation 0.3 (7–45 d); gearbox / generator-bearing drift 0.7 (3–21 d, 60 % end in a failure
stop); anemometer bias/stuck/erratic 1.0 (2–20 d); comm gaps 10 short (≤ 12 h) + 0.4 long (1–7 d), 30 % carrying
integral error codes.

Labels follow `fault_labels` (data.md). Onset and end are exact, on the 15-min grid. Curtailment and icing are labelled
as **whole episodes**. `params` records whether power was actually cut (`power_cut`, `cut_steps`, `cut_onset`,
`cut_end`). M6 trains on `effective` step labels: the span where power was cut, with episodes that never bit
dropped. A cap above the available power leaves no trace in SCADA, so no detector can see it. Set
`SENTINEL_WIND_LABELS=episode` to score on the whole episodes instead.

## Consumers
- `sentinel.model.wind.data.load_wind()` reads this cache first and falls back to the quick stand-in
  `sentinel/model/wind/synth.py` (which needs no ERA5) only when the cache is missing.
- `sentinel.data.wind.quality` holds WindMon's six quality rules + OK / DEGRADED / SUSPECT flags. The manifest stores
  the flag shares per park.
- Splits (M6 convention): test = 2025; in 2024, days ≤ 19 of each month train and days ≥ 22 validate; WP2 is held out.
