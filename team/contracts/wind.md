# Wind contract: PPC HackITall schema

Code: `sentinel/contracts/wind.py`. Real PPC data must drop in unchanged: if PPC's sample uses other sensor names,
change only `SENSORS` there (+ announce).

## Tags
`WP<park>.TID<turbine>.<sensor>`, e.g. `WP1.TID07.act_pwt`. `parse_tag()` → `Tag(park, turbine, sensor)`;
`Tag.turbine_id` = `WP1.TID07` is the asset id used in the envelope, statuses and SQLite. Parsing is lenient on
unknown sensors (PPC may add some); `strict=True` rejects them.

| sensor | unit | meaning |
|---|---|---|
| `act_pwt` | kW (per-turbine unit: `power_unit` / `power_scale` in `turbines`) | active power |
| `wd_spd` | km/h | nacelle anemometer wind speed (convert with `kmh_to_ms`) |
| `conv_t`, `gearbox_t`, `gen1_t`, `gen2_t`, `transformer_t` | °C | converter, gearbox, generator bearings 1 and 2, transformer |
| `turbine_t` | °C | ambient temperature at the turbine |

These are exactly the sensors in PPC's DataDictionary (WindMon repo; Andrei's review on #7). PPC data has **no**
wind direction, yaw, rotor speed, pitch or status tags; direction-dependent effects use ERA5 direction from the
meteo table. Turbine and park ids are hex-like strings in real data (`WP98FA99B8.TID374A8025`); `TAG_RE` accepts
them and `make_tag` passes string ids through unchanged (int zero-padding is only for synthetic parks).

## Error codes
PPC's raw `current_value` is float64, so codes arrive as **integral floats** (`5021.0`). `is_error_code(sensor,
value)`: on temps and `wd_spd` any integral value is a code; on `act_pwt` only integral **non-zero** values are codes
(0.0 kW is a stopped turbine). Stored in long format with `is_code = true` and the code in `value`; consumers must
drop or mask codes before using the series. Null = data gap.

## Raw PPC files (reader input, unchanged)
- Telemetry: `datetime_current_value` (epoch ms UTC), `current_value` (float64), `tag_name`.
- Meteo, **per turbine**: `data_timestamp`, `fk_wtg_id`, `wind_speed_mean_5min` (m/s, arrives as a string).
- Turbines: `pk_wtg_id`, `fk_wtg_model_id`, `fk_park_id`, `fk_turbine_id`, `lat_y`, `long_x`.

## Tables
- `wind_telemetry` (long): `ts, tag, value, is_code`. 15 min (simulator); PPC data at its native rate.
- Wide per-turbine view (built by #10, not a contract): one row per `ts, turbine_id`, one column per sensor.
- Meteo wind: a 5-min series per **turbine** (`ts, turbine_id, wind_ms, wdir_deg?`); the simulator derives it from ERA5
  100 m wind via shear (`wdir_deg` from ERA5, absent in PPC data).
- `turbines`: `turbine_id, park_id, model, rated_kw, hub_height_m, rotor_d_m, lat, lon, county, power_unit, power_scale`.
- Labels: `fault_labels` (data.md) with `asset_kind = turbine` and a `WindAnomaly` class.

Parks (synthetic RO, #22): two in Dobrogea (Constanța), one in Vaslui; ~120 turbines; generic 2.5 MW / 6.1 MW
class curves. One whole park is held out of training.
