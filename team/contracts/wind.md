# Wind contract: PPC HackITall schema

Code: `sentinel/contracts/wind.py`. Real PPC data must drop in unchanged: if PPC's sample uses other sensor names,
change only `SENSORS` there (+ announce).

## Tags
`WP<park>.TID<turbine>.<sensor>`, e.g. `WP1.TID07.act_pwt`. `parse_tag()` → `Tag(park, turbine, sensor)`;
`Tag.turbine_id` = `WP1.TID07` is the asset id used in the envelope, statuses and SQLite. Parsing is lenient on
unknown sensors (PPC may add some); `strict=True` rejects them.

| sensor | unit | meaning |
|---|---|---|
| `act_pwt` | kW | active power |
| `wd_spd` | km/h | nacelle wind speed (convert with `kmh_to_ms`) |
| `wd_dir`, `nac_dir` | deg | wind direction, nacelle yaw |
| `rot_spd` | rpm | rotor speed |
| `pitch` | deg | blade pitch |
| `conv_tmp`, `gbx_tmp`, `gen_brg_tmp`, `trf_tmp`, `amb_tmp` | °C | converter, gearbox, generator bearing, transformer, ambient |
| `status` | code | turbine state / error code |

Names other than `act_pwt` / `wd_spd` follow WindMon and are to be confirmed against PPC's sample (Andrei).

## Error codes
An **integer** value on a continuous sensor is an error code, not a reading (`is_error_code(sensor, value)`).
Stored in long format with `is_code = true` and the code in `value`; consumers must drop or mask it before using the
series. Null = data gap.

## Tables
- `wind_telemetry` (long): `ts, tag, value, is_code`. 15 min (simulator); PPC data at its native rate.
- Wide per-turbine view (built by #10, not a contract): one row per `ts, turbine_id`, one column per sensor.
- Meteo wind: a 5-min series per park (`ts, park_id, wind_ms, wdir_deg`) — from ERA5 100 m wind via shear.
- `turbines`: `turbine_id, park_id, model, rated_kw, hub_height_m, rotor_d_m, lat, lon, county`.
- Labels: `fault_labels` (data.md) with `asset_kind = turbine` and a `WindAnomaly` class.

Parks (synthetic RO, #22): two in Dobrogea (Constanța), one in Vaslui; ~120 turbines; generic 2.5 MW / 6.1 MW
class curves. One whole park is held out of training.
