# data-v2 prosumers (card #172, spec I steps 2-3)

`python -m sentinel.sim.prosumers_v2` writes `$SENTINEL_CACHE/prosumers-v2/v1-<digest>/` (default cache root
`~/.cache/sentinel`; `--out DIR` overrides). Deterministic for a given config + weather + prices file, and the same
bytes with the Rust or the Python kernel and any number of processes (`SENTINEL_PV2_WORKERS`, default cpus-1, max 8).
Default: 5000 homes in all 33 RO localities, local years 2022-2025 (splits: train 2022-23, val 2024, test 2025;
`split = unseen` homes, 30 %, never in training). `--quick`: 240 homes, 8 localities, 2024.

A new dataset next to the fleet (`sentinel.sim.fleet`, data-v4), which is unchanged (tests/test_data_v2.py).
PV and household load use the fleet's physics on new homes; batteries run the K8 kernel
(`sentinel.kernels.battery_strategies`, numpy reference + Rust copy, bit-identical).

```python
from sentinel.sim.prosumers_v2 import ProsumerConfig, generate, load_training, load_truth
out = generate(ProsumerConfig())            # cached; regenerates on a worker that doesn't have it
tr = load_training(out, years=[2024])       # what PPC sees -> models
gt = load_truth(out, years=[2024])          # labels / evaluation only
```

Times are UTC (`timestamp[ns, UTC]`, interval start, 15 min); a `year=YYYY` file holds one Europe/Bucharest year.
Energies are kWh per 15-min interval, powers kW (interval mean), all rounded to 1 Wh / 1 W.

## Training files (what PPC sees; never contain `manifest.forbidden_training_columns`)

| file | rows | columns |
|---|---|---|
| `homes_public.parquet` | home | home_id, locality_id, locality, county (= solar region), lat, lon, **kwp** (as recorded at connection), tilt_recorded, azimuth_recorded, inverter_kw, install_year, **contract** (`standard` / `maxcontrol`), **meter** (`smart` / `profile`), **battery_declared**, **enrolled** (level-3 aggregator), split |
| `meter_observed/year=YYYY/part-CCC.parquet` | smart home x 15 min | ts, home_id, import_kwh, export_kwh, **available_at** (= local D+1 00:00, in UTC). Null on ~0.3 % of home-days (readings that never arrived). Profile homes have no rows |
| `meter_monthly.parquet` | home x local month | home_id, month (`YYYY-MM`), meter, import_kwh, export_kwh, available_at (start of next local month). All homes |
| `settlement/year=YYYY.parquet` | county x 15 min | ts, county, smart_import_kwh, smart_export_kwh (measured curves), profile_import_kwh, profile_export_kwh (profile homes' monthly totals spread on the standard profile: load shape for import, solar shape for export). Smart + profile = everything PPC is billed on |
| `prices.parquet` | 15 min | ts, pzu_lei_mwh, price_source (`opcom_pzu_hourly_x4` / `opcom_pzu_15min` / `proxy_shift_371d` for 2022), customer_price_lei_mwh (MaxControl: PZU clipped to 0..999), day_rank (0 = the local day's most expensive quarter-hour) |
| `weather.parquet` | locality x 15 min | as the fleet's (ts, point_id = locality_id, ghi/dni/dhi, temp, wind, snow, source) |
| `standard_profile.parquet` | 15 min | ts, load_coef, solar_coef, profile_source (`builtin_proxy` until I3 #173 commits the distributor profile) |

## Hidden files (ground truth: labels and evaluation only)

| file | rows | columns |
|---|---|---|
| `homes.parquet` | home | the public columns + **has_battery, strategy** (`none` / `self_consumption` / `price_driven` / `enrolled`), **cap_kwh, p_kw, reserve_frac, k_top, eta**, and the fleet's hidden PV/household truth (kwp_true, tilt_true, archetype, heating, has_ev, true_*) |
| `truth/home_15min/year=YYYY/part-CCC.parquet` | home x 15 min | ts, home_id, pv_kw, load_kw, batt_kw (> 0 charging), soc_kwh (end of interval; null without battery), import_kwh, export_kwh. `import - export = (load - pv + batt) * 0.25` |
| `truth/portfolio/year=YYYY.parquet` | (county, meter, strategy) x 15 min | ts, county, meter, strategy, n_homes, import_kwh, export_kwh, pv_kwh, load_kwh, batt_kwh |

Parts are chunks of `chunk_homes` homes in home_id order; `load_training` / `load_truth` concatenate them in home,
then time order. `manifest.json`: config, digest, training/hidden file lists, realised shares, summer 18-21 export
share per strategy, source labels, per-stage timing.

## How homes are drawn (config.py; labels: real RO / synthetic)

- has_battery 20 % (real RO: ANRE ~66,000 / 332,684); contract `maxcontrol` for 25 % of battery homes (synthetic,
  sweep 0.10 / 0.25 / 0.50 with `--share-maxcontrol-of-battery`) and 3 % of the others; MaxControl needs a smart
  meter, the rest get one so the total is 60 % (real RO: Retele Electrice 50-59 %).
- strategy: no battery -> none; battery + MaxControl -> price_driven (90 %, else self_consumption); battery +
  standard + smart -> enrolled (10 %); other batteries -> self_consumption. battery_declared: 70 % of battery
  homes, always when enrolled.
- battery: cap 5-15 kWh, P 3-5 kW, reserve 10-40 % of cap, k_top 8-20 quarter-hours, eta 0.95 one way.
- price_driven homes ignore the rule (run self-consumption) on 10 % of days (`noncompliance_prob`).

Strategy rules (kernel docstring, `sentinel/kernels/strategies.py`): self-consumption charges from PV surplus and
covers the house, never exports from the battery; price-driven charges from PV first, discharges at full power in
the home's k_top most expensive quarter-hours of the local day (exporting), covers the house when the price is above
the day's median, and holds the battery below it; enrolled = self-consumption until I8 dispatches it.
