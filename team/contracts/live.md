# Live feed contract: 15-min loop, forecast vs actual (card #128, read by #130)

EXPERIMENT (merge=never until Tudor keeps it). Readings are **simulated** (#11 fleet via `SimWorld`, or `MockWorld`
in tests) and every run row says so. Nothing here touches `demo/sentinel.sqlite` or replay#506.

Writer: `python scripts/live_feed.py` (`sentinel/live/`). Reader: the app's live view (#130), read-only.

## File

One SQLite file, WAL mode, written by one process, read by many:
`$SENTINEL_LIVE_DB`, default `demo/live.sqlite` (git-ignored). The writer creates the tables below if missing
(`CREATE TABLE IF NOT EXISTS`, no Drizzle migration: the app reads it with a separate `better-sqlite3`/libsql handle
opened read-only). A row becomes visible when its tick commits: each tick is one transaction.

## Clock

Every tick handles one 15-min step. Times are TEXT ISO-8601 UTC `YYYY-MM-DDTHH:MM:SSZ`, step **start** (as data.md).
At tick `T` (`T` aligned to 15 min) the step `[T-15min, T)` has just finished:
1. readings for `ts = T-15min` arrive (`live_readings`);
2. the prediction made at the previous tick for `ts = T-15min` gets its `actual` and `err` (`live_predictions`);
3. a new prediction is stored for `ts = T` (the next 15 min), `t0 = T` (= information cut-off: nothing at or after
   `t0` was read).

`ts` is on the **simulated clock** (the world's axis, e.g. the #506 demo week 2025-06-02..08), not wall time:
real time mode maps the wall clock onto that week by whole weeks (same weekday and UTC time of day); `--speed N`
runs one tick every N seconds from `--start`. `live_ticks.wall_at` is the wall clock when the tick was written.
UI: draw the "now" line at the latest `live_ticks.ts + 15 min`; actuals left of it, predictions right of it.

## Tables

`live_runs` (one row per run; a restart continues the same `run_id`)
| column | meaning |
|---|---|
| `run_id` TEXT PK | e.g. `live-sim-F7` |
| `source` | `simulated` (vendor adapters later, #131) |
| `label` | `synthetic RO, simulated readings` — show it next to every number |
| `world` | `sim` / `mock` |
| `tick_s` | step length in seconds (900) |
| `clock` | `realtime` or `speed:<s>` |
| `feeders_json` | `[{"feeder_id","limit_kw","site_ids":[…]}]` |
| `model_versions_json`, `notes`, `created_at` | provenance |

`live_ticks` (one row per handled tick; PK `(run_id, ts)` so a restart never duplicates a tick)
`run_id, ts, wall_at, status (ok | gap), n_readings`. `gap` = the process was down or the source had nothing;
gap ticks have no readings and no predictions.

`live_readings` (PK `(run_id, ts, site_id)`): what arrived for step `ts`.
`run_id, ts, site_id, feeder_id, pv_kw, load_kw, batt_kw, soc, net_kw`. kW; `batt_kw` + = charging (the
self-consumption rule, data.md); `net_kw = pv_kw - load_kw - batt_kw`, + = export. Missing values are NULL.

`live_predictions` (PK `(run_id, ts, level, entity_id, quantity, method)`)
| column | meaning |
|---|---|
| `ts` | target step start (= `t0` for this 1-step-ahead feed) |
| `t0` | information cut-off |
| `level`, `entity_id`, `feeder_id` | `site` + site id, or `feeder` + feeder id (`entity_id = feeder_id`) |
| `quantity` | `pv`, `load`, `net` (net = export after the battery rule, + = export) |
| `method` | `corrected` (headline: model + 0.9 x last residual), `model` (M1/M2 day-ahead step), `persistence` (PV: clear-sky-index, load: last value) |
| `q10, q50, q90` | kW. Feeder rows: `q50` = sum of site q50; `q10`/`q90` = sum of site quantiles (**conservative band, not a feeder quantile**, say so in the UI) |
| `limit_kw` | feeder rows only: `G` of the feeder |
| `flag` | feeder `net` rows only: `none / watch / congested / severe` (one-step version of #120 notify.md: `severe` P10 > limit or P50 ≥ 1.3·limit; `congested` P50 > limit; `watch` P90 > limit or P50 ≥ 0.9·limit) |
| `actual` | filled at the next tick (NULL until then, or if the reading is missing) |
| `err` | `actual - q50` |
| `err_flag` | 1 if the miss is large: actual outside [q10, q90] **and** `abs(err)` > max(0.25·max(|q50|, |actual|), 1 kW·n_sites) — show as a marker |
| `model_version`, `is_fallback` | provenance (`is_fallback = 1` when the model was not available and the mock answered) |

The app's "F7 is about to go over its limit" = the latest `ts` row with `level='feeder' AND quantity='net' AND
method='corrected' AND flag IN ('congested','severe')`.

## Honesty notes (show in the UI)
- Simulated readings on the #506 demo week, re-played on a mapped clock; not live telemetry.
- The nets are day-ahead (not autoregressive); the latest reading enters only through the residual correction.
- Real use needs vendor adapters, site-to-feeder mapping, feeder limits, retrained models and customer consent.
