# Replay output: what lands in SQLite

The Drizzle schema in `packages/db` (#38, Tudor) is the source of truth for table and column names; Python applies
its generated SQL (`packages/db/drizzle/*.sql`) with `sqlite3` and only inserts rows. Agreed on #38 (notes 42, 48):
#38 ports `demo/fixtures/schema_draft.sql` @ bd61d3b (card/30) 1:1 to Drizzle, so until #38 merges that draft is the
column reference. Conventions: times are **TEXT ISO-8601 UTC** in `ts`, plus `step` 0..95 within the local day;
every output row carries `run_id`; baseline and Sentinel are separate rows (`policy`); JSON in `*_json` TEXT columns.

`python replay.py --policy baseline,sentinel --out $LAB_OUTPUT_DIR/sentinel.sqlite` writes, for one test week and
each policy:

| Output | Table | Notes |
|---|---|---|
| Run manifest | `runs` | one row per policy; `label` (synthetic RO / real RO / non-RO), `model_versions_json`, components real vs fallback in `notes` |
| KPIs | `kpis` | `peak_reduction`, `minutes_over_limit`, `p10_hit_rate`, `comfort_breaches`, `lei_per_home`, `alerts_per_turbine_day` per policy |
| Topology | `feeders`, `parks`, `assets` | one `assets` table with `asset_kind` (site / turbine); GIS-recorded vs true metadata |
| Expected vs actual | `asset_steps` | `actual_kw`, `q10/q50/q90` per asset per step (site view, power-curve panels) |
| Raw measurements | `measurements` | prosumer telemetry + wind `act_pwt`, `wd_spd`, meteo wind, temps, error codes |
| Model outputs | `predictions` | envelope fields not in `asset_steps` (fault_probs, anomaly_class, ood_score, member_std, `is_fallback`) |
| Asset status per tick | `statuses` | `status, headroom_factor, reason, telemetry_age_s` |
| Feeder curves | `feeder_curves` | per policy: `export_kw, import_kw, limit_kw, slack_kw` |
| `feeder_commitment` | `commitments` | `nominal_kw >= available_kw >= committed_kw`, `realised_kw`, `haircut_h`, `hit` |
| `battery_schedule` | `schedules` | per policy: `batt_kw, soc, comfort_reserve_kwh, clipped` |
| `work_order` | `work_orders` | crew type, fault, score, rank, lost kWh / flex MWh / lei, MW at risk, days to visit |
| `alert` | `alerts` | incl. curtailment log (`creates_work_order = 0`, `curtailed_mwh`) and icing safety alerts |
| `human_review` | `review_queue` | reason, ood_score, interval_width, member_std |
| `gis_flag` | `gis_flags` | recorded vs suggested feeder/phase, tau |
| Park forecast | `park_forecasts` | expected MWh, availability, curtailed MWh, data quality |
| `park_report` | `reports` | daily TSO-style park report |
| `zone_demand` | `zone_demand` | feeder -> locality -> county, P10/P50/P90 MW, `peak_stress`, `is_peak_hour` |
| Every action's audit | `audit_log` | `action_id, action_type, model_version, input_hash, prediction_json, threshold, status, schedule_json` |

Every action row has its `action_id` in `audit_log`, so the UI's audit panel joins on it.
