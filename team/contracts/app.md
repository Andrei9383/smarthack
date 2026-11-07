# Application & Database Contract

This document defines the contract between the Python backend (simulators, ML models, optimizer, agent loop) and the Next.js web application (Drizzle ORM, oRPC routers, MapLibre UI).

The single source of truth for the database schema is defined in `packages/db/src/schema/sentinel.ts` using Drizzle ORM.
Migration SQL files generated via `drizzle-kit generate` reside in `packages/db/drizzle/<timestamp>_<name>/migration.sql`.

> **Conventions:**
> - **Times**: TEXT ISO-8601 UTC (`ts`: e.g. `'2025-06-03T11:15:00Z'`), 15-min grid, rendered in **Europe/Bucharest** in the UI. Date buckets (`day`/`date`) use `'YYYY-MM-DD'`.
> - **Provenance**: Every table that holds model, replay, or telemetry data carries `run_id` -> `runs.id`.
> - **Units**: Explicitly suffixed in column names (`_kw`, `_kwh`, `_mw`, `_mwh`, `_v`, `_c`, `_kmh`, `_pct`). Money is in Romanian lei (`lei_at_risk`).
> - **JSON fields**: Stored as TEXT containing valid JSON (`*_json`).
> - **Python usage**: Python services apply `packages/db/drizzle/*/migration.sql` with standard `sqlite3` and only insert rows; Python never edits the schema.
> - **No `step` column**: bucket by `ts` converted to Europe/Bucharest.

---

## 1. SQLite Database Tables

| Table | Written By | Read By | Purpose |
|---|---|---|---|
| `runs` | Infra Manifest (#47), Replay (#17) | UI (Run selector / Demo mode #33), All queries | Manifest of replay and experiment runs with labels (`synthetic RO`, `real RO`, `non-RO`) |
| `kpis` | Replay (#17), Fixtures (#30) | UI (Header KPI strip, Closing summary) | Replay aggregate KPIs (`peak_reduction`, `minutes_over_limit`, `p10_hit_rate`, etc.) |
| `feeders` | Simulators (#11), Fixtures (#30) | UI (Feeders mode #19, Demand mode #40), LP (#18) | Distribution feeder limits (`limit_kw`), trust (`tau`, `gis_trust`), and map line `geometry_json` |
| `parks` | Simulators (#22), Fixtures (#30) | UI (Production mode #39, O&M mode #32) | Wind parks metadata (location, rated MW, turbine count) |
| `assets` | Simulators (#11, #22), Fixtures (#30) | UI (all modes), Agent (#17, #26), LP (#18) | Fleet registry: prosumer sites and wind turbines (`asset_kind`: `'prosumer'` \| `'turbine'`) |
| `measurements` | Simulators (#11, #22), SCADA/IoT | UI (Site view #31, Turbine drawer #32) | Per-tick measured actuals (PV, load, battery, inverter, wind speed, temperatures) |
| `predictions` | ML Inference (#14, #41), MockPredictor (#7) | Agent Qualify (#26), UI (Site/Turbine drawer) | Prediction envelope per (model, t0, ts): q10/q50/q90, fault probs, anomaly class, OOD score, is_fallback |
| `asset_steps` | Replay (#17), Fixtures (#30) | UI (Site view #31) | 96-step site view comparing measured actuals against quantile forecast (`q10_kw`, `q50_kw`, `q90_kw`) |
| `statuses` | Agent Qualify (#26), Triage (#26) | LP (#18), UI (O&M queue #32, Feeders #19) | Per-tick asset operational qualification (`eligible`, `derated`, `excluded`, `human_review`) |
| `feeder_curves` | Feeder LP (#18), Replay (#17) | UI (Feeders mode #19) | Long-format 96-step feeder export curves per policy (`baseline` vs `sentinel`), limits, and slack |
| `commitments` | Commitment (#27), Replay (#17) | UI (Feeders mode #19, Flex panel #31) | Feeder P10 guaranteed export commitments (`nominal_kw`, `available_kw`, `committed_kw`, `realised_kw`) |
| `schedules` | Feeder LP (#18), Rollout kernel (#50) | Replay (#17), UI (Site view #31) | 96-step battery dispatch per prosumer site (`batt_kw`, `soc`, `comfort_reserve_kwh`, `clipped`) |
| `work_orders` | Agent Triage (#26), Operator via UI | UI (O&M queue #32) | Ranked maintenance work orders per crew_type (residential/wind) with score and lei at risk |
| `alerts` | Agent Loop (#17), Baselines (#24) | UI (O&M queue #32, Header KPI strip) | Real-time and historical anomaly alerts (`creates_work_order: 0` for curtailment) |
| `review_queue` | Agent Loop (#17), Qualify (#26) | UI (O&M human-review queue #32) | Cases flagged for human review (OOD, wide interval, ensemble disagreement) |
| `gis_flags` | Grid-Twin (#13), M8 | UI (Feeders mode #19) | Suspected phase/feeder topology mismatches for hatched map rendering |
| `park_forecasts`| WindPower (#42), Simulators (#22) | UI (Production mode #39) | Wind park generation forecasts (`p10_mw`, `p50_mw`, `p90_mw`), expected MWh, availability |
| `zone_demand` | DemandNet (#44), Replay (#17) | UI (Demand mode #40) | County / locality / feeder hierarchical demand forecasts, peak hour, and stress score |
| `reports` | Daily Report Gen (#17, #16) | UI (O&M reports drawer #32) | Daily TSO / park performance summaries (production MWh, availability, curtailed MWh) |
| `audit_log` | Agent Loop (#17), Qualify (#26) | UI (Audit trail #32) | Immutable audit record joining on `action_id` with inputs, model versions, thresholds, decisions |

---

## 2. Table Column Specifications

### 2.1 Provenance & Top-Level
- **`runs`**: `id` (PK), `kind` (fixtures \| replay \| eval \| train), `policy` (baseline \| sentinel), `label` ('synthetic RO' \| 'real RO' \| 'non-RO'), `week_start`, `git_sha`, `model_versions_json`, `created_at` (ISO-8601), `finished_at`, `notes`.
- **`kpis`**: `run_id` (FK), `policy` (baseline \| sentinel), `name`, `value`, `unit`, `label`. PK: `(run_id, policy, name)`.

### 2.2 Topology
- **`feeders`**: `id` (PK, e.g. `'F7'`), `name`, `county`, `locality`, `limit_kw`, `g_max_kw`, `g_conservative_kw`, `tau`, `gis_trust` ('ok' \| 'low'), `geometry_json` (GeoJSON line).
- **`parks`**: `id` (PK, e.g. `'WP01'`), `name`, `county`, `lat`, `lon`, `rated_mw`, `turbine_count`.
- **`assets`**: `id` (PK), `asset_kind` ('prosumer' \| 'turbine'), `name`, `county`, `locality`, `lat`, `lon`, `feeder_id` (FK), `park_id` (FK), `gis_phase`, `gis_trust` ('ok' \| 'low'), `rated_kw`, `has_battery`, `battery_kwh`, `battery_kw`, `comfort_reserve_kwh`, `recorded_kwp`, `true_kwp`, `recorded_tilt_deg`, `true_tilt_deg`, `recorded_azimuth_deg`, `true_azimuth_deg`, `turbine_model`, `hub_height_m`, `rotor_diameter_m`, `metadata_json`.

### 2.3 Telemetry & Predictions
- **`measurements`**: `run_id` (FK), `asset_id` (FK), `ts` (ISO-8601 UTC), `pv_ac_kw`, `load_kw`, `batt_kw`, `soc`, `inv_voltage_v`, `inv_status`, `act_pwt_kw`, `wd_spd_kmh`, `wd_dir_deg`, `meteo_wind_kmh`, `ambient_temp_c`, `conv_temp_c`, `gearbox_temp_c`, `gen_temp_c`, `transformer_temp_c`, `error_code`, `data_quality`. PK: `(run_id, asset_id, ts)`.
- **`predictions`**: `run_id` (FK), `asset_id` (FK), `model` (M1–M8), `model_version`, `t0` (ISO-8601 UTC, when prediction was made), `ts` (ISO-8601 UTC, step predicted), `horizon_idx`, `q10`, `q50`, `q90`, `fault_probs_json`, `anomaly_class`, `ood_score`, `member_std`, `is_fallback` (1 = fallback model produced this). PK: `(run_id, asset_id, model, t0, ts)`.
- **`asset_steps`**: `run_id` (FK), `asset_id` (FK), `ts` (ISO-8601 UTC), `actual_kw`, `q10_kw`, `q50_kw`, `q90_kw`. PK: `(run_id, asset_id, ts)`.

### 2.4 Decision & Dispatch
- **`statuses`**: `run_id` (FK), `asset_id` (FK), `ts` (ISO-8601 UTC), `status` ('eligible' \| 'derated' \| 'excluded' \| 'human_review'), `headroom_factor`, `reason`, `telemetry_age_s`. PK: `(run_id, asset_id, ts)`.
- **`feeder_curves`**: `run_id` (FK), `feeder_id` (FK), `policy` ('baseline' \| 'sentinel'), `ts` (ISO-8601 UTC), `export_kw`, `import_kw`, `limit_kw`, `slack_kw`, `is_violation`. PK: `(run_id, feeder_id, policy, ts)`.
- **`commitments`**: `run_id` (FK), `feeder_id` (FK), `ts` (ISO-8601 UTC), `nominal_kw`, `available_kw`, `committed_kw`, `realised_kw`, `haircut_h`, `hit`. PK: `(run_id, feeder_id, ts)`.
- **`schedules`**: `run_id` (FK), `asset_id` (FK), `policy` ('sentinel'), `ts` (ISO-8601 UTC), `batt_kw`, `soc`, `comfort_reserve_kwh`, `comfort_breached`, `clipped`. PK: `(run_id, asset_id, policy, ts)`.

### 2.5 Actions & Audit
- **`work_orders`**: `action_id` (PK), `run_id` (FK), `asset_id` (FK), `asset_kind`, `crew_type` ('residential' \| 'wind'), `action` ('truck' \| 'remote_check' \| 'inspection'), `ts` (ISO-8601 UTC), `day` (YYYY-MM-DD), `fault` (M4/M6 class), `confidence`, `lost_kwh_day`, `lost_flex_mwh_day`, `lost_lei_day`, `mw_at_risk`, `days_to_visit`, `score`, `rank`, `status`, `evidence_json`.
- **`alerts`**: `action_id` (PK), `run_id` (FK), `asset_id` (FK), `park_id` (FK), `feeder_id` (FK), `ts` (ISO-8601 UTC), `code`, `severity`, `creates_work_order`, `curtailed_mwh`, `message`, `evidence_json`.
- **`review_queue`**: `action_id` (PK), `run_id` (FK), `asset_id` (FK), `ts` (ISO-8601 UTC), `reason`, `ood_score`, `interval_width`, `member_std`, `resolved`.
- **`gis_flags`**: `action_id` (PK), `run_id` (FK), `section_id`, `recorded_feeder`, `recorded_phase`, `suggested_feeder`, `suggested_phase`, `tau`, `geometry_json`.
- **`park_forecasts`**: `run_id` (FK), `park_id` (FK), `ts` (ISO-8601 UTC), `p10_mw`, `p50_mw`, `p90_mw`, `expected_mwh`, `availability_pct`, `curtailed_mwh`, `data_quality_pct`. PK: `(run_id, park_id, ts)`.
- **`zone_demand`**: `run_id` (FK), `zone_id`, `level` ('county' \| 'locality' \| 'feeder'), `zone_name`, `parent_zone_id`, `ts` (ISO-8601 UTC), `q10_mw`, `q50_mw`, `q90_mw`, `peak_stress`, `is_peak_hour`, `label`. PK: `(run_id, level, zone_id, ts)`.
- **`reports`**: `action_id` (PK), `run_id` (FK), `park_id` (FK), `date` (YYYY-MM-DD), `energy_mwh`, `expected_mwh`, `availability_pct`, `curtailed_mwh`, `data_quality_pct`, `issues_json`.
- **`audit_log`**: `id` (PK), `run_id` (FK), `action_id`, `action_type`, `ts` (ISO-8601 UTC), `policy`, `actor`, `asset_id` (FK), `feeder_id` (FK), `park_id` (FK), `model`, `model_version`, `input_hash`, `prediction_json`, `threshold`, `status`, `schedule_json`, `committed_kw`, `realised_kw`, `note`.

---

## 3. oRPC API Contract & Routers

The application uses `@orpc/contract` and `@orpc/server` with Zod schemas defined under `packages/api/src/contract/`:

- **`feeders`**: `list`, `getCurves`, `getCommitments`, `listGisFlags`
- **`om`**: `listAssets`, `listAlerts`, `listWorkOrders`, `listReviewQueue`, `getTurbineDrawer`, `listReports`
- **`production`**: `getRun`, `listParks`, `getParkForecasts`, `getCountyTotals` (wind field is static: `/windfield/meta.json`)
- **`demand`**: `getZones`, `getTopStressed`
- **`flex`**: `getOverview`, `getFeedersFlex`
- **`sites`**: `getDetail`, `getTelemetry`, `getSteps`, `getPredictions`
- **`audit`**: `listLogs`
- **`manifest`**: `getRuns`, `getKpis`

Stub handlers returning empty arrays or default types are provided in `packages/api/src/contract/index.ts` so that pages and tests build cleanly.
