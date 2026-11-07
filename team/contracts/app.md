# Application & Database Contract

This document defines the contract between the Python backend (simulators, ML models, optimizer, agent loop) and the Next.js web application (Drizzle ORM, oRPC routers, MapLibre UI).

The single source of truth for the database schema is defined in `packages/db/src/schema/sentinel.ts` using Drizzle ORM.
Migration SQL files generated via `drizzle-kit generate` reside in `packages/db/drizzle/*.sql`.

> **Rule:** Python services (`sentinel/agent/replay.py`, `demo/fixtures/seed.py`) apply these generated SQL files using standard `sqlite3` without altering table schemas. Times are stored in **UTC (ISO-8601 strings or Unix epoch ms)** and formatted for display in **Europe/Bucharest** in the UI.

---

## 1. SQLite Database Tables

| Table | Written By | Read By | Purpose |
|---|---|---|---|
| `assets` | Seed (#30), Simulators (#11, #22) | UI (all modes), Agent (#17, #26), LP (#18) | Fleet registry: prosumer sites and wind turbines |
| `predictions` | ML Inference (#14, #41), MockPredictor (#7) | Agent Qualify (#26), UI (Site/Turbine drawer) | Prediction envelope (q10/q50/q90, fault probs, anomaly class) |
| `statuses` | Agent Qualify (#26), Triage (#26) | LP (#18), UI (O&M queue #32, Feeders #19) | Asset qualification (eligible, derated, excluded, human_review) |
| `feeder_curves` | Feeder LP (#18), Replay (#17) | UI (Feeders mode #19) | 96-step feeder export curves: baseline vs Sentinel, limit, slack |
| `commitments` | Commitment (#27), Replay (#17) | UI (Feeders mode #19, Flex panel #31) | Feeder P10 commitment: nominal, available, committed, realised |
| `schedules` | Feeder LP (#18), Rollout kernel (#50) | Replay (#17), UI (Site view #31) | 96-step battery dispatch per prosumer site (battery_w, soc) |
| `work_orders` | Agent Triage (#26), Operator via UI | UI (O&M queue #32) | Dispatched work orders for field technicians |
| `alerts` | Agent Loop (#17), Baselines (#24) | UI (O&M queue #32, Header KPI strip) | Real-time and historical anomaly alerts |
| `zone_demand` | DemandNet (#44), Replay (#17) | UI (Demand mode #40) | County & zone demand forecasts (P10/P50/P90), peak hour & stress |
| `park_forecasts`| WindPower (#42), Simulators (#22) | UI (Production mode #39) | Wind park generation forecasts (P10/P50/P90) and expected MWh |
| `reports` | Daily Report Gen (#17, #16) | UI (O&M reports drawer #32) | Daily TSO / park performance summaries (JSON content) |
| `audit_log` | Agent Loop (#17), Qualify (#26) | UI (Audit trail #32) | Immutable record of decisions, model versions, thresholds, inputs |
| `runs` | Infra Manifest (#47), Replay (#17) | UI (Run selector / Demo mode #33) | Manifest of replay/training runs with labels (`synthetic RO`, `real RO`, `non-RO`) |

---

## 2. Table Schemas & Column Definitions

### 2.1 `assets`
Registry of all physical assets managed by Sentinel (prosumer sites and wind turbines).
- `id` (TEXT, PK): E.g. `site_ro_bv_042` or `WP1.TID04`
- `asset_type` (TEXT, NOT NULL): `'prosumer'` | `'turbine'`
- `name` (TEXT, NOT NULL): Human-readable name
- `lat` (REAL, NOT NULL): Latitude in WGS84
- `lon` (REAL, NOT NULL): Longitude in WGS84
- `feeder_id` (TEXT): ID of distribution feeder (for prosumers)
- `park_id` (TEXT): ID of wind park (for turbines, e.g. `'WP1'`)
- `county` (TEXT, NOT NULL): Romanian county code (e.g. `'B'`, `'BV'`, `'CT'`, `'TL'`)
- `locality` (TEXT): Locality name (e.g. `'Brasov'`, `'Cogealac'`)
- `rated_power_kw` (REAL, NOT NULL): kWp for PV or rated kW for wind turbine
- `metadata_json` (TEXT): JSON containing true/recorded tilt, azimuth, battery kWh, GIS phase trust, turbine model, etc.
- `created_at` (TEXT, NOT NULL, DEFAULT `(CURRENT_TIMESTAMP)`)

### 2.2 `predictions`
Prediction envelopes output by models M1–M8 or MockPredictor.
- `id` (INTEGER, PK, AUTOINCREMENT)
- `asset_id` (TEXT, NOT NULL, FK -> `assets.id`)
- `model` (TEXT, NOT NULL): E.g. `'PVNet'`, `'LoadNet'`, `'SiteHealth'`, `'WindNBM'`, `'MockPredictor'`
- `model_version` (TEXT, NOT NULL): Commit hash or model tag
- `t` (TEXT, NOT NULL): Timestamp forecast is made for (ISO-8601 UTC)
- `horizon_step` (INTEGER, NOT NULL): Step index in horizon (e.g. 0 to 95 for 15-min 24h)
- `q10` (REAL, NOT NULL): 10th percentile forecast (kW)
- `q50` (REAL, NOT NULL): 50th percentile (median forecast in kW)
- `q90` (REAL, NOT NULL): 90th percentile forecast (kW)
- `fault_probs_json` (TEXT): JSON map of fault class probabilities
- `anomaly_class` (TEXT): Predicted anomaly class (e.g. `'normal'`, `'inverter_clipping'`, `'curtailment'`, `'sensor_freeze'`)
- `ood_score` (REAL): Out-of-distribution score (0.0 to 1.0)
- `member_std` (REAL): Ensemble spread / standard deviation
- `created_at` (TEXT, NOT NULL, DEFAULT `(CURRENT_TIMESTAMP)`)

### 2.3 `statuses`
Current operational qualification assigned to an asset.
- `asset_id` (TEXT, PK, FK -> `assets.id`)
- `status` (TEXT, NOT NULL): `'eligible'` | `'derated'` | `'excluded'` | `'human_review'`
- `headroom_factor` (REAL, NOT NULL, DEFAULT 1.0): Multiplier on available flex (e.g. 0.0 if excluded, 0.5 if derated)
- `reason` (TEXT, NOT NULL): E.g. `'healthy'`, `'suspected_fault:soiling'`, `'telemetry_stale'`, `'ood_weather'`
- `updated_at` (TEXT, NOT NULL, DEFAULT `(CURRENT_TIMESTAMP)`)

### 2.4 `feeder_curves`
Aggregate power curves for each feeder over a 24-hour simulation day (96 15-min intervals).
- `id` (INTEGER, PK, AUTOINCREMENT)
- `feeder_id` (TEXT, NOT NULL)
- `t` (TEXT, NOT NULL): Interval timestamp (ISO-8601 UTC)
- `step` (INTEGER, NOT NULL): 0 to 95
- `baseline_export_kw` (REAL, NOT NULL): Export without coordinated battery dispatch
- `sentinel_export_kw` (REAL, NOT NULL): Export with Sentinel LP battery dispatch
- `limit_kw` (REAL, NOT NULL): Feeder thermal / reverse power capacity limit (G_max or G_conservative)
- `slack_kw` (REAL, NOT NULL): Margin under limit (limit - export)
- `run_id` (TEXT, NOT NULL, FK -> `runs.run_id`)

### 2.5 `commitments`
P10 feeder export commitments offered to the distribution grid.
- `id` (INTEGER, PK, AUTOINCREMENT)
- `feeder_id` (TEXT, NOT NULL)
- `t` (TEXT, NOT NULL): Interval timestamp (ISO-8601 UTC)
- `step` (INTEGER, NOT NULL): 0 to 95
- `nominal_kw` (REAL, NOT NULL): Unconstrained scheduled export
- `available_kw` (REAL, NOT NULL): Total flexible battery headroom
- `committed_kw` (REAL, NOT NULL): Firm P10 guaranteed export cap
- `realised_kw` (REAL): Actual measured export during replay tick (NULL until evaluated)
- `run_id` (TEXT, NOT NULL, FK -> `runs.run_id`)

### 2.6 `schedules`
Individual asset battery dispatch schedules generated by the per-feeder LP.
- `id` (INTEGER, PK, AUTOINCREMENT)
- `asset_id` (TEXT, NOT NULL, FK -> `assets.id`)
- `t` (TEXT, NOT NULL): Timestamp (ISO-8601 UTC)
- `step` (INTEGER, NOT NULL): 0 to 95
- `battery_w` (REAL, NOT NULL): Battery power (positive = discharging, negative = charging)
- `soc` (REAL, NOT NULL): Expected State of Charge (0.0 to 1.0)
- `run_id` (TEXT, NOT NULL, FK -> `runs.run_id`)

### 2.7 `work_orders`
Actionable maintenance work orders for field technicians.
- `id` (TEXT, PK): E.g. `'WO-2026-001'`
- `asset_id` (TEXT, NOT NULL, FK -> `assets.id`)
- `issue_type` (TEXT, NOT NULL): E.g. `'inverter_tripped'`, `'converter_overtemp'`, `'bearing_wear'`
- `priority` (TEXT, NOT NULL): `'critical'` | `'high'` | `'medium'` | `'low'`
- `status` (TEXT, NOT NULL): `'open'` | `'assigned'` | `'in_progress'` | `'resolved'` | `'cancelled'`
- `scheduled_for` (TEXT): Target service date/time
- `technician` (TEXT): Assigned field team
- `estimated_lost_mwh` (REAL): Potential lost generation without intervention
- `notes` (TEXT): Context and recommended actions
- `created_at` (TEXT, NOT NULL, DEFAULT `(CURRENT_TIMESTAMP)`)
- `updated_at` (TEXT, NOT NULL, DEFAULT `(CURRENT_TIMESTAMP)`)

### 2.8 `alerts`
System-wide alerts for anomalies, faults, and limit warnings.
- `id` (TEXT, PK): E.g. `'ALT-10492'`
- `asset_id` (TEXT, NOT NULL, FK -> `assets.id`)
- `severity` (TEXT, NOT NULL): `'critical'` | `'warning'` | `'info'`
- `message` (TEXT, NOT NULL): Alert description
- `anomaly_class` (TEXT): Associated fault class
- `acknowledged` (INTEGER, NOT NULL, DEFAULT 0): 0 = unread, 1 = acknowledged
- `timestamp` (TEXT, NOT NULL): Event timestamp (ISO-8601 UTC)

### 2.9 `zone_demand`
Regional demand forecast and stress score across Romanian counties/localities.
- `id` (INTEGER, PK, AUTOINCREMENT)
- `zone_id` (TEXT, NOT NULL): County code or locality ID
- `county` (TEXT, NOT NULL): County name / code
- `t` (TEXT, NOT NULL): Interval timestamp (ISO-8601 UTC)
- `demand_p10_mw` (REAL, NOT NULL)
- `demand_p50_mw` (REAL, NOT NULL)
- `demand_p90_mw` (REAL, NOT NULL)
- `peak_hour` (INTEGER, NOT NULL): Estimated peak hour (0–23)
- `peak_mw` (REAL, NOT NULL): Forecast peak load
- `stress_score` (REAL, NOT NULL): 0.0 to 1.0 stress score for map coloring
- `run_id` (TEXT, NOT NULL, FK -> `runs.run_id`)

### 2.10 `park_forecasts`
Wind park aggregated generation forecasts.
- `id` (INTEGER, PK, AUTOINCREMENT)
- `park_id` (TEXT, NOT NULL): E.g. `'WP1'`, `'WP2'`
- `t` (TEXT, NOT NULL): Interval timestamp (ISO-8601 UTC)
- `p10_mw` (REAL, NOT NULL)
- `p50_mw` (REAL, NOT NULL)
- `p90_mw` (REAL, NOT NULL)
- `expected_mwh` (REAL, NOT NULL): Total expected energy generation in period
- `run_id` (TEXT, NOT NULL, FK -> `runs.run_id`)

### 2.11 `reports`
TSO-style and O&M daily park summaries.
- `id` (TEXT, PK): E.g. `'REP-WP1-20261107'`
- `park_id` (TEXT, NOT NULL)
- `report_date` (TEXT, NOT NULL): YYYY-MM-DD
- `content_json` (TEXT, NOT NULL): Deduplicated anomalies, availability, lost energy, active tickets
- `created_at` (TEXT, NOT NULL, DEFAULT `(CURRENT_TIMESTAMP)`)

### 2.12 `audit_log`
Complete traceability audit log for every automated action.
- `id` (INTEGER, PK, AUTOINCREMENT)
- `action_type` (TEXT, NOT NULL): `'battery_schedule'` | `'feeder_commitment'` | `'work_order'` | `'alert'` | `'gis_flag'` | `'human_review'` | `'park_report'` | `'zone_demand'`
- `asset_id` (TEXT, NOT NULL)
- `model_version` (TEXT, NOT NULL): Commit or model identifier
- `input_hash` (TEXT, NOT NULL): SHA-256 hash of input window / features
- `prediction_json` (TEXT, NOT NULL): Model output envelope
- `threshold` (REAL): Threshold used for the decision
- `status` (TEXT, NOT NULL): Resulting status or decision code
- `schedule_json` (TEXT): Associated schedule or details if applicable
- `created_at` (TEXT, NOT NULL, DEFAULT `(CURRENT_TIMESTAMP)`)

### 2.13 `runs`
Manifest of replay and experiment runs displayed in the demo UI.
- `id` (INTEGER, PK, AUTOINCREMENT)
- `run_id` (TEXT, NOT NULL, UNIQUE): E.g. `'replay_20261107_v1'`
- `label` (TEXT, NOT NULL): `'synthetic RO'` | `'real RO'` | `'non-RO'`
- `description` (TEXT, NOT NULL)
- `val_metrics_json` (TEXT): Map of evaluation metrics (val_nmae, val_pinball, hours_to_detect)
- `created_at` (TEXT, NOT NULL, DEFAULT `(CURRENT_TIMESTAMP)`)

---

## 3. oRPC API Router Procedures

The backend provides type-safe RPC procedures via `@orpc/server` and `@orpc/contract`:

| Router | Procedure | Input | Output | Purpose |
|---|---|---|---|---|
| `feeders` | `list` | `{ runId?: string }` | `FeederSummary[]` | Feeder list with max export and stress indicators |
| `feeders` | `getCurves` | `{ feederId: string, runId?: string }` | `FeederCurvePoint[]` | 96-step baseline vs Sentinel vs limit |
| `feeders` | `getCommitment` | `{ feederId: string, runId?: string }` | `CommitmentPoint[]` | Nominal, available, committed, realised |
| `om` | `listAssets` | `{ status?: string, assetType?: string }` | `AssetListItem[]` | Assets filtered by qualification status |
| `om` | `listAlerts` | `{ acknowledged?: boolean }` | `AlertItem[]` | Active fault/warning alert queue |
| `om` | `listWorkOrders` | `{ status?: string, priority?: string }` | `WorkOrderItem[]` | Open maintenance orders |
| `om` | `getTurbineDrawer` | `{ assetId: string }` | `TurbineDrawerData` | Power curve scatter + temp residual trends |
| `om` | `getReports` | `{ parkId?: string }` | `ParkReportItem[]` | Daily TSO / park summaries |
| `production` | `getWindfield` | `{ t?: string }` | `WindfieldMeta` | Particles wind field grid metadata |
| `production` | `getParkForecasts`| `{ runId?: string }` | `ParkForecastItem[]` | Wind park fan charts and Dobrogea/Moldova totals |
| `demand` | `getZones` | `{ t?: string, runId?: string }` | `ZoneDemandItem[]` | County/zone stress score and P50/P90 load |
| `demand` | `getTopStressed` | `{ runId?: string, limit?: number }` | `StressedFeederItem[]`| Top stressed feeders requiring curtailment/shaving |
| `flex` | `getOverview` | `{ runId?: string }` | `FlexOverview` | Total fleet battery capacity and dispatch |
| `sites` | `getDetail` | `{ assetId: string }` | `SiteDetail` | Single site telemetry, PV/load forecast, dispatch |
| `audit` | `listLogs` | `{ assetId?: string, actionType?: string, limit?: number }` | `AuditLogItem[]` | Traceable decision audit log entries |
| `manifest` | `getRuns` | `{}` | `RunItem[]` | Available dataset runs and evaluation metrics |
