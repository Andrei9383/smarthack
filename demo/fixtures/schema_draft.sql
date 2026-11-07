-- DRAFT schema for the shared SQLite — card #30 (Alexandra) input to card #38 (Tudor).
--
-- NOT the source of truth. #38's Drizzle schema in packages/db/ is, and `drizzle-kit generate`
-- emits packages/db/drizzle/*.sql. This file exists only so make_fixtures.py can build
-- demo/fixtures.sqlite before #38 lands; the moment it does, make_fixtures.py applies
-- packages/db/drizzle/*.sql instead and this file is deleted (plan §3.4: Python never edits
-- the schema, it only inserts rows).
--
-- Derived from: #38's table list, plan §3.4 (interfaces), §3.5 (decision maths), §3.2 (model
-- registry), and what #30/#31/#32/#40 actually have to render.
--
-- Conventions proposed for #38:
--   * times: INTEGER unix seconds UTC (Drizzle `integer({ mode: 'timestamp' })`), rendered
--     Europe/Bucharest in the UI (R19). Alternative is TEXT ISO-8601 Z — either is fine, but it
--     has to be one of them everywhere, because Python writes these columns directly.
--   * `t` = step start, 15-min grid, 96 steps/day. `step` = 0..95 index within the local day,
--     kept alongside `t` so the UI can bucket without doing timezone maths in the browser.
--   * power in kW, energy in kWh for prosumer rows; MW/MWh for park, feeder-aggregate and zone
--     rows (columns are suffixed, never bare numbers).
--   * money in lei (RON).
--   * every table that holds model or replay output carries `run_id` -> runs.id, so a panel can
--     always say which run and which label it is showing.
--   * `*_json` columns are TEXT holding JSON.

-- ---------------------------------------------------------------- provenance

CREATE TABLE runs (
  id              TEXT PRIMARY KEY,
  kind            TEXT NOT NULL,    -- fixtures | replay | eval | train
  policy          TEXT,             -- baseline | sentinel | NULL
  label           TEXT NOT NULL,    -- 'synthetic RO' | 'real RO' | 'non-RO'   (§12: every number labelled)
  git_commit      TEXT,
  model_versions_json TEXT,         -- {"M1":"pvnet@abc123", ...}
  started_at      INTEGER NOT NULL,
  finished_at     INTEGER,
  notes           TEXT
);

-- ---------------------------------------------------------------- topology
-- Not in #38's list; proposed as additions. Demand mode (#40) needs county -> locality -> feeder
-- to roll up, and feeder_curves needs somewhere to keep the limit's provenance.

CREATE TABLE feeders (
  id              TEXT PRIMARY KEY,         -- 'F7'
  name            TEXT NOT NULL,
  county          TEXT NOT NULL,            -- 'CT'
  locality        TEXT NOT NULL,
  limit_kw        REAL NOT NULL,            -- G_f, §3.5
  g_max_kw        REAL,                     -- grid-twin optimistic
  g_conservative_kw REAL,
  tau             REAL,                     -- trust weight; low trust only tightens (§3.5)
  gis_trust       TEXT                      -- ok | low   (M8 / #13; drives the hatched LV sections)
);

CREATE TABLE parks (
  id              TEXT PRIMARY KEY,         -- 'WP01'
  name            TEXT NOT NULL,
  county          TEXT NOT NULL,            -- 2x Constanta, 1x Vaslui (#30)
  lat             REAL NOT NULL,
  lon             REAL NOT NULL,
  rated_mw        REAL NOT NULL,
  turbine_count   INTEGER NOT NULL
);

-- ---------------------------------------------------------------- assets

CREATE TABLE assets (
  id              TEXT PRIMARY KEY,         -- '1042' for the demo-story site
  asset_type      TEXT NOT NULL,            -- prosumer | turbine
  name            TEXT,
  lat             REAL NOT NULL,
  lon             REAL NOT NULL,
  county          TEXT NOT NULL,
  locality        TEXT,
  feeder_id       TEXT REFERENCES feeders(id),   -- as the GIS records it (§3.4), may be wrong: that is M8's job
  phase           TEXT,                          -- also as recorded
  park_id         TEXT REFERENCES parks(id),
  -- prosumer metadata; §3.4 wants recorded vs true side by side so #13/#8 can show GIS error
  kwp_recorded    REAL,
  kwp_true        REAL,
  tilt_recorded   REAL,
  tilt_true       REAL,
  azimuth_recorded REAL,
  azimuth_true    REAL,
  batt_capacity_kwh REAL,
  batt_power_kw   REAL,
  batt_reserve_kwh REAL,                     -- R_i, comfort reserve (§3.5)
  -- turbine metadata (§3.4 wind telemetry)
  turbine_model   TEXT,
  hub_height_m    REAL,
  rated_kw        REAL,
  commissioned_on INTEGER
);

CREATE INDEX assets_type_idx   ON assets(asset_type);
CREATE INDEX assets_feeder_idx ON assets(feeder_id);
CREATE INDEX assets_park_idx   ON assets(park_id);

-- ---------------------------------------------------------------- measurements
-- GAP in #38's list: there is no table holding actuals, but #31's acceptance is "site 1042 shows
-- the divergence on day 4" (actual vs P50 + band) and #32's turbine drawer is a scatter of
-- measured points over the learned power curve. Both need per-asset measured values. Columns
-- follow §3.4 prosumer telemetry (15-min) and the PPC/WindMon wind tags.

CREATE TABLE measurements (
  run_id          TEXT NOT NULL REFERENCES runs(id),
  asset_id        TEXT NOT NULL REFERENCES assets(id),
  t               INTEGER NOT NULL,
  step            INTEGER NOT NULL,
  -- prosumer
  pv_ac_w         REAL,
  load_w          REAL,
  batt_w          REAL,                      -- + charge / - discharge
  soc             REAL,                      -- 0..1
  inv_voltage_v   REAL,
  inv_status      TEXT,
  -- turbine (act_pwt kW, wd_spd km/h per §3.4; integer sentinel values are error codes)
  act_pwt_kw      REAL,
  wd_spd_kmh      REAL,
  wd_dir_deg      REAL,
  meteo_wind_kmh  REAL,                      -- the 5-min meteo series, resampled
  temps_json      TEXT,                      -- {converter, gearbox, generator_bearing, transformer, ambient}
  error_code      INTEGER,
  quality         TEXT,                      -- ok | gap | suspect | frozen   (#28 drills)
  PRIMARY KEY (run_id, asset_id, t)
);

-- ---------------------------------------------------------------- model output

-- One row per (asset, model, t0, horizon step): the §3.4 prediction envelope.
CREATE TABLE predictions (
  run_id          TEXT NOT NULL REFERENCES runs(id),
  asset_id        TEXT NOT NULL REFERENCES assets(id),
  model           TEXT NOT NULL,             -- M1 | M2 | M3 | M4 | M5 | M6 | M7 | M8  (§3.2)
  model_version   TEXT NOT NULL,
  t0              INTEGER NOT NULL,          -- when the prediction was made
  t               INTEGER NOT NULL,          -- the step predicted
  horizon_idx     INTEGER NOT NULL,
  q10             REAL,
  q50             REAL,
  q90             REAL,
  fault_probs_json TEXT,                     -- M4: 8 classes
  anomaly_class   TEXT,                      -- M6: fault_stop | curtailment | icing | underperformance | overheating | anemometer_fault | data_gap
  ood_score       REAL,
  member_std      REAL,
  is_fallback     INTEGER NOT NULL DEFAULT 0,  -- 1 = the model slipped and its fallback produced this (§3.2)
  PRIMARY KEY (run_id, asset_id, model, t0, t)
);

CREATE INDEX predictions_asset_t_idx ON predictions(run_id, asset_id, t);

-- qualify() output, per asset per tick (§3.4, §3.5)
CREATE TABLE statuses (
  run_id          TEXT NOT NULL REFERENCES runs(id),
  asset_id        TEXT NOT NULL REFERENCES assets(id),
  t               INTEGER NOT NULL,
  status          TEXT NOT NULL,             -- eligible | derated | excluded | human_review
  headroom_factor REAL NOT NULL,
  reason          TEXT NOT NULL,             -- stale_telemetry | sensor_fault | data_gap | pv_fault | battery_fault
                                             -- | ood | wide_interval | ensemble_disagreement | curtailment | icing | ok
  telemetry_age_s INTEGER,
  PRIMARY KEY (run_id, asset_id, t)
);

-- #32's human-review queue is this table filtered to status='human_review'; the reason enum above
-- carries the three reasons the card names (OOD, wide interval, ensemble disagreement), so no
-- separate review_queue table is needed. Flagging in case #38 would rather have one.

-- ---------------------------------------------------------------- feeder layer

-- Per feeder per step, one row per policy, so Feeders mode (#19) can toggle baseline/sentinel.
CREATE TABLE feeder_curves (
  run_id          TEXT NOT NULL REFERENCES runs(id),
  feeder_id       TEXT NOT NULL REFERENCES feeders(id),
  policy          TEXT NOT NULL,             -- baseline | sentinel
  t               INTEGER NOT NULL,
  step            INTEGER NOT NULL,
  export_kw       REAL NOT NULL,             -- worst plausible export, §3.5 feeder constraint
  import_kw       REAL,
  limit_kw        REAL NOT NULL,
  slack_kw        REAL NOT NULL,             -- s_t
  PRIMARY KEY (run_id, feeder_id, policy, t)
);

-- The four numbers that are never conflated (§3.5): nominal >= available >= committed; realised measured after.
CREATE TABLE commitments (
  run_id          TEXT NOT NULL REFERENCES runs(id),
  feeder_id       TEXT NOT NULL REFERENCES feeders(id),
  t               INTEGER NOT NULL,
  step            INTEGER NOT NULL,
  nominal_kw      REAL NOT NULL,
  available_kw    REAL NOT NULL,
  committed_kw    REAL NOT NULL,             -- (1 - h) * Q10(deliverable_t)
  realised_kw     REAL,                      -- NULL until the next tick logs it
  haircut_h       REAL,                      -- telemetry-loss haircut
  hit             INTEGER,                   -- 1 if realised >= committed; drives #31's hit rate
  PRIMARY KEY (run_id, feeder_id, t)
);

-- battery_schedule action, per site per step (§3.5 LP variables c_it, d_it, e_it)
CREATE TABLE schedules (
  run_id          TEXT NOT NULL REFERENCES runs(id),
  asset_id        TEXT NOT NULL REFERENCES assets(id),
  t               INTEGER NOT NULL,
  step            INTEGER NOT NULL,
  charge_kw       REAL NOT NULL DEFAULT 0,
  discharge_kw    REAL NOT NULL DEFAULT 0,
  soc_kwh         REAL,
  clipped         INTEGER NOT NULL DEFAULT 0,  -- the per-tick feasibility clip fired (R4)
  PRIMARY KEY (run_id, asset_id, t)
);

-- ---------------------------------------------------------------- O&M layer

-- Top-K per day per crew type (§3.5 triage score). Sensor faults are 'remote_check', never a truck.
CREATE TABLE work_orders (
  id              TEXT PRIMARY KEY,
  run_id          TEXT NOT NULL REFERENCES runs(id),
  asset_id        TEXT NOT NULL REFERENCES assets(id),
  crew_type       TEXT NOT NULL,             -- residential | wind
  action          TEXT NOT NULL,             -- truck | remote_check | inspection
  created_at      INTEGER NOT NULL,
  day             INTEGER NOT NULL,          -- local day bucket, for "top-K per day"
  fault_class     TEXT,                      -- M4 class, prosumer
  anomaly_class   TEXT,                      -- M6 class, turbine
  confidence      REAL NOT NULL,             -- P_fault,i
  lost_kwh_day    REAL,
  lost_mwh_day    REAL,
  lost_flex_mwh_day REAL,
  lei_at_risk     REAL,
  mw_at_risk      REAL,
  days_to_visit   INTEGER,                   -- D_i
  score           REAL NOT NULL,
  rank_k          INTEGER,                   -- position in the day's top-K for this crew
  status          TEXT NOT NULL DEFAULT 'open',  -- open | acknowledged | done | dismissed
  evidence_json   TEXT
);

CREATE INDEX work_orders_queue_idx ON work_orders(run_id, crew_type, day, rank_k);

-- alert / gis_flag actions. Curtailment is logged here and MUST NOT produce a work order (§3.5, R12).
CREATE TABLE alerts (
  id              TEXT PRIMARY KEY,
  run_id          TEXT NOT NULL REFERENCES runs(id),
  asset_id        TEXT REFERENCES assets(id),
  park_id         TEXT REFERENCES parks(id),  -- park-wide curtailment has no single asset
  feeder_id       TEXT REFERENCES feeders(id),
  t               INTEGER NOT NULL,
  kind            TEXT NOT NULL,             -- curtailment | icing_safety | data_gap | anemometer_fault | gis_flag | ood_review
  severity        TEXT NOT NULL,             -- info | warn | critical
  creates_work_order INTEGER NOT NULL DEFAULT 0,  -- 0 for curtailment, by contract
  curtailed_mwh   REAL,
  message         TEXT NOT NULL,
  evidence_json   TEXT
);

CREATE INDEX alerts_run_t_idx ON alerts(run_id, t);

-- ---------------------------------------------------------------- forecast layers

-- park_report action + Production mode (#39)
CREATE TABLE park_forecasts (
  run_id          TEXT NOT NULL REFERENCES runs(id),
  park_id         TEXT NOT NULL REFERENCES parks(id),
  t               INTEGER NOT NULL,
  step            INTEGER NOT NULL,
  p10_mw          REAL,
  p50_mw          REAL,
  p90_mw          REAL,
  expected_mwh    REAL,
  availability_pct REAL,
  curtailed_mwh   REAL,
  data_quality_pct REAL,
  PRIMARY KEY (run_id, park_id, t)
);

-- zone_demand action; #44 writes it, #30 fixtures until then. Demand mode (#40) rolls county ->
-- locality -> feeder, so one table with a zone_type discriminator rather than three tables.
CREATE TABLE zone_demand (
  run_id          TEXT NOT NULL REFERENCES runs(id),
  zone_type       TEXT NOT NULL,             -- county | locality | feeder
  zone_id         TEXT NOT NULL,             -- county code 'CT' | locality name | feeder id
  zone_name       TEXT NOT NULL,
  parent_zone_id  TEXT,
  t               INTEGER NOT NULL,
  step            INTEGER NOT NULL,
  p10_mw          REAL,
  p50_mw          REAL NOT NULL,
  p90_mw          REAL,
  peak_stress     REAL NOT NULL,             -- 0..1, drives the red scale
  is_peak_hour    INTEGER NOT NULL DEFAULT 0,
  label           TEXT NOT NULL,             -- 'synthetic RO' (households) | 'real RO' (ENTSO-E load)
  PRIMARY KEY (run_id, zone_type, zone_id, t)
);

CREATE INDEX zone_demand_t_idx ON zone_demand(run_id, zone_type, t);

-- daily park report (#32, SHOULD)
CREATE TABLE reports (
  id              TEXT PRIMARY KEY,
  run_id          TEXT NOT NULL REFERENCES runs(id),
  park_id         TEXT REFERENCES parks(id),
  day             INTEGER NOT NULL,
  production_mwh  REAL,
  availability_pct REAL,
  curtailed_mwh   REAL,
  data_quality_pct REAL,
  body_json       TEXT
);

-- ---------------------------------------------------------------- audit
-- §3.4: every action gets an audit_record (model_version, input_hash, prediction, threshold,
-- status, schedule). #32's audit trail reads this, and compares realised vs committed.

CREATE TABLE audit_log (
  id              TEXT PRIMARY KEY,
  run_id          TEXT NOT NULL REFERENCES runs(id),
  t               INTEGER NOT NULL,
  actor           TEXT NOT NULL,             -- agent | human
  action_type     TEXT NOT NULL,             -- battery_schedule | feeder_commitment | work_order | alert
                                             -- | gis_flag | human_review | park_report | zone_demand
  ref_id          TEXT,                      -- work_orders.id / alerts.id / reports.id
  asset_id        TEXT REFERENCES assets(id),
  feeder_id       TEXT REFERENCES feeders(id),
  park_id         TEXT REFERENCES parks(id),
  model           TEXT,
  model_version   TEXT,
  input_hash      TEXT,
  prediction_json TEXT,
  threshold       REAL,
  status          TEXT,
  committed_kw    REAL,
  realised_kw     REAL,
  note            TEXT
);

CREATE INDEX audit_log_run_t_idx ON audit_log(run_id, t);
