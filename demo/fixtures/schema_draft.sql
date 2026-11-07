-- DRAFT schema for the shared SQLite — card #30 (Alexandra), input to card #38 (Tudor).
--
-- NOT the source of truth. #38's Drizzle schema in packages/db/ is, and `drizzle-kit generate`
-- emits packages/db/drizzle/*.sql. This file exists only so make_fixtures.py can build
-- demo/fixtures.sqlite before #38 lands; the moment it does, make_fixtures.py applies the
-- generated SQL instead and this file is deleted (plan §3.4: Python never edits the schema, it
-- only inserts rows).
--
-- Reconciled against two inputs:
--   * #38's card: the authoritative table list (assets, predictions, statuses, feeder_curves,
--     commitments, schedules, work_orders, alerts, zone_demand, park_forecasts, reports,
--     audit_log, runs). Those names are used verbatim here.
--   * team/contracts/replay.md (yolooo332, card/7, note [32] on #38): what `replay.py` (#17)
--     actually writes. Its conventions win over my earlier guesses, because #17 produces the real
--     rows these fixtures imitate -- so: times are UTC ISO-8601 strings, the column is `ts`, every
--     action carries an `action_id` that `audit_log` joins on, and arrays are JSON text.
--     replay.md proposes different table names for four of #38's (feeder_steps/battery_steps/
--     asset_status/asset_steps, and sites+turbines split instead of one assets table). #38 is
--     source of truth, so #38's names are used; see the note on #38 for the tie-break.
--
-- Conventions:
--   * `ts` TEXT, ISO-8601 UTC ('2025-06-03T11:15:00Z'), 15-min grid, rendered Europe/Bucharest in
--     the UI (R19). Sortable as text, so BETWEEN works for the 24 h slider.
--   * `day`/`date` TEXT 'YYYY-MM-DD', local (Europe/Bucharest) day bucket, for "top-K per day".
--   * power kW / energy kWh for prosumer rows; MW / MWh for park, feeder-aggregate and zone rows.
--     Units are always in the column name.
--   * money in lei (RON).
--   * every table holding model or replay output carries `run_id` -> runs.id, so a panel can
--     always say which run and which label (§12: every number labelled).
--   * `*_json` columns are TEXT holding JSON.
--   * no invented convenience columns: if #17 will not write it, fixtures do not have it either,
--     or the UI would work on fixtures and break on real rows.

-- ---------------------------------------------------------------- provenance

CREATE TABLE runs (
  id                  TEXT PRIMARY KEY,
  kind                TEXT NOT NULL,     -- fixtures | replay | eval | train
  policy              TEXT,              -- baseline | sentinel | NULL
  label               TEXT NOT NULL,     -- 'synthetic RO' | 'real RO' | 'non-RO'
  week_start          TEXT,              -- 'YYYY-MM-DD', the replayed test week
  git_sha             TEXT,
  model_versions_json TEXT,              -- {"M1":"pvnet@abc123", ...}
  created_at          TEXT NOT NULL,
  finished_at         TEXT,
  notes               TEXT
);

-- Demo KPI strip (§7 step 1) and the closing summary (§7 step 7). In replay.md, absent from #38's
-- list; the fixtures need it or the KPI strip has nowhere to read from.
CREATE TABLE kpis (
  run_id          TEXT NOT NULL REFERENCES runs(id),
  policy          TEXT NOT NULL,         -- baseline | sentinel
  name            TEXT NOT NULL,         -- peak_reduction | minutes_over_limit | p10_hit_rate
                                         -- | comfort_breaches | lei_per_home | alerts_per_turbine_day
  value           REAL NOT NULL,
  unit            TEXT,
  label           TEXT NOT NULL,
  PRIMARY KEY (run_id, policy, name)
);

-- ---------------------------------------------------------------- topology
-- feeders/parks are not in #38's list; replay.md has feeders + zones. Demand mode (#40) needs the
-- county -> locality -> feeder rollup, and feeder_curves needs the limit's provenance.

CREATE TABLE feeders (
  id                TEXT PRIMARY KEY,    -- 'F7'
  name              TEXT NOT NULL,
  county            TEXT NOT NULL,       -- 'CT'
  locality          TEXT NOT NULL,
  limit_kw          REAL NOT NULL,       -- G_f (§3.5)
  g_max_kw          REAL,                -- grid-twin optimistic
  g_conservative_kw REAL,
  tau               REAL,                -- trust weight; low trust only ever tightens (§3.5)
  gis_trust         TEXT,                -- ok | low  (drives #19's hatched LV sections)
  geometry_json     TEXT                 -- GeoJSON, where the map needs a line
);

CREATE TABLE parks (
  id              TEXT PRIMARY KEY,      -- 'WP01'
  name            TEXT NOT NULL,
  county          TEXT NOT NULL,         -- 2x Constanta, 1x Vaslui (#30)
  lat             REAL NOT NULL,
  lon             REAL NOT NULL,
  rated_mw        REAL NOT NULL,
  turbine_count   INTEGER NOT NULL
);

-- ---------------------------------------------------------------- assets
-- One table with an asset_kind discriminator, per #38's card ("assets (prosumer sites + turbines,
-- asset_type, ...)"). replay.md splits sites/turbines; see the note on #38.

CREATE TABLE assets (
  id               TEXT PRIMARY KEY,     -- '1042' is the demo-story site
  asset_kind       TEXT NOT NULL,        -- prosumer | turbine
  name             TEXT,
  lat              REAL NOT NULL,
  lon              REAL NOT NULL,
  county           TEXT NOT NULL,
  locality         TEXT,
  feeder_id        TEXT REFERENCES feeders(id),  -- as the GIS records it (§3.4); may be wrong, that is M8's job
  phase            TEXT,                         -- also as recorded
  park_id          TEXT REFERENCES parks(id),
  -- prosumer metadata: §3.4 wants recorded vs true side by side so #13/M8 can show GIS error
  kwp_recorded     REAL,
  kwp_true         REAL,
  tilt_recorded    REAL,
  tilt_true        REAL,
  azimuth_recorded REAL,
  azimuth_true     REAL,
  batt_capacity_kwh REAL,
  batt_power_kw    REAL,
  comfort_reserve_kwh REAL,              -- R_i (§3.5)
  -- turbine metadata (§3.4 wind telemetry)
  turbine_model    TEXT,
  hub_height_m     REAL,
  rated_kw         REAL,
  commissioned_on  TEXT
);

CREATE INDEX assets_kind_idx   ON assets(asset_kind);
CREATE INDEX assets_feeder_idx ON assets(feeder_id);
CREATE INDEX assets_park_idx   ON assets(park_id);

-- ---------------------------------------------------------------- per-asset series

-- replay.md's asset_steps: expected vs actual, the series #31's site view draws. Keeping actual_kw
-- next to the quantiles (rather than in a separate actuals table) because every panel that shows
-- one shows the other, and it is one read per site.
CREATE TABLE asset_steps (
  run_id          TEXT NOT NULL REFERENCES runs(id),
  asset_id        TEXT NOT NULL REFERENCES assets(id),
  ts              TEXT NOT NULL,
  actual_kw       REAL,
  q10_kw          REAL,
  q50_kw          REAL,
  q90_kw          REAL,
  PRIMARY KEY (run_id, asset_id, ts)
);

-- The rest of the §3.4 envelope, which asset_steps has no room for: fault probabilities, anomaly
-- class, OOD, ensemble spread, and which model version produced the row. One row per
-- (asset, model, t0, ts) so a re-forecast does not overwrite the earlier one.
CREATE TABLE predictions (
  run_id           TEXT NOT NULL REFERENCES runs(id),
  asset_id         TEXT NOT NULL REFERENCES assets(id),
  model            TEXT NOT NULL,        -- M1 | M2 | M3 | M4 | M5 | M6 | M7 | M8 (§3.2)
  model_version    TEXT NOT NULL,
  t0               TEXT NOT NULL,        -- when the prediction was made
  ts               TEXT NOT NULL,        -- the step predicted
  horizon_idx      INTEGER NOT NULL,
  q10              REAL,
  q50              REAL,
  q90              REAL,
  fault_probs_json TEXT,                 -- M4: 8 classes
  anomaly_class    TEXT,                 -- M6: fault_stop | curtailment | icing | underperformance
                                         -- | overheating | anemometer_fault | data_gap
  ood_score        REAL,
  member_std       REAL,
  is_fallback      INTEGER NOT NULL DEFAULT 0,  -- 1 = the model slipped, its fallback produced this (§3.2)
  PRIMARY KEY (run_id, asset_id, model, t0, ts)
);

CREATE INDEX predictions_asset_ts_idx ON predictions(run_id, asset_id, ts);

-- Raw telemetry. Not in either list, but #32's turbine drawer is a scatter of measured points over
-- the learned power curve and a component temperature-residual trend: neither can be drawn from
-- actual_kw alone, both need wind speed and component temps. Columns follow §3.4 (prosumer 15-min
-- telemetry; PPC/WindMon wind tags, where integer sentinel values are error codes).
CREATE TABLE measurements (
  run_id          TEXT NOT NULL REFERENCES runs(id),
  asset_id        TEXT NOT NULL REFERENCES assets(id),
  ts              TEXT NOT NULL,
  -- prosumer
  pv_ac_w         REAL,
  load_w          REAL,
  batt_w          REAL,                  -- + charge / - discharge
  soc             REAL,                  -- 0..1
  inv_voltage_v   REAL,
  inv_status      TEXT,
  -- turbine
  act_pwt_kw      REAL,
  wd_spd_kmh      REAL,
  wd_dir_deg      REAL,
  meteo_wind_kmh  REAL,                  -- the 5-min meteo series, resampled
  temps_json      TEXT,                  -- {converter, gearbox, generator_bearing, transformer, ambient}
  error_code      INTEGER,
  quality         TEXT,                  -- ok | gap | suspect | frozen (#28 drills)
  PRIMARY KEY (run_id, asset_id, ts)
);

-- qualify() output, per asset per tick (§3.4, §3.5)
CREATE TABLE statuses (
  run_id          TEXT NOT NULL REFERENCES runs(id),
  asset_id        TEXT NOT NULL REFERENCES assets(id),
  ts              TEXT NOT NULL,
  status          TEXT NOT NULL,         -- eligible | derated | excluded | human_review
  headroom_factor REAL NOT NULL,
  reason          TEXT NOT NULL,         -- ok | stale_telemetry | sensor_fault | data_gap | pv_fault
                                         -- | battery_fault | ood | wide_interval
                                         -- | ensemble_disagreement | curtailment | icing
  telemetry_age_s INTEGER,
  PRIMARY KEY (run_id, asset_id, ts)
);

-- ---------------------------------------------------------------- feeder layer

-- One row per policy per step, so Feeders mode (#19) can toggle baseline/sentinel.
CREATE TABLE feeder_curves (
  run_id          TEXT NOT NULL REFERENCES runs(id),
  feeder_id       TEXT NOT NULL REFERENCES feeders(id),
  policy          TEXT NOT NULL,         -- baseline | sentinel
  ts              TEXT NOT NULL,
  export_kw       REAL NOT NULL,         -- worst plausible export (§3.5 feeder constraint)
  import_kw       REAL,
  limit_kw        REAL NOT NULL,
  slack_kw        REAL NOT NULL,         -- s_t
  PRIMARY KEY (run_id, feeder_id, policy, ts)
);

-- The four numbers that are never conflated (§3.5): nominal >= available >= committed; realised
-- is measured afterwards. #31's panel is exactly this table.
CREATE TABLE commitments (
  run_id          TEXT NOT NULL REFERENCES runs(id),
  feeder_id       TEXT NOT NULL REFERENCES feeders(id),
  ts              TEXT NOT NULL,
  nominal_kw      REAL NOT NULL,
  available_kw    REAL NOT NULL,
  committed_kw    REAL NOT NULL,         -- (1 - h) * Q10(deliverable_t)
  realised_kw     REAL,                  -- NULL until the next tick logs it
  haircut_h       REAL,                  -- telemetry-loss haircut
  hit             INTEGER,               -- 1 if realised >= committed; drives #31's hit rate
  PRIMARY KEY (run_id, feeder_id, ts)
);

-- battery_schedule action, per site per step (§3.5 LP variables c_it, d_it, e_it)
CREATE TABLE schedules (
  run_id          TEXT NOT NULL REFERENCES runs(id),
  asset_id        TEXT NOT NULL REFERENCES assets(id),
  policy          TEXT NOT NULL,         -- baseline | sentinel
  ts              TEXT NOT NULL,
  batt_kw         REAL NOT NULL DEFAULT 0,  -- + charge / - discharge, replay.md's single column
  soc             REAL,                     -- 0..1
  comfort_reserve_kwh REAL,
  clipped         INTEGER NOT NULL DEFAULT 0,  -- the per-tick feasibility clip fired (R4)
  PRIMARY KEY (run_id, asset_id, policy, ts)
);

-- ---------------------------------------------------------------- O&M layer

-- Top-K per day per crew type (§3.5 triage score). Sensor faults are 'remote_check', never a truck.
CREATE TABLE work_orders (
  action_id       TEXT PRIMARY KEY,      -- joins audit_log.action_id
  run_id          TEXT NOT NULL REFERENCES runs(id),
  asset_id        TEXT NOT NULL REFERENCES assets(id),
  asset_kind      TEXT NOT NULL,         -- prosumer | turbine
  crew_type       TEXT NOT NULL,         -- residential | wind
  action          TEXT NOT NULL,         -- truck | remote_check | inspection
  ts              TEXT NOT NULL,
  day             TEXT NOT NULL,         -- local day bucket, for "top-K per day"
  fault           TEXT,                  -- M4 class (prosumer) or M6 anomaly class (turbine)
  confidence      REAL NOT NULL,         -- P_fault,i
  lost_kwh_day    REAL,
  lost_flex_mwh_day REAL,
  lost_lei_day    REAL,
  mw_at_risk      REAL,
  days_to_visit   INTEGER,               -- D_i
  score           REAL NOT NULL,
  rank            INTEGER,               -- position in the day's top-K for this crew
  status          TEXT NOT NULL DEFAULT 'open',  -- open | acknowledged | done | dismissed
  evidence_json   TEXT
);

CREATE INDEX work_orders_queue_idx ON work_orders(run_id, crew_type, day, rank);

-- alert action. Curtailment is logged here and MUST NOT produce a work order (§3.5, R12);
-- icing is a safety alert with no truck in a storm.
CREATE TABLE alerts (
  action_id       TEXT PRIMARY KEY,
  run_id          TEXT NOT NULL REFERENCES runs(id),
  asset_id        TEXT REFERENCES assets(id),
  park_id         TEXT REFERENCES parks(id),   -- park-wide curtailment has no single asset
  feeder_id       TEXT REFERENCES feeders(id),
  ts              TEXT NOT NULL,
  code            TEXT NOT NULL,         -- curtailment | icing_safety | data_gap | anemometer_fault
  severity        TEXT NOT NULL,         -- info | warn | critical
  creates_work_order INTEGER NOT NULL DEFAULT 0,  -- 0 for curtailment, by contract
  curtailed_mwh   REAL,
  message         TEXT NOT NULL,
  evidence_json   TEXT
);

CREATE INDEX alerts_run_ts_idx ON alerts(run_id, ts);

-- human_review action. replay.md gives this its own table rather than deriving it from
-- statuses(status='human_review'); following replay.md, since #17 writes the rows. #32's
-- human-review queue reads it.
CREATE TABLE review_queue (
  action_id       TEXT PRIMARY KEY,
  run_id          TEXT NOT NULL REFERENCES runs(id),
  asset_id        TEXT NOT NULL REFERENCES assets(id),
  ts              TEXT NOT NULL,
  reason          TEXT NOT NULL,         -- ood | wide_interval | ensemble_disagreement
  ood_score       REAL,
  interval_width  REAL,
  member_std      REAL,
  resolved        INTEGER NOT NULL DEFAULT 0
);

-- gis_flag action (M8 / #13, SHOULD). Feeders mode hatches low-trust sections from this.
CREATE TABLE gis_flags (
  action_id       TEXT PRIMARY KEY,
  run_id          TEXT NOT NULL REFERENCES runs(id),
  section_id      TEXT NOT NULL,
  recorded_feeder TEXT,
  recorded_phase  TEXT,
  suggested_feeder TEXT,
  suggested_phase TEXT,
  tau             REAL,
  geometry_json   TEXT
);

-- ---------------------------------------------------------------- forecast layers

-- Production mode (#39) + park availability
CREATE TABLE park_forecasts (
  run_id           TEXT NOT NULL REFERENCES runs(id),
  park_id          TEXT NOT NULL REFERENCES parks(id),
  ts               TEXT NOT NULL,
  p10_mw           REAL,
  p50_mw           REAL,
  p90_mw           REAL,
  expected_mwh     REAL,
  availability_pct REAL,
  curtailed_mwh    REAL,
  data_quality_pct REAL,
  PRIMARY KEY (run_id, park_id, ts)
);

-- zone_demand action; #44 writes it, #30 fixtures until then. Demand mode (#40) rolls
-- county -> locality -> feeder, so one table with a `level` discriminator (replay.md) rather
-- than three tables.
CREATE TABLE zone_demand (
  run_id          TEXT NOT NULL REFERENCES runs(id),
  zone_id         TEXT NOT NULL,         -- county code 'CT' | locality name | feeder id
  level           TEXT NOT NULL,         -- county | locality | feeder
  zone_name       TEXT NOT NULL,
  parent_zone_id  TEXT,
  ts              TEXT NOT NULL,
  q10_mw          REAL,
  q50_mw          REAL NOT NULL,
  q90_mw          REAL,
  peak_stress     REAL NOT NULL,         -- 0..1, drives the red scale
  is_peak_hour    INTEGER NOT NULL DEFAULT 0,
  label           TEXT NOT NULL,         -- 'synthetic RO' (households) | 'real RO' (ENTSO-E load)
  PRIMARY KEY (run_id, level, zone_id, ts)
);

CREATE INDEX zone_demand_ts_idx ON zone_demand(run_id, level, ts);

-- park_report action: the daily park report (#32, SHOULD)
CREATE TABLE reports (
  action_id        TEXT PRIMARY KEY,
  run_id           TEXT NOT NULL REFERENCES runs(id),
  park_id          TEXT REFERENCES parks(id),
  date             TEXT NOT NULL,        -- 'YYYY-MM-DD'
  energy_mwh       REAL,
  expected_mwh     REAL,
  availability_pct REAL,
  curtailed_mwh    REAL,
  data_quality_pct REAL,
  issues_json      TEXT
);

-- ---------------------------------------------------------------- audit
-- §3.4: every action gets an audit_record (model_version, input_hash, prediction, threshold,
-- status, schedule). #32's audit panel joins on action_id; realised vs committed is shown here.

CREATE TABLE audit_log (
  id              TEXT PRIMARY KEY,
  run_id          TEXT NOT NULL REFERENCES runs(id),
  action_id       TEXT,                  -- work_orders / alerts / review_queue / gis_flags / reports
  action_type     TEXT NOT NULL,         -- battery_schedule | feeder_commitment | work_order | alert
                                         -- | gis_flag | human_review | park_report | zone_demand
  ts              TEXT NOT NULL,
  policy          TEXT,
  actor           TEXT NOT NULL,         -- agent | human
  asset_id        TEXT REFERENCES assets(id),
  feeder_id       TEXT REFERENCES feeders(id),
  park_id         TEXT REFERENCES parks(id),
  model           TEXT,
  model_version   TEXT,
  input_hash      TEXT,
  prediction_json TEXT,
  threshold       REAL,
  status          TEXT,
  schedule_json   TEXT,
  committed_kw    REAL,
  realised_kw     REAL,
  note            TEXT
);

CREATE INDEX audit_log_run_ts_idx ON audit_log(run_id, ts);
CREATE INDEX audit_log_action_idx ON audit_log(action_id);
