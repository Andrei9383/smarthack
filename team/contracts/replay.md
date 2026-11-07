# Replay output: what lands in SQLite

The Drizzle schema in `packages/db` (#38, Tudor) is the source of truth for table and column names; Python applies
its generated SQL with `sqlite3` and only inserts rows. This file maps our outputs onto those tables. Until #38
merges, the names below are the **proposal** #38 starts from and #30 (Alexandra) seeds fixtures in; when #38 lands,
this file is updated to its exact names.

`python replay.py --policy baseline,sentinel --out $LAB_OUTPUT_DIR/sentinel.sqlite` writes everything below for a
test week, for each policy. Times UTC ISO strings; arrays as JSON text.

| Output | Table (proposed) | Key columns |
|---|---|---|
| Run manifest | `runs` | `run_id, created_at, policy, git_sha, model_versions (JSON), label, week_start` |
| Sites, feeders, turbines, zones | `sites`, `feeders`, `turbines`, `zones` | static metadata (data.md / wind.md) + `geometry` GeoJSON where needed |
| Feeder curves per step | `feeder_steps` | `run_id, policy, feeder_id, ts, export_kw, import_kw, limit_kw, slack_kw` |
| `feeder_commitment` | `commitments` | `run_id, feeder_id, ts, nominal_kw, available_kw, committed_kw, realised_kw` |
| `battery_schedule` | `battery_steps` | `run_id, policy, site_id, ts, batt_kw, soc, comfort_reserve_kwh` |
| Asset status per tick | `asset_status` | `run_id, asset_id, asset_kind, ts, status, headroom_factor, reason` |
| Site view (expected vs actual) | `asset_steps` | `run_id, asset_id, ts, actual_kw, q10_kw, q50_kw, q90_kw` |
| `work_order` | `work_orders` | `run_id, action_id, ts, asset_id, asset_kind, fault, crew_type, score, lost_kwh_day, lost_lei_day, rank` |
| `alert` | `alerts` | `run_id, action_id, ts, asset_id, severity, code, message` |
| `human_review` | `review_queue` | `run_id, action_id, ts, asset_id, reason` |
| `gis_flag` | `gis_flags` | `run_id, section_id, recorded_*, suggested_*, tau, geometry` |
| `park_report` | `park_reports` | `run_id, park_id, date, energy_mwh, expected_mwh, availability, curtailed_mwh, issues (JSON)` |
| `zone_demand` | `zone_demand` | `run_id, zone_id, level, ts, q10_mw, q50_mw, q90_mw, peak_stress` |
| Every action's audit | `audit` | `run_id, action_id, type, ts, policy, model_version, input_hash, prediction (JSON), threshold, status, schedule (JSON)` |
| KPIs | `kpis` | `run_id, policy, name, value, label` (`peak_reduction`, `minutes_over_limit`, `p10_hit_rate`, `comfort_breaches`, `lei_per_home`, `alerts_per_turbine_day`) |

Every row an action produces also has its `action_id` in `audit`, so the UI's audit panel joins on it.
