# Replay output: what lands in SQLite

The Drizzle schema in `packages/db` (#38, Tudor; final at d362a39, column list in `team/contracts/app.md`) is the
source of truth for table and column names. Python applies its generated SQL (`packages/db/drizzle/*/migration.sql`,
or until #38 merges the byte copy `sentinel/agent/schema_fallback.sql`) with `sqlite3` and only inserts rows.
Conventions: times are **TEXT ISO-8601 UTC** in `ts` (`day`/`date` as `YYYY-MM-DD`); there is **no `step` column**,
the UI buckets `ts` in Europe/Bucharest; every output row carries `run_id`; baseline and Sentinel are separate runs
(`policy`); JSON in `*_json` TEXT columns; units in column names (W from the simulator are converted to kW).

`python replay.py --policy baseline,sentinel --out $LAB_OUTPUT_DIR/sentinel.sqlite [--world mock|sim] [--days 7]
[--seed 0] [--full] [--no-strict]` writes, for one test week and each policy:

| Output | Table | Columns written |
|---|---|---|
| Run manifest | `runs` | one row per policy: `kind='replay'`, `policy`, `label` (synthetic RO / real RO / non-RO), `week_start`, `git_sha`, `model_versions_json`, `created_at`, `finished_at`, `notes` (components real vs fallback, errors, world, inputs hash) |
| KPIs | `kpis` | `peak_reduction`, `minutes_over_limit`, `p10_hit_rate`, `comfort_breaches`, `lei_per_home`, `alerts_per_turbine_day` (+ `peak_reduction_congested`, `clip_rate`) per policy, with `unit`, `label` |
| Topology | `feeders`, `parks`, `assets` | feeders `limit_kw, g_max_kw, g_conservative_kw, tau, gis_trust`; parks `rated_mw, turbine_count`; assets `asset_kind` (prosumer / turbine), `gis_phase, gis_trust, battery_kwh, battery_kw, comfort_reserve_kwh, recorded_/true_ kwp, tilt_deg, azimuth_deg` |
| Expected vs actual | `asset_steps` | `actual_kw, q10_kw, q50_kw, q90_kw` per site per step (default DB omits night rows whose neighbours are also all-zero; read a missing step as 0) |
| Raw measurements | `measurements` | sites `pv_ac_kw, load_kw, batt_kw, soc, inv_voltage_v, inv_status, data_quality`; turbines `act_pwt_kw, wd_spd_kmh, ambient_temp_c, gearbox_temp_c, error_code, data_quality`; written once per DB (under the Sentinel run) |
| Model outputs | `predictions` | `model, model_version, t0, ts, horizon_idx, q10/q50/q90, fault_probs_json, ood_score, member_std, is_fallback`; default DB: daily site-health rows; `--full` adds per-step PV/load rows |
| Asset status per tick | `statuses` | `status, headroom_factor, reason, telemetry_age_s` (on change + day start) |
| Feeder curves | `feeder_curves` | per policy: `export_kw, import_kw, limit_kw, slack_kw, is_violation` |
| `feeder_commitment` | `commitments` | `nominal_kw >= available_kw >= committed_kw`, `realised_kw`, `haircut_h`, `hit` |
| `battery_schedule` | `schedules` | `batt_kw, soc, comfort_reserve_kwh, comfort_breached, clipped`; default DB: Sentinel only, `--full` adds baseline |
| `work_order` | `work_orders` | `crew_type, action, ts, day, fault, confidence, lost_kwh_day, lost_flex_mwh_day, lost_lei_day, mw_at_risk, days_to_visit, score, rank, status, evidence_json` |
| `alert` | `alerts` | `code, severity, creates_work_order, curtailed_mwh, message, evidence_json`; curtailment log has `creates_work_order = 0` |
| `human_review` | `review_queue` | `reason, ood_score, interval_width, member_std, resolved = 0` |
| `gis_flag` | `gis_flags` | `section_id`, recorded vs suggested feeder/phase, `tau`, `geometry_json` |
| Park forecast | `park_forecasts` | `p10/p50/p90_mw`, expected MWh, availability, curtailed MWh, data quality |
| `park_report` | `reports` | daily: `date, energy_mwh, expected_mwh, availability_pct, curtailed_mwh, data_quality_pct, issues_json` |
| `zone_demand` | `zone_demand` | feeder -> locality -> county: `level, zone_name, parent_zone_id, q10/q50/q90_mw, peak_stress, is_peak_hour, label` |
| Every action's audit | `audit_log` | `action_id, action_type, ts, policy, actor, asset/feeder/park_id, model_version, input_hash, prediction_json, threshold, status, schedule_json, note` |

Every action row has its `action_id` in `audit_log`, so the UI's audit panel joins on it.

**Strict mode** (default in `replay.py` and the tests; `--no-strict` to only warn): a field the agent produces that
has no column (after the name aliases in `sentinel/agent/audit.py`), a NOT NULL column left without a value, or a
missing table raises `SchemaMismatch`. The only exceptions are listed, with reasons, in `audit.ALLOWED_DROPS`:
`step` (any table), `feeders.n_sites / installed_pv_kw / installed_batt_kwh` (derivable from `assets`),
`gis_flags.confidence`, `zone_demand.stress` (per-step) and `zone_demand.model_version` (in `audit_log`).

**Size**: the default 7-day mock DB is ~47 MB (R18 < 50 MB). `--full` keeps per-step PV/load predictions, baseline
schedules and night `asset_steps` rows (~115 MB).
