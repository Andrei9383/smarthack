-- Copy of the first Drizzle migration (packages/db/drizzle/20261107131320_panoramic_anita_blake/migration.sql).
-- audit.py applies packages/db/drizzle/ when present; this file is used only if that directory is missing.
-- Never edit columns here: change the Drizzle schema in packages/db/src/schema/ and regenerate.

CREATE TABLE `account` (
	`id` text PRIMARY KEY,
	`account_id` text NOT NULL,
	`provider_id` text NOT NULL,
	`user_id` text NOT NULL,
	`access_token` text,
	`refresh_token` text,
	`id_token` text,
	`access_token_expires_at` integer,
	`refresh_token_expires_at` integer,
	`scope` text,
	`password` text,
	`created_at` integer DEFAULT (cast(unixepoch('subsecond') * 1000 as integer)) NOT NULL,
	`updated_at` integer NOT NULL,
	CONSTRAINT `fk_account_user_id_user_id_fk` FOREIGN KEY (`user_id`) REFERENCES `user`(`id`) ON DELETE CASCADE
);
--> statement-breakpoint
CREATE TABLE `session` (
	`id` text PRIMARY KEY,
	`expires_at` integer NOT NULL,
	`token` text NOT NULL UNIQUE,
	`created_at` integer DEFAULT (cast(unixepoch('subsecond') * 1000 as integer)) NOT NULL,
	`updated_at` integer NOT NULL,
	`ip_address` text,
	`user_agent` text,
	`user_id` text NOT NULL,
	CONSTRAINT `fk_session_user_id_user_id_fk` FOREIGN KEY (`user_id`) REFERENCES `user`(`id`) ON DELETE CASCADE
);
--> statement-breakpoint
CREATE TABLE `user` (
	`id` text PRIMARY KEY,
	`name` text NOT NULL,
	`email` text NOT NULL UNIQUE,
	`email_verified` integer DEFAULT false NOT NULL,
	`image` text,
	`created_at` integer DEFAULT (cast(unixepoch('subsecond') * 1000 as integer)) NOT NULL,
	`updated_at` integer DEFAULT (cast(unixepoch('subsecond') * 1000 as integer)) NOT NULL
);
--> statement-breakpoint
CREATE TABLE `verification` (
	`id` text PRIMARY KEY,
	`identifier` text NOT NULL,
	`value` text NOT NULL,
	`expires_at` integer NOT NULL,
	`created_at` integer DEFAULT (cast(unixepoch('subsecond') * 1000 as integer)) NOT NULL,
	`updated_at` integer DEFAULT (cast(unixepoch('subsecond') * 1000 as integer)) NOT NULL
);
--> statement-breakpoint
CREATE TABLE `alerts` (
	`action_id` text PRIMARY KEY,
	`run_id` text NOT NULL,
	`asset_id` text,
	`park_id` text,
	`feeder_id` text,
	`ts` text NOT NULL,
	`code` text NOT NULL,
	`severity` text NOT NULL,
	`creates_work_order` integer DEFAULT 0 NOT NULL,
	`curtailed_mwh` real,
	`message` text NOT NULL,
	`evidence_json` text,
	CONSTRAINT `fk_alerts_run_id_runs_id_fk` FOREIGN KEY (`run_id`) REFERENCES `runs`(`id`),
	CONSTRAINT `fk_alerts_asset_id_assets_id_fk` FOREIGN KEY (`asset_id`) REFERENCES `assets`(`id`),
	CONSTRAINT `fk_alerts_park_id_parks_id_fk` FOREIGN KEY (`park_id`) REFERENCES `parks`(`id`),
	CONSTRAINT `fk_alerts_feeder_id_feeders_id_fk` FOREIGN KEY (`feeder_id`) REFERENCES `feeders`(`id`)
);
--> statement-breakpoint
CREATE TABLE `asset_steps` (
	`run_id` text NOT NULL,
	`asset_id` text NOT NULL,
	`ts` text NOT NULL,
	`actual_kw` real,
	`q10_kw` real,
	`q50_kw` real,
	`q90_kw` real,
	CONSTRAINT `asset_steps_pk` PRIMARY KEY(`run_id`, `asset_id`, `ts`),
	CONSTRAINT `fk_asset_steps_run_id_runs_id_fk` FOREIGN KEY (`run_id`) REFERENCES `runs`(`id`),
	CONSTRAINT `fk_asset_steps_asset_id_assets_id_fk` FOREIGN KEY (`asset_id`) REFERENCES `assets`(`id`)
);
--> statement-breakpoint
CREATE TABLE `assets` (
	`id` text PRIMARY KEY,
	`asset_kind` text NOT NULL,
	`name` text NOT NULL,
	`county` text NOT NULL,
	`locality` text,
	`lat` real NOT NULL,
	`lon` real NOT NULL,
	`feeder_id` text,
	`park_id` text,
	`gis_phase` text,
	`gis_trust` text,
	`rated_kw` real NOT NULL,
	`has_battery` integer DEFAULT 0 NOT NULL,
	`battery_kwh` real,
	`battery_kw` real,
	`comfort_reserve_kwh` real,
	`recorded_kwp` real,
	`true_kwp` real,
	`recorded_tilt_deg` real,
	`true_tilt_deg` real,
	`recorded_azimuth_deg` real,
	`true_azimuth_deg` real,
	`turbine_model` text,
	`hub_height_m` real,
	`rotor_diameter_m` real,
	`metadata_json` text,
	CONSTRAINT `fk_assets_feeder_id_feeders_id_fk` FOREIGN KEY (`feeder_id`) REFERENCES `feeders`(`id`),
	CONSTRAINT `fk_assets_park_id_parks_id_fk` FOREIGN KEY (`park_id`) REFERENCES `parks`(`id`)
);
--> statement-breakpoint
CREATE TABLE `audit_log` (
	`id` text PRIMARY KEY,
	`run_id` text NOT NULL,
	`action_id` text,
	`action_type` text NOT NULL,
	`ts` text NOT NULL,
	`policy` text,
	`actor` text NOT NULL,
	`asset_id` text,
	`feeder_id` text,
	`park_id` text,
	`model` text,
	`model_version` text,
	`input_hash` text,
	`prediction_json` text,
	`threshold` real,
	`status` text,
	`schedule_json` text,
	`committed_kw` real,
	`realised_kw` real,
	`note` text,
	CONSTRAINT `fk_audit_log_run_id_runs_id_fk` FOREIGN KEY (`run_id`) REFERENCES `runs`(`id`),
	CONSTRAINT `fk_audit_log_asset_id_assets_id_fk` FOREIGN KEY (`asset_id`) REFERENCES `assets`(`id`),
	CONSTRAINT `fk_audit_log_feeder_id_feeders_id_fk` FOREIGN KEY (`feeder_id`) REFERENCES `feeders`(`id`),
	CONSTRAINT `fk_audit_log_park_id_parks_id_fk` FOREIGN KEY (`park_id`) REFERENCES `parks`(`id`)
);
--> statement-breakpoint
CREATE TABLE `commitments` (
	`run_id` text NOT NULL,
	`feeder_id` text NOT NULL,
	`ts` text NOT NULL,
	`nominal_kw` real NOT NULL,
	`available_kw` real NOT NULL,
	`committed_kw` real NOT NULL,
	`realised_kw` real,
	`haircut_h` real,
	`hit` integer,
	CONSTRAINT `commitments_pk` PRIMARY KEY(`run_id`, `feeder_id`, `ts`),
	CONSTRAINT `fk_commitments_run_id_runs_id_fk` FOREIGN KEY (`run_id`) REFERENCES `runs`(`id`),
	CONSTRAINT `fk_commitments_feeder_id_feeders_id_fk` FOREIGN KEY (`feeder_id`) REFERENCES `feeders`(`id`)
);
--> statement-breakpoint
CREATE TABLE `feeder_curves` (
	`run_id` text NOT NULL,
	`feeder_id` text NOT NULL,
	`policy` text NOT NULL,
	`ts` text NOT NULL,
	`export_kw` real NOT NULL,
	`import_kw` real DEFAULT 0 NOT NULL,
	`limit_kw` real NOT NULL,
	`slack_kw` real NOT NULL,
	`is_violation` integer DEFAULT 0 NOT NULL,
	CONSTRAINT `feeder_curves_pk` PRIMARY KEY(`run_id`, `feeder_id`, `policy`, `ts`),
	CONSTRAINT `fk_feeder_curves_run_id_runs_id_fk` FOREIGN KEY (`run_id`) REFERENCES `runs`(`id`),
	CONSTRAINT `fk_feeder_curves_feeder_id_feeders_id_fk` FOREIGN KEY (`feeder_id`) REFERENCES `feeders`(`id`)
);
--> statement-breakpoint
CREATE TABLE `feeders` (
	`id` text PRIMARY KEY,
	`name` text NOT NULL,
	`county` text NOT NULL,
	`locality` text NOT NULL,
	`limit_kw` real NOT NULL,
	`g_max_kw` real,
	`g_conservative_kw` real,
	`tau` real,
	`gis_trust` text,
	`geometry_json` text
);
--> statement-breakpoint
CREATE TABLE `gis_flags` (
	`action_id` text PRIMARY KEY,
	`run_id` text NOT NULL,
	`section_id` text NOT NULL,
	`recorded_feeder` text,
	`recorded_phase` text,
	`suggested_feeder` text,
	`suggested_phase` text,
	`tau` real,
	`geometry_json` text,
	CONSTRAINT `fk_gis_flags_run_id_runs_id_fk` FOREIGN KEY (`run_id`) REFERENCES `runs`(`id`)
);
--> statement-breakpoint
CREATE TABLE `kpis` (
	`run_id` text NOT NULL,
	`policy` text NOT NULL,
	`name` text NOT NULL,
	`value` real NOT NULL,
	`unit` text,
	`label` text NOT NULL,
	CONSTRAINT `kpis_pk` PRIMARY KEY(`run_id`, `policy`, `name`),
	CONSTRAINT `fk_kpis_run_id_runs_id_fk` FOREIGN KEY (`run_id`) REFERENCES `runs`(`id`)
);
--> statement-breakpoint
CREATE TABLE `measurements` (
	`run_id` text NOT NULL,
	`asset_id` text NOT NULL,
	`ts` text NOT NULL,
	`pv_ac_kw` real,
	`load_kw` real,
	`batt_kw` real,
	`soc` real,
	`inv_voltage_v` real,
	`inv_status` text,
	`act_pwt_kw` real,
	`wd_spd_kmh` real,
	`wd_dir_deg` real,
	`meteo_wind_kmh` real,
	`ambient_temp_c` real,
	`conv_temp_c` real,
	`gearbox_temp_c` real,
	`gen_temp_c` real,
	`transformer_temp_c` real,
	`error_code` integer,
	`data_quality` text DEFAULT 'ok',
	CONSTRAINT `measurements_pk` PRIMARY KEY(`run_id`, `asset_id`, `ts`),
	CONSTRAINT `fk_measurements_run_id_runs_id_fk` FOREIGN KEY (`run_id`) REFERENCES `runs`(`id`),
	CONSTRAINT `fk_measurements_asset_id_assets_id_fk` FOREIGN KEY (`asset_id`) REFERENCES `assets`(`id`)
);
--> statement-breakpoint
CREATE TABLE `park_forecasts` (
	`run_id` text NOT NULL,
	`park_id` text NOT NULL,
	`ts` text NOT NULL,
	`p10_mw` real,
	`p50_mw` real,
	`p90_mw` real,
	`expected_mwh` real,
	`availability_pct` real,
	`curtailed_mwh` real,
	`data_quality_pct` real,
	CONSTRAINT `park_forecasts_pk` PRIMARY KEY(`run_id`, `park_id`, `ts`),
	CONSTRAINT `fk_park_forecasts_run_id_runs_id_fk` FOREIGN KEY (`run_id`) REFERENCES `runs`(`id`),
	CONSTRAINT `fk_park_forecasts_park_id_parks_id_fk` FOREIGN KEY (`park_id`) REFERENCES `parks`(`id`)
);
--> statement-breakpoint
CREATE TABLE `parks` (
	`id` text PRIMARY KEY,
	`name` text NOT NULL,
	`county` text NOT NULL,
	`lat` real NOT NULL,
	`lon` real NOT NULL,
	`rated_mw` real NOT NULL,
	`turbine_count` integer NOT NULL
);
--> statement-breakpoint
CREATE TABLE `predictions` (
	`run_id` text NOT NULL,
	`asset_id` text NOT NULL,
	`model` text NOT NULL,
	`model_version` text NOT NULL,
	`t0` text NOT NULL,
	`ts` text NOT NULL,
	`horizon_idx` integer NOT NULL,
	`q10` real,
	`q50` real,
	`q90` real,
	`fault_probs_json` text,
	`anomaly_class` text,
	`ood_score` real,
	`member_std` real,
	`is_fallback` integer DEFAULT 0 NOT NULL,
	CONSTRAINT `predictions_pk` PRIMARY KEY(`run_id`, `asset_id`, `model`, `t0`, `ts`),
	CONSTRAINT `fk_predictions_run_id_runs_id_fk` FOREIGN KEY (`run_id`) REFERENCES `runs`(`id`),
	CONSTRAINT `fk_predictions_asset_id_assets_id_fk` FOREIGN KEY (`asset_id`) REFERENCES `assets`(`id`)
);
--> statement-breakpoint
CREATE TABLE `reports` (
	`action_id` text PRIMARY KEY,
	`run_id` text NOT NULL,
	`park_id` text,
	`date` text NOT NULL,
	`energy_mwh` real,
	`expected_mwh` real,
	`availability_pct` real,
	`curtailed_mwh` real,
	`data_quality_pct` real,
	`issues_json` text,
	CONSTRAINT `fk_reports_run_id_runs_id_fk` FOREIGN KEY (`run_id`) REFERENCES `runs`(`id`),
	CONSTRAINT `fk_reports_park_id_parks_id_fk` FOREIGN KEY (`park_id`) REFERENCES `parks`(`id`)
);
--> statement-breakpoint
CREATE TABLE `review_queue` (
	`action_id` text PRIMARY KEY,
	`run_id` text NOT NULL,
	`asset_id` text NOT NULL,
	`ts` text NOT NULL,
	`reason` text NOT NULL,
	`ood_score` real,
	`interval_width` real,
	`member_std` real,
	`resolved` integer DEFAULT 0 NOT NULL,
	CONSTRAINT `fk_review_queue_run_id_runs_id_fk` FOREIGN KEY (`run_id`) REFERENCES `runs`(`id`),
	CONSTRAINT `fk_review_queue_asset_id_assets_id_fk` FOREIGN KEY (`asset_id`) REFERENCES `assets`(`id`)
);
--> statement-breakpoint
CREATE TABLE `runs` (
	`id` text PRIMARY KEY,
	`kind` text NOT NULL,
	`policy` text,
	`label` text NOT NULL,
	`week_start` text,
	`git_sha` text,
	`model_versions_json` text,
	`created_at` text NOT NULL,
	`finished_at` text,
	`notes` text
);
--> statement-breakpoint
CREATE TABLE `schedules` (
	`run_id` text NOT NULL,
	`asset_id` text NOT NULL,
	`policy` text DEFAULT 'sentinel' NOT NULL,
	`ts` text NOT NULL,
	`batt_kw` real NOT NULL,
	`soc` real NOT NULL,
	`comfort_reserve_kwh` real,
	`comfort_breached` integer DEFAULT 0 NOT NULL,
	`clipped` integer DEFAULT 0 NOT NULL,
	CONSTRAINT `schedules_pk` PRIMARY KEY(`run_id`, `asset_id`, `policy`, `ts`),
	CONSTRAINT `fk_schedules_run_id_runs_id_fk` FOREIGN KEY (`run_id`) REFERENCES `runs`(`id`),
	CONSTRAINT `fk_schedules_asset_id_assets_id_fk` FOREIGN KEY (`asset_id`) REFERENCES `assets`(`id`)
);
--> statement-breakpoint
CREATE TABLE `statuses` (
	`run_id` text NOT NULL,
	`asset_id` text NOT NULL,
	`ts` text NOT NULL,
	`status` text NOT NULL,
	`headroom_factor` real DEFAULT 1 NOT NULL,
	`reason` text NOT NULL,
	`telemetry_age_s` real,
	CONSTRAINT `statuses_pk` PRIMARY KEY(`run_id`, `asset_id`, `ts`),
	CONSTRAINT `fk_statuses_run_id_runs_id_fk` FOREIGN KEY (`run_id`) REFERENCES `runs`(`id`),
	CONSTRAINT `fk_statuses_asset_id_assets_id_fk` FOREIGN KEY (`asset_id`) REFERENCES `assets`(`id`)
);
--> statement-breakpoint
CREATE TABLE `work_orders` (
	`action_id` text PRIMARY KEY,
	`run_id` text NOT NULL,
	`asset_id` text NOT NULL,
	`asset_kind` text NOT NULL,
	`crew_type` text NOT NULL,
	`action` text NOT NULL,
	`ts` text NOT NULL,
	`day` text NOT NULL,
	`fault` text,
	`confidence` real NOT NULL,
	`lost_kwh_day` real,
	`lost_flex_mwh_day` real,
	`lost_lei_day` real,
	`mw_at_risk` real,
	`days_to_visit` integer,
	`score` real NOT NULL,
	`rank` integer,
	`status` text DEFAULT 'open' NOT NULL,
	`evidence_json` text,
	CONSTRAINT `fk_work_orders_run_id_runs_id_fk` FOREIGN KEY (`run_id`) REFERENCES `runs`(`id`),
	CONSTRAINT `fk_work_orders_asset_id_assets_id_fk` FOREIGN KEY (`asset_id`) REFERENCES `assets`(`id`)
);
--> statement-breakpoint
CREATE TABLE `zone_demand` (
	`run_id` text NOT NULL,
	`zone_id` text NOT NULL,
	`level` text NOT NULL,
	`zone_name` text NOT NULL,
	`parent_zone_id` text,
	`ts` text NOT NULL,
	`q10_mw` real,
	`q50_mw` real NOT NULL,
	`q90_mw` real,
	`peak_stress` real NOT NULL,
	`is_peak_hour` integer DEFAULT 0 NOT NULL,
	`label` text NOT NULL,
	CONSTRAINT `zone_demand_pk` PRIMARY KEY(`run_id`, `level`, `zone_id`, `ts`),
	CONSTRAINT `fk_zone_demand_run_id_runs_id_fk` FOREIGN KEY (`run_id`) REFERENCES `runs`(`id`)
);
--> statement-breakpoint
CREATE INDEX `account_userId_idx` ON `account` (`user_id`);--> statement-breakpoint
CREATE INDEX `session_userId_idx` ON `session` (`user_id`);--> statement-breakpoint
CREATE INDEX `verification_identifier_idx` ON `verification` (`identifier`);--> statement-breakpoint
CREATE INDEX `alerts_run_ts_idx` ON `alerts` (`run_id`,`ts`);--> statement-breakpoint
CREATE INDEX `asset_steps_asset_ts_idx` ON `asset_steps` (`asset_id`,`ts`);--> statement-breakpoint
CREATE INDEX `assets_feeder_idx` ON `assets` (`feeder_id`);--> statement-breakpoint
CREATE INDEX `assets_park_idx` ON `assets` (`park_id`);--> statement-breakpoint
CREATE INDEX `assets_county_idx` ON `assets` (`county`);--> statement-breakpoint
CREATE INDEX `audit_log_run_ts_idx` ON `audit_log` (`run_id`,`ts`);--> statement-breakpoint
CREATE INDEX `audit_log_action_idx` ON `audit_log` (`action_id`);--> statement-breakpoint
CREATE INDEX `commitments_feeder_ts_idx` ON `commitments` (`feeder_id`,`ts`);--> statement-breakpoint
CREATE INDEX `feeder_curves_feeder_policy_idx` ON `feeder_curves` (`feeder_id`,`policy`);--> statement-breakpoint
CREATE INDEX `kpis_run_policy_idx` ON `kpis` (`run_id`,`policy`);--> statement-breakpoint
CREATE INDEX `measurements_asset_ts_idx` ON `measurements` (`asset_id`,`ts`);--> statement-breakpoint
CREATE INDEX `park_forecasts_ts_idx` ON `park_forecasts` (`park_id`,`ts`);--> statement-breakpoint
CREATE INDEX `predictions_asset_ts_idx` ON `predictions` (`asset_id`,`ts`);--> statement-breakpoint
CREATE INDEX `predictions_run_model_idx` ON `predictions` (`run_id`,`model`);--> statement-breakpoint
CREATE INDEX `schedules_asset_policy_idx` ON `schedules` (`asset_id`,`policy`);--> statement-breakpoint
CREATE INDEX `statuses_status_idx` ON `statuses` (`status`);--> statement-breakpoint
CREATE INDEX `work_orders_queue_idx` ON `work_orders` (`run_id`,`crew_type`,`day`,`rank`);--> statement-breakpoint
CREATE INDEX `zone_demand_ts_idx` ON `zone_demand` (`run_id`,`level`,`ts`);