CREATE TABLE `park_battery` (
	`run_id` text NOT NULL,
	`park_id` text NOT NULL,
	`variant` text NOT NULL,
	`ts` text NOT NULL,
	`sold_mwh` real,
	`actual_mwh` real,
	`charge_mwh` real,
	`discharge_mwh` real,
	`soc_mwh` real,
	`delivered_mwh` real,
	`dev_none_mwh` real,
	`dev_mwh` real,
	`avoidable_lei` real,
	`avoidable_none_lei` real,
	`net_cost_lei` real,
	CONSTRAINT `park_battery_pk` PRIMARY KEY(`run_id`, `park_id`, `variant`, `ts`),
	CONSTRAINT `fk_park_battery_run_id_runs_id_fk` FOREIGN KEY (`run_id`) REFERENCES `runs`(`id`),
	CONSTRAINT `fk_park_battery_park_id_parks_id_fk` FOREIGN KEY (`park_id`) REFERENCES `parks`(`id`)
);
--> statement-breakpoint
CREATE TABLE `turbine_availability` (
	`run_id` text NOT NULL,
	`asset_id` text NOT NULL,
	`park_id` text,
	`day` text NOT NULL,
	`rule` text,
	`availability_mean` real NOT NULL,
	`availability_min` real,
	`uncertain` integer DEFAULT 0 NOT NULL,
	`m6_class` text,
	`m6_verdict` text,
	`m6_source` text,
	`m7_p_fail_7d` real,
	CONSTRAINT `turbine_availability_pk` PRIMARY KEY(`run_id`, `asset_id`, `day`),
	CONSTRAINT `fk_turbine_availability_run_id_runs_id_fk` FOREIGN KEY (`run_id`) REFERENCES `runs`(`id`),
	CONSTRAINT `fk_turbine_availability_asset_id_assets_id_fk` FOREIGN KEY (`asset_id`) REFERENCES `assets`(`id`),
	CONSTRAINT `fk_turbine_availability_park_id_parks_id_fk` FOREIGN KEY (`park_id`) REFERENCES `parks`(`id`)
);
--> statement-breakpoint
CREATE TABLE `wind_positions` (
	`run_id` text NOT NULL,
	`park_id` text NOT NULL,
	`policy` text NOT NULL,
	`ts` text NOT NULL,
	`cap_mw` real,
	`actual_mwh` real,
	`sold_da_mwh` real NOT NULL,
	`sold_id_mwh` real,
	`fc_p10_mwh` real,
	`fc_p50_mwh` real,
	`fc_p90_mwh` real,
	`pzu_lei_mwh` real,
	`dev_mwh` real,
	`avoidable_lei` real,
	`removed_turbines` integer,
	CONSTRAINT `wind_positions_pk` PRIMARY KEY(`run_id`, `park_id`, `policy`, `ts`),
	CONSTRAINT `fk_wind_positions_run_id_runs_id_fk` FOREIGN KEY (`run_id`) REFERENCES `runs`(`id`),
	CONSTRAINT `fk_wind_positions_park_id_parks_id_fk` FOREIGN KEY (`park_id`) REFERENCES `parks`(`id`)
);
--> statement-breakpoint
ALTER TABLE `work_orders` ADD `lei_per_day` real;--> statement-breakpoint
ALTER TABLE `work_orders` ADD `triage_score` real;--> statement-breakpoint
ALTER TABLE `work_orders` ADD `best_repair_day` text;--> statement-breakpoint
CREATE INDEX `park_battery_park_ts_idx` ON `park_battery` (`park_id`,`ts`);--> statement-breakpoint
CREATE INDEX `turbine_availability_park_day_idx` ON `turbine_availability` (`park_id`,`day`);--> statement-breakpoint
CREATE INDEX `wind_positions_park_ts_idx` ON `wind_positions` (`park_id`,`ts`);