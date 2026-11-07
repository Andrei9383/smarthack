CREATE TABLE `judge_verdicts` (
	`run_id` text NOT NULL,
	`day` text NOT NULL,
	`t0` text,
	`asset_id` text NOT NULL,
	`model` text NOT NULL,
	`model_version` text,
	`bucket` text,
	`verdict` text NOT NULL,
	`p_valid` real,
	`p_valid_sd` real,
	`trust` real,
	`n_eff` real,
	`band_scale` real,
	`substitute` text,
	`reasons_json` text,
	`rule` text NOT NULL,
	`judge_version` text NOT NULL,
	CONSTRAINT `judge_verdicts_pk` PRIMARY KEY(`run_id`, `day`, `asset_id`, `model`),
	CONSTRAINT `fk_judge_verdicts_run_id_runs_id_fk` FOREIGN KEY (`run_id`) REFERENCES `runs`(`id`)
);
--> statement-breakpoint
CREATE TABLE `outcomes` (
	`id` integer PRIMARY KEY AUTOINCREMENT,
	`run_id` text NOT NULL,
	`model` text NOT NULL,
	`model_version` text,
	`asset_id` text NOT NULL,
	`day` text NOT NULL,
	`score_name` text NOT NULL,
	`score` real,
	`weight` real DEFAULT 1 NOT NULL,
	`source` text NOT NULL,
	`observed_at` text NOT NULL,
	`gated` integer DEFAULT 0 NOT NULL,
	`bucket` text,
	CONSTRAINT `fk_outcomes_run_id_runs_id_fk` FOREIGN KEY (`run_id`) REFERENCES `runs`(`id`)
);
--> statement-breakpoint
CREATE TABLE `trust` (
	`run_id` text NOT NULL,
	`day` text NOT NULL,
	`model` text NOT NULL,
	`model_version` text NOT NULL,
	`bucket` text NOT NULL,
	`score_name` text NOT NULL,
	`prior` real,
	`n_eff` real DEFAULT 0 NOT NULL,
	`trust` real,
	`lo80` real,
	`hi80` real,
	`band_scale` real DEFAULT 1 NOT NULL,
	`health_alert` integer DEFAULT 0 NOT NULL,
	CONSTRAINT `trust_pk` PRIMARY KEY(`run_id`, `day`, `model`, `model_version`, `bucket`, `score_name`),
	CONSTRAINT `fk_trust_run_id_runs_id_fk` FOREIGN KEY (`run_id`) REFERENCES `runs`(`id`)
);
--> statement-breakpoint
CREATE INDEX `judge_verdicts_asset_idx` ON `judge_verdicts` (`asset_id`,`day`);--> statement-breakpoint
CREATE INDEX `judge_verdicts_run_verdict_idx` ON `judge_verdicts` (`run_id`,`verdict`);--> statement-breakpoint
CREATE INDEX `outcomes_run_model_day_idx` ON `outcomes` (`run_id`,`model`,`day`);--> statement-breakpoint
CREATE INDEX `outcomes_asset_idx` ON `outcomes` (`asset_id`,`day`);--> statement-breakpoint
CREATE INDEX `trust_run_model_idx` ON `trust` (`run_id`,`model`,`day`);