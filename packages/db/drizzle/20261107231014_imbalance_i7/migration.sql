CREATE TABLE `imbalance_kpis` (
	`run_id` text,
	`policy` text NOT NULL,
	`name` text NOT NULL,
	`value` real,
	`unit` text,
	`label` text,
	`date_from` text,
	`date_to` text
);
--> statement-breakpoint
CREATE TABLE `portfolio_forecast` (
	`issue_time` text NOT NULL,
	`ts` text NOT NULL,
	`policy` text NOT NULL,
	`horizon` text,
	`p10` real,
	`p50` real,
	`p90` real,
	`run_id` text
);
--> statement-breakpoint
CREATE TABLE `positions` (
	`ts` text NOT NULL,
	`policy` text NOT NULL,
	`bought_da` real,
	`traded_id` real,
	`actual` real,
	`dev_mwh` real,
	`intraday_lei` real,
	`imbalance_lei` real,
	`avoidable_lei` real,
	`pzu_lei_mwh` real,
	`pi_lei_mwh` real,
	`import_mwh` real,
	`export_mwh` real,
	`run_id` text
);
--> statement-breakpoint
CREATE TABLE `site_strategy` (
	`site_id` text PRIMARY KEY,
	`strategy` text,
	`confidence` real
);
--> statement-breakpoint
ALTER TABLE `assets` ADD `contract` text;--> statement-breakpoint
ALTER TABLE `assets` ADD `meter` text;--> statement-breakpoint
CREATE INDEX `imbalance_kpis_idx` ON `imbalance_kpis` (`policy`,`name`);--> statement-breakpoint
CREATE INDEX `portfolio_forecast_idx` ON `portfolio_forecast` (`ts`,`policy`);--> statement-breakpoint
CREATE INDEX `positions_idx` ON `positions` (`ts`,`policy`);