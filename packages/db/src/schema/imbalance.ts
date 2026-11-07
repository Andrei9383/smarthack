import { index, real, sqliteTable, text } from "drizzle-orm/sqlite-core";

// Feature I (#177, spec I step 9a). Written by sentinel/eval/imbalance_replay.py (#176, `TABLES`): it drops and
// recreates these tables, so columns and index names mirror that writer. ts / issue_time = ISO-8601 UTC text,
// quantities MWh per 15-min step, + = the portfolio takes from the grid.

export const portfolioForecast = sqliteTable(
  "portfolio_forecast",
  {
    issueTime: text("issue_time").notNull(),
    ts: text("ts").notNull(),
    policy: text("policy").notNull(), // profile | persistence | forecast_only | sentinel
    horizon: text("horizon"), // da (D-1) | id (15:00 re-forecast)
    p10: real("p10"), // NULL for point policies
    p50: real("p50"),
    p90: real("p90"),
    runId: text("run_id"),
  },
  (table) => [index("portfolio_forecast_idx").on(table.ts, table.policy)],
);

export const positions = sqliteTable(
  "positions",
  {
    ts: text("ts").notNull(),
    policy: text("policy").notNull(), // + sentinel_id (level 2: trades the 15:00 correction on PI)
    boughtDa: real("bought_da"),
    tradedId: real("traded_id"),
    actual: real("actual"),
    devMwh: real("dev_mwh"), // actual - bought_da - traded_id (> 0 short)
    intradayLei: real("intraday_lei"),
    imbalanceLei: real("imbalance_lei"),
    avoidableLei: real("avoidable_lei"),
    pzuLeiMwh: real("pzu_lei_mwh"),
    piLeiMwh: real("pi_lei_mwh"),
    importMwh: real("import_mwh"),
    exportMwh: real("export_mwh"),
    runId: text("run_id"),
  },
  (table) => [index("positions_idx").on(table.ts, table.policy)],
);

// Totals over the replayed days, one row per (policy, name).
export const imbalanceKpis = sqliteTable(
  "imbalance_kpis",
  {
    runId: text("run_id"),
    policy: text("policy").notNull(),
    name: text("name").notNull(),
    value: real("value"),
    unit: text("unit"),
    label: text("label"),
    dateFrom: text("date_from"),
    dateTo: text("date_to"),
  },
  (table) => [index("imbalance_kpis_idx").on(table.policy, table.name)],
);

// I4 #174 battery strategy per home. No writer yet: the API falls back to assets.has_battery.
export const siteStrategy = sqliteTable("site_strategy", {
  siteId: text("site_id").primaryKey(),
  strategy: text("strategy"), // none | self_consumption | price_driven | enrolled
  confidence: real("confidence"),
});
