import { implement } from "@orpc/server";
import { createDb, type Database } from "@wattsentry/db";
import { imbalanceKpis, portfolioForecast, positions, siteStrategy } from "@wattsentry/db/schema/imbalance";
import { assets } from "@wattsentry/db/schema/replay";
import { and, asc, desc, eq, gte, lt, sql, type SQL } from "drizzle-orm";
import { existsSync } from "node:fs";
import { resolve } from "node:path";

import type { Context } from "../context";
import { imbalanceContract, type PolicyTotals } from "../contract/imbalance";
import { emptyIfNoTables, inWindow as tsWindow } from "./runs";

const os = implement(imbalanceContract).$context<Context>();
const DAY_MS = 86_400_000;
const iso = (ms: number) => new Date(ms).toISOString().replace(".000Z", "Z");

// The imbalance replay's tables (positions, portfolio_forecast) live in their own file when one exists:
// WATTSENTRY_IMBALANCE_DB, else demo/imbalance.sqlite (kept out of demo/wattsentry.sqlite to keep that file
// unchanged and small). Otherwise they are read from the app DB.
const IMBALANCE_FILE = [process.env.WATTSENTRY_IMBALANCE_DB, "demo/imbalance.sqlite"]
  .filter((p): p is string => Boolean(p))
  .flatMap((p) => [resolve(p), resolve("../..", p)])
  .find(existsSync);
let imbalanceDb: Database | undefined;
const store = (db: Database): Database =>
  IMBALANCE_FILE ? (imbalanceDb ??= createDb({ DATABASE_URL: `file:${IMBALANCE_FILE}` })) : db;

// Bucharest local hour in SQL: EEST (+3) between the last Sundays of March and October, approximated by
// 30 Mar - 25 Oct (exact in 2025, the replayed year), else EET (+2).
const localHour = sql`CAST(strftime('%H', ${positions.ts}, CASE WHEN substr(${positions.ts}, 6, 5) BETWEEN '03-30' AND '10-25' THEN '+3 hours' ELSE '+2 hours' END) AS INTEGER)`;
const f = (expr: SQL) => sql<number>`COALESCE(${expr}, 0)`.mapWith(Number);

const inWindow = (from?: string, to?: string) => tsWindow(positions.ts, from, to);

async function totals(db: Database, where?: SQL): Promise<PolicyTotals[]> {
  const rows = await emptyIfNoTables(() =>
    db
      .select({
        policy: positions.policy,
        avoidableLei: f(sql`SUM(${positions.avoidableLei})`),
        intradayLei: f(sql`SUM(${positions.intradayLei})`),
        imbalanceLei: f(sql`SUM(${positions.imbalanceLei})`),
        absDevMwh: f(sql`SUM(ABS(${positions.devMwh}))`),
        exportMwh: f(sql`SUM(${positions.exportMwh})`),
        doubleCover1821Mwh: f(sql`SUM(CASE WHEN ${localHour} BETWEEN 18 AND 20 AND ${positions.devMwh} < 0 THEN -${positions.devMwh} END)`),
        days: f(sql`COUNT(*) / 96.0`), // 15-min steps per day
      })
      .from(positions)
      .where(where)
      .groupBy(positions.policy),
  );
  return rows.map((r) => ({ ...r, leiPerMwhInjected: r.exportMwh > 0 ? r.avoidableLei / r.exportMwh : null }));
}

// Whole-replay totals as the replay wrote them (imbalance_kpis = the run's LAB_METRIC headline). The demo copy of
// positions only covers the days around the demo week, so summing it would not be the whole replay.
const KPI_OF = {
  avoidable_cost_lei: "avoidableLei",
  intraday_lei: "intradayLei",
  imbalance_lei: "imbalanceLei",
  abs_dev_mwh: "absDevMwh",
  export_mwh: "exportMwh",
  double_cover_mwh_18_21: "doubleCover1821Mwh",
  lei_per_mwh_injected: "leiPerMwhInjected",
  steps: "days",
} as const;

async function replayTotals(db: Database): Promise<PolicyTotals[]> {
  const rows = await emptyIfNoTables(() =>
    db.select({ policy: imbalanceKpis.policy, name: imbalanceKpis.name, value: imbalanceKpis.value }).from(imbalanceKpis),
  );
  const by = new Map<string, PolicyTotals>();
  for (const { policy, name, value } of rows) {
    const key = KPI_OF[name as keyof typeof KPI_OF];
    if (!key || value == null) continue;
    const t = by.get(policy) ?? {
      policy,
      avoidableLei: 0,
      intradayLei: 0,
      imbalanceLei: 0,
      absDevMwh: 0,
      exportMwh: 0,
      leiPerMwhInjected: null,
      doubleCover1821Mwh: 0,
      days: 0,
    };
    t[key] = key === "days" ? value / 96 : value; // steps of 15 min
    by.set(policy, t);
  }
  return [...by.values()];
}

// The replayed day nearest to `from`, shifted by whole days so it keeps the caller's local-midnight offset.
async function nearestWindow(db: Database, from: string) {
  const pick = (where: SQL, order: SQL) =>
    emptyIfNoTables(() => db.select({ ts: positions.ts }).from(positions).where(where).orderBy(order).limit(1));
  const [after] = await pick(gte(positions.ts, from), asc(positions.ts));
  const [before] = after ? [] : await pick(lt(positions.ts, from), desc(positions.ts));
  const hit = after ?? before;
  if (!hit) return null;
  const start = Date.parse(from) + Math.floor((Date.parse(hit.ts) - Date.parse(from)) / DAY_MS) * DAY_MS;
  return { from: iso(start), to: iso(start + DAY_MS) };
}

export const imbalanceRouter = os.router({
  getImbalanceDay: os.getImbalanceDay.handler(async ({ context, input }) => {
    const db = store(context.db);
    const from = input.from ?? "2025-06-02T21:00:00Z";
    let win = { from, to: input.to ?? iso(Date.parse(from) + DAY_MS) };
    const has = await emptyIfNoTables(() =>
      db.select({ ts: positions.ts }).from(positions).where(inWindow(win.from, win.to)).limit(1),
    );
    if (!has.length) {
      const near = await nearestWindow(db, from);
      if (!near) return { from: null, to: null, steps: [], totals: [] };
      win = near;
    }
    const where = and(inWindow(win.from, win.to), input.policy ? eq(positions.policy, input.policy) : undefined);
    // Each policy's own day-ahead band; sentinel_id reuses sentinel's.
    const band = sql`(${portfolioForecast.ts} = ${positions.ts} AND ${portfolioForecast.horizon} = 'da' AND ${portfolioForecast.policy} = CASE WHEN ${positions.policy} = 'sentinel_id' THEN 'sentinel' ELSE ${positions.policy} END)`;
    const [steps, tot] = await Promise.all([
      emptyIfNoTables(() =>
        db
          .select({
            ts: positions.ts,
            policy: positions.policy,
            p10: portfolioForecast.p10,
            p50: portfolioForecast.p50,
            p90: portfolioForecast.p90,
            bought: sql<number | null>`${positions.boughtDa} + COALESCE(${positions.tradedId}, 0)`,
            actual: positions.actual,
            devMwh: positions.devMwh,
            avoidableLei: positions.avoidableLei,
          })
          .from(positions)
          .leftJoin(portfolioForecast, band)
          .where(where)
          .orderBy(asc(positions.ts)),
      ),
      totals(db, where),
    ]);
    return { ...win, steps, totals: tot };
  }),

  getPolicyComparison: os.getPolicyComparison.handler(async ({ context, input }) => {
    const db = store(context.db);
    if (!input?.from && !input?.to) {
      const whole = await replayTotals(db);
      if (whole.length) return whole;
    }
    return totals(db, inWindow(input?.from, input?.to));
  }),

  getStrategyBreakdown: os.getStrategyBreakdown.handler(async ({ context: { db } }) => {
    const classified = await emptyIfNoTables(() =>
      db
        .select({
          strategy: sql<string>`COALESCE(${siteStrategy.strategy}, 'unknown')`,
          homes: sql<number>`COUNT(*)`.mapWith(Number),
          confidence: sql<number | null>`AVG(${siteStrategy.confidence})`,
        })
        .from(siteStrategy)
        .groupBy(siteStrategy.strategy),
    );
    if (classified.length) return { source: "site_strategy" as const, rows: classified };
    // Not classified yet (no strategy writer in the replay DB): split the fleet by declared battery only.
    const declared = await emptyIfNoTables(() =>
      db
        .select({
          strategy: sql<string>`CASE WHEN ${assets.hasBattery} = 1 THEN 'battery' ELSE 'none' END`,
          homes: sql<number>`COUNT(*)`.mapWith(Number),
          confidence: sql<number | null>`NULL`,
        })
        .from(assets)
        .where(eq(assets.assetKind, "prosumer"))
        .groupBy(sql`1`),
    );
    return { source: declared.length ? ("assets" as const) : ("none" as const), rows: declared };
  }),
});
