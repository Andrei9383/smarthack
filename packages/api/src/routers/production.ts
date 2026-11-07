import { implement } from "@orpc/server";
import type { Database } from "@sentinel-app/db";
import { assets, assetSteps, parkForecasts, parks, runs } from "@sentinel-app/db/schema/sentinel";
import { and, asc, desc, eq, gte, lt, sql, type SQL } from "drizzle-orm";

import type { Context } from "../context";
import { productionContract } from "../contract/production";

const os = implement(productionContract).$context<Context>();

const STEP_HOURS = 0.25;

// A clean checkout runs on an empty in-memory DB (env.server.ts): no rows, not an error.
async function hasSchema(db: Database) {
  const rows = await db.all(sql`SELECT 1 FROM sqlite_master WHERE type = 'table' AND name = 'runs'`);
  return rows.length > 0;
}

// Replay writes shared inputs into only one of its per-policy runs, so ask per table.
async function latestRunWith(db: Database, table: typeof parkForecasts | typeof assetSteps) {
  if (!(await hasSchema(db))) return null;
  const [run] = await db
    .select({ runId: runs.id, label: runs.label })
    .from(runs)
    .where(sql`EXISTS (SELECT 1 FROM ${table} WHERE ${table.runId} = ${runs.id})`)
    .orderBy(desc(runs.createdAt))
    .limit(1);
  return run ?? null;
}

function inWindow(ts: typeof assetSteps.ts | typeof parkForecasts.ts, from?: string, to?: string) {
  const bounds: SQL[] = [];
  if (from) bounds.push(gte(ts, from));
  if (to) bounds.push(lt(ts, to));
  return bounds;
}

export const productionRouter = os.router({
  getRun: os.getRun.handler(({ context }) => latestRunWith(context.db, parkForecasts)),

  listParks: os.listParks.handler(async ({ context }) => {
    if (!(await hasSchema(context.db))) return [];
    return context.db.select().from(parks).orderBy(asc(parks.id));
  }),

  getParkForecasts: os.getParkForecasts.handler(async ({ context, input }) => {
    const runId = input?.runId ?? (await latestRunWith(context.db, parkForecasts))?.runId;
    if (!runId) return [];
    return context.db
      .select({
        parkId: parkForecasts.parkId,
        ts: parkForecasts.ts,
        p10Mw: parkForecasts.p10Mw,
        p50Mw: parkForecasts.p50Mw,
        p90Mw: parkForecasts.p90Mw,
        expectedMwh: parkForecasts.expectedMwh,
        availabilityPct: parkForecasts.availabilityPct,
        curtailedMwh: parkForecasts.curtailedMwh,
      })
      .from(parkForecasts)
      .where(
        and(
          eq(parkForecasts.runId, runId),
          input?.parkId ? eq(parkForecasts.parkId, input.parkId) : undefined,
        ),
      )
      .orderBy(asc(parkForecasts.ts));
  }),

  getCountyTotals: os.getCountyTotals.handler(async ({ context, input }) => {
    const { db } = context;
    const from = input?.from;
    const to = input?.to;
    const windRun = input?.runId ?? (await latestRunWith(db, parkForecasts))?.runId;
    // asset_steps, not predictions: replay's lite DB has no per-step PV predictions.
    const solarRun = input?.runId ?? (await latestRunWith(db, assetSteps))?.runId;

    const wind = windRun
      ? await db
          .select({
            county: parks.county,
            lat: sql<number>`avg(${parks.lat})`,
            lon: sql<number>`avg(${parks.lon})`,
            mwh: sql<number>`sum(coalesce(${parkForecasts.expectedMwh}, ${parkForecasts.p50Mw} * ${STEP_HOURS}, 0))`,
          })
          .from(parkForecasts)
          .innerJoin(parks, eq(parks.id, parkForecasts.parkId))
          .where(and(eq(parkForecasts.runId, windRun), ...inWindow(parkForecasts.ts, from, to)))
          .groupBy(parks.county)
      : [];

    const solar = solarRun
      ? await db
          .select({
            county: assets.county,
            lat: sql<number>`avg(${assets.lat})`,
            lon: sql<number>`avg(${assets.lon})`,
            mwh: sql<number>`sum(coalesce(${assetSteps.q50Kw}, 0)) * ${STEP_HOURS} / 1000`,
          })
          .from(assetSteps)
          .innerJoin(assets, eq(assets.id, assetSteps.assetId))
          .where(
            and(
              eq(assetSteps.runId, solarRun),
              eq(assets.assetKind, "prosumer"),
              ...inWindow(assetSteps.ts, from, to),
            ),
          )
          .groupBy(assets.county)
      : [];

    const totals = new Map<string, { county: string; lat: number; lon: number; solarMwh: number; windMwh: number }>();
    for (const row of wind) {
      totals.set(row.county, { county: row.county, lat: row.lat, lon: row.lon, solarMwh: 0, windMwh: row.mwh });
    }
    for (const row of solar) {
      // Sit the solar marker on the county's homes, not on its parks.
      const windMwh = totals.get(row.county)?.windMwh ?? 0;
      totals.set(row.county, { county: row.county, lat: row.lat, lon: row.lon, solarMwh: row.mwh, windMwh });
    }
    return [...totals.values()]
      .map((row) => ({ ...row, expectedMwh: row.solarMwh + row.windMwh }))
      .sort((a, b) => a.county.localeCompare(b.county));
  }),
});
