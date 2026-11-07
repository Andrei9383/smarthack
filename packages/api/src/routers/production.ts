import { implement } from "@orpc/server";
import {
  assets,
  assetSteps,
  parkBattery,
  parkForecasts,
  parks,
  runs,
  windPositions,
} from "@sentinel-app/db/schema/sentinel";
import { and, asc, desc, eq, gte, lt, sql, type SQL } from "drizzle-orm";

import type { Context } from "../context";
import { productionContract } from "../contract/production";
import { emptyIfNoTables, latestRunWith } from "./runs";

const os = implement(productionContract).$context<Context>();

const STEP_HOURS = 0.25;

type TsColumn = typeof assetSteps.ts | typeof parkForecasts.ts | typeof windPositions.ts | typeof parkBattery.ts;

function inWindow(ts: TsColumn, from?: string, to?: string) {
  const bounds: SQL[] = [];
  if (from) bounds.push(gte(ts, from));
  if (to) bounds.push(lt(ts, to));
  return bounds;
}

// Newest run that wrote W3/W5 rows; empty until the wind writers fill these tables.
async function windRun(db: Context["db"], table: typeof windPositions | typeof parkBattery, runId?: string) {
  if (runId) return runId;
  const [run] = await emptyIfNoTables(() =>
    db
      .select({ id: runs.id })
      .from(runs)
      .where(sql`EXISTS (SELECT 1 FROM ${table} WHERE ${table.runId} = ${runs.id})`)
      .orderBy(desc(runs.createdAt))
      .limit(1),
  );
  return run?.id;
}

export const productionRouter = os.router({
  getRun: os.getRun.handler(({ context }) => latestRunWith(context.db, parkForecasts)),

  listParks: os.listParks.handler(async ({ context }) => {
    return emptyIfNoTables(() => context.db.select().from(parks).orderBy(asc(parks.id)));
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

  getWindPositions: os.getWindPositions.handler(async ({ context: { db }, input }) => {
    const runId = await windRun(db, windPositions, input?.runId);
    if (!runId) return [];
    return db
      .select({
        parkId: windPositions.parkId,
        policy: windPositions.policy,
        ts: windPositions.ts,
        actualMwh: windPositions.actualMwh,
        soldDaMwh: windPositions.soldDaMwh,
        fcP10Mwh: windPositions.fcP10Mwh,
        fcP50Mwh: windPositions.fcP50Mwh,
        fcP90Mwh: windPositions.fcP90Mwh,
        avoidableLei: windPositions.avoidableLei,
        removedTurbines: windPositions.removedTurbines,
      })
      .from(windPositions)
      .where(
        and(
          eq(windPositions.runId, runId),
          eq(windPositions.policy, input?.policy ?? "sentinel"),
          input?.parkId ? eq(windPositions.parkId, input.parkId) : undefined,
          ...inWindow(windPositions.ts, input?.from, input?.to),
        ),
      )
      .orderBy(asc(windPositions.ts));
  }),

  getParkBattery: os.getParkBattery.handler(async ({ context: { db }, input }) => {
    const runId = await windRun(db, parkBattery, input?.runId);
    if (!runId) return [];
    return db
      .select({
        parkId: parkBattery.parkId,
        variant: parkBattery.variant,
        ts: parkBattery.ts,
        chargeMwh: parkBattery.chargeMwh,
        dischargeMwh: parkBattery.dischargeMwh,
        socMwh: parkBattery.socMwh,
        devNoneMwh: parkBattery.devNoneMwh,
        devMwh: parkBattery.devMwh,
        avoidableLei: parkBattery.avoidableLei,
        avoidableNoneLei: parkBattery.avoidableNoneLei,
      })
      .from(parkBattery)
      .where(
        and(
          eq(parkBattery.runId, runId),
          input?.parkId ? eq(parkBattery.parkId, input.parkId) : undefined,
          ...inWindow(parkBattery.ts, input?.from, input?.to),
        ),
      )
      .orderBy(asc(parkBattery.variant), asc(parkBattery.ts));
  }),
});
