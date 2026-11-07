import { implement } from "@orpc/server";
import { assets, assetSteps, measurements, parkForecasts, parks, runs } from "@sentinel-app/db/schema/sentinel";
import { type AnyColumn, and, asc, desc, eq, gte, isNotNull, lt, sql, type SQL } from "drizzle-orm";

import type { CountySeriesPoint } from "../contract/production";

import type { Context } from "../context";
import { productionContract } from "../contract/production";
import { emptyIfNoTables, latestRunWith } from "./runs";

const os = implement(productionContract).$context<Context>();

const STEP_HOURS = 0.25;

function inWindow(ts: typeof assetSteps.ts | typeof parkForecasts.ts | typeof measurements.ts, from?: string, to?: string) {
  const bounds: SQL[] = [];
  if (from) bounds.push(gte(ts, from));
  if (to) bounds.push(lt(ts, to));
  return bounds;
}

// 'YYYY-MM-DDTHH:00:00Z', the hour a stored step falls in (stored ts are UTC text).
const hourOf = (ts: AnyColumn) => sql<string>`substr(${ts}, 1, 13) || ':00:00Z'`;

// Mean MW over the hour of a kW column summed across assets: the sum over all rows divided by
// the number of 15-min steps seen in that hour.
const meanMw = (column: AnyColumn, ts: AnyColumn) =>
  sql<number | null>`sum(${column}) / count(distinct ${ts}) / 1000.0`;

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

  getCountySeries: os.getCountySeries.handler(async ({ context, input }) => {
    const { db } = context;
    const from = input?.from;
    const to = input?.to;
    const solarRun = input?.runId ?? (await latestRunWith(db, assetSteps))?.runId;
    // Measured park output lives in turbine telemetry; take the newest run that has some.
    const [windRow] = input?.runId
      ? [{ runId: input.runId }]
      : await emptyIfNoTables(() =>
          db
            .select({ runId: runs.id })
            .from(runs)
            .where(sql`EXISTS (SELECT 1 FROM ${measurements} WHERE ${measurements.runId} = ${runs.id} AND ${measurements.actPwtKw} IS NOT NULL)`)
            .orderBy(desc(runs.createdAt))
            .limit(1),
        );
    const windRun = windRow?.runId;

    const solar = solarRun
      ? await emptyIfNoTables(() =>
          db
            .select({
              county: assets.county,
              ts: hourOf(assetSteps.ts),
              actual: meanMw(assetSteps.actualKw, assetSteps.ts),
              p10: meanMw(assetSteps.q10Kw, assetSteps.ts),
              p50: meanMw(assetSteps.q50Kw, assetSteps.ts),
              p90: meanMw(assetSteps.q90Kw, assetSteps.ts),
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
            .groupBy(assets.county, hourOf(assetSteps.ts)),
        )
      : [];

    const wind = windRun
      ? await emptyIfNoTables(() =>
          db
            .select({
              county: parks.county,
              ts: hourOf(measurements.ts),
              actual: meanMw(measurements.actPwtKw, measurements.ts),
            })
            .from(measurements)
            .innerJoin(assets, eq(assets.id, measurements.assetId))
            .innerJoin(parks, eq(parks.id, assets.parkId))
            .where(
              and(
                eq(measurements.runId, windRun),
                eq(assets.assetKind, "turbine"),
                isNotNull(measurements.actPwtKw),
                ...inWindow(measurements.ts, from, to),
              ),
            )
            .groupBy(parks.county, hourOf(measurements.ts)),
        )
      : [];

    // Turbines idle a few watts below zero; a park never exports negative power.
    const clamp = (value: number | null) => (value === null ? null : Math.max(0, value));
    const byCounty = new Map<string, Map<string, CountySeriesPoint>>();
    const pointFor = (county: string, ts: string) => {
      const hours = byCounty.get(county) ?? new Map<string, CountySeriesPoint>();
      byCounty.set(county, hours);
      const point = hours.get(ts) ?? {
        ts,
        solarActualMw: null,
        solarP10Mw: null,
        solarP50Mw: null,
        solarP90Mw: null,
        windActualMw: null,
      };
      hours.set(ts, point);
      return point;
    };
    for (const row of solar) {
      const point = pointFor(row.county, row.ts);
      point.solarActualMw = clamp(row.actual);
      point.solarP10Mw = clamp(row.p10);
      point.solarP50Mw = clamp(row.p50);
      point.solarP90Mw = clamp(row.p90);
    }
    for (const row of wind) pointFor(row.county, row.ts).windActualMw = clamp(row.actual);

    return [...byCounty.entries()]
      .map(([county, hours]) => ({
        county,
        points: [...hours.values()].sort((a, b) => a.ts.localeCompare(b.ts)),
      }))
      .sort((a, b) => a.county.localeCompare(b.county));
  }),
});
