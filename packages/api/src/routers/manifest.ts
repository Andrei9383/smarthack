import { implement } from "@orpc/server";
import {
  assets,
  feeders,
  kpis,
  measurements,
  parkForecasts,
  parks,
  runs,
  workOrders,
} from "@wattsentry/db/schema/replay";
import { and, count, desc, eq, gte, inArray, lt, lte, sql } from "drizzle-orm";

import type { Context } from "../context";
import { manifestContract } from "../contract/manifest";
import { currentRunIds, emptyIfNoTables } from "./runs";

const os = implement(manifestContract).$context<Context>();

const STEP_HOURS = 0.25;
const EMPTY_OVERVIEW = {
  homes: 0,
  feeders: 0,
  turbines: 0,
  parks: 0,
  productionMwh: null,
  avgWindKmh: null,
  availabilityPct: null,
  activeIssues: null,
};

export const manifestRouter = os.router({
  getRuns: os.getRuns.handler(({ context }) =>
    emptyIfNoTables(() => context.db.select().from(runs).orderBy(desc(runs.createdAt))),
  ),
  getKpis: os.getKpis.handler(({ context, input }) =>
    emptyIfNoTables(async () =>
      context.db
        .select()
        .from(kpis)
        .where(inArray(kpis.runId, await currentRunIds(context.db, input?.runId))),
    ),
  ),
  getOverview: os.getOverview.handler(async ({ context, input }) => {
    const { db } = context;
    const [row] = await emptyIfNoTables(async () => {
      const runIds = await currentRunIds(db, undefined);
      const fleet = await db
        .select({ kind: assets.assetKind, total: count() })
        .from(assets)
        .groupBy(assets.assetKind);
      const [feederTotal] = await db.select({ total: count() }).from(feeders);
      const [parkTotal] = await db.select({ total: count() }).from(parks);
      // Same formula as the Production panel, so the two can't disagree.
      const [production] = await db
        .select({
          mwh: sql<number | null>`sum(coalesce(${parkForecasts.expectedMwh}, ${parkForecasts.p50Mw} * ${STEP_HOURS}))`,
          availability: sql<number | null>`avg(${parkForecasts.availabilityPct})`,
        })
        .from(parkForecasts)
        .where(
          and(
            inArray(parkForecasts.runId, runIds),
            gte(parkForecasts.ts, input.from),
            lt(parkForecasts.ts, input.to),
          ),
        );
      const [wind] = await db
        .select({ kmh: sql<number | null>`avg(${measurements.wdSpdKmh})` })
        .from(measurements)
        .where(
          and(
            inArray(measurements.runId, runIds),
            gte(measurements.ts, input.from),
            lt(measurements.ts, input.to),
          ),
        );
      const [issues] = await db
        .select({ total: count() })
        .from(workOrders)
        .where(
          and(
            inArray(workOrders.runId, runIds),
            eq(workOrders.status, "open"),
            lte(workOrders.ts, input.now),
          ),
        );
      const fleetCount = (kind: string) => fleet.find((f) => f.kind === kind)?.total ?? 0;
      return [
        {
          homes: fleetCount("prosumer"),
          feeders: feederTotal?.total ?? 0,
          turbines: fleetCount("turbine"),
          parks: parkTotal?.total ?? 0,
          productionMwh: production?.mwh ?? null,
          avgWindKmh: wind?.kmh ?? null,
          availabilityPct: production?.availability ?? null,
          activeIssues: runIds.length > 0 ? (issues?.total ?? 0) : null,
        },
      ];
    });
    return row ?? EMPTY_OVERVIEW;
  }),
});
