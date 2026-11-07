import { implement } from "@orpc/server";
import {
  assets,
  commitments,
  feederCurves,
  feeders,
  gisFlags,
  schedules,
} from "@sentinel-app/db/schema/sentinel";
import { and, asc, eq, gt, gte, inArray, lt, sql, type SQL } from "drizzle-orm";

import type { Context } from "../context";
import { feedersContract } from "../contract/feeders";
import { currentRunIds, emptyIfNoTables } from "./runs";

const os = implement(feedersContract).$context<Context>();

const STEP_MINUTES = 15;
const STEP_HOURS = STEP_MINUTES / 60;
const MAX_ABSORBING_SITES = 8;

function inWindow(ts: Parameters<typeof gte>[0], from?: string, to?: string): SQL[] {
  return [from ? gte(ts, from) : undefined, to ? lt(ts, to) : undefined].filter(
    (clause): clause is SQL => clause !== undefined,
  );
}

function quantile(sorted: number[], q: number): number {
  if (sorted.length === 0) return 0;
  const position = (sorted.length - 1) * q;
  const low = Math.floor(position);
  const high = Math.ceil(position);
  const a = sorted[low] ?? 0;
  const b = sorted[high] ?? a;
  return a + (b - a) * (position - low);
}

export const feedersRouter = os.router({
  list: os.list.handler(({ context }) =>
    emptyIfNoTables(async () => {
      const rows = await context.db.select().from(feeders).orderBy(asc(feeders.id));
      return rows.map((row) => ({
        ...row,
        gisTrust: row.gisTrust === "ok" || row.gisTrust === "low" ? row.gisTrust : null,
      }));
    }),
  ),

  getCurves: os.getCurves.handler(({ context, input }) =>
    emptyIfNoTables(async () =>
      context.db
        .select({
          ts: feederCurves.ts,
          exportKw: feederCurves.exportKw,
          importKw: sql<number>`coalesce(${feederCurves.importKw}, 0)`,
          limitKw: feederCurves.limitKw,
          slackKw: feederCurves.slackKw,
          policy: feederCurves.policy,
          isViolation: sql<number>`coalesce(${feederCurves.isViolation}, 0)`,
        })
        .from(feederCurves)
        .where(
          and(
            inArray(feederCurves.runId, await currentRunIds(context.db, input.runId)),
            eq(feederCurves.feederId, input.feederId),
            ...inWindow(feederCurves.ts, input.from, input.to),
          ),
        )
        .orderBy(asc(feederCurves.ts)),
    ),
  ),

  getOverLimit: os.getOverLimit.handler(({ context, input }) =>
    emptyIfNoTables(async () =>
      context.db
        .select({
          feederId: feederCurves.feederId,
          policy: feederCurves.policy,
          minutesOver: sql<number>`sum(${feederCurves.exportKw} > ${feederCurves.limitKw}) * ${STEP_MINUTES}`,
          peakExportKw: sql<number>`max(${feederCurves.exportKw})`,
        })
        .from(feederCurves)
        .where(
          and(
            inArray(feederCurves.runId, await currentRunIds(context.db, input?.runId)),
            ...inWindow(feederCurves.ts, input?.from, input?.to),
          ),
        )
        .groupBy(feederCurves.feederId, feederCurves.policy),
    ),
  ),

  getSocFan: os.getSocFan.handler(({ context, input }) =>
    emptyIfNoTables(async () => {
      const rows = await context.db
        .select({ ts: schedules.ts, soc: schedules.soc })
        .from(schedules)
        .innerJoin(assets, eq(assets.id, schedules.assetId))
        .where(
          and(
            inArray(schedules.runId, await currentRunIds(context.db, input.runId)),
            eq(assets.feederId, input.feederId),
            eq(schedules.policy, input.policy),
            ...inWindow(schedules.ts, input.from, input.to),
          ),
        )
        .orderBy(asc(schedules.ts));
      const byTs = new Map<string, number[]>();
      for (const { ts, soc } of rows) {
        if (soc === null) continue;
        const values = byTs.get(ts) ?? [];
        values.push(soc);
        byTs.set(ts, values);
      }
      return [...byTs].map(([ts, values]) => {
        const sorted = values.sort((a, b) => a - b);
        return {
          ts,
          socQ10: quantile(sorted, 0.1),
          socQ50: quantile(sorted, 0.5),
          socQ90: quantile(sorted, 0.9),
        };
      });
    }),
  ),

  getAbsorption: os.getAbsorption.handler(({ context, input }) =>
    emptyIfNoTables(async () => {
      const runIds = await currentRunIds(context.db, input.runId);
      const overSteps = await context.db
        .select({ ts: feederCurves.ts })
        .from(feederCurves)
        .where(
          and(
            inArray(feederCurves.runId, runIds),
            eq(feederCurves.feederId, input.feederId),
            eq(feederCurves.policy, "baseline"),
            gt(feederCurves.exportKw, feederCurves.limitKw),
            ...inWindow(feederCurves.ts, input.from, input.to),
          ),
        );
      if (overSteps.length === 0) return [];
      const charging = await context.db
        .select({
          assetId: schedules.assetId,
          absorbedKwh: sql<number>`sum(${schedules.battKw}) * ${STEP_HOURS}`,
        })
        .from(schedules)
        .innerJoin(assets, eq(assets.id, schedules.assetId))
        .where(
          and(
            inArray(schedules.runId, runIds),
            eq(assets.feederId, input.feederId),
            eq(schedules.policy, "sentinel"),
            gt(schedules.battKw, 0), // + = charging
            inArray(
              schedules.ts,
              overSteps.map((step) => step.ts),
            ),
          ),
        )
        .groupBy(schedules.assetId)
        .orderBy(sql`2 desc`)
        .limit(MAX_ABSORBING_SITES);
      return charging;
    }),
  ),

  getCommitments: os.getCommitments.handler(({ context, input }) =>
    emptyIfNoTables(async () =>
      context.db
        .select({
          ts: commitments.ts,
          nominalKw: commitments.nominalKw,
          availableKw: commitments.availableKw,
          committedKw: commitments.committedKw,
          realisedKw: commitments.realisedKw,
          haircutH: commitments.haircutH,
          hit: commitments.hit,
        })
        .from(commitments)
        .where(
          and(
            inArray(commitments.runId, await currentRunIds(context.db, input.runId)),
            eq(commitments.feederId, input.feederId),
          ),
        )
        .orderBy(asc(commitments.ts)),
    ),
  ),

  listGisFlags: os.listGisFlags.handler(({ context, input }) =>
    emptyIfNoTables(async () =>
      context.db
        .select({
          actionId: gisFlags.actionId,
          sectionId: gisFlags.sectionId,
          recordedFeeder: gisFlags.recordedFeeder,
          recordedPhase: gisFlags.recordedPhase,
          suggestedFeeder: gisFlags.suggestedFeeder,
          suggestedPhase: gisFlags.suggestedPhase,
          tau: gisFlags.tau,
          geometryJson: gisFlags.geometryJson,
        })
        .from(gisFlags)
        .where(inArray(gisFlags.runId, await currentRunIds(context.db, input?.runId))),
    ),
  ),
});
