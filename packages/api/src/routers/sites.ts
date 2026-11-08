// Site view (#31): one prosumer site over the replay week, read from the shared SQLite.
// Shapes come from contract/sites.ts. getStatuses, getSchedule and getWorkOrders are additions
// the site drawer needs that the contract does not cover yet (noted on #31).
import {
  assetColumns,
  assets,
  assetSteps,
  measurements,
  predictions,
  runs,
  schedules,
  statuses,
  workOrders,
} from "@wattsentry/db/schema/index";
import type { Database } from "@wattsentry/db";
import { ORPCError } from "@orpc/server";
import { and, asc, desc, eq, sql } from "drizzle-orm";
import { z } from "zod";

import {
  assetStepPointSchema,
  siteDetailSchema,
  sitePredictionPointSchema,
  siteTelemetryPointSchema,
  workOrderItemSchema,
} from "../contract";
import { publicProcedure } from "../index";
import { emptyIfNoTables } from "./runs";

const siteInput = z.object({
  assetId: z.string(),
  runId: z.string().optional(),
});

// Without an explicit run, read the newest run, preferring Sentinel over baseline: the site view
// shows what Sentinel did. Fixtures have policy NULL, which sorts with Sentinel.
export async function resolveRunId(
  db: Database,
  runId?: string,
): Promise<string | null> {
  if (runId) return runId;
  // a clean checkout has no tables yet: no run, so every procedure returns no data
  const [row] = await emptyIfNoTables(() =>
    db
      .select({ id: runs.id })
      .from(runs)
      .orderBy(
        sql`case when ${runs.policy} = 'baseline' then 1 else 0 end`,
        desc(runs.createdAt),
      )
      .limit(1),
  );
  return row?.id ?? null;
}

export const statusSegmentSchema = z.object({
  from: z.string(),
  to: z.string(),
  status: z.string(),
  reason: z.string(),
  headroomFactor: z.number(),
});
export type StatusSegment = z.infer<typeof statusSegmentSchema>;

export const schedulePointSchema = z.object({
  ts: z.string(),
  battKw: z.number(),
  soc: z.number(),
  comfortReserveKwh: z.number().nullable(),
  comfortBreached: z.number(),
  clipped: z.number(),
});
export type SchedulePoint = z.infer<typeof schedulePointSchema>;

export const runLabelSchema = z.object({
  id: z.string(),
  label: z.string(),
  kind: z.string(),
});

export const sitesRouter = {
  // The run the panels read, so each can say on screen whether it is synthetic, real RO or non-RO.
  getRun: publicProcedure
    .input(z.object({ runId: z.string().optional() }).optional())
    .output(runLabelSchema.nullable())
    .handler(async ({ input, context: { db } }) => {
      const runId = await resolveRunId(db, input?.runId);
      if (!runId) return null;
      const [row] = await db
        .select({ id: runs.id, label: runs.label, kind: runs.kind })
        .from(runs)
        .where(eq(runs.id, runId))
        .limit(1);
      return row ?? null;
    }),

  getDetail: publicProcedure
    .input(siteInput)
    .output(siteDetailSchema)
    .handler(async ({ input, context: { db } }) => {
      const runId = await resolveRunId(db, input.runId);
      const [asset] = await emptyIfNoTables(() =>
        db.select(assetColumns).from(assets).where(eq(assets.id, input.assetId)).limit(1),
      );
      if (!asset)
        throw new ORPCError("NOT_FOUND", {
          message: `no asset ${input.assetId}`,
        });
      const [last] = runId
        ? await db
            .select({ status: statuses.status, reason: statuses.reason })
            .from(statuses)
            .where(
              and(
                eq(statuses.runId, runId),
                eq(statuses.assetId, input.assetId),
              ),
            )
            .orderBy(desc(statuses.ts))
            .limit(1)
        : [];
      return {
        id: asset.id,
        name: asset.name,
        assetKind: asset.assetKind,
        county: asset.county,
        locality: asset.locality,
        ratedKw: asset.ratedKw,
        hasBattery: asset.hasBattery,
        batteryKwh: asset.batteryKwh,
        status: last?.status ?? "unknown",
        reason: last?.reason,
        gisTrust:
          asset.gisTrust === "ok" || asset.gisTrust === "low"
            ? asset.gisTrust
            : null,
      };
    }),

  getTelemetry: publicProcedure
    .input(siteInput)
    .output(z.array(siteTelemetryPointSchema))
    .handler(async ({ input, context: { db } }) => {
      const runId = await resolveRunId(db, input.runId);
      if (!runId) return [];
      return db
        .select({
          ts: measurements.ts,
          pvAcKw: measurements.pvAcKw,
          loadKw: measurements.loadKw,
          battKw: measurements.battKw,
          soc: measurements.soc,
        })
        .from(measurements)
        .where(
          and(
            eq(measurements.runId, runId),
            eq(measurements.assetId, input.assetId),
          ),
        )
        .orderBy(asc(measurements.ts));
    }),

  getSteps: publicProcedure
    .input(siteInput)
    .output(z.array(assetStepPointSchema))
    .handler(async ({ input, context: { db } }) => {
      const runId = await resolveRunId(db, input.runId);
      if (!runId) return [];
      return db
        .select({
          ts: assetSteps.ts,
          actualKw: assetSteps.actualKw,
          q10Kw: assetSteps.q10Kw,
          q50Kw: assetSteps.q50Kw,
          q90Kw: assetSteps.q90Kw,
        })
        .from(assetSteps)
        .where(
          and(
            eq(assetSteps.runId, runId),
            eq(assetSteps.assetId, input.assetId),
          ),
        )
        .orderBy(asc(assetSteps.ts));
    }),

  getPredictions: publicProcedure
    .input(siteInput.extend({ model: z.string().optional() }))
    .output(z.array(sitePredictionPointSchema))
    .handler(async ({ input, context: { db } }) => {
      const runId = await resolveRunId(db, input.runId);
      if (!runId) return [];
      const where = [
        eq(predictions.runId, runId),
        eq(predictions.assetId, input.assetId),
      ];
      if (input.model) where.push(eq(predictions.model, input.model));
      return db
        .select({
          ts: predictions.ts,
          t0: predictions.t0,
          horizonIdx: predictions.horizonIdx,
          model: predictions.model,
          modelVersion: predictions.modelVersion,
          q10: predictions.q10,
          q50: predictions.q50,
          q90: predictions.q90,
          faultProbsJson: predictions.faultProbsJson,
          anomalyClass: predictions.anomalyClass,
          oodScore: predictions.oodScore,
          memberStd: predictions.memberStd,
          isFallback: predictions.isFallback,
        })
        .from(predictions)
        .where(and(...where))
        .orderBy(asc(predictions.ts));
    }),

  // Status as runs of equal state, so the timeline is a handful of segments, not 672 rows.
  getStatuses: publicProcedure
    .input(siteInput)
    .output(z.array(statusSegmentSchema))
    .handler(async ({ input, context: { db } }) => {
      const runId = await resolveRunId(db, input.runId);
      if (!runId) return [];
      const rows = await db
        .select({
          ts: statuses.ts,
          status: statuses.status,
          reason: statuses.reason,
          headroomFactor: statuses.headroomFactor,
        })
        .from(statuses)
        .where(
          and(eq(statuses.runId, runId), eq(statuses.assetId, input.assetId)),
        )
        .orderBy(asc(statuses.ts));
      const segments: StatusSegment[] = [];
      for (const row of rows) {
        const prev = segments.at(-1);
        if (prev && prev.status === row.status && prev.reason === row.reason) {
          prev.to = row.ts;
        } else {
          segments.push({
            from: row.ts,
            to: row.ts,
            status: row.status,
            reason: row.reason,
            headroomFactor: row.headroomFactor,
          });
        }
      }
      return segments;
    }),

  getSchedule: publicProcedure
    .input(siteInput)
    .output(z.array(schedulePointSchema))
    .handler(async ({ input, context: { db } }) => {
      const runId = await resolveRunId(db, input.runId);
      if (!runId) return [];
      return db
        .select({
          ts: schedules.ts,
          battKw: schedules.battKw,
          soc: schedules.soc,
          comfortReserveKwh: schedules.comfortReserveKwh,
          comfortBreached: schedules.comfortBreached,
          clipped: schedules.clipped,
        })
        .from(schedules)
        .where(
          and(
            eq(schedules.runId, runId),
            eq(schedules.assetId, input.assetId),
            eq(schedules.policy, "sentinel"),
          ),
        )
        .orderBy(asc(schedules.ts));
    }),

  getWorkOrders: publicProcedure
    .input(siteInput)
    .output(z.array(workOrderItemSchema))
    .handler(async ({ input, context: { db } }) => {
      const runId = await resolveRunId(db, input.runId);
      if (!runId) return [];
      return db
        .select({
          actionId: workOrders.actionId,
          assetId: workOrders.assetId,
          assetKind: workOrders.assetKind,
          crewType: workOrders.crewType,
          action: workOrders.action,
          status: workOrders.status,
          score: workOrders.score,
          day: workOrders.day,
          rank: workOrders.rank,
          ts: workOrders.ts,
          fault: workOrders.fault,
          confidence: workOrders.confidence,
          lostKwhDay: workOrders.lostKwhDay,
          lostFlexMwhDay: workOrders.lostFlexMwhDay,
          lostLeiDay: workOrders.lostLeiDay,
          mwAtRisk: workOrders.mwAtRisk,
          daysToVisit: workOrders.daysToVisit,
          evidenceJson: workOrders.evidenceJson,
        })
        .from(workOrders)
        .where(
          and(
            eq(workOrders.runId, runId),
            eq(workOrders.assetId, input.assetId),
          ),
        )
        .orderBy(asc(workOrders.ts));
    }),
};
