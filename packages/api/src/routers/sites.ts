// Site view: one prosumer site over the replay week.
import { implement, ORPCError } from "@orpc/server";
import {
  assetColumns,
  assets,
  assetSteps,
  predictions,
  schedules,
  statuses,
  workOrders,
} from "@wattsentry/db/schema/replay";
import { and, asc, desc, eq } from "drizzle-orm";

import type { Context } from "../context";
import { type StatusSegment, sitesContract } from "../contract/sites";
import { emptyIfNoTables, resolveRunId } from "./runs";

const os = implement(sitesContract).$context<Context>();

export const sitesRouter = os.router({
  getDetail: os.getDetail.handler(async ({ input, context: { db } }) => {
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
          .where(and(eq(statuses.runId, runId), eq(statuses.assetId, input.assetId)))
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
      gisTrust: asset.gisTrust === "ok" || asset.gisTrust === "low" ? asset.gisTrust : null,
    };
  }),

  getSteps: os.getSteps.handler(async ({ input, context: { db } }) => {
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
      .where(and(eq(assetSteps.runId, runId), eq(assetSteps.assetId, input.assetId)))
      .orderBy(asc(assetSteps.ts));
  }),

  getPredictions: os.getPredictions.handler(async ({ input, context: { db } }) => {
    const runId = await resolveRunId(db, input.runId);
    if (!runId) return [];
    const where = [eq(predictions.runId, runId), eq(predictions.assetId, input.assetId)];
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

  getStatuses: os.getStatuses.handler(async ({ input, context: { db } }) => {
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
      .where(and(eq(statuses.runId, runId), eq(statuses.assetId, input.assetId)))
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

  getSchedule: os.getSchedule.handler(async ({ input, context: { db } }) => {
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
      .where(and(eq(schedules.runId, runId), eq(schedules.assetId, input.assetId), eq(schedules.policy, "sentinel")))
      .orderBy(asc(schedules.ts));
  }),

  getWorkOrders: os.getWorkOrders.handler(async ({ input, context: { db } }) => {
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
      .where(and(eq(workOrders.runId, runId), eq(workOrders.assetId, input.assetId)))
      .orderBy(asc(workOrders.ts));
  }),
});
