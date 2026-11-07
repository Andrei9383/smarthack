// O&M mode (#32): sites and turbines by status, the crews' work-order queues, the human-review
// queue, alerts (curtailment is logged here, never as a work order), turbine drawer, park reports.
import { implement } from "@orpc/server";
import {
  alerts,
  assets,
  measurements,
  predictions,
  reports,
  reviewQueue,
  statuses,
  workOrders,
} from "@sentinel-app/db/schema/sentinel";
import { and, asc, desc, eq, lte, sql, type SQL } from "drizzle-orm";

import type { Context } from "../context";
import { omContract } from "../contract/om";
import { emptyIfNoTables } from "./runs";
import { resolveRunId } from "./sites";

const os = implement(omContract).$context<Context>();

const defined = (clauses: (SQL | undefined)[]) =>
  clauses.filter((c): c is SQL => c !== undefined);

export const omRouter = os.router({
  // Every asset with its status at `at` (the replay clock), or at the newest step.
  listAssets: os.listAssets.handler(({ context: { db }, input }) =>
    emptyIfNoTables(async () => {
      const runId = await resolveRunId(db, input?.runId);
      if (!runId) return [];
      // Replay writes a status on change and at day start, so read each asset's latest row.
      const before = input?.at ? sql`and s2.ts <= ${input.at}` : sql``;
      const at = sql`(select max(s2.ts) from statuses s2 where s2.run_id = ${runId} and s2.asset_id = ${assets.id} ${before})`;
      const rows = await db
        .select({
          id: assets.id,
          name: assets.name,
          assetKind: assets.assetKind,
          status: sql<string>`coalesce(${statuses.status}, 'unknown')`,
          headroomFactor: sql<number>`coalesce(${statuses.headroomFactor}, 0)`,
          reason: sql<string>`coalesce(${statuses.reason}, 'no_status')`,
          county: assets.county,
          locality: assets.locality,
          gisTrust: assets.gisTrust,
          lat: assets.lat,
          lon: assets.lon,
          feederId: assets.feederId,
          parkId: assets.parkId,
          ratedKw: assets.ratedKw,
        })
        .from(assets)
        .leftJoin(
          statuses,
          and(
            eq(statuses.assetId, assets.id),
            eq(statuses.runId, runId),
            sql`${statuses.ts} = ${at}`,
          ),
        )
        .where(
          and(
            ...defined([
              input?.assetKind
                ? eq(assets.assetKind, input.assetKind)
                : undefined,
              input?.status ? eq(statuses.status, input.status) : undefined,
            ]),
          ),
        )
        .orderBy(asc(assets.id));
      return rows.map((row) => ({
        ...row,
        gisTrust:
          row.gisTrust === "ok" || row.gisTrust === "low" ? row.gisTrust : null,
      }));
    }),
  ),

  listAlerts: os.listAlerts.handler(({ context: { db }, input }) =>
    emptyIfNoTables(async () => {
      const runId = await resolveRunId(db, input?.runId);
      if (!runId) return [];
      return db
        .select({
          actionId: alerts.actionId,
          assetId: alerts.assetId,
          parkId: alerts.parkId,
          feederId: alerts.feederId,
          severity: alerts.severity,
          code: alerts.code,
          message: alerts.message,
          ts: alerts.ts,
          createsWorkOrder: alerts.createsWorkOrder,
          curtailedMwh: alerts.curtailedMwh,
          evidenceJson: alerts.evidenceJson,
        })
        .from(alerts)
        .where(
          and(
            ...defined([
              eq(alerts.runId, runId),
              input?.severity ? eq(alerts.severity, input.severity) : undefined,
              input?.parkId ? eq(alerts.parkId, input.parkId) : undefined,
              input?.until ? lte(alerts.ts, input.until) : undefined,
            ]),
          ),
        )
        .orderBy(desc(alerts.ts));
    }),
  ),

  // Ranked per crew: score order, best first (plan §3.5 top-K triage).
  listWorkOrders: os.listWorkOrders.handler(({ context: { db }, input }) =>
    emptyIfNoTables(async () => {
      const runId = await resolveRunId(db, input?.runId);
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
            ...defined([
              eq(workOrders.runId, runId),
              input?.status ? eq(workOrders.status, input.status) : undefined,
              input?.crewType
                ? eq(workOrders.crewType, input.crewType)
                : undefined,
              input?.assetId
                ? eq(workOrders.assetId, input.assetId)
                : undefined,
              input?.until ? lte(workOrders.ts, input.until) : undefined,
            ]),
          ),
        )
        .orderBy(asc(workOrders.crewType), desc(workOrders.score));
    }),
  ),

  listReviewQueue: os.listReviewQueue.handler(({ context: { db }, input }) =>
    emptyIfNoTables(async () => {
      const runId = await resolveRunId(db, input?.runId);
      if (!runId) return [];
      return db
        .select({
          actionId: reviewQueue.actionId,
          assetId: reviewQueue.assetId,
          ts: reviewQueue.ts,
          reason: reviewQueue.reason,
          oodScore: reviewQueue.oodScore,
          intervalWidth: reviewQueue.intervalWidth,
          memberStd: reviewQueue.memberStd,
          resolved: reviewQueue.resolved,
        })
        .from(reviewQueue)
        .where(
          and(
            ...defined([
              eq(reviewQueue.runId, runId),
              input?.resolved === undefined
                ? undefined
                : eq(reviewQueue.resolved, input.resolved ? 1 : 0),
              input?.until ? lte(reviewQueue.ts, input.until) : undefined,
            ]),
          ),
        )
        .orderBy(desc(reviewQueue.ts));
    }),
  ),

  // Measured points over the expected curve (M6 P50 at the same step), and hourly temperatures.
  getTurbineDrawer: os.getTurbineDrawer.handler(
    async ({ context: { db }, input }) => {
      const empty = {
        assetId: input.assetId,
        name: input.assetId,
        powerCurve: [],
        tempTrends: [],
      };
      const [asset] = await emptyIfNoTables(() =>
        db.select().from(assets).where(eq(assets.id, input.assetId)).limit(1),
      );
      const runId = await resolveRunId(db, input.runId);
      if (!asset || !runId) return empty;
      const rows = await db
        .select({
          ts: measurements.ts,
          wdSpdKmh: measurements.wdSpdKmh,
          actPwtKw: measurements.actPwtKw,
          expectedPwtKw: predictions.q50,
          ambientTempC: measurements.ambientTempC,
          convTempC: measurements.convTempC,
          gearboxTempC: measurements.gearboxTempC,
          genTempC: measurements.genTempC,
          transformerTempC: measurements.transformerTempC,
        })
        .from(measurements)
        .leftJoin(
          predictions,
          and(
            eq(predictions.runId, measurements.runId),
            eq(predictions.assetId, measurements.assetId),
            eq(predictions.ts, measurements.ts),
            eq(predictions.model, "M6"),
          ),
        )
        .where(
          and(
            eq(measurements.runId, runId),
            eq(measurements.assetId, input.assetId),
          ),
        )
        .orderBy(asc(measurements.ts));
      return {
        assetId: asset.id,
        name: asset.name,
        parkId: asset.parkId,
        ratedKw: asset.ratedKw,
        turbineModel: asset.turbineModel,
        powerCurve: rows
          .filter((r) => r.wdSpdKmh != null && r.actPwtKw != null)
          .map((r) => ({
            wdSpdKmh: r.wdSpdKmh as number,
            actPwtKw: r.actPwtKw as number,
            expectedPwtKw: r.expectedPwtKw,
          })),
        // one point per hour is plenty for a week-long trend
        tempTrends: rows
          .filter((r) => r.ts.slice(14, 16) === "00")
          .map((r) => ({
            ts: r.ts,
            ambientTempC: r.ambientTempC,
            convTempC: r.convTempC,
            gearboxTempC: r.gearboxTempC,
            genTempC: r.genTempC,
            transformerTempC: r.transformerTempC,
          })),
      };
    },
  ),

  listReports: os.listReports.handler(({ context: { db }, input }) =>
    emptyIfNoTables(async () => {
      const runId = await resolveRunId(db, input?.runId);
      if (!runId) return [];
      return db
        .select({
          actionId: reports.actionId,
          parkId: reports.parkId,
          date: reports.date,
          energyMwh: reports.energyMwh,
          expectedMwh: reports.expectedMwh,
          availabilityPct: reports.availabilityPct,
          curtailedMwh: reports.curtailedMwh,
          dataQualityPct: reports.dataQualityPct,
          issuesJson: reports.issuesJson,
        })
        .from(reports)
        .where(
          and(
            ...defined([
              eq(reports.runId, runId),
              input?.parkId ? eq(reports.parkId, input.parkId) : undefined,
            ]),
          ),
        )
        .orderBy(asc(reports.parkId), asc(reports.date));
    }),
  ),
});
