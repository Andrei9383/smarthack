// O&M mode (#32): sites and turbines by status, the crews' work-order queues, the human-review
// queue, alerts (curtailment is logged here, never as a work order), turbine drawer, park reports.
import { implement } from "@orpc/server";
import {
  alerts,
  assetColumns,
  assets,
  measurements,
  predictions,
  reports,
  reviewQueue,
  statuses,
  turbineAvailability,
  workOrders,
} from "@sentinel-app/db/schema/sentinel";
import { and, asc, desc, eq, lte, sql, type SQL } from "drizzle-orm";

import type { Context } from "../context";
import { omContract } from "../contract/om";
import { emptyIfNoTables } from "./runs";
import { resolveRunId } from "./sites";

const os = implement(omContract).$context<Context>();

// DBs written before #170 lack the W4/W6 columns: retry without them and read evidence_json instead.
async function withoutMissingColumns<T>(full: () => Promise<T[]>, base: () => Promise<T[]>) {
  try {
    return await full();
  } catch (error) {
    const cause = (error as { cause?: { message?: string } }).cause?.message ?? "";
    if (`${(error as Error).message} ${cause}`.includes("no such column")) return base();
    throw error;
  }
}

const num = (v: unknown) => (typeof v === "number" && Number.isFinite(v) ? v : null);

function parseEvidence(text: string | null): Record<string, unknown> | null {
  try {
    return text ? (JSON.parse(text) as Record<string, unknown>) : null;
  } catch {
    return null;
  }
}

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

  // Per crew, the costliest first: lei/day, then triage score (W4 #167, plan §3.5 top-K).
  listWorkOrders: os.listWorkOrders.handler(({ context: { db }, input }) =>
    emptyIfNoTables(async () => {
      const runId = await resolveRunId(db, input?.runId);
      if (!runId) return [];
      const base = {
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
      };
      const where = and(
        ...defined([
          eq(workOrders.runId, runId),
          input?.status ? eq(workOrders.status, input.status) : undefined,
          input?.crewType ? eq(workOrders.crewType, input.crewType) : undefined,
          input?.assetId ? eq(workOrders.assetId, input.assetId) : undefined,
          input?.until ? lte(workOrders.ts, input.until) : undefined,
        ]),
      );
      const baseQuery = () => db.select(base).from(workOrders).where(where);
      type Row = Awaited<ReturnType<typeof baseQuery>>[number] & {
        leiPerDay?: number | null;
        triageScore?: number | null;
        bestRepairDay?: string | null;
      };
      const rows: Row[] = await withoutMissingColumns<Row>(
        () =>
          db
            .select({
              ...base,
              leiPerDay: workOrders.leiPerDay,
              triageScore: workOrders.triageScore,
              bestRepairDay: workOrders.bestRepairDay,
            })
            .from(workOrders)
            .where(where),
        baseQuery,
      );
      return rows
        .map((row) => {
          const ev = parseEvidence(row.evidenceJson);
          return {
            ...row,
            leiPerDay: row.leiPerDay ?? num(ev?.lei_per_day) ?? row.lostLeiDay ?? null,
            triageScore: row.triageScore ?? num(ev?.triage_score),
            bestRepairDay:
              row.bestRepairDay ?? (typeof ev?.best_repair_day === "string" ? ev.best_repair_day : null),
          };
        })
        .sort(
          (a, b) =>
            a.crewType.localeCompare(b.crewType) ||
            (b.leiPerDay ?? -1) - (a.leiPerDay ?? -1) ||
            b.score - a.score,
        );
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
        db.select(assetColumns).from(assets).where(eq(assets.id, input.assetId)).limit(1),
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
      // the day-ahead view as of the replay clock: up to tomorrow's delivery day (sold at D-1)
      const tomorrow = input.at
        ? new Date(Date.parse(input.at) + 86_400_000).toISOString().slice(0, 10)
        : undefined;
      const [avail] = await emptyIfNoTables(() =>
        db
          .select()
          .from(turbineAvailability)
          .where(
            and(
              ...defined([
                eq(turbineAvailability.runId, runId),
                eq(turbineAvailability.assetId, input.assetId),
                tomorrow ? lte(turbineAvailability.day, tomorrow) : undefined,
              ]),
            ),
          )
          .orderBy(desc(turbineAvailability.day))
          .limit(1),
      );
      return {
        availability: avail
          ? {
              day: avail.day,
              rule: avail.rule,
              availabilityMean: avail.availabilityMean,
              availabilityMin: avail.availabilityMin,
              uncertain: avail.uncertain === 1,
              m6Class: avail.m6Class,
              m7PFail7d: avail.m7PFail7d,
            }
          : null,
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
