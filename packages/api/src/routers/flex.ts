// Flexibility panel: what the fleet can offer the DSO, and what it actually promised.
// The four numbers are kept apart on purpose: nominal >= available >= committed,
// and realised is measured afterwards. "We never sell the nominal number."
import { implement } from "@orpc/server";
import type { Database } from "@wattsentry/db";
import { assets, commitments, statuses } from "@wattsentry/db/schema/replay";
import { and, eq, sql } from "drizzle-orm";

import type { Context } from "../context";
import { type FeederFlexItem, flexContract } from "../contract/flex";
import { emptyIfNoTables, resolveRunId } from "./runs";

const os = implement(flexContract).$context<Context>();

// Status per site at the newest step of the run.
async function latestStatuses(db: Database, runId: string) {
  return db
    .select({
      assetId: statuses.assetId,
      status: statuses.status,
      headroomFactor: statuses.headroomFactor,
    })
    .from(statuses)
    .where(
      and(
        eq(statuses.runId, runId),
        // Replay writes a status on change and at day start, so read each asset's latest row.
        sql`${statuses.ts} = (select max(s2.ts) from statuses s2 where s2.run_id = ${runId} and s2.asset_id = ${statuses.assetId})`,
      ),
    );
}

export const flexRouter = os.router({
  getOverview: os.getOverview.handler(async ({ input, context: { db } }) => {
    const runId = await resolveRunId(db, input?.runId);
    const [fleet] = await emptyIfNoTables(() =>
      db
        .select({
          totalBatteries: sql<number>`coalesce(sum(${assets.hasBattery}), 0)`,
          totalKwh: sql<number>`coalesce(sum(${assets.batteryKwh}), 0)`,
          totalKw: sql<number>`coalesce(sum(${assets.batteryKw}), 0)`,
        })
        .from(assets)
        .where(eq(assets.assetKind, "prosumer")),
    );
    // Peak hour of the week: the most the fleet committed, and what was available then.
    const [peak] = runId
      ? await db
          .select({
            committed: sql<number>`sum(${commitments.committedKw})`,
            available: sql<number>`sum(${commitments.availableKw})`,
          })
          .from(commitments)
          .where(eq(commitments.runId, runId))
          .groupBy(commitments.ts)
          .orderBy(sql`sum(${commitments.committedKw}) desc`)
          .limit(1)
      : [];
    return {
      totalBatteries: Number(fleet?.totalBatteries ?? 0),
      totalKwh: Number(fleet?.totalKwh ?? 0),
      totalKw: Number(fleet?.totalKw ?? 0),
      dispatchedKw: Number(peak?.committed ?? 0),
      availableKw: Number(peak?.available ?? 0),
    };
  }),

  getFeedersFlex: os.getFeedersFlex.handler(async ({ input, context: { db } }) => {
    const runId = await resolveRunId(db, input?.runId);
    if (!runId) return [];
    const [sites, latest] = await Promise.all([
      db
        .select({
          id: assets.id,
          feederId: assets.feederId,
          batteryKw: assets.batteryKw,
        })
        .from(assets)
        .where(eq(assets.assetKind, "prosumer")),
      latestStatuses(db, runId),
    ]);
    const statusOf = new Map(latest.map((row) => [row.assetId, row]));
    const byFeeder = new Map<string, FeederFlexItem>();
    for (const site of sites) {
      if (!site.feederId) continue;
      const item = byFeeder.get(site.feederId) ?? {
        feederId: site.feederId,
        flexibleHeadroomKw: 0,
        eligibleSites: 0,
        deratedSites: 0,
        excludedSites: 0,
      };
      const st = statusOf.get(site.id);
      if (st?.status === "eligible") item.eligibleSites += 1;
      else if (st?.status === "derated") item.deratedSites += 1;
      else item.excludedSites += 1; // excluded, in human review, or no status: out of the pool
      if (st && (st.status === "eligible" || st.status === "derated")) {
        item.flexibleHeadroomKw += (site.batteryKw ?? 0) * st.headroomFactor;
      }
      byFeeder.set(site.feederId, item);
    }
    return [...byFeeder.values()]
      .map((item) => ({
        ...item,
        flexibleHeadroomKw: Math.round(item.flexibleHeadroomKw * 10) / 10,
      }))
      .sort((a, b) => a.feederId.localeCompare(b.feederId, "en", { numeric: true }));
  }),

  getCommitments: os.getCommitments.handler(async ({ input, context: { db } }) => {
    const runId = await resolveRunId(db, input?.runId);
    if (!runId) return [];
    const where = [eq(commitments.runId, runId)];
    if (input?.feederId) where.push(eq(commitments.feederId, input.feederId));
    // Sum over feeders per step first, then average the steps within each hour.
    const perStep = db
      .select({
        ts: commitments.ts,
        nominal: sql<number>`sum(${commitments.nominalKw})`.as("nominal"),
        available: sql<number>`sum(${commitments.availableKw})`.as("available"),
        committed: sql<number>`sum(${commitments.committedKw})`.as("committed"),
        realised: sql<number | null>`sum(${commitments.realisedKw})`.as("realised"),
        // the scope delivered if its total realised met its total commitment
        hit: sql<number | null>`case when sum(${commitments.committedKw}) > 0
          then sum(${commitments.realisedKw}) >= sum(${commitments.committedKw}) end`.as("hit"),
      })
      .from(commitments)
      .where(and(...where))
      .groupBy(commitments.ts)
      .as("per_step");
    const rows = await db
      .select({
        hour: sql<string>`substr(${perStep.ts}, 1, 13) || ':00:00Z'`,
        nominalKw: sql<number>`avg(${perStep.nominal})`,
        availableKw: sql<number>`avg(${perStep.available})`,
        committedKw: sql<number>`avg(${perStep.committed})`,
        realisedKw: sql<number | null>`avg(${perStep.realised})`,
        hitRate: sql<number | null>`avg(${perStep.hit})`,
      })
      .from(perStep)
      .groupBy(sql`substr(${perStep.ts}, 1, 13)`)
      .orderBy(sql`substr(${perStep.ts}, 1, 13)`);
    const r1 = (v: number | null) => (v == null ? null : Math.round(Number(v) * 10) / 10);
    return rows.map((row) => ({
      hour: row.hour,
      nominalKw: r1(row.nominalKw) ?? 0,
      availableKw: r1(row.availableKw) ?? 0,
      committedKw: r1(row.committedKw) ?? 0,
      realisedKw: r1(row.realisedKw),
      hitRate: row.hitRate == null ? null : Number(row.hitRate),
    }));
  }),
});
