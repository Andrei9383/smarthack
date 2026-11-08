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
import { existsSync, readFileSync } from "node:fs";
import { join, resolve } from "node:path";

import type { Context } from "../context";
import { feedersContract, type GridResults } from "../contract/feeders";
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

// Feature G documents: SENTINEL_GRID_DIR holds hosting_capacity.json (G1), voltage.json (G2) and
// dso_value.json (G3); also tried: each writer's default --out, then the committed demo copies (demo/grid, #193),
// from the repo root or apps/web.
type Doc = Record<string, any>; // eslint-disable-line @typescript-eslint/no-explicit-any
const GRID_DIRS = [process.env.SENTINEL_GRID_DIR, "out/grid", "out/grid/g1", "out/hosting_capacity", ".lab-out/voltage", "demo/grid", "."]
  .filter((d): d is string => Boolean(d))
  .flatMap((d) => [resolve(d), resolve("../..", d)]);

function readDoc(name: string): Doc | null {
  const path = GRID_DIRS.map((dir) => join(dir, name)).find(existsSync);
  if (!path) return null;
  try {
    return JSON.parse(readFileSync(path, "utf8")) as Doc;
  } catch {
    return null;
  }
}

const num = (v: unknown) => (typeof v === "number" && Number.isFinite(v) ? v : null);
const pct = (x: unknown) => (typeof x === "number" ? `${(x * 100).toFixed(1)}%` : "—");
const lei = (x: unknown) => (typeof x === "number" ? `${Math.round(x).toLocaleString("en-GB")} lei` : "—");

function hostingOf(doc: Doc | null, variant = "batteries", tol = "30"): GridResults["hosting"] {
  if (doc?.schema !== "hosting_capacity/v1") return null;
  const total = doc.total?.capacity?.[variant]?.[tol] ?? {};
  return {
    label: String(doc.label ?? "synthetic RO"),
    runId: String(doc.run_id ?? "local"),
    tolMin: Number(tol),
    variant,
    totalWithout: Number(total.self_consumption ?? 0),
    totalThreshold: num(total.threshold),
    totalWith: Number(total.sentinel ?? 0),
    feeders: (doc.feeders as Doc[]).map((f) => {
      const cap = f.capacity?.[variant]?.[tol] ?? {};
      return {
        feederId: String(f.feeder_id),
        nHomes: Number(f.n_homes ?? 0),
        without: Number(cap.self_consumption ?? 0),
        threshold: num(cap.threshold),
        with: Number(cap.sentinel ?? 0),
        censored: Boolean(f.censored?.[variant]?.[tol]?.sentinel),
      };
    }),
  };
}

function voltageOf(doc: Doc | null): GridResults["voltage"] {
  const bySource = (doc?.by_v_source ?? {}) as Record<string, Doc>;
  const r = bySource["1.03"] ?? Object.values(bySource)[0];
  if (!doc || !r?.baseline || !r?.sentinel) return null;
  const pol = (p: Doc) => ({
    violations: Number(p.voltage_violations ?? 0),
    minutes: Number(p.voltage_violation_minutes ?? 0),
    tripKwh: Number(p.inverter_trip_kwh ?? 0),
    tripLei: Number(p.inverter_trip_lei ?? 0),
    vMax: Number(p.v_max ?? 0),
  });
  return {
    label: String(doc.label ?? "non-RO topology"),
    runId: String(doc.run_id ?? "local"),
    vLimit: Number(doc.v_limit ?? 253),
    nHomes: Number(doc.n_unique_homes ?? 55),
    baseline: pol(r.baseline),
    sentinel: pol(r.sentinel),
  };
}

function valueOf(doc: Doc | null): Pick<GridResults, "value" | "twin"> {
  if (!doc) return { value: null, twin: null };
  const v = doc.dso_value as Doc | undefined;
  const a = (v?.assumptions ?? {}) as Doc;
  const deo = (a.deo_cost_per_prosumer_lei ?? []) as number[];
  const m = doc.moved_home as Doc | undefined;
  return {
    value: v
      ? {
          label: String(v.label ?? "proxy"),
          labelDeo: String(v.label_deo ?? ""),
          runId: String(doc.run_id ?? "local"),
          deferredCapexLei: Number(v.deferred_capex_lei ?? 0),
          deoBasisLei: ((v.deferred_capex_lei_deo_basis ?? []) as number[]).map(Number),
          gain: Number(v.hosting_capacity_gain ?? 0),
          feeders: ((v.feeders ?? []) as Doc[]).map((f) => ({
            feederId: String(f.feeder_id),
            gain: Number(f.hosting_capacity_gain ?? 0),
            yearsDeferred: Number(f.years_deferred ?? 0),
            lei: Number(f.deferred_capex_lei ?? 0),
          })),
          assumptions: [
            {
              label: "Reinforcement cost per feeder",
              value: `${lei(a.reinforcement_cost_lei)}${a.cost_is_placeholder ? " (placeholder)" : ""}`,
            },
            { label: "Discount rate", value: pct(a.discount_rate) },
            { label: "Prosumer growth per year", value: pct(a.growth_rate) },
            { label: "Hosting capacity from", value: String(a.hosting_capacity ?? "—") },
            {
              label: "DEO cost per prosumer (other operator)",
              value: deo.length === 2 ? `${lei(deo[0])} – ${lei(deo[1])}` : "—",
            },
            { label: "Formula", value: "cost × (1 − 1 / (1 + r)^years deferred)" },
          ],
        }
      : null,
    twin: {
      recoveryRate: num(doc.recovery?.gis_recovery_rate),
      nCorrupted: num(doc.recovery?.n_corrupted),
      moved: m?.found
        ? {
            siteId: String(m.site_id),
            recordedFeeder: String(m.recorded_feeder),
            suggestedFeeder: String(m.suggested_feeder),
            peakGisKw: num(m.planned_peak_gis_kw),
            peakM8Kw: num(m.planned_peak_m8_kw),
            sitesChanged: Number(m.n_sites_changed ?? 0),
          }
        : null,
    },
  };
}

export const feedersRouter = os.router({
  getGrid: os.getGrid.handler(() => ({
    hosting: hostingOf(readDoc("hosting_capacity.json")),
    voltage: voltageOf(readDoc("voltage.json")),
    ...valueOf(readDoc("dso_value.json")),
  })),

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

  getOverLimitSteps: os.getOverLimitSteps.handler(({ context, input }) =>
    emptyIfNoTables(async () =>
      context.db
        .select({
          ts: feederCurves.ts,
          feeders: sql<number>`count(distinct ${feederCurves.feederId})`.mapWith(Number),
        })
        .from(feederCurves)
        .where(
          and(
            inArray(feederCurves.runId, await currentRunIds(context.db)),
            eq(feederCurves.policy, input.policy),
            sql`${feederCurves.exportKw} > ${feederCurves.limitKw}`,
            ...inWindow(feederCurves.ts, input.from, input.to),
          ),
        )
        .groupBy(feederCurves.ts)
        .orderBy(asc(feederCurves.ts)),
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
