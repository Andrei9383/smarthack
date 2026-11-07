// Congestion notices (card #120): day-ahead congestion forecast per feeder and household emails via Resend.
import { implement } from "@orpc/server";
import type { Database } from "@sentinel-app/db";
import {
  assetColumns,
  assets,
  assetSteps,
  feederCurves,
  feeders,
  measurements,
  runs,
  schedules,
} from "@sentinel-app/db/schema/sentinel";
import { and, asc, desc, eq, gte, inArray, isNotNull, lt } from "drizzle-orm";

import type { Context } from "../context";
import {
  notifyContract,
  type FeederNotice,
  type HouseholdItem,
  type SentLogItem,
} from "../contract/notify";
import { renderNotice } from "../notify/email";
import {
  gradeForecast,
  inWindow,
  LEVEL_RANK,
  loadForecaster,
  STEP_HOURS,
  type ForecastStep,
} from "../notify/forecast";
import { maskEmail, notifyConfig, sendBatch } from "../notify/resend";
import { currentRunIds, emptyIfNoTables } from "./runs";

const os = implement(notifyContract).$context<Context>();

type DayInput = { from: string; to: string; runId?: string };

const metaOf = (json: string | null): Record<string, unknown> => {
  try {
    return json ? (JSON.parse(json) as Record<string, unknown>) : {};
  } catch {
    return {};
  }
};

async function pvModelVersion(db: Database, runIds: string[]): Promise<string> {
  const rows = await db
    .select({ json: runs.modelVersionsJson })
    .from(runs)
    .where(inArray(runs.id, runIds))
    .orderBy(desc(runs.createdAt));
  for (const row of rows) {
    const pv = metaOf(row.json).pv;
    if (typeof pv === "string") return pv;
  }
  return "pvnet (version not recorded)";
}

async function buildNotice(db: Database, feederId: string, day: DayInput): Promise<FeederNotice | null> {
  const [feeder] = await db.select().from(feeders).where(eq(feeders.id, feederId)).limit(1);
  if (!feeder) return null;
  const runIds = await currentRunIds(db, day.runId);
  const homes = await db
    .select(assetColumns)
    .from(assets)
    .where(and(eq(assets.assetKind, "prosumer"), eq(assets.feederId, feederId)))
    .orderBy(asc(assets.id));
  const homeIds = homes.map((h) => h.id);
  if (homeIds.length === 0) return null;

  const [pvRows, loadRows, actualRows, planRows, modelVersion] = await Promise.all([
    db
      .select({ assetId: assetSteps.assetId, ts: assetSteps.ts, q10: assetSteps.q10Kw, q50: assetSteps.q50Kw, q90: assetSteps.q90Kw })
      .from(assetSteps)
      .where(and(inArray(assetSteps.runId, runIds), inArray(assetSteps.assetId, homeIds), gte(assetSteps.ts, day.from), lt(assetSteps.ts, day.to))),
    db
      .select({ assetId: measurements.assetId, ts: measurements.ts, load: measurements.loadKw })
      .from(measurements)
      .where(and(inArray(measurements.runId, runIds), inArray(measurements.assetId, homeIds), isNotNull(measurements.loadKw))),
    db
      .select({ ts: feederCurves.ts, exportKw: feederCurves.exportKw, limitKw: feederCurves.limitKw })
      .from(feederCurves)
      .where(and(inArray(feederCurves.runId, runIds), eq(feederCurves.feederId, feederId), eq(feederCurves.policy, "baseline"), gte(feederCurves.ts, day.from), lt(feederCurves.ts, day.to))),
    db
      .select({ assetId: schedules.assetId, ts: schedules.ts, battKw: schedules.battKw })
      .from(schedules)
      .where(and(inArray(schedules.runId, runIds), inArray(schedules.assetId, homeIds), eq(schedules.policy, "sentinel"), gte(schedules.ts, day.from), lt(schedules.ts, day.to))),
    pvModelVersion(db, runIds),
  ]);

  // Per-home load series (for the household view) and the feeder sum (for the export forecast).
  const homeLoad = new Map<string, Map<string, number>>();
  const feederLoad = new Map<string, number>();
  for (const r of loadRows) {
    const kw = r.load ?? 0;
    const series = homeLoad.get(r.assetId) ?? new Map<string, number>();
    series.set(r.ts, kw);
    homeLoad.set(r.assetId, series);
    feederLoad.set(r.ts, (feederLoad.get(r.ts) ?? 0) + kw);
  }
  const feederLoadAt = loadForecaster(feederLoad);

  const pvByTs = new Map<string, { q10: number; q50: number; q90: number }>();
  for (const r of pvRows) {
    const acc = pvByTs.get(r.ts) ?? { q10: 0, q50: 0, q90: 0 };
    acc.q10 += r.q10 ?? 0;
    acc.q50 += r.q50 ?? 0;
    acc.q90 += r.q90 ?? 0;
    pvByTs.set(r.ts, acc);
  }
  const actualByTs = new Map(actualRows.map((r) => [r.ts, r]));
  const allTs = [...new Set([...pvByTs.keys(), ...actualByTs.keys()])].sort();

  const steps: ForecastStep[] = allTs.map((ts) => {
    const pv = pvByTs.get(ts) ?? { q10: 0, q50: 0, q90: 0 };
    const load = feederLoadAt(ts);
    return {
      ts,
      q10Kw: pv.q10 - load.kw,
      q50Kw: pv.q50 - load.kw,
      q90Kw: pv.q90 - load.kw,
      limitKw: actualByTs.get(ts)?.limitKw ?? feeder.limitKw,
      loadSource: load.source,
    };
  });
  const f = gradeForecast(steps);

  const actualOver = actualRows.filter((r) => r.exportKw > r.limitKw);
  const households: HouseholdItem[] = homes.map((h) => {
    const meta = metaOf(h.metadataJson);
    const loadAt = loadForecaster(homeLoad.get(h.id) ?? new Map());
    let pvWindow = 0;
    let exportWindow = 0;
    for (const r of pvRows) {
      if (r.assetId !== h.id || !inWindow(r.ts, f.windowFrom, f.windowTo)) continue;
      const pv = r.q50 ?? 0;
      pvWindow += pv * STEP_HOURS;
      exportWindow += Math.max(0, pv - loadAt(r.ts).kw) * STEP_HOURS;
    }
    const plan = planRows
      .filter((r) => r.assetId === h.id && inWindow(r.ts, f.windowFrom, f.windowTo))
      .reduce((sum, r) => sum + Math.max(0, r.battKw ?? 0) * STEP_HOURS, 0);
    return {
      assetId: h.id,
      locality: h.locality ?? feeder.locality,
      county: h.county,
      feederId,
      kwp: h.recordedKwp ?? h.ratedKw,
      hasBattery: Boolean(h.hasBattery) && (h.batteryKwh ?? 0) > 0,
      batteryKwh: h.batteryKwh ?? 0,
      comfortReserveKwh: h.comfortReserveKwh ?? 0,
      hasEv: meta.has_ev === true,
      hasAc: meta.has_ac === true,
      hasBoiler: meta.has_electric_boiler === true,
      worksFromHome: meta.works_from_home === true,
      pvWindowKwh: pvWindow,
      exportWindowKwh: exportWindow,
      batteryPlanKwh: plan,
    };
  });
  households.sort((a, b) => b.exportWindowKwh - a.exportWindowKwh || a.assetId.localeCompare(b.assetId));

  return {
    feederId,
    feederName: feeder.name,
    limitKw: feeder.limitKw,
    level: f.level,
    reason: f.reason,
    windowFrom: f.windowFrom,
    windowTo: f.windowTo,
    stepsOverP10: f.stepsOverP10,
    stepsOverP50: f.stepsOverP50,
    stepsOverP90: f.stepsOverP90,
    peakP50Kw: f.peakP50Kw,
    peakP90Kw: f.peakP90Kw,
    overLimitP50Kwh: f.overLimitP50Kwh,
    loadSource: f.loadSource,
    modelVersion,
    actualStepsOver: actualOver.length,
    actualStepsInWindow: actualOver.filter((r) => inWindow(r.ts, f.windowFrom, f.windowTo)).length,
    actualPeakKw: Math.max(0, ...actualRows.map((r) => r.exportKw)),
    steps: steps.map((s) => ({
      ts: s.ts,
      q10Kw: s.q10Kw,
      q50Kw: s.q50Kw,
      q90Kw: s.q90Kw,
      limitKw: s.limitKw,
      actualKw: actualByTs.get(s.ts)?.exportKw ?? null,
    })),
    households,
  };
}

function render(notice: FeederNotice, home: HouseholdItem) {
  return renderNotice(home, {
    feederId: notice.feederId,
    feederName: notice.feederName,
    limitKw: notice.limitKw,
    modelVersion: notice.modelVersion,
    forecast: {
      level: notice.level,
      reason: notice.reason,
      windowFrom: notice.windowFrom,
      windowTo: notice.windowTo,
      stepsOverP10: notice.stepsOverP10,
      stepsOverP50: notice.stepsOverP50,
      stepsOverP90: notice.stepsOverP90,
      peakP50Kw: notice.peakP50Kw,
      peakP90Kw: notice.peakP90Kw,
      overLimitP50Kwh: notice.overLimitP50Kwh,
      loadSource: notice.loadSource,
      steps: [],
    },
  });
}

// Sent log lives in memory: the demo SQLite is a committed, read-only artefact.
const SENT_LOG: SentLogItem[] = [];
const LOG_MAX = 50;

const bucharestDay = (ts: string) =>
  new Intl.DateTimeFormat("en-CA", { timeZone: "Europe/Bucharest" }).format(new Date(Date.parse(ts) + 60_000));

export const notifyRouter = os.router({
  getConfig: os.getConfig.handler(() => {
    const cfg = notifyConfig();
    return {
      enabled: Boolean(cfg.apiKey) && cfg.demoTo.length > 0,
      from: cfg.from,
      demoTo: cfg.demoTo.map(maskEmail),
      hourlyCap: cfg.hourlyCap,
    };
  }),

  getForecast: os.getForecast.handler(async ({ context, input }) => {
    const [notice] = await emptyIfNoTables(async () => [await buildNotice(context.db, input.feederId, input)]);
    return notice ?? null;
  }),

  getFleet: os.getFleet.handler(({ context, input }) =>
    emptyIfNoTables(async () => {
      const ids = await context.db.select({ id: feeders.id }).from(feeders).orderBy(asc(feeders.id));
      const notices = await Promise.all(ids.map(({ id }) => buildNotice(context.db, id, input)));
      return notices.flatMap((n) => {
        if (!n) return [];
        const { steps: _steps, households, ...rest } = n;
        return [{ ...rest, householdCount: households.length }];
      });
    }),
  ),

  preview: os.preview.handler(async ({ context, input }) => {
    const notice = await buildNotice(context.db, input.feederId, input);
    const home = notice?.households.find((h) => h.assetId === input.assetId);
    return notice && home ? render(notice, home) : null;
  }),

  send: os.send.handler(async ({ context, input }) => {
    const cfg = notifyConfig();
    const notice = await buildNotice(context.db, input.feederId, input);
    const base = {
      at: new Date().toISOString(),
      feederId: input.feederId,
      day: bucharestDay(input.from),
      level: notice?.level ?? ("none" as const),
      to: cfg.demoTo.map(maskEmail),
    };
    const log = (entry: SentLogItem) => {
      SENT_LOG.unshift(entry);
      SENT_LOG.length = Math.min(SENT_LOG.length, LOG_MAX);
      return entry;
    };
    if (!notice) return log({ ...base, count: 0, ids: [], ok: false, error: "no data for this feeder and day" });
    if (LEVEL_RANK[notice.level] < LEVEL_RANK[input.minLevel])
      return log({ ...base, count: 0, ids: [], ok: false, error: `level "${notice.level}" is below "${input.minLevel}": nothing to send` });
    if (cfg.demoTo.length === 0)
      return log({ ...base, count: 0, ids: [], ok: false, error: "NOTIFY_DEMO_TO is not set on the server" });

    const wanted = input.assetIds ? new Set(input.assetIds) : null;
    const emails = notice.households
      .filter((h) => !wanted || wanted.has(h.assetId))
      .flatMap((h) => {
        const mail = render(notice, h);
        // Synthetic homes have no address: every notice goes to the demo inbox, tagged with its home.
        return mail
          ? [{ to: cfg.demoTo, subject: `[${h.assetId}] ${mail.subject}`, html: mail.html, text: mail.text,
               tags: [{ name: "feeder", value: notice.feederId }, { name: "level", value: notice.level }] }]
          : [];
      });
    const result = await sendBatch(emails);
    return log({
      ...base,
      count: result.ok ? result.ids.length : 0,
      ids: result.ok ? result.ids : [],
      ok: result.ok,
      error: result.ok ? null : result.error,
    });
  }),

  listSent: os.listSent.handler(() => SENT_LOG),
});
