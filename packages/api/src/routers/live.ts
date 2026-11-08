// Live view (card #130): the #128 15-min loop's feeder rows, or the replay day when there is no feed.
// Contract and the provisional live table: packages/api/src/contract/live.ts.
import { call, implement } from "@orpc/server";
import { createDb, type Database } from "@sentinel-app/db";
import { feeders } from "@sentinel-app/db/schema/sentinel";
import { eq, sql } from "drizzle-orm";

import type { Context } from "../context";
import { LIVE_TICK_SECONDS, liveContract, type LiveFeeder, type LivePoint } from "../contract/live";
import { notifyRouter } from "./notify";
import { emptyIfNoTables } from "./runs";

const os = implement(liveContract).$context<Context>();

/** A feed whose last write is older than this many ticks is treated as stopped. */
const STALE_TICKS = 3;
/** Live rows shown: the last 24 h of simulated time plus the forecast ahead. */
const LIVE_STEPS = 96 + 8;

type Row = Record<string, unknown>;
const num = (v: unknown) => (v == null || v === "" ? null : Number(v));

let liveDb: { path: string; db: Database } | null = null;
function openLiveDb(): Database | null {
  const path = process.env.LIVE_DB;
  if (!path) return null;
  if (liveDb?.path !== path) {
    const url = path.startsWith("file:") || path.includes("://") ? path : `file:${path}`;
    liveDb = { path, db: createDb({ DATABASE_URL: url }) };
  }
  return liveDb.db;
}

async function fromLiveFeed(feederId: string): Promise<LiveFeeder | { stale: string } | null> {
  const db = openLiveDb();
  if (!db) return null;
  let rows: Row[];
  try {
    rows = await emptyIfNoTables(() =>
      db.all<Row>(
        sql`select ts, actual_kw, q10_kw, q50_kw, q90_kw, persistence_kw, limit_kw, tick_at, speed
            from live_feeder_steps where feeder_id = ${feederId} order by ts desc limit ${LIVE_STEPS}`,
      ),
    );
  } catch {
    return null; // unreadable file: fall back to replay rather than break the panel
  }
  if (rows.length === 0) return null;
  rows.reverse();
  const withActual = rows.filter((r) => r.actual_kw != null);
  const writtenAt = rows.reduce<string | null>((max, r) => {
    const t = r.tick_at == null ? null : String(r.tick_at);
    return t && (!max || t > max) ? t : max;
  }, null);
  const speed = num(rows.at(-1)?.speed) ?? 1;
  const ageSec = writtenAt ? Math.max(0, (Date.now() - Date.parse(writtenAt)) / 1000) : null;
  const tickSeconds = LIVE_TICK_SECONDS / Math.max(speed, 1e-6);
  if (ageSec == null || ageSec > STALE_TICKS * tickSeconds) {
    const mins = ageSec == null ? "?" : Math.round(ageSec / 60);
    return { stale: `live feed stale (last write ${mins} min ago)` };
  }
  const points: LivePoint[] = rows.map((r) => ({
    ts: String(r.ts),
    actualKw: num(r.actual_kw),
    q10Kw: num(r.q10_kw),
    q50Kw: num(r.q50_kw),
    q90Kw: num(r.q90_kw),
    persistenceKw: num(r.persistence_kw),
  }));
  return {
    feederId,
    limitKw: num(rows.at(-1)?.limit_kw) ?? 0,
    source: "live",
    simulated: true,
    sourceNote: "live feed",
    tickSeconds,
    latestTs: withActual.length ? String(withActual.at(-1)?.ts) : null,
    writtenAt,
    ageSec,
    speed,
    forecastLabel: "next-15-min forecast (#128 loop)",
    points,
  };
}

export const liveRouter = os.router({
  getFeeder: os.getFeeder.handler(async ({ context, input }) => {
    const live = await fromLiveFeed(input.feederId);
    if (live && !("stale" in live)) {
      if (live.limitKw > 0) return live;
      const [feeder] = await emptyIfNoTables(() =>
        context.db.select({ limitKw: feeders.limitKw }).from(feeders).where(eq(feeders.id, input.feederId)).limit(1),
      );
      return { ...live, limitKw: feeder?.limitKw ?? 0 };
    }
    // No live feed, or a stopped one: the replay day stands in (actual = baseline export,
    // forecast = #120's day-ahead feeder export quantiles).
    const notice = await call(notifyRouter.getForecast, input, { context });
    if (!notice) return null;
    return {
      feederId: input.feederId,
      limitKw: notice.limitKw,
      source: "replay",
      simulated: true,
      sourceNote: live ? live.stale : "no live feed",
      tickSeconds: LIVE_TICK_SECONDS,
      latestTs: null,
      writtenAt: null,
      ageSec: null,
      speed: null,
      forecastLabel: `day-ahead forecast (${notice.modelVersion})`,
      points: notice.steps.map((s) => ({
        ts: s.ts,
        actualKw: s.actualKw,
        q10Kw: s.q10Kw,
        q50Kw: s.q50Kw,
        q90Kw: s.q90Kw,
        persistenceKw: null,
      })),
    };
  }),
});
