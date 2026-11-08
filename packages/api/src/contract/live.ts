import { oc } from "@orpc/contract";
import { z } from "zod";

import type { CongestionLevelValue } from "./notify";

// Live view: actual feeder export up to "now", the next-15-min forecast ahead of it,
// the error of the last prediction, and a warning when the forecast crosses the feeder limit.
//
// Two sources, one shape:
// - "live": the 15-min live loop's own SQLite file (env LIVE_DB, read-only):
//     live_feeder_steps(feeder_id TEXT, ts TEXT 'YYYY-MM-DDTHH:MM:SSZ', actual_kw REAL NULL,
//       q10_kw REAL, q50_kw REAL, q90_kw REAL, persistence_kw REAL NULL, limit_kw REAL,
//       tick_at TEXT -- wall-clock ISO time the row was written --, speed REAL NULL)
//   A tick writes the prediction for ts = now + 15 min (actual NULL) and fills actual_kw for ts = now.
// - "replay": no live feed (or a stale one). The replay day stands in: the actual is the baseline
//   export, the forecast is the congestion notice's day-ahead P10/P50/P90 feeder export. The client masks actuals
//   after its replay clock, so nothing after "now" is shown as measured.
// Readings are simulated in both cases and the UI says so.

export const LIVE_TICK_SECONDS = 900;

const livePointSchema = z.object({
  ts: z.string(),
  actualKw: z.number().nullable(),
  q10Kw: z.number().nullable(),
  q50Kw: z.number().nullable(),
  q90Kw: z.number().nullable(),
  persistenceKw: z.number().nullable(),
});
export type LivePoint = z.infer<typeof livePointSchema>;

const liveSourceSchema = z.enum(["live", "replay"]);

const liveFeederSchema = z.object({
  feederId: z.string(),
  limitKw: z.number(),
  source: liveSourceSchema,
  /** Always true today: the readings come from the simulated fleet, never a real meter. */
  simulated: z.literal(true),
  /** Why this source: "live feed", "no live feed", "live feed stale (last tick 52 min ago)". */
  sourceNote: z.string(),
  tickSeconds: z.number(),
  /** Live only: the latest step with a reading. Replay: null, the client's replay clock is "now". */
  latestTs: z.string().nullable(),
  /** Live only: wall-clock time of the last write, and its age in seconds when served. */
  writtenAt: z.string().nullable(),
  ageSec: z.number().nullable(),
  /** Live only: simulated-time speed-up (e.g. 90 = one 15-min tick every 10 s). */
  speed: z.number().nullable(),
  forecastLabel: z.string(),
  points: z.array(livePointSchema),
});
export type LiveFeeder = z.infer<typeof liveFeederSchema>;

export const liveContract = {
  getFeeder: oc
    .input(
      z.object({
        feederId: z.string(),
        from: z.string(), // ISO-8601 UTC, replay day window (used by the replay fallback)
        to: z.string(),
        runId: z.string().optional(),
      }),
    )
    .output(liveFeederSchema.nullable()),
};

// ---- Pure view logic for the panel: slice at "now", score, warn. ----

/** A miss is flagged when it is off by more than 25% of the prediction and 5% of the limit. */
const MISS_REL = 0.25;
const MISS_OF_LIMIT = 0.05;

export type LiveTick = {
  ts: string;
  predictedKw: number;
  actualKw: number;
  errorKw: number; // actual - predicted
  flagged: boolean;
};

type LiveWarning = {
  level: CongestionLevelValue;
  /** First forecast step over the limit at the warning's quantile, if any. */
  firstOverTs: string | null;
  peakQ50Kw: number;
  peakQ90Kw: number;
};

export type LiveView = {
  nowTs: string;
  past: LivePoint[]; // actual known, ts <= now
  ahead: LivePoint[]; // forecast only, ts > now, within the horizon
  next: LivePoint | null; // the next 15-min step
  lastTick: LiveTick | null;
  misses: LiveTick[]; // flagged past ticks, newest first
  meanAbsErrorKw: number | null;
  warning: LiveWarning;
};

function scoreTick(p: LivePoint, limitKw: number): LiveTick | null {
  if (p.actualKw == null || p.q50Kw == null) return null;
  const errorKw = p.actualKw - p.q50Kw;
  const flagged =
    Math.abs(errorKw) > MISS_REL * Math.max(Math.abs(p.q50Kw), 1) &&
    Math.abs(errorKw) > MISS_OF_LIMIT * limitKw;
  return { ts: p.ts, predictedKw: p.q50Kw, actualKw: p.actualKw, errorKw, flagged };
}

/** The congestion notice's grades on the forecast ahead: severe = P10 over, congested = P50 over, watch = P90 over. */
function warnFor(ahead: LivePoint[], limitKw: number): LiveWarning {
  const over = (key: "q10Kw" | "q50Kw" | "q90Kw") => ahead.find((p) => (p[key] ?? -Infinity) > limitKw);
  const severe = over("q10Kw");
  const congested = over("q50Kw");
  const watch = over("q90Kw");
  const hit = severe ?? congested ?? watch;
  return {
    level: severe ? "severe" : congested ? "congested" : watch ? "watch" : "none",
    firstOverTs: hit?.ts ?? null,
    peakQ50Kw: Math.max(0, ...ahead.map((p) => p.q50Kw ?? 0)),
    peakQ90Kw: Math.max(0, ...ahead.map((p) => p.q90Kw ?? 0)),
  };
}

/**
 * Cut the series at `nowTs` (timestamps compare as text). Actuals after now are dropped, so the
 * replay fallback never shows a future reading. `horizonSteps` forecast steps are kept ahead.
 */
export function liveView(feed: LiveFeeder, nowTs: string, horizonSteps = 8): LiveView {
  const past = feed.points.filter((p) => p.ts <= nowTs);
  const ahead = feed.points
    .filter((p) => p.ts > nowTs)
    .slice(0, horizonSteps)
    .map((p) => ({ ...p, actualKw: null }));
  const scored = past.map((p) => scoreTick(p, feed.limitKw)).filter((t): t is LiveTick => t !== null);
  const lastTick = scored.at(-1) ?? null;
  const misses = scored.filter((t) => t.flagged).reverse();
  const meanAbsErrorKw = scored.length
    ? scored.reduce((sum, t) => sum + Math.abs(t.errorKw), 0) / scored.length
    : null;
  return {
    nowTs,
    past,
    ahead,
    next: ahead[0] ?? null,
    lastTick,
    misses,
    meanAbsErrorKw,
    warning: warnFor(ahead, feed.limitKw),
  };
}
