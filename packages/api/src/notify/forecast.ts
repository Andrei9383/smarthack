// Day-ahead feeder congestion forecast (card #120, team/contracts/notify.md).
//
// Export forecast per 15-min step = sum over the feeder's homes of M1 PVNet PV quantiles (asset_steps)
// minus yesterday's measured load at the same time (persistence load forecast). Everything used is known
// the evening before the day, so the notice can go out then. Pure functions: the router feeds them rows.

export const STEP_HOURS = 0.25;
const DAY_MS = 24 * 60 * 60 * 1000;

export type CongestionLevel = "none" | "watch" | "congested" | "severe";
export const LEVEL_RANK: Record<CongestionLevel, number> = { none: 0, watch: 1, congested: 2, severe: 3 };

// Thresholds (why: see team/contracts/notify.md).
export const THRESHOLDS = {
  /** congested: P50 export over the limit for at least this many steps (30 min) */
  congestedSteps: 2,
  /** severe: P10 export over the limit for at least this many steps (1 h), i.e. over even on a dull day */
  severeP10Steps: 4,
  /** severe: or the P50 peak at least this far over the limit */
  severePeakRatio: 1.3,
  /** watch: P90 crosses the limit, or P50 comes within this share of it */
  watchRatio: 0.9,
};

export type LoadSource = "yesterday" | "weekly_profile" | "none";

export type ForecastStep = {
  ts: string;
  q10Kw: number;
  q50Kw: number;
  q90Kw: number;
  limitKw: number;
  loadSource: LoadSource;
};

export type FeederForecast = {
  level: CongestionLevel;
  windowFrom: string | null; // first step of the congestion window (UTC)
  windowTo: string | null; // end of the last step (exclusive, UTC)
  stepsOverP10: number;
  stepsOverP50: number;
  stepsOverP90: number;
  peakP50Kw: number;
  peakP90Kw: number;
  overLimitP50Kwh: number;
  loadSource: LoadSource;
  reason: string;
  steps: ForecastStep[];
};

export const shiftTs = (ts: string, ms: number) =>
  new Date(Date.parse(ts) + ms).toISOString().replace(".000Z", "Z");
export const yesterday = (ts: string) => shiftTs(ts, -DAY_MS);
const minuteOfDay = (ts: string) => {
  const d = new Date(ts);
  return d.getUTCHours() * 60 + d.getUTCMinutes();
};

/**
 * Load forecast for each step: yesterday's measured value; if yesterday is missing (first replay day),
 * the mean of the same time of day over earlier days, then over the whole week (flagged); else 0.
 */
export function loadForecaster(loadByTs: Map<string, number>) {
  const byMinute = new Map<number, { sum: number; n: number }>();
  for (const [ts, kw] of loadByTs) {
    const m = minuteOfDay(ts);
    const acc = byMinute.get(m) ?? { sum: 0, n: 0 };
    acc.sum += kw;
    acc.n += 1;
    byMinute.set(m, acc);
  }
  return (ts: string): { kw: number; source: LoadSource } => {
    const prev = loadByTs.get(yesterday(ts));
    if (prev !== undefined) return { kw: prev, source: "yesterday" };
    const profile = byMinute.get(minuteOfDay(ts));
    if (profile && profile.n > 0) return { kw: profile.sum / profile.n, source: "weekly_profile" };
    return { kw: 0, source: "none" };
  };
}

function worstSource(steps: ForecastStep[]): LoadSource {
  if (steps.some((s) => s.loadSource === "none")) return "none";
  if (steps.some((s) => s.loadSource === "weekly_profile")) return "weekly_profile";
  return "yesterday";
}

const fmtKw = (kw: number) => `${Math.round(kw)} kW`;

export function gradeForecast(steps: ForecastStep[]): FeederForecast {
  const over = (pick: (s: ForecastStep) => number) => steps.filter((s) => pick(s) > s.limitKw);
  const overP10 = over((s) => s.q10Kw);
  const overP50 = over((s) => s.q50Kw);
  const overP90 = over((s) => s.q90Kw);
  const limit = steps[0]?.limitKw ?? 0;
  const peakP50 = Math.max(0, ...steps.map((s) => s.q50Kw));
  const peakP90 = Math.max(0, ...steps.map((s) => s.q90Kw));
  const overKwh = steps.reduce((sum, s) => sum + Math.max(0, s.q50Kw - s.limitKw) * STEP_HOURS, 0);
  const t = THRESHOLDS;

  let level: CongestionLevel = "none";
  let reason = `P90 peak ${fmtKw(peakP90)} stays under the ${fmtKw(limit)} limit`;
  if (overP10.length >= t.severeP10Steps || (overP50.length >= t.congestedSteps && limit > 0 && peakP50 >= t.severePeakRatio * limit)) {
    level = "severe";
    reason =
      overP10.length >= t.severeP10Steps
        ? `even the P10 (dull-day) forecast is over the limit for ${overP10.length * 15} min`
        : `P50 peak ${fmtKw(peakP50)} is ${Math.round((peakP50 / limit - 1) * 100)}% over the ${fmtKw(limit)} limit`;
  } else if (overP50.length >= t.congestedSteps) {
    level = "congested";
    reason = `P50 export over the ${fmtKw(limit)} limit for ${overP50.length * 15} min, peak ${fmtKw(peakP50)}`;
  } else if (overP90.length > 0 || (limit > 0 && peakP50 >= t.watchRatio * limit)) {
    level = "watch";
    reason =
      overP90.length > 0
        ? `only the P90 (sunny-side) forecast crosses the limit, ${overP90.length * 15} min`
        : `P50 peak ${fmtKw(peakP50)} within ${Math.round((1 - t.watchRatio) * 100)}% of the limit`;
  }

  const windowSteps = level === "watch" ? overP90 : level === "none" ? [] : overP50;
  const first = windowSteps[0];
  const last = windowSteps[windowSteps.length - 1];
  return {
    level,
    windowFrom: first?.ts ?? null,
    windowTo: last ? shiftTs(last.ts, STEP_HOURS * 60 * 60 * 1000) : null,
    stepsOverP10: overP10.length,
    stepsOverP50: overP50.length,
    stepsOverP90: overP90.length,
    peakP50Kw: peakP50,
    peakP90Kw: peakP90,
    overLimitP50Kwh: overKwh,
    loadSource: worstSource(steps),
    reason,
    steps,
  };
}

export const inWindow = (ts: string, from: string | null, to: string | null) =>
  from !== null && to !== null && ts >= from && ts < to;
