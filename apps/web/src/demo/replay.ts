// The replay clock: one test week in 15-min steps, starting 00:00 Europe/Bucharest.
export const STEP_MINUTES = 15;
export const STEPS_PER_DAY = 96;
export const DAYS = 7;
export const STEPS_IN_WEEK = STEPS_PER_DAY * DAYS;
export const NOON_STEP = 48;
// F7's peak day in the demo week; the fault at site 1042 is the day after (#30).
export const HERO_DAY = 2;
export const FAULT_DAY = 3;
export const DEFAULT_STEP = HERO_DAY * STEPS_PER_DAY + NOON_STEP;

// 2025-06-02 00:00 EEST: the week #30 seeds, no DST change (R19). Align with runs.week_start (#56).
const WEEK_START_UTC = Date.UTC(2025, 5, 1, 21, 0);

export function stepToDate(step: number): Date {
  return new Date(WEEK_START_UTC + step * STEP_MINUTES * 60_000);
}

export const dayOfStep = (step: number) => Math.floor(step / STEPS_PER_DAY);
export const stepOfDay = (step: number) => step % STEPS_PER_DAY;
export const clampStep = (step: number) => Math.max(0, Math.min(STEPS_IN_WEEK - 1, step));

const TIME = new Intl.DateTimeFormat("en-GB", {
  timeZone: "Europe/Bucharest",
  hour: "2-digit",
  minute: "2-digit",
  hourCycle: "h23",
});
const DAY = new Intl.DateTimeFormat("en-GB", {
  timeZone: "Europe/Bucharest",
  weekday: "short",
  day: "numeric",
});

export const timeLabel = (step: number) => TIME.format(stepToDate(step));
export const dayLabel = (day: number) => DAY.format(stepToDate(day * STEPS_PER_DAY));

// Stored timestamps are 'YYYY-MM-DDTHH:MM:SSZ', compared as text, so match that format exactly.
export const stepToTs = (step: number) => `${stepToDate(step).toISOString().slice(0, 19)}Z`;

export function dayWindow(step: number) {
  const day = dayOfStep(step);
  return { from: stepToTs(day * STEPS_PER_DAY), to: stepToTs((day + 1) * STEPS_PER_DAY) };
}

/** Inverse of stepToTs for stored timestamps; may fall outside the week (callers clamp or filter). */
export const tsToStep = (ts: string) =>
  Math.round((Date.parse(ts) - WEEK_START_UTC) / (STEP_MINUTES * 60_000));

/** "Wed 4 · 12:00", the SCADA-style clock label. */
export const clockLabel = (step: number) => `${dayLabel(dayOfStep(step))} · ${timeLabel(step)}`;
