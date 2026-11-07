// The replay clock: one test week in 15-min steps, starting 00:00 Europe/Bucharest.
export const STEP_MINUTES = 15;
export const STEPS_PER_DAY = 96;
export const DAYS = 7;
export const STEPS_IN_WEEK = STEPS_PER_DAY * DAYS;
export const NOON_STEP = 48;

const WEEK_START_UTC = Date.UTC(2025, 5, 2, 21, 0); // 2025-06-03 00:00 EEST, no DST change (R19)

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
