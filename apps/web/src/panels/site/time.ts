import { STEPS_PER_DAY, stepToTs } from "@/demo/replay";

// Shared by the site and flexibility panels. Storage is UTC ISO with 'Z'; the replay clock and
// every label are Europe/Bucharest (plan R19).

const BUCHAREST = new Intl.DateTimeFormat("en-GB", {
  timeZone: "Europe/Bucharest",
  year: "numeric",
  month: "numeric",
  day: "numeric",
  hour: "numeric",
  minute: "numeric",
  hourCycle: "h23",
});

// Charts format dates in the browser's zone, so hand them Bucharest wall-clock time.
export function wallClock(ts: string): Date {
  const parts = Object.fromEntries(
    BUCHAREST.formatToParts(new Date(ts)).map((part) => [
      part.type,
      Number(part.value),
    ]),
  );
  return new Date(
    parts.year as number,
    (parts.month as number) - 1,
    parts.day as number,
    parts.hour as number,
    parts.minute as number,
  );
}

export const hhmm = (date: Date) =>
  `${String(date.getHours()).padStart(2, "0")}:${String(date.getMinutes()).padStart(2, "0")}`;

/** The stored `ts` string for a replay step. */
export const stepTs = stepToTs;

/** [from, to) of the replay day that contains `step`, as stored `ts` strings. */
export function dayRange(step: number): [string, string] {
  const day = Math.floor(step / STEPS_PER_DAY);
  return [stepTs(day * STEPS_PER_DAY), stepTs((day + 1) * STEPS_PER_DAY)];
}

export const inRange = (ts: string, [from, to]: [string, string]) =>
  ts >= from && ts < to;

const DAY_LABEL = new Intl.DateTimeFormat("en-GB", {
  timeZone: "Europe/Bucharest",
  weekday: "short",
  day: "numeric",
  month: "short",
});
export const dayName = (ts: string) => DAY_LABEL.format(new Date(ts));
