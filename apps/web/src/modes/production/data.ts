import type { ParkForecastItem } from "@sentinel-app/api/contract/production";

import { stepToDate } from "@/demo/replay";

const HOUR_MS = 3_600_000;
const WINDOW_HOURS = 24;
const STEP_HOURS = 0.25;

export type HourlyForecast = { ts: string; p10: number; p50: number; p90: number };

// Stored ts has no milliseconds, and rows are compared to these bounds as strings.
const isoSeconds = (ms: number) => new Date(ms).toISOString().replace(".000Z", "Z");

// The 24 h after the replay clock, on whole hours so scrubbing inside an hour refetches nothing.
export function forecastWindow(step: number) {
  const start = Math.floor(stepToDate(step).getTime() / HOUR_MS) * HOUR_MS;
  return { from: isoSeconds(start), to: isoSeconds(start + WINDOW_HOURS * HOUR_MS) };
}

export function inWindow(rows: ParkForecastItem[], { from, to }: { from: string; to: string }) {
  return rows.filter((row) => row.ts >= from && row.ts < to);
}

export function expectedMwh(rows: ParkForecastItem[]) {
  return rows.reduce((total, row) => total + (row.expectedMwh ?? (row.p50Mw ?? 0) * STEP_HOURS), 0);
}

// Mean MW per hour, summed across parks when rows hold several: the 15-min rows are too dense to chart.
export function hourly(rows: ParkForecastItem[]): HourlyForecast[] {
  const hours = new Map<string, Map<string, ParkForecastItem[]>>();
  for (const row of rows) {
    const hour = `${row.ts.slice(0, 13)}:00:00Z`;
    const byPark = hours.get(hour) ?? new Map<string, ParkForecastItem[]>();
    byPark.set(row.parkId, [...(byPark.get(row.parkId) ?? []), row]);
    hours.set(hour, byPark);
  }
  const sum = (byPark: Map<string, ParkForecastItem[]>, pick: (row: ParkForecastItem) => number | null | undefined) =>
    [...byPark.values()].reduce(
      (total, parkRows) => total + parkRows.reduce((s, row) => s + (pick(row) ?? 0), 0) / parkRows.length,
      0,
    );
  return [...hours.entries()]
    .sort(([a], [b]) => a.localeCompare(b))
    .map(([ts, byPark]) => ({
      ts,
      p10: sum(byPark, (row) => row.p10Mw),
      p50: sum(byPark, (row) => row.p50Mw),
      p90: sum(byPark, (row) => row.p90Mw),
    }));
}
