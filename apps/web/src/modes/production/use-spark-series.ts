import type { CountySeriesItem, ParkForecastItem, ParkItem } from "@wattsentry/api/contract/production";
import { useQuery } from "@tanstack/react-query";
import { useMemo } from "react";

import type { SparkPoint } from "@/components/map-spark/map-spark";
import { stepToDate } from "@/demo/replay";
import { useReplayStep } from "@/demo/use-replay";
import { orpc } from "@/utils/orpc";

import { hourly } from "./data";

const FOREVER = Number.POSITIVE_INFINITY;
const HOUR_MS = 3_600_000;
const HALF_HOUR_MS = HOUR_MS / 2;
// The map sparklines look 6 h back and 24 h ahead of the replay clock: the same day ahead the
// inspector's fan chart and "strongest wind around …" sentence cover.
export const SPARK_PAST_HOURS = 6;
export const SPARK_AHEAD_HOURS = 24;

const BUCHAREST_HOUR = new Intl.DateTimeFormat("en-GB", {
  timeZone: "Europe/Bucharest",
  hour: "2-digit",
  minute: "2-digit",
  hourCycle: "h23",
});

export const sparkTime = (ms: number) => BUCHAREST_HOUR.format(new Date(ms));

// Hourly buckets plot at their middle and are labelled by their start ("11:00"). A bucket counts
// as measured only once it has ended, so the measured line always stops at or before "now".
type Bucket = { ts: string; actual?: number | null; p10?: number | null; p50?: number | null; p90?: number | null };

function toSpark(buckets: Bucket[], now: number): SparkPoint[] {
  return buckets.map((bucket) => {
    const start = Date.parse(bucket.ts);
    const ended = start + HOUR_MS <= now;
    return {
      t: start + HALF_HOUR_MS,
      label: sparkTime(start),
      actual: ended ? (bucket.actual ?? null) : null,
      p10: bucket.p10,
      p50: bucket.p50,
      p90: bucket.p90,
    };
  });
}

function parkSeries(
  park: ParkItem,
  forecasts: ParkForecastItem[],
  measured: Map<string, number | null> | undefined,
  now: number,
): SparkPoint[] {
  const hours = hourly(forecasts.filter((row) => row.parkId === park.id));
  return toSpark(
    hours.map((hour) => ({ ...hour, actual: measured?.get(hour.ts) ?? null })),
    now,
  );
}

// Rooftop solar per county is tens to hundreds of kW, so its sparkline reads in kW. Replay stores
// no rows for dark hours; those are zero output, not missing data.
const KW_PER_MW = 1000;

function solarSeries(item: CountySeriesItem, window: { start: string; end: string }, now: number): SparkPoint[] {
  const byTs = new Map(item.points.map((point) => [point.ts, point]));
  const kw = (value: number | null | undefined) => (value == null ? 0 : value * KW_PER_MW);
  const buckets: Bucket[] = [];
  for (let ms = Date.parse(window.start); ms < Date.parse(window.end); ms += HOUR_MS) {
    const ts = `${new Date(ms).toISOString().slice(0, 19)}Z`;
    const point = byTs.get(ts);
    buckets.push({
      ts,
      actual: kw(point?.solarActualMw),
      p10: kw(point?.solarP10Mw),
      p50: kw(point?.solarP50Mw),
      p90: kw(point?.solarP90Mw),
    });
  }
  return toSpark(buckets, now);
}

/** Per-park and per-county sparkline series around the replay clock (whole run fetched once). */
export function useProductionSparks(parks: ParkItem[]) {
  const [step] = useReplayStep();
  const forecasts = useQuery(orpc.production.getParkForecasts.queryOptions({ staleTime: FOREVER }));
  const series = useQuery(orpc.production.getCountySeries.queryOptions({ staleTime: FOREVER }));
  const now = stepToDate(step).getTime();
  const from = now - SPARK_PAST_HOURS * HOUR_MS;
  const to = now + SPARK_AHEAD_HOURS * HOUR_MS;

  // Slice the run to the window once per hour of clock, not per 15-min step.
  const hourKey = Math.floor(now / HOUR_MS);
  const window = useMemo(() => {
    const iso = (hours: number) => `${new Date((hourKey + hours) * HOUR_MS).toISOString().slice(0, 19)}Z`;
    return { start: iso(-SPARK_PAST_HOURS - 2), end: iso(SPARK_AHEAD_HOURS + 2) };
  }, [hourKey]);

  const parkRows = useMemo(
    () => (forecasts.data ?? []).filter((row) => row.ts >= window.start && row.ts < window.end),
    [forecasts.data, window],
  );
  const countyItems = useMemo(
    () =>
      (series.data ?? []).map((item) => ({
        ...item,
        solar: item.points.some((point) => point.solarP50Mw !== null),
        points: item.points.filter((point) => point.ts >= window.start && point.ts < window.end),
      })),
    [series.data, window],
  );

  return useMemo(() => {
    // Turbine telemetry is summed per county; it is the park's own output only when the park is
    // alone in its county.
    const parksPerCounty = new Map<string, number>();
    for (const park of parks) parksPerCounty.set(park.county, (parksPerCounty.get(park.county) ?? 0) + 1);
    const windByCounty = new Map(
      countyItems.map((item) => [item.county, new Map(item.points.map((point) => [point.ts, point.windActualMw]))]),
    );
    const parkSparks = new Map(
      parks.map((park) => [
        park.id,
        parkSeries(
          park,
          parkRows,
          parksPerCounty.get(park.county) === 1 ? windByCounty.get(park.county) : undefined,
          now,
        ),
      ]),
    );
    const countySparks = new Map(
      countyItems
        .filter((item) => item.solar)
        .map((item) => [item.county, solarSeries(item, window, now)]),
    );
    return { now, from, to, parkSparks, countySparks };
  }, [parks, parkRows, countyItems, window, now, from, to]);
}
