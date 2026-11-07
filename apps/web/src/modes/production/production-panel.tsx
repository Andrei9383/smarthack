"use client";

import { useQuery } from "@tanstack/react-query";
import { curveMonotoneX } from "d3-shape";
import { useMemo } from "react";

import { Band } from "@/components/charts/band";
import { Grid } from "@/components/charts/grid";
import { Line, LineChart } from "@/components/charts/line-chart";
import { ChartTooltip } from "@/components/charts/tooltip";
import { XAxis } from "@/components/charts/x-axis";

import { DATA_LABEL, expectedMwh, PARKS, parkForecast, type Region } from "./data";
import { useHour, useSelectedPark } from "./use-production-state";
import { fetchWindMeta } from "./wind-field";

const FAN_COLOR = "#2dd4bf";
const CHART_MARGIN = { left: 8, right: 8, top: 8, bottom: 24 };
const REGIONS: Region[] = ["Dobrogea", "Moldova"];
const BEST_HOURS = 3;

const BUCHAREST_HOUR = new Intl.DateTimeFormat("en-GB", {
  timeZone: "Europe/Bucharest",
  hour: "2-digit",
  minute: "2-digit",
  hourCycle: "h23",
});
const BUCHAREST_PARTS = new Intl.DateTimeFormat("en-GB", {
  timeZone: "Europe/Bucharest",
  year: "numeric",
  month: "numeric",
  day: "numeric",
  hour: "numeric",
  hourCycle: "h23",
});

const localLabel = (ts: string) => BUCHAREST_HOUR.format(new Date(ts));

// Charts format dates in the browser's zone, so hand them Bucharest wall-clock time (plan R19).
function wallClock(ts: string) {
  const parts = Object.fromEntries(
    BUCHAREST_PARTS.formatToParts(new Date(ts)).map((part) => [part.type, Number(part.value)]),
  );
  return new Date(parts.year as number, (parts.month as number) - 1, parts.day as number, parts.hour as number);
}

export function ProductionPanel() {
  const [hour, setHour] = useHour();
  const [selected] = useSelectedPark();
  const { data: meta } = useQuery({
    queryKey: ["windfield-meta"],
    queryFn: fetchWindMeta,
    staleTime: Number.POSITIVE_INFINITY,
  });
  const park = PARKS.find((p) => p.id === selected) ?? PARKS[0];

  const { fan, labels, totals, bestHours, parkMwh } = useMemo(() => {
    const rows = park ? parkForecast(park) : [];
    const allParks = PARKS.map((p) => ({ park: p, rows: parkForecast(p) }));
    const byHour = rows.map((row, index) => ({
      ts: row.ts,
      mw: allParks.reduce((sum, entry) => sum + (entry.rows[index]?.p50_mw ?? 0), 0),
    }));
    return {
      fan: rows.map((row) => ({
        date: wallClock(row.ts),
        p10: row.p10_mw,
        p50: row.p50_mw,
        p90: row.p90_mw,
      })),
      labels: rows.map((row) => localLabel(row.ts)),
      totals: REGIONS.map((region) => ({
        region,
        mwh: allParks
          .filter((entry) => entry.park.region === region)
          .reduce((sum, entry) => sum + expectedMwh(entry.rows), 0),
      })),
      bestHours: [...byHour].sort((a, b) => b.mw - a.mw).slice(0, BEST_HOURS),
      parkMwh: expectedMwh(rows),
    };
  }, [park]);

  const hourTs = meta?.hours[hour]?.ts;
  const windLabel = meta?.source === "mock" ? "mock wind field" : "real RO, ERA5 100 m wind";

  return (
    <aside className="absolute top-3 right-3 bottom-3 z-10 flex w-[440px] max-w-[calc(100%-1.5rem)] flex-col gap-3 overflow-y-auto rounded-md border bg-background/95 p-3 backdrop-blur">
      <header>
        <h2 className="font-semibold">
          {park?.id} · {park?.name}
        </h2>
        <p className="text-xs text-muted-foreground">
          {park?.county} · {park?.rated_mw} MW rated · {Math.round(parkMwh)} MWh next 24 h ·{" "}
          <span className="font-medium text-amber-400">{DATA_LABEL}</span>
        </p>
      </header>

      <section className="space-y-1">
        <label htmlFor="wind-hour" className="flex justify-between text-sm font-medium">
          <span>Wind at 100 m</span>
          <span className="text-xs font-normal text-muted-foreground">
            {hourTs ? `${localLabel(hourTs)} Bucharest` : "—"} · {meta ? windLabel : "loading"}
          </span>
        </label>
        <input
          id="wind-hour"
          type="range"
          min={0}
          max={(meta?.hours.length ?? 24) - 1}
          value={hour}
          onChange={(event) => setHour(Number(event.target.value))}
          className="w-full accent-teal-400"
        />
      </section>

      <section className="space-y-1">
        <h3 className="text-sm font-medium">Park output, P10–P90 (MW)</h3>
        <LineChart data={fan} aspectRatio="5 / 2" margin={CHART_MARGIN} animationDuration={0}>
          <Grid />
          <Band upperKey="p90" lowerKey="p10" label="P10–P90" fill={FAN_COLOR} curve={curveMonotoneX} />
          <Line dataKey="p50" stroke={FAN_COLOR} curve={curveMonotoneX} fadeEdges={false} />
          <XAxis numTicks={5} labels={labels} />
          <ChartTooltip />
        </LineChart>
      </section>

      <section className="grid grid-cols-2 gap-2 rounded border p-2">
        {totals.map(({ region, mwh }) => (
          <div key={region}>
            <div className="text-[11px] text-muted-foreground">{region}, next 24 h</div>
            <div className="text-base font-semibold">{Math.round(mwh)} MWh</div>
          </div>
        ))}
      </section>

      <section className="space-y-1">
        <h3 className="text-sm font-medium">Best production hours, all parks</h3>
        <ul className="space-y-1 text-sm">
          {bestHours.map(({ ts, mw }) => (
            <li key={ts} className="flex justify-between">
              <span>{localLabel(ts)}</span>
              <span className="text-muted-foreground">{Math.round(mw)} MW</span>
            </li>
          ))}
        </ul>
      </section>
    </aside>
  );
}
