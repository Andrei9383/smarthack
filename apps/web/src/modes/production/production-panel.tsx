"use client";

import { useQuery } from "@tanstack/react-query";
import { curveMonotoneX } from "d3-shape";
import { useMemo } from "react";

import { Band } from "@/components/charts/band";
import { Grid } from "@/components/charts/grid";
import { Line, LineChart } from "@/components/charts/line-chart";
import { ChartTooltip } from "@/components/charts/tooltip";
import { XAxis } from "@/components/charts/x-axis";

import { expectedMwh, hourly } from "./data";
import { useProductionData } from "./use-production-data";
import { useHour, useSelectedPark } from "./use-production-state";
import { fetchWindMeta } from "./wind-field";

// Sentinel's own forecast: primary green line over a brand-500 P10–P90 band (ui-style §3).
const FAN_FILL = "var(--brand-500)";
const FAN_LINE = "var(--primary)";
const SECTION_LABEL = "text-[11px] font-medium tracking-[0.04em] text-muted-foreground uppercase";
const CHART_MARGIN = { left: 8, right: 8, top: 8, bottom: 24 };
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
  const hour = useHour();
  const [selected, setSelected] = useSelectedPark();
  const { data: meta } = useQuery({
    queryKey: ["windfield-meta"],
    queryFn: fetchWindMeta,
    staleTime: Number.POSITIVE_INFINITY,
  });
  const { label, parks, rows, counties, isLoading } = useProductionData();
  const park = parks.find((p) => p.id === selected) ?? parks[0];

  const { fan, labels, bestHours, parkMwh } = useMemo(() => {
    const parkRows = rows.filter((row) => row.parkId === park?.id);
    const parkHours = hourly(parkRows);
    return {
      fan: parkHours.map((row) => ({
        date: wallClock(row.ts),
        p10: row.p10,
        p50: row.p50,
        p90: row.p90,
      })),
      labels: parkHours.map((row) => localLabel(row.ts)),
      bestHours: hourly(rows)
        .sort((a, b) => b.p50 - a.p50)
        .slice(0, BEST_HOURS),
      parkMwh: expectedMwh(parkRows),
    };
  }, [park?.id, rows]);
  const windCounties = counties.filter((county) => county.windMwh > 0);

  const hourTs = meta?.hours[hour]?.ts;
  const windLabel = meta?.source === "mock" ? "mock wind field" : "real RO, ERA5 100 m wind";

  return (
    <aside className="absolute top-3 right-3 bottom-3 z-10 flex w-[440px] max-w-[calc(100%-1.5rem)] flex-col gap-5 overflow-y-auto rounded-lg border border-border bg-card p-5 text-card-foreground shadow-card">
      <header className="space-y-1.5">
        <h2 className="text-base font-semibold text-foreground">
          {park ? `${park.id} · ${park.name}` : "Production"}
        </h2>
        <p className="text-[13px] leading-relaxed text-text-secondary tabular-nums">
          {park && `${park.county} · ${park.ratedMw} MW rated · ${Math.round(parkMwh)} MWh next 24 h · `}
          <span className="inline-flex items-center rounded-full border border-border bg-muted px-2 py-px text-[11px] font-medium text-text-secondary">
            {label ?? (isLoading ? "loading" : "no run loaded")}
          </span>
        </p>
      </header>

      {/* Neighbouring parks stack at country zoom, so their markers alone can't select them all. */}
      {parks.length > 1 && (
        <nav className="flex flex-wrap gap-1.5" aria-label="Parks">
          {parks.map((p) => (
            <button
              key={p.id}
              type="button"
              aria-pressed={p.id === park?.id}
              onClick={() => setSelected(p.id)}
              className={`rounded-full border px-2.5 py-1 text-xs font-medium tabular-nums transition-colors ${
                p.id === park?.id
                  ? "border-brand-100 bg-brand-50 text-brand-900"
                  : "border-border bg-card text-text-secondary hover:bg-muted hover:text-foreground"
              }`}
            >
              {p.id}
            </button>
          ))}
        </nav>
      )}

      <p className="flex items-baseline justify-between gap-3">
        <span className={SECTION_LABEL}>Wind at 100 m</span>
        <span className="text-xs text-text-secondary tabular-nums">
          {hourTs ? `${localLabel(hourTs)} Bucharest` : "—"} · {meta ? windLabel : "loading"}
        </span>
      </p>

      {fan.length === 0 ? (
        !isLoading && (
          <p className="rounded-lg bg-muted p-4 text-[13px] text-text-secondary">
            No park forecast rows for this time in the loaded run.
          </p>
        )
      ) : (
        <>
          <section className="space-y-2">
            <h3 className={SECTION_LABEL}>Park output, P10–P90 (MW)</h3>
            <LineChart data={fan} aspectRatio="5 / 2" margin={CHART_MARGIN} animationDuration={0}>
              <Grid />
              <Band
                upperKey="p90"
                lowerKey="p10"
                label="P10–P90"
                fill={FAN_FILL}
                fillOpacity={0.16}
                strokeOpacity={0.5}
                curve={curveMonotoneX}
              />
              <Line dataKey="p50" stroke={FAN_LINE} curve={curveMonotoneX} fadeEdges={false} />
              <XAxis numTicks={5} labels={labels} />
              <ChartTooltip />
            </LineChart>
          </section>

          <section className="grid grid-cols-2 gap-x-6 gap-y-4 rounded-lg bg-sidebar p-4">
            {windCounties.map(({ county, windMwh }) => (
              <div key={county} className="space-y-1">
                <div className={SECTION_LABEL}>{county} wind, next 24 h</div>
                <div className="text-[28px] leading-none font-semibold text-foreground tabular-nums">
                  {Math.round(windMwh)} <span className="text-[13px] font-normal text-muted-foreground">MWh</span>
                </div>
              </div>
            ))}
          </section>

          <section className="space-y-2">
            <h3 className={SECTION_LABEL}>Best production hours, all parks</h3>
            <ul className="divide-y divide-border text-[13px] tabular-nums">
              {bestHours.map(({ ts, p50 }) => (
                <li key={ts} className="flex justify-between py-1.5">
                  <span className="font-medium text-foreground">{localLabel(ts)}</span>
                  <span className="text-text-secondary">{Math.round(p50)} MW</span>
                </li>
              ))}
            </ul>
          </section>
        </>
      )}
    </aside>
  );
}
