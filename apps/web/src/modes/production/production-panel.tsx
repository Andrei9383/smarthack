"use client";

import { useQuery } from "@tanstack/react-query";
import { curveMonotoneX } from "d3-shape";
import { useMemo, useState } from "react";

import { Band } from "@/components/charts/band";
import { Grid } from "@/components/charts/grid";
import { Line, LineChart } from "@/components/charts/line-chart";
import { ChartTooltip } from "@/components/charts/tooltip";
import { stepToDate } from "@/lib/replay";
import { useReplayStep } from "@/lib/use-replay";
import { Term } from "@/components/help";
import { DetailsSheet, Metric, MetricRow, PopList, RowButton, SectionHeader, SheetSection } from "@/components/panels/disclosure";

import { hm } from "@/lib/time";
import { expectedMwh, hourly } from "./data";
import { FanHourTicks, FanNowMarker, FanYAxis } from "./fan-chart-overlays";
import { useProductionData } from "./use-production-data";
import { useHour, useSelectedPark } from "./use-production-state";
import { WindActions } from "./wind-actions";
import { fetchWindMeta } from "./wind-field";

// WattSentry's own forecast: primary green line over a brand-500 P10–P90 band.
const FAN_FILL = "var(--brand-500)";
const FAN_LINE = "var(--sentinel)";
const Y_TICKS = 4;
const TICK_HOURS = 6;
const CHART_MARGIN = { left: 32, right: 8, top: 26, bottom: 24 };
const BEST_HOURS = 3;
/** Counties on the glance; the rest are in Details. */
const TOP_COUNTIES = 5;

const BUCHAREST_PARTS = new Intl.DateTimeFormat("en-GB", {
  timeZone: "Europe/Bucharest",
  year: "numeric",
  month: "numeric",
  day: "numeric",
  hour: "numeric",
  hourCycle: "h23",
});


// Charts format dates in the browser's zone, so hand them Bucharest wall-clock time.
function wallClock(ts: string) {
  const parts = Object.fromEntries(
    BUCHAREST_PARTS.formatToParts(new Date(ts)).map((part) => [part.type, Number(part.value)]),
  );
  return new Date(parts.year as number, (parts.month as number) - 1, parts.day as number, parts.hour as number);
}

// Park names carry a trailing "(synthetic RO)"-style tag; the panel shows the name without it.
function splitTag(name: string): [string, string | null] {
  const match = /^(.*?)\s*(\([^()]+\))$/.exec(name);
  return match ? [match[1] ?? name, match[2] ?? null] : [name, null];
}

export function ProductionPanel() {
  const [step] = useReplayStep();
  const hour = useHour();
  const [selected, setSelected] = useSelectedPark();
  const { data: meta } = useQuery({
    queryKey: ["windfield-meta"],
    queryFn: fetchWindMeta,
    staleTime: Number.POSITIVE_INFINITY,
  });
  const { parks, rows, counties, isLoading } = useProductionData();
  const [detailsOpen, setDetailsOpen] = useState(false);
  const park = parks.find((p) => p.id === selected) ?? parks[0];

  const { fan, bestHours, parkMwh } = useMemo(() => {
    const parkRows = rows.filter((row) => row.parkId === park?.id);
    const parkHours = hourly(parkRows);
    return {
      fan: parkHours.map((row) => ({
        date: wallClock(row.ts),
        ts: row.ts,
        p10: row.p10,
        p50: row.p50,
        p90: row.p90,
      })),
      bestHours: hourly(rows)
        .sort((a, b) => b.p50 - a.p50)
        .slice(0, BEST_HOURS),
      parkMwh: expectedMwh(parkRows),
    };
  }, [park?.id, rows]);
  const windCounties = counties.filter((county) => county.windMwh > 0).sort((a, b) => b.windMwh - a.windMwh);
  const topCounty = windCounties[0]?.windMwh ?? 0;

  const peakMw = Math.max(...bestHours.map((row) => row.p50), 0);
  const best = bestHours[0];
  const nowRow = fan[0];
  const now = stepToDate(step);
  const nowDate = new Date(wallClock(now.toISOString()).getTime() + now.getUTCMinutes() * 60_000);
  const [parkName] = park ? splitTag(park.name) : [""];

  const hourTs = meta?.hours[hour]?.ts;
  const mwFmt = (v: number) => (v >= 10 ? Math.round(v).toString() : v.toFixed(1));

  return (
    <aside className="absolute top-3 right-3 bottom-3 z-10 flex w-[440px] max-w-[calc(100%-1.5rem)] flex-col overflow-y-auto rounded-xl bg-card px-5 pt-4 pb-3 text-card-foreground shadow-floating">
      <header className="space-y-1">
        <div className="flex items-start justify-between gap-3">
          <h2 className="text-title leading-snug text-foreground">
            {park ? (
              <>
                {`${park.id} · ${parkName}`}
              </>
            ) : (
              "Production"
            )}
          </h2>
        </div>
        {park && (
          <p className="text-caption text-muted-foreground tabular-nums">
            {park.county} · {park.ratedMw} MW rated
          </p>
        )}
      </header>

      {/* Neighbouring parks stack at country zoom, so their markers alone can't select them all. */}
      {parks.length > 1 && (
        <nav className="mt-3 flex flex-wrap gap-1" aria-label="Parks">
          {parks.map((p) => (
            <button
              key={p.id}
              type="button"
              aria-pressed={p.id === park?.id}
              onClick={() => setSelected(p.id)}
              className={`rounded-md px-2 py-1 text-caption font-medium tabular-nums transition-[background-color,color] duration-150 ${
                p.id === park?.id
                  ? "bg-primary text-primary-foreground"
                  : "bg-surface-raised text-text-secondary hover:text-foreground"
              }`}
            >
              {p.id}
            </button>
          ))}
        </nav>
      )}

      {fan.length === 0 ? (
        !isLoading && (
          <p className="mt-3 text-dense text-text-secondary">No park forecast rows for this time in the loaded run.</p>
        )
      ) : (
        <>
          <MetricRow className="mt-4 py-1">
            <Metric
              label="Next 24 h"
              value={Math.round(parkMwh)}
              unit="MWh"
              sub={`${park?.id} expected`}
              popTitle="Expected energy, next 24 hours"
              pop={
                <div className="space-y-2">
                  <PopList
                    rows={[
                      { key: "park", label: `${park?.id} park`, value: `${Math.round(parkMwh)} MWh` },
                      ...windCounties.map(({ county, windMwh }) => ({
                        key: county,
                        label: `${county} wind, all parks`,
                        value: `${Math.round(windMwh)} MWh`,
                      })),
                    ]}
                  />
                  <p className="text-caption text-muted-foreground">
                    The sum of the P50 (middle) forecast, 15 minutes at a time.
                  </p>
                </div>
              }
            />
            <Metric
              label="Now"
              value={mwFmt(nowRow?.p50 ?? 0)}
              unit="MW"
              sub={nowRow ? `range ${mwFmt(nowRow.p10)}–${mwFmt(nowRow.p90)}` : undefined}
              popTitle="How sure is this?"
              pop={
                <div className="space-y-2">
                  <PopList
                    rows={[
                      { key: "p90", label: "P90, high case", value: `${mwFmt(nowRow?.p90 ?? 0)} MW` },
                      { key: "p50", label: "P50, middle", value: `${mwFmt(nowRow?.p50 ?? 0)} MW` },
                      { key: "p10", label: "P10, low case", value: `${mwFmt(nowRow?.p10 ?? 0)} MW` },
                    ]}
                  />
                  <p className="text-caption text-muted-foreground">
                    Eight times in ten the park lands inside the shaded band on the chart.
                  </p>
                </div>
              }
            />
            <Metric
              label="Best hour"
              value={best ? hm(best.ts) : "—"}
              sub={best ? `${Math.round(best.p50)} MW, all parks` : undefined}
              popTitle="Best production hours, all parks"
              pop={
                <ol className="space-y-1 text-dense tabular-nums">
                  {bestHours.map(({ ts, p50 }, index) => {
                    const top = index === 0;
                    return (
                      <li key={ts} className="flex items-center gap-2.5">
                        <span className={`w-3 text-caption ${top ? "font-semibold text-status-ok" : "text-muted-foreground"}`}>
                          {index + 1}
                        </span>
                        <span className={`w-11 font-medium ${top ? "font-semibold text-foreground" : "text-foreground"}`}>{hm(ts)}</span>
                        <span className="h-1.5 flex-1 overflow-hidden rounded-full bg-muted">
                          <span
                            className={`block h-full rounded-full ${top ? "bg-sentinel" : "bg-brand-500/50"}`}
                            style={{ width: `${peakMw > 0 ? (p50 / peakMw) * 100 : 0}%` }}
                          />
                        </span>
                        <span className={`w-12 text-right ${top ? "font-semibold text-status-ok" : "text-text-secondary"}`}>
                          {Math.round(p50)} <span className="text-caption font-normal text-muted-foreground">MW</span>
                        </span>
                      </li>
                    );
                  })}
                </ol>
              }
            />
          </MetricRow>

          <section className="mt-5 space-y-1">
            <SectionHeader
              as="h3"
              title={
                <>
                  Park output, <Term id="forecast-band">P10–P90</Term>
                </>
              }
              aside={
              <ul className="flex items-center gap-3 text-caption text-muted-foreground">
                <li className="flex items-center gap-1.5">
                  <span className="h-0.5 w-3.5 rounded-full bg-sentinel" aria-hidden />
                  P50
                </li>
                <li className="flex items-center gap-1.5">
                  <span className="h-2.5 w-3.5 rounded-sm border border-brand-500/50 bg-brand-500/15" aria-hidden />
                  P10–P90
                </li>
              </ul>
              }
            />
            <LineChart data={fan} aspectRatio="5 / 2" margin={CHART_MARGIN}>
              <Grid numTicksRows={Y_TICKS} strokeDasharray="none" />
              <Band
                upperKey="p90"
                lowerKey="p10"
                label="P10–P90"
                fill={FAN_FILL}
                fillOpacity={0.16}
                strokeOpacity={0.5}
                curve={curveMonotoneX}
              />
              <Line dataKey="p50" stroke={FAN_LINE} curve={curveMonotoneX} />
              <FanYAxis numTicks={Y_TICKS} unit="MW" />
              <FanHourTicks everyHours={TICK_HOURS} />
              <FanNowMarker date={nowDate} label="now" />
              <ChartTooltip />
            </LineChart>
          </section>

          {windCounties.length > 0 ? (
            <section className="mt-5">
              <SectionHeader as="h3" lead="Wind by county," title="next 24 h" />
              <ol className="mt-1 divide-y divide-border">
                {windCounties.slice(0, TOP_COUNTIES).map(({ county, windMwh }) => (
                  <li key={county} className="flex items-center gap-3 py-2 text-dense">
                    <span className="w-24 truncate text-foreground">{county}</span>
                    <span className="h-1.5 flex-1 overflow-hidden rounded-full bg-muted">
                      <span
                        className="block h-full rounded-full bg-brand-500/60"
                        style={{ width: `${topCounty > 0 ? (windMwh / topCounty) * 100 : 0}%` }}
                      />
                    </span>
                    <span className="w-16 text-right font-medium text-foreground tabular-nums">
                      {Math.round(windMwh)} <span className="text-caption font-normal text-muted-foreground">MWh</span>
                    </span>
                  </li>
                ))}
              </ol>
            </section>
          ) : null}

          <div className="mt-3 border-t border-border">
            <RowButton onClick={() => setDetailsOpen(true)} aside="sold vs made, battery, wind field">
              <span className="font-medium">Details</span>
            </RowButton>
          </div>

          <DetailsSheet
            open={detailsOpen}
            onOpenChange={setDetailsOpen}
            title={park ? `${park.id} · details` : "Production details"}
            description="What WattSentry sold and what the park made, the park battery, and the wind field."
            wide
          >
            <WindActions parkId={park?.id} clock={now} wallClock={wallClock} />
            <SheetSection title="Wind field and counties">
              <dl className="divide-y divide-border text-dense">
                <div className="flex items-center justify-between gap-3 py-2.5">
                  <dt className="text-text-secondary">Wind at 100 m</dt>
                  <dd className="font-semibold text-foreground tabular-nums">{hourTs ? hm(hourTs) : "—"}</dd>
                </div>
                {windCounties.map(({ county, windMwh }) => (
                  <div key={county} className="flex items-baseline justify-between gap-3 py-2.5">
                    <dt className="text-text-secondary">{county} wind · next 24 h</dt>
                    <dd className="font-semibold text-foreground tabular-nums">
                      {Math.round(windMwh)} <span className="text-caption font-normal text-muted-foreground">MWh</span>
                    </dd>
                  </div>
                ))}
              </dl>
            </SheetSection>
          </DetailsSheet>
        </>
      )}
    </aside>
  );
}
