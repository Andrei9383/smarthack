"use client";

import { useQuery } from "@tanstack/react-query";
import { curveMonotoneX } from "d3-shape";
import { useMemo } from "react";

import { Band } from "@/components/charts/band";
import { Grid } from "@/components/charts/grid";
import { Line, LineChart } from "@/components/charts/line-chart";
import { ChartTooltip } from "@/components/charts/tooltip";
import { stepToDate } from "@/demo/replay";
import { useReplayStep } from "@/demo/use-replay";
import { useShowOwnPill } from "@/components/shell/inspector-embedded";
import { Term } from "@/components/help";
import { Metric, MetricRow, PopList } from "@/panels/disclosure";

import { expectedMwh, hourly } from "./data";
import { FanHourTicks, FanNowMarker, FanYAxis } from "./fan-chart-overlays";
import { useProductionData } from "./use-production-data";
import { useHour, useSelectedPark } from "./use-production-state";
import { fetchWindMeta } from "./wind-field";

// Sentinel's own forecast: primary green line over a brand-500 P10–P90 band (ui-style §3).
const FAN_FILL = "var(--brand-500)";
const FAN_LINE = "var(--primary)";
const SECTION_LABEL = "text-[11px] font-medium tracking-[0.04em] text-text-secondary uppercase";
const PILL = "inline-flex items-center gap-1.5 rounded-full bg-muted px-2 py-0.5 text-[11px] font-medium whitespace-nowrap text-text-secondary";
const Y_TICKS = 4;
const TICK_HOURS = 6;
const CHART_MARGIN = { left: 32, right: 8, top: 26, bottom: 24 };
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

// Same dot colours as the shared provenance pill (ui-pass2): synthetic grey, real RO green, non-RO blue.
function provenanceDot(text: string | undefined) {
  if (!text) return "bg-status-unknown";
  if (/non-RO/i.test(text)) return "bg-status-forecast";
  if (/synthetic|mock/i.test(text)) return "bg-status-unknown";
  if (/real RO/i.test(text)) return "bg-primary";
  return "bg-status-unknown";
}

function Pill({ text }: { text: string }) {
  return (
    <span className={PILL}>
      <span className={`size-1.5 shrink-0 rounded-full ${provenanceDot(text)}`} aria-hidden />
      {text}
    </span>
  );
}

// Park names carry a trailing "(synthetic RO)"-style tag that the pill already shows; mute it.
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
  const { label, parks, rows, counties, isLoading } = useProductionData();
  const showPill = useShowOwnPill(label);
  const park = parks.find((p) => p.id === selected) ?? parks[0];

  const { fan, bestHours, parkMwh, parkPeak } = useMemo(() => {
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
      parkPeak: [...parkHours].sort((a, b) => b.p50 - a.p50)[0],
    };
  }, [park?.id, rows]);
  const windCounties = counties.filter((county) => county.windMwh > 0);

  const peakMw = Math.max(...bestHours.map((row) => row.p50), 0);
  const best = bestHours[0];
  const nowRow = fan[0];
  const now = stepToDate(step);
  const nowDate = new Date(wallClock(now.toISOString()).getTime() + now.getUTCMinutes() * 60_000);
  const [parkName, parkTag] = park ? splitTag(park.name) : ["", null];

  const hourTs = meta?.hours[hour]?.ts;
  const windLabel = meta?.source === "mock" ? "mock wind field" : "real RO, ERA5 100 m wind";
  const mwFmt = (v: number) => (v >= 10 ? Math.round(v).toString() : v.toFixed(1));

  return (
    <aside className="absolute top-3 right-3 bottom-3 z-10 flex w-[440px] max-w-[calc(100%-1.5rem)] flex-col overflow-y-auto rounded-xl bg-card px-5 pt-4 pb-3 text-card-foreground shadow-floating">
      <header className="space-y-1">
        <div className="flex items-start justify-between gap-3">
          <h2 className="text-base leading-snug font-semibold text-foreground">
            {park ? (
              <>
                {`${park.id} · ${parkName}`}
                {/* The tag repeats the provenance pill; hide it when the shell pill already says it. */}
                {parkTag && showPill && <span className="text-sm font-normal text-muted-foreground"> {parkTag}</span>}
              </>
            ) : (
              "Production"
            )}
          </h2>
          {showPill ? (
            <span className="shrink-0 pt-0.5">
              <Pill text={label ?? (isLoading ? "loading" : "no run loaded")} />
            </span>
          ) : null}
        </div>
        {park && (
          <p className="text-xs text-text-secondary tabular-nums">
            {park.county} · {park.ratedMw} MW rated
          </p>
        )}
      </header>

      {/* Neighbouring parks stack at country zoom, so their markers alone can't select them all. */}
      {parks.length > 1 && (
        <nav className="mt-2.5 flex flex-wrap gap-1.5" aria-label="Parks">
          {parks.map((p) => (
            <button
              key={p.id}
              type="button"
              aria-pressed={p.id === park?.id}
              onClick={() => setSelected(p.id)}
              className={`rounded-full px-2.5 py-1 text-xs font-medium tabular-nums transition-[background-color,box-shadow,color] duration-150 ${
                p.id === park?.id
                  ? "bg-brand-50 text-brand-900 shadow-raised"
                  : "bg-card text-text-secondary shadow-raised hover:bg-sidebar hover:text-foreground"
              }`}
            >
              {p.id}
            </button>
          ))}
        </nav>
      )}

      {fan.length === 0 ? (
        !isLoading && (
          <p className="mt-3 text-[13px] text-text-secondary">No park forecast rows for this time in the loaded run.</p>
        )
      ) : (
        <>
          <p className="pt-3 pb-3 text-[14px] leading-snug text-foreground">
            {park?.id} makes about {mwFmt(nowRow?.p50 ?? 0)} MW right now. Expect about {Math.round(parkMwh)} MWh over
            the next 24 hours, with the strongest wind around {parkPeak ? localLabel(parkPeak.ts) : "—"}.
          </p>

          <MetricRow className="border-y border-border py-2">
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
                  <p className="text-[11px] text-muted-foreground">
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
                  <p className="text-[11px] text-muted-foreground">
                    Eight times in ten the park lands inside the shaded band on the chart.
                  </p>
                </div>
              }
            />
            <Metric
              label="Best hour"
              value={best ? localLabel(best.ts) : "—"}
              sub={best ? `${Math.round(best.p50)} MW, all parks` : undefined}
              popTitle="Best production hours, all parks"
              pop={
                <ol className="space-y-1 text-[13px] tabular-nums">
                  {bestHours.map(({ ts, p50 }, index) => {
                    const top = index === 0;
                    return (
                      <li key={ts} className="flex items-center gap-2.5">
                        <span className={`w-3 font-mono text-[11px] ${top ? "font-semibold text-brand-700" : "text-muted-foreground"}`}>
                          {index + 1}
                        </span>
                        <span className={`w-11 font-medium ${top ? "text-brand-900" : "text-foreground"}`}>{localLabel(ts)}</span>
                        <span className="h-1.5 flex-1 overflow-hidden rounded-full bg-muted">
                          <span
                            className={`block h-full rounded-full ${top ? "bg-primary" : "bg-brand-500/50"}`}
                            style={{ width: `${peakMw > 0 ? (p50 / peakMw) * 100 : 0}%` }}
                          />
                        </span>
                        <span className={`w-12 text-right ${top ? "font-semibold text-brand-900" : "text-text-secondary"}`}>
                          {Math.round(p50)} <span className="text-[11px] font-normal text-muted-foreground">MW</span>
                        </span>
                      </li>
                    );
                  })}
                </ol>
              }
            />
          </MetricRow>

          <section className="space-y-1.5 pt-3.5">
            <div className="flex items-center justify-between gap-3">
              <h3 className={SECTION_LABEL}>
                Park output, <Term id="forecast-band">P10–P90</Term> (MW)
              </h3>
              <ul className="flex items-center gap-3 text-[11px] text-text-secondary">
                <li className="flex items-center gap-1.5">
                  <span className="h-0.5 w-3.5 rounded-full bg-primary" aria-hidden />
                  P50
                </li>
                <li className="flex items-center gap-1.5">
                  <span className="h-2.5 w-3.5 rounded-sm border border-brand-500/50 bg-brand-500/15" aria-hidden />
                  P10–P90
                </li>
              </ul>
            </div>
            <LineChart data={fan} aspectRatio="5 / 2" margin={CHART_MARGIN} animationDuration={0}>
              <Grid numTicksRows={Y_TICKS} fadeHorizontal={false} strokeDasharray="none" />
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
              <FanYAxis numTicks={Y_TICKS} unit="MW" />
              <FanHourTicks everyHours={TICK_HOURS} />
              <FanNowMarker date={nowDate} label="now" />
              <ChartTooltip />
            </LineChart>
          </section>

          {/* Facts as hairline rows, not boxes inside the inspector card. */}
          <dl className="mt-2 divide-y divide-border border-t border-border text-[13px]">
            <div className="flex items-center justify-between gap-3 py-2.5">
              <dt className="text-text-secondary">Wind at 100 m</dt>
              <dd className="flex items-center gap-2">
                <span className="font-semibold text-foreground tabular-nums">{hourTs ? localLabel(hourTs) : "—"}</span>
                <Pill text={meta ? windLabel : "loading"} />
              </dd>
            </div>
            {windCounties.map(({ county, windMwh }) => (
              <div key={county} className="flex items-baseline justify-between gap-3 py-2.5">
                <dt className="text-text-secondary">{county} wind · next 24 h</dt>
                <dd className="font-semibold text-foreground tabular-nums">
                  {Math.round(windMwh)} <span className="text-[11px] font-normal text-muted-foreground">MWh</span>
                </dd>
              </div>
            ))}
          </dl>
        </>
      )}
    </aside>
  );
}
