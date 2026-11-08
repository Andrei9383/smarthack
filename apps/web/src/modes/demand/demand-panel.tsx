"use client";

import { curveMonotoneX } from "d3-shape";
import { MapPinned } from "lucide-react";
import { useMemo, useState } from "react";

import { Band } from "@/components/charts/band";
import { Grid } from "@/components/charts/grid";
import { Line, LineChart } from "@/components/charts/line-chart";
import { ChartTooltip } from "@/components/charts/tooltip";
import { DetailsSheet, LABEL, Metric, MetricRow, SectionHeader, SheetSection, StatusPill } from "@/panels/disclosure";

import { hhmm, wallClock } from "@/panels/site/time";

import { AXIS_MARGIN, ChartAxes, ChartLegend, EmptyNote, yAxisFor } from "../feeders/ui";
import { DEMAND_SERIES, rampColor } from "./colors";
import { byStress, formatPower, type Zone } from "./data";
import { useDemandData, useSelectedZone } from "./use-demand-data";

const TOP_FEEDERS = 5;
/** Zones on the glance; the full ranking is behind "See all". */
const TOP_ZONES = 5;

/** Stress in words: P90 demand over what the zone's cables can carry. */
function stressWords(stress: number) {
  if (stress >= 1) return "over what its cables can carry";
  if (stress >= 0.8) return "close to what its cables can carry";
  if (stress >= 0.5) return "about half of what its cables can carry";
  return "well within what its cables can carry";
}

function peakLabel(zone: Zone) {
  return `${hhmm(wallClock(zone.peakTs))}, ${formatPower(zone.peakMw)}`;
}

/** 0–1 stress as a short bar coloured along the stress ramp; values above 1 fill it red. */
function StressMeter({ stress, className }: { stress: number; className?: string }) {
  return (
    <span
      aria-hidden
      className={`block h-1.5 overflow-hidden rounded-full bg-muted ${className ?? "w-full"}`}
    >
      <span
        className="block h-full rounded-full transition-[width] duration-150"
        style={{ width: `${Math.min(1, Math.max(0, stress)) * 100}%`, backgroundColor: rampColor(stress) }}
      />
    </span>
  );
}

function ZoneList({
  zones,
  selectedId,
  onSelect,
  compact = false,
}: {
  zones: Zone[];
  selectedId: string | null;
  onSelect: (id: string) => void;
  /** Glance version: name and stress only, no peak column. */
  compact?: boolean;
}) {
  const cols = compact ? "grid-cols-[1.25rem_minmax(0,1fr)_5.5rem]" : "grid-cols-[1.25rem_minmax(0,1fr)_6rem_4.5rem]";
  return (
    <ol className="-mx-2">
      {zones.map((zone, index) => {
        const isSelected = zone.id === selectedId;
        return (
          <li key={zone.id}>
            <button
              type="button"
              onClick={() => onSelect(zone.id)}
              aria-current={isSelected || undefined}
              className={`grid w-full ${cols} items-center gap-x-3 rounded-lg px-2 py-2 text-left text-dense transition-colors duration-150 focus-visible:outline-none ${
                isSelected ? "bg-surface-raised font-medium text-foreground" : "hover:bg-surface-raised focus-visible:bg-surface-raised"
              }`}
            >
              <span className="text-xs text-muted-foreground tabular-nums">{index + 1}</span>
              <span className="truncate font-medium">{zone.name}</span>
              {compact ? null : (
                <span className="text-right text-xs whitespace-nowrap text-text-secondary tabular-nums">{peakLabel(zone)}</span>
              )}
              <span className="flex items-center gap-1.5">
                <StressMeter stress={zone.peakStress} className={compact ? "w-10" : "w-8"} />
                <span className="text-xs font-medium text-foreground tabular-nums">
                  {zone.peakStress.toFixed(2)}
                </span>
              </span>
            </button>
          </li>
        );
      })}
    </ol>
  );
}

function ListHeader({ title }: { title: string }) {
  return (
    <div className="grid grid-cols-[1.25rem_minmax(0,1fr)_6rem_4.5rem] items-end gap-x-3">
      <h3 className={`col-span-2 ${LABEL}`}>{title}</h3>
      <span className="text-right text-caption text-muted-foreground">Peak</span>
      <span className="text-caption text-muted-foreground">Stress</span>
    </div>
  );
}

export function DemandPanel() {
  const [selectedId, setSelected] = useSelectedZone();
  const [listOpen, setListOpen] = useState(false);
  const { zones, counties, feeders, all, ts, isLoading } = useDemandData();
  const selected = selectedId ? zones.get(selectedId) : undefined;
  // Nothing picked: chart the most stressed county, without zooming to it.
  const shown = selected ?? counties[0];

  const trail = useMemo(() => {
    const path: Zone[] = [];
    let current = selected;
    while (current) {
      path.unshift(current);
      current = current.parentId ? zones.get(current.parentId) : undefined;
    }
    return path;
  }, [selected, zones]);

  const children = useMemo(
    () => (selected ? byStress(all.filter((zone) => zone.parentId === selected.id)) : counties),
    [all, counties, selected],
  );

  const { fan, dates } = useMemo(() => {
    const rows = shown?.rows ?? [];
    const dates = rows.map((row) => wallClock(row.ts));
    // Fixture zones are small: chart kW below 1 MW so the axis and tooltip stay readable.
    const scale = (shown?.peakMw ?? 0) < 1 ? 1000 : 1;
    return {
      fan: rows.map((row, index) => ({
        date: dates[index] as Date,
        p10: (row.q10Mw ?? row.q50Mw) * scale,
        p50: row.q50Mw * scale,
        p90: (row.q90Mw ?? row.q50Mw) * scale,
      })),
      dates,
    };
  }, [shown]);
  const axis = useMemo(() => yAxisFor(fan.map((row) => row.p90)), [fan]);

  const now = shown?.rows.find((row) => row.ts === ts);
  const stressNow = shown?.stressByTs.get(ts);
  const unit = (shown?.peakMw ?? 0) < 1 ? "kW" : "MW";
  const [nowValue, nowUnit] = now ? formatPower(now.q50Mw).split(" ") : ["—", undefined];
  const [peakValue] = shown ? [hhmm(wallClock(shown.peakTs))] : ["—"];
  const listTitle = selected ? `Inside ${selected.name}` : "Counties by peak stress";
  const select = (id: string) => {
    setListOpen(false);
    void setSelected(id);
  };

  return (
    <aside className="absolute top-3 right-3 bottom-3 z-10 flex w-[440px] max-w-[calc(100%-1.5rem)] flex-col overflow-y-auto overscroll-contain rounded-xl bg-card text-card-foreground shadow-floating">
      <header className="px-5 pt-4">
        <nav className="flex flex-wrap items-center gap-1 text-caption text-muted-foreground" aria-label="Zone">
          <button type="button" className="transition-colors duration-150 hover:text-foreground" onClick={() => setSelected(null)}>
            Romania
          </button>
          {trail.map((zone) => (
            <span key={zone.id} className="flex items-center gap-1">
              <span aria-hidden>›</span>
              <button type="button" className="transition-colors duration-150 hover:text-foreground" onClick={() => setSelected(zone.id)}>
                {zone.name}
              </button>
            </span>
          ))}
          {!selected && shown && (
            <span className="flex items-center gap-1">
              <span aria-hidden>›</span>
              <span>{shown.name}</span>
            </span>
          )}
        </nav>
        <div className="mt-1 flex flex-wrap items-center gap-x-2.5 gap-y-1">
          <h2 className="text-title leading-tight tracking-tight">{shown ? shown.name : "Demand"}</h2>
          {shown && (
            <StatusPill status={selected ? "unknown" : "warn"}>{selected ? shown.level : "Most stressed county"}</StatusPill>
          )}
        </div>
      </header>

      {!shown ? (
        !isLoading && (
          <div className="p-5">
            <EmptyNote icon={MapPinned}>No zone demand rows for this day in the loaded run.</EmptyNote>
          </div>
        )
      ) : (
        <div className="flex flex-col px-5 pb-2">
          {/* Words only: the numbers sit in the metrics below. */}
          <p className="pt-3 text-body leading-snug text-text-secondary">
            Today&apos;s peak is {stressWords(shown.peakStress)}.
          </p>

          <MetricRow className="mt-4 py-1">
            <Metric
              label="Stress now"
              value={stressNow?.toFixed(2) ?? "—"}
              sub="of 1.00"
              popTitle="What stress means"
              pop={
                <div className="space-y-2.5">
                  <p>
                    P90 demand divided by what the zone&apos;s cables can carry. At 1.00 a high-case evening fills them.
                  </p>
                  {stressNow !== undefined ? (
                    <div className="space-y-1">
                      <StressMeter stress={stressNow} />
                      <div className="flex justify-between text-caption text-muted-foreground tabular-nums">
                        <span>now {stressNow.toFixed(2)}</span>
                        <span>peak {shown.peakStress.toFixed(2)}</span>
                      </div>
                    </div>
                  ) : null}
                </div>
              }
            />
            <Metric label="Demand now" value={nowValue} unit={nowUnit} sub="P50 forecast" />
            <Metric label="Peak today" value={peakValue} sub={formatPower(shown.peakMw)} />
          </MetricRow>

          <section className="mt-5 space-y-1">
            <SectionHeader
              as="h3"
              lead="Demand today,"
              title={unit}
              aside={
                <ChartLegend
                  items={[
                    { label: "P50", color: DEMAND_SERIES },
                    { label: "P10–P90", color: DEMAND_SERIES, dashed: true },
                  ]}
                />
              }
            />
            <div className="relative">
              <LineChart
                data={fan}
                aspectRatio="5 / 2"
                margin={AXIS_MARGIN}
                animationDuration={0}
                yScaleDomainMax={axis.domainMax}
              >
                <Grid rowTickValues={axis.ticks} fadeHorizontal={false} />
                <Band upperKey="p90" lowerKey="p10" label="P10–P90" fill={DEMAND_SERIES} fillOpacity={0.14} strokeOpacity={0.35} curve={curveMonotoneX} />
                <Line dataKey="p50" stroke={DEMAND_SERIES} curve={curveMonotoneX} fadeEdges={false} />
                <ChartTooltip />
              </LineChart>
              <ChartAxes dates={dates} top={axis.top} ticks={axis.ticks} unit={unit} now={wallClock(ts)} />
            </div>
          </section>

          {children.length > 0 ? (
            <section className="mt-5">
              <SectionHeader
                as="h3"
                title={listTitle}
                aside={
                  children.length > TOP_ZONES || feeders.length > 0 ? (
                    <button
                      type="button"
                      onClick={() => setListOpen(true)}
                      className="rounded-md px-1.5 py-0.5 text-caption font-medium text-foreground underline-offset-2 transition-colors duration-150 hover:bg-surface-raised hover:underline focus-visible:outline-2 focus-visible:outline-ring"
                    >
                      See all
                    </button>
                  ) : null
                }
              />
              <ZoneList zones={children.slice(0, TOP_ZONES)} selectedId={selectedId} onSelect={select} compact />
            </section>
          ) : feeders.length > 0 ? (
            <button
              type="button"
              onClick={() => setListOpen(true)}
              className="mt-4 self-start rounded-md px-1.5 py-0.5 text-caption font-medium text-foreground underline-offset-2 hover:underline"
            >
              Top stressed feeders
            </button>
          ) : null}

          <DetailsSheet
            open={listOpen}
            onOpenChange={setListOpen}
            title={selected ? `Inside ${selected.name}` : "Where demand peaks"}
            description="Ranked by peak stress today. Pick a row to chart it and zoom the map."
          >
            {children.length > 0 && (
              <SheetSection title={listTitle}>
                <ListHeader title="" />
                <ZoneList zones={children} selectedId={selectedId} onSelect={select} />
              </SheetSection>
            )}
            {feeders.length > 0 && (
              <SheetSection title="Top stressed feeders">
                <ListHeader title="" />
                <ZoneList zones={feeders.slice(0, TOP_FEEDERS)} selectedId={selectedId} onSelect={select} />
              </SheetSection>
            )}
          </DetailsSheet>
        </div>
      )}
    </aside>
  );
}
