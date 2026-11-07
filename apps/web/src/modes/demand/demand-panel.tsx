"use client";

import { curveMonotoneX } from "d3-shape";
import { MapPinned } from "lucide-react";
import { useMemo } from "react";

import { Band } from "@/components/charts/band";
import { Grid } from "@/components/charts/grid";
import { Line, LineChart } from "@/components/charts/line-chart";
import { ChartTooltip } from "@/components/charts/tooltip";
import { useShowOwnPill } from "@/components/shell/inspector-embedded";

import { hhmm, wallClock } from "@/panels/site/time";

import {
  AXIS_MARGIN,
  ChartAxes,
  ChartLegend,
  EmptyNote,
  ProvenancePill,
  SECTION_LABEL,
  SectionHeader,
  yAxisFor,
} from "../feeders/ui";
import { DEMAND_SERIES, rampColor } from "./colors";
import { byStress, countyOf, formatPower, type Zone } from "./data";
import { useDemandData, useSelectedZone } from "./use-demand-data";

const TOP_FEEDERS = 5;
const KPI_LABEL = "text-[11px] leading-none whitespace-nowrap text-text-secondary";
const KPI_VALUE = "text-[22px] leading-none font-semibold tracking-tight tabular-nums";

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
}: {
  zones: Zone[];
  selectedId: string | null;
  onSelect: (id: string) => void;
}) {
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
              className={`grid w-full grid-cols-[1.25rem_minmax(0,1fr)_6rem_4.5rem] items-center gap-x-3 rounded-sm px-2 py-2 text-left text-[13px] transition-colors duration-150 focus-visible:outline-none ${
                isSelected ? "bg-brand-50 text-brand-900" : "hover:bg-muted focus-visible:bg-muted"
              }`}
            >
              <span className="text-xs text-muted-foreground tabular-nums">{index + 1}</span>
              <span className="truncate font-medium">{zone.name}</span>
              <span className="text-right text-xs whitespace-nowrap text-text-secondary tabular-nums">{peakLabel(zone)}</span>
              <span className="flex items-center gap-1.5">
                <StressMeter stress={zone.peakStress} className="w-8" />
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
      <h3 className={`col-span-2 ${SECTION_LABEL}`}>{title}</h3>
      <span className="text-right text-[11px] text-muted-foreground">Peak</span>
      <span className="text-[11px] text-muted-foreground">Stress</span>
    </div>
  );
}

export function DemandPanel() {
  const [selectedId, setSelected] = useSelectedZone();
  const { zones, counties, feeders, all, ts, isLoading } = useDemandData();
  const selected = selectedId ? zones.get(selectedId) : undefined;
  // Nothing picked: chart the most stressed county, without zooming to it.
  const shown = selected ?? counties[0];
  const showPill = useShowOwnPill(shown?.label);

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

  return (
    <aside className="absolute top-3 right-3 bottom-3 z-10 flex w-[440px] max-w-[calc(100%-1.5rem)] flex-col overflow-y-auto overscroll-contain rounded-lg border border-border bg-card text-card-foreground shadow-card">
      <header className="sticky top-0 z-10 border-b border-border bg-card/95 px-5 pt-4 pb-3 backdrop-blur-sm">
        <nav className="flex flex-wrap items-center gap-1 text-xs text-muted-foreground" aria-label="Zone">
          <button type="button" className="transition-colors duration-150 hover:text-brand-700" onClick={() => setSelected(null)}>
            Romania
          </button>
          {trail.map((zone) => (
            <span key={zone.id} className="flex items-center gap-1">
              <span aria-hidden>›</span>
              <button type="button" className="transition-colors duration-150 hover:text-brand-700" onClick={() => setSelected(zone.id)}>
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
        <div className="mt-1.5 flex flex-wrap items-center gap-2">
          <h2 className="text-xl leading-tight font-semibold tracking-tight">{shown ? shown.name : "Demand"}</h2>
          {shown && (
            <span
              className={`inline-flex rounded-full px-2 py-0.5 text-[11px] font-medium ${
                selected ? "bg-muted text-text-secondary" : "bg-status-warn/10 text-status-warn"
              }`}
            >
              {selected ? shown.level : "most stressed county"}
            </span>
          )}
          {showPill ? (
            <span className="ml-auto">
              <ProvenancePill label={shown?.label ?? (isLoading ? "loading" : "no run loaded")} />
            </span>
          ) : null}
        </div>
      </header>

      {!shown ? (
        !isLoading && (
          <div className="p-5">
            <EmptyNote icon={MapPinned}>No zone demand rows for this day in the loaded run.</EmptyNote>
          </div>
        )
      ) : (
        <div className="flex flex-col divide-y divide-border px-5">
          <section className="grid grid-cols-3 gap-x-5 py-5">
            <div className="flex flex-col gap-2">
              <div className={KPI_LABEL}>Stress now</div>
              <div className="flex items-center gap-2">
                <span className={KPI_VALUE}>{stressNow?.toFixed(2) ?? "—"}</span>
                {stressNow !== undefined && <StressMeter stress={stressNow} className="w-full max-w-12" />}
              </div>
              <div className="text-[11px] leading-none text-muted-foreground tabular-nums">of 1.00</div>
            </div>
            <div className="flex flex-col gap-2">
              <div className={KPI_LABEL}>Demand now</div>
              <div className={`${KPI_VALUE} whitespace-nowrap`}>{now ? formatPower(now.q50Mw) : "—"}</div>
            </div>
            <div className="flex flex-col gap-2">
              <div className={KPI_LABEL}>Peak today</div>
              <div className={`${KPI_VALUE} whitespace-nowrap`}>{hhmm(wallClock(shown.peakTs))}</div>
              <div className="text-[11px] leading-none text-muted-foreground tabular-nums">{formatPower(shown.peakMw)}</div>
            </div>
          </section>

          <section className="space-y-3 py-6">
            <SectionHeader title={`Demand, P50 with P10–P90 (${unit})`}>
              <ChartLegend
                items={[
                  { label: "P50", color: DEMAND_SERIES },
                  { label: "P10–P90", color: DEMAND_SERIES, dashed: true },
                ]}
              />
            </SectionHeader>
            <div className="relative">
              <LineChart
                data={fan}
                aspectRatio="2 / 1"
                margin={AXIS_MARGIN}
                animationDuration={0}
                yScaleDomainMax={axis.domainMax}
              >
                <Grid rowTickValues={axis.ticks} fadeHorizontal={false} />
                <Band upperKey="p90" lowerKey="p10" label="P10–P90" fill={DEMAND_SERIES} fillOpacity={0.14} strokeOpacity={0.35} curve={curveMonotoneX} />
                <Line dataKey="p50" stroke={DEMAND_SERIES} curve={curveMonotoneX} fadeEdges={false} />
                <ChartTooltip />
              </LineChart>
              <ChartAxes dates={dates} top={axis.top} ticks={axis.ticks} unit={unit} />
            </div>
          </section>

          {children.length > 0 && (
            <section className="space-y-2 py-6">
              <ListHeader title={selected ? "Inside this zone" : "Counties by peak stress"} />
              <ZoneList zones={children} selectedId={selectedId} onSelect={setSelected} />
            </section>
          )}

          {feeders.length > 0 && (
            <section className="space-y-2 py-6">
              <ListHeader title="Top stressed feeders" />
              <ZoneList zones={feeders.slice(0, TOP_FEEDERS)} selectedId={selectedId} onSelect={setSelected} />
            </section>
          )}
        </div>
      )}
    </aside>
  );
}
