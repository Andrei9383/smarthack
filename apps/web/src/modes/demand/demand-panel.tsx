"use client";

import { curveMonotoneX } from "d3-shape";
import { useMemo } from "react";

import { Band } from "@/components/charts/band";
import { Grid } from "@/components/charts/grid";
import { Line, LineChart } from "@/components/charts/line-chart";
import { ChartTooltip } from "@/components/charts/tooltip";
import { XAxis } from "@/components/charts/x-axis";

import { hhmm, wallClock } from "@/panels/site/time";

import { DEMAND_SERIES } from "./colors";
import { byStress, countyOf, formatPower, type Zone } from "./data";
import { useDemandData, useSelectedZone } from "./use-demand-data";

const CHART_MARGIN = { left: 8, right: 8, top: 8, bottom: 24 };
const TOP_FEEDERS = 5;
const SECTION_LABEL = "text-[11px] font-medium tracking-[0.04em] text-muted-foreground uppercase";
const KPI_LABEL = "text-[11px] leading-snug text-muted-foreground";
const KPI_VALUE = "text-[22px] leading-none font-semibold tracking-tight tabular-nums";

function peakLabel(zone: Zone) {
  return `${hhmm(wallClock(zone.peakTs))}, ${formatPower(zone.peakMw)}`;
}

function ZoneRow({ zone, onSelect }: { zone: Zone; onSelect: (id: string) => void }) {
  return (
    <li>
      <button
        type="button"
        onClick={() => onSelect(zone.id)}
        className="flex w-full items-baseline justify-between gap-3 rounded-sm px-2 py-1.5 text-left text-[13px] font-medium text-foreground transition-colors hover:bg-brand-50 focus-visible:bg-brand-50 focus-visible:outline-none"
      >
        <span>{zone.name}</span>
        <span className="text-xs font-normal whitespace-nowrap text-text-secondary tabular-nums">
          {peakLabel(zone)} · stress {zone.peakStress.toFixed(2)}
        </span>
      </button>
    </li>
  );
}

export function DemandPanel() {
  const [selectedId, setSelected] = useSelectedZone();
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

  const { fan, labels } = useMemo(() => {
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
      labels: dates.map(hhmm),
    };
  }, [shown]);

  const now = shown?.rows.find((row) => row.ts === ts);
  const stressNow = shown?.stressByTs.get(ts);
  const unit = (shown?.peakMw ?? 0) < 1 ? "kW" : "MW";

  return (
    <aside className="absolute top-3 right-3 bottom-3 z-10 flex w-[440px] max-w-[calc(100%-1.5rem)] flex-col gap-5 overflow-y-auto rounded-lg border border-border bg-card p-5 text-card-foreground shadow-card">
      <header>
        <nav className="flex flex-wrap items-center gap-1 text-xs text-muted-foreground" aria-label="Zone">
          <button type="button" className="transition-colors hover:text-brand-700" onClick={() => setSelected(null)}>
            Romania
          </button>
          {trail.map((zone) => (
            <span key={zone.id} className="flex items-center gap-1">
              <span aria-hidden>›</span>
              <button type="button" className="transition-colors hover:text-brand-700" onClick={() => setSelected(zone.id)}>
                {zone.name}
              </button>
            </span>
          ))}
        </nav>
        <h2 className="mt-1.5 text-base font-semibold">{shown ? shown.name : "Demand"}</h2>
        <p className="mt-1 text-xs leading-relaxed text-text-secondary">
          {shown && `${selected ? shown.level : "most stressed county"} · `}
          <span className="inline-flex rounded-full bg-status-warn/10 px-2 py-0.5 text-[11px] font-medium text-status-warn">
            {shown?.label ?? (isLoading ? "loading" : "no run loaded")}
          </span>
        </p>
      </header>

      {!shown ? (
        !isLoading && (
          <p className="rounded-lg bg-muted p-4 text-sm text-text-secondary">
            No zone demand rows for this day in the loaded run.
          </p>
        )
      ) : (
        <>
          <section className="grid grid-cols-3 gap-x-6 rounded-lg bg-sidebar p-4">
            <div className="flex flex-col gap-1">
              <div className={KPI_LABEL}>Stress now</div>
              <div className={KPI_VALUE}>{stressNow?.toFixed(2) ?? "—"}</div>
            </div>
            <div className="flex flex-col gap-1">
              <div className={KPI_LABEL}>Demand now</div>
              <div className={`${KPI_VALUE} whitespace-nowrap`}>{now ? formatPower(now.q50Mw) : "—"}</div>
            </div>
            <div className="flex flex-col gap-1">
              <div className={KPI_LABEL}>Peak today</div>
              <div className={`${KPI_VALUE} whitespace-nowrap`}>
                {hhmm(wallClock(shown.peakTs))}
              </div>
              <div className="text-[13px] text-muted-foreground tabular-nums">{formatPower(shown.peakMw)}</div>
            </div>
          </section>

          <section className="space-y-2">
            <h3 className={SECTION_LABEL}>Demand, P50 with P10–P90 ({unit})</h3>
            <LineChart data={fan} aspectRatio="5 / 2" margin={CHART_MARGIN} animationDuration={0}>
              <Grid />
              <Band upperKey="p90" lowerKey="p10" label="P10–P90" fill={DEMAND_SERIES} fillOpacity={0.14} strokeOpacity={0.35} curve={curveMonotoneX} />
              <Line dataKey="p50" stroke={DEMAND_SERIES} curve={curveMonotoneX} fadeEdges={false} />
              <XAxis numTicks={5} labels={labels} />
              <ChartTooltip />
            </LineChart>
          </section>

          {children.length > 0 && (
            <section className="space-y-2">
              <h3 className={SECTION_LABEL}>{selected ? "Inside this zone" : "Counties by peak stress"}</h3>
              <ul className="-mx-2">
                {children.map((zone) => (
                  <ZoneRow key={zone.id} zone={zone} onSelect={setSelected} />
                ))}
              </ul>
            </section>
          )}

          {feeders.length > 0 && (
            <section className="space-y-2">
              <h3 className={SECTION_LABEL}>Top stressed feeders</h3>
              <ul className="-mx-2">
                {feeders.slice(0, TOP_FEEDERS).map((zone) => (
                  <ZoneRow key={zone.id} zone={zone} onSelect={setSelected} />
                ))}
              </ul>
            </section>
          )}
        </>
      )}
    </aside>
  );
}
