"use client";

import { curveMonotoneX } from "d3-shape";
import { useMemo } from "react";

import { Band } from "@/components/charts/band";
import { Grid } from "@/components/charts/grid";
import { Line, LineChart } from "@/components/charts/line-chart";
import { ChartTooltip } from "@/components/charts/tooltip";
import { XAxis } from "@/components/charts/x-axis";

import { hhmm, wallClock } from "@/panels/site/time";

import { byStress, countyOf, formatPower, type Zone } from "./data";
import { useDemandData, useSelectedZone } from "./use-demand-data";

const FAN_COLOR = "#ef4444";
const CHART_MARGIN = { left: 8, right: 8, top: 8, bottom: 24 };
const TOP_FEEDERS = 5;

function peakLabel(zone: Zone) {
  return `${hhmm(wallClock(zone.peakTs))}, ${formatPower(zone.peakMw)}`;
}

function ZoneRow({ zone, onSelect }: { zone: Zone; onSelect: (id: string) => void }) {
  return (
    <li>
      <button
        type="button"
        onClick={() => onSelect(zone.id)}
        className="flex w-full items-baseline justify-between gap-3 rounded px-1 py-0.5 text-left text-sm hover:bg-muted"
      >
        <span>{zone.name}</span>
        <span className="text-xs text-muted-foreground whitespace-nowrap">
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
    <aside className="absolute top-3 right-3 bottom-3 z-10 flex w-[440px] max-w-[calc(100%-1.5rem)] flex-col gap-3 overflow-y-auto rounded-md border bg-background/95 p-3 backdrop-blur">
      <header>
        <nav className="flex flex-wrap items-center gap-1 text-xs text-muted-foreground" aria-label="Zone">
          <button type="button" className="hover:text-foreground" onClick={() => setSelected(null)}>
            Romania
          </button>
          {trail.map((zone) => (
            <span key={zone.id} className="flex items-center gap-1">
              <span aria-hidden>›</span>
              <button type="button" className="hover:text-foreground" onClick={() => setSelected(zone.id)}>
                {zone.name}
              </button>
            </span>
          ))}
        </nav>
        <h2 className="mt-1 font-semibold">{shown ? shown.name : "Demand"}</h2>
        <p className="text-xs text-muted-foreground">
          {shown && `${selected ? shown.level : "most stressed county"} · `}
          <span className="font-medium text-amber-400">
            {shown?.label ?? (isLoading ? "loading" : "no run loaded")}
          </span>
        </p>
      </header>

      {!shown ? (
        !isLoading && (
          <p className="rounded border p-3 text-sm text-muted-foreground">
            No zone demand rows for this day in the loaded run.
          </p>
        )
      ) : (
        <>
          <section className="grid grid-cols-3 gap-2 rounded border p-2">
            <div>
              <div className="text-[11px] text-muted-foreground">Stress now</div>
              <div className="text-base font-semibold">{stressNow?.toFixed(2) ?? "—"}</div>
            </div>
            <div>
              <div className="text-[11px] text-muted-foreground">Demand now</div>
              <div className="text-base font-semibold">{now ? formatPower(now.q50Mw) : "—"}</div>
            </div>
            <div>
              <div className="text-[11px] text-muted-foreground">Peak today</div>
              <div className="text-base font-semibold whitespace-nowrap">
                {hhmm(wallClock(shown.peakTs))}
              </div>
              <div className="text-[11px] text-muted-foreground">{formatPower(shown.peakMw)}</div>
            </div>
          </section>

          <section className="space-y-1">
            <h3 className="text-sm font-medium">Demand, P50 with P10–P90 ({unit})</h3>
            <LineChart data={fan} aspectRatio="5 / 2" margin={CHART_MARGIN} animationDuration={0}>
              <Grid />
              <Band upperKey="p90" lowerKey="p10" label="P10–P90" fill={FAN_COLOR} curve={curveMonotoneX} />
              <Line dataKey="p50" stroke={FAN_COLOR} curve={curveMonotoneX} fadeEdges={false} />
              <XAxis numTicks={5} labels={labels} />
              <ChartTooltip />
            </LineChart>
          </section>

          {children.length > 0 && (
            <section className="space-y-1">
              <h3 className="text-sm font-medium">{selected ? "Inside this zone" : "Counties by peak stress"}</h3>
              <ul>
                {children.map((zone) => (
                  <ZoneRow key={zone.id} zone={zone} onSelect={setSelected} />
                ))}
              </ul>
            </section>
          )}

          {feeders.length > 0 && (
            <section className="space-y-1">
              <h3 className="text-sm font-medium">Top stressed feeders</h3>
              <ul>
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
