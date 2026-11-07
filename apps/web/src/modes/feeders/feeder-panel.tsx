"use client";

import { curveMonotoneX } from "d3-shape";
import { useMemo } from "react";

import { Band } from "@/components/charts/band";
import { Grid } from "@/components/charts/grid";
import { Line, LineChart } from "@/components/charts/line-chart";
import { ChartTooltip } from "@/components/charts/tooltip";
import { XAxis } from "@/components/charts/x-axis";
import { stepOfDay, timeLabel } from "@/demo/replay";
import { usePolicy, useReplayStep } from "@/demo/use-replay";

import {
  absorbingSites,
  DATA_LABEL,
  FEEDERS,
  feederCurves,
  feederSoc,
  minutesOverLimit,
  peakExport,
} from "./data";
import { useSelectedFeeder } from "./use-feeder-state";

const BUCHAREST = new Intl.DateTimeFormat("en-GB", {
  timeZone: "Europe/Bucharest",
  year: "numeric",
  month: "numeric",
  day: "numeric",
  hour: "numeric",
  minute: "numeric",
  hourCycle: "h23",
});

// Charts format dates in the browser's zone, so hand them Bucharest wall-clock time (plan R19).
function bucharestWallClock(ts: string) {
  const parts = Object.fromEntries(
    BUCHAREST.formatToParts(new Date(ts)).map((part) => [part.type, Number(part.value)]),
  );
  return new Date(
    parts.year as number,
    (parts.month as number) - 1,
    parts.day as number,
    parts.hour as number,
    parts.minute as number,
  );
}

const hhmm = (date: Date) =>
  `${String(date.getHours()).padStart(2, "0")}:${String(date.getMinutes()).padStart(2, "0")}`;

const BASELINE_COLOR = "#f59e0b";
const SENTINEL_COLOR = "#2dd4bf";
const LIMIT_COLOR = "#ef4444";
const SOC_COLOR = "#818cf8";
const CHART_MARGIN = { left: 8, right: 8, top: 8, bottom: 24 };

function Stat({ label, value, tone }: { label: string; value: string; tone?: string }) {
  return (
    <div>
      <div className="text-[11px] text-muted-foreground">{label}</div>
      <div className="text-base font-semibold" style={{ color: tone }}>
        {value}
      </div>
    </div>
  );
}

function ChartLegend({ items }: { items: { label: string; color: string; dashed?: boolean }[] }) {
  return (
    <div className="flex flex-wrap gap-3 text-[11px] text-muted-foreground">
      {items.map(({ label, color, dashed }) => (
        <span key={label} className="flex items-center gap-1">
          <span
            className="inline-block w-3 border-t-2"
            style={{ borderColor: color, borderStyle: dashed ? "dashed" : "solid" }}
          />
          {label}
        </span>
      ))}
    </div>
  );
}

export function FeederPanel() {
  const [policy] = usePolicy();
  const [step] = useReplayStep();
  const [selected] = useSelectedFeeder();
  const feeder = FEEDERS.find((f) => f.id === selected) ?? FEEDERS[0];
  const feederId = feeder?.id ?? "";

  const { exportData, socData, baselineOver, sentinelOver, baselinePeak, sentinelPeak, labels } = useMemo(() => {
    const rows = feederCurves(feederId);
    const sentinelByTs = new Map(
      rows.filter((r) => r.policy === "sentinel").map((r) => [r.ts, r]),
    );
    const exportRows = rows
      .filter((r) => r.policy === "baseline")
      .map((r) => {
        const sentinel = sentinelByTs.get(r.ts);
        return {
          date: bucharestWallClock(r.ts),
          baseline_export: r.export_kw,
          sentinel_export: sentinel?.export_kw ?? 0,
          limit: r.limit_kw,
          baseline_import: r.import_kw,
          sentinel_import: sentinel?.import_kw ?? 0,
        };
      });
    return {
      exportData: exportRows,
      socData: feederSoc(feederId).map(({ ts, ...soc }) => ({
        date: bucharestWallClock(ts),
        ...soc,
      })),
      baselineOver: minutesOverLimit(rows, "baseline"),
      sentinelOver: minutesOverLimit(rows, "sentinel"),
      baselinePeak: peakExport(rows, "baseline"),
      sentinelPeak: peakExport(rows, "sentinel"),
      labels: exportRows.map((row) => hhmm(row.date)),
    };
  }, [feederId]);

  const sites = absorbingSites(feederId);
  const now = exportData[stepOfDay(step)];
  const nowExport = policy === "baseline" ? now?.baseline_export : now?.sentinel_export;

  return (
    <aside className="absolute top-3 right-3 bottom-3 z-10 flex w-[440px] max-w-[calc(100%-1.5rem)] flex-col gap-3 overflow-y-auto rounded-md border bg-background/95 p-3 backdrop-blur">
      <header className="flex items-start justify-between gap-2">
        <div>
          <h2 className="font-semibold">
            {feeder?.id} · {feeder?.name}
          </h2>
          <p className="text-xs text-muted-foreground">
            {feeder?.county} · limit {feeder?.limit_kw} kW · GIS trust {feeder?.gis_trust} ·{" "}
            <span className="font-medium text-amber-400">{DATA_LABEL}</span>
          </p>
        </div>
      </header>

      <div className="grid grid-cols-2 gap-2 rounded border p-2">
        <Stat
          label="Baseline, minutes over limit"
          value={`${baselineOver} min`}
          tone={baselineOver > 0 ? LIMIT_COLOR : undefined}
        />
        <Stat
          label="Sentinel, minutes over limit"
          value={`${sentinelOver} min`}
          tone={sentinelOver > 0 ? LIMIT_COLOR : SENTINEL_COLOR}
        />
        <Stat
          label={`At ${timeLabel(step)}, ${policy} export`}
          value={`${Math.round(nowExport ?? 0)} kW`}
          tone={now && (nowExport ?? 0) > now.limit ? LIMIT_COLOR : undefined}
        />
        <Stat label="Limit" value={`${feeder?.limit_kw ?? 0} kW`} />
        <Stat label="Baseline, peak export" value={`${Math.round(baselinePeak)} kW`} />
        <Stat label="Sentinel, peak export" value={`${Math.round(sentinelPeak)} kW`} />
      </div>

      <section className="space-y-1">
        <h3 className="text-sm font-medium">Export vs limit (kW)</h3>
        <ChartLegend
          items={[
            { label: "Baseline", color: BASELINE_COLOR },
            { label: "Sentinel", color: SENTINEL_COLOR },
            { label: "Limit", color: LIMIT_COLOR, dashed: true },
          ]}
        />
        <LineChart data={exportData} aspectRatio="5 / 2" margin={CHART_MARGIN} animationDuration={0}>
          <Grid />
          <Line dataKey="baseline_export" stroke={BASELINE_COLOR} curve={curveMonotoneX} fadeEdges={false} />
          <Line dataKey="sentinel_export" stroke={SENTINEL_COLOR} curve={curveMonotoneX} fadeEdges={false} />
          <Line dataKey="limit" stroke={LIMIT_COLOR} strokeWidth={1.5} fadeEdges={false} dashFromIndex={0} />
          <XAxis numTicks={5} labels={labels} />
          <ChartTooltip />
        </LineChart>
      </section>

      <section className="space-y-1">
        <h3 className="text-sm font-medium">Evening import (kW)</h3>
        <ChartLegend
          items={[
            { label: "Baseline", color: BASELINE_COLOR },
            { label: "Sentinel", color: SENTINEL_COLOR },
          ]}
        />
        <LineChart data={exportData} aspectRatio="5 / 2" margin={CHART_MARGIN} animationDuration={0}>
          <Grid />
          <Line dataKey="baseline_import" stroke={BASELINE_COLOR} curve={curveMonotoneX} fadeEdges={false} />
          <Line dataKey="sentinel_import" stroke={SENTINEL_COLOR} curve={curveMonotoneX} fadeEdges={false} />
          <XAxis numTicks={5} labels={labels} />
          <ChartTooltip />
        </LineChart>
      </section>

      <section className="space-y-1">
        <h3 className="text-sm font-medium">Battery SoC fan (%)</h3>
        <LineChart data={socData} aspectRatio="5 / 2" margin={CHART_MARGIN} animationDuration={0}>
          <Grid />
          <Band upperKey="soc_q90" lowerKey="soc_q10" label="P10–P90" fill={SOC_COLOR} curve={curveMonotoneX} />
          <Line dataKey="soc_q50" stroke={SOC_COLOR} curve={curveMonotoneX} fadeEdges={false} />
          <XAxis numTicks={5} labels={labels} />
          <ChartTooltip />
        </LineChart>
      </section>

      <section className="space-y-1">
        <h3 className="text-sm font-medium">Sites that carried the absorption</h3>
        {sites.length === 0 ? (
          <p className="text-xs text-muted-foreground">No absorption needed on this feeder.</p>
        ) : (
          <ul className="space-y-1 text-sm">
            {sites.map((site) => (
              <li key={site.asset_id} className="flex justify-between">
                <span>Site {site.asset_id}</span>
                <span className="text-muted-foreground">{site.absorbed_kwh} kWh</span>
              </li>
            ))}
          </ul>
        )}
      </section>
    </aside>
  );
}
