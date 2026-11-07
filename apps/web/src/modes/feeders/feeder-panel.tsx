"use client";

import { useQuery } from "@tanstack/react-query";
import { curveMonotoneX } from "d3-shape";
import { useMemo } from "react";

import { Band } from "@/components/charts/band";
import { Grid } from "@/components/charts/grid";
import { Line, LineChart } from "@/components/charts/line-chart";
import { ChartTooltip } from "@/components/charts/tooltip";
import { XAxis } from "@/components/charts/x-axis";
import { stepToTs, timeLabel } from "@/demo/replay";
import { usePolicy, useReplayStep } from "@/demo/use-replay";
import { orpc } from "@/utils/orpc";


import { useDayWindow, useFeederList, useOverLimit, useRunLabel } from "./use-feeder-data";
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
  const [selected, setSelected] = useSelectedFeeder();
  const window = useDayWindow();
  const { data: feeders, isLoading } = useFeederList();
  const { data: overLimit } = useOverLimit();
  const label = useRunLabel();
  const feeder = feeders?.find((f) => f.id === selected) ?? feeders?.[0];
  const feederId = feeder?.id ?? "";
  const input = { feederId, ...window };

  const { data: curves } = useQuery(
    orpc.feeders.getCurves.queryOptions({ input, enabled: Boolean(feederId) }),
  );
  const { data: soc } = useQuery(
    orpc.feeders.getSocFan.queryOptions({
      input: { ...input, policy: "sentinel" }, // replay.py stores schedules for Sentinel only
      enabled: Boolean(feederId),
    }),
  );
  const { data: sites } = useQuery(
    orpc.feeders.getAbsorption.queryOptions({ input, enabled: Boolean(feederId) }),
  );

  const { exportData, labels } = useMemo(() => {
    const rows = curves ?? [];
    const sentinelByTs = new Map(
      rows.filter((r) => r.policy === "sentinel").map((r) => [r.ts, r]),
    );
    const exportRows = rows
      .filter((r) => r.policy === "baseline")
      .map((r) => {
        const sentinel = sentinelByTs.get(r.ts);
        return {
          ts: r.ts,
          date: bucharestWallClock(r.ts),
          baseline_export: r.exportKw,
          sentinel_export: sentinel?.exportKw ?? 0,
          limit: r.limitKw,
          baseline_import: r.importKw,
          sentinel_import: sentinel?.importKw ?? 0,
        };
      });
    return { exportData: exportRows, labels: exportRows.map((row) => hhmm(row.date)) };
  }, [curves]);

  const socData = useMemo(
    () =>
      (soc ?? []).map((row) => ({
        date: bucharestWallClock(row.ts),
        soc_q10: row.socQ10 * 100,
        soc_q50: row.socQ50 * 100,
        soc_q90: row.socQ90 * 100,
      })),
    [soc],
  );
  const socLabels = useMemo(() => socData.map((row) => hhmm(row.date)), [socData]);

  const statFor = (name: string) =>
    overLimit?.find((row) => row.feederId === feederId && row.policy === name);
  const baselineOver = statFor("baseline")?.minutesOver ?? 0;
  const sentinelOver = statFor("sentinel")?.minutesOver ?? 0;
  const now = exportData.find((row) => row.ts === stepToTs(step));
  const nowExport = policy === "baseline" ? now?.baseline_export : now?.sentinel_export;

  if (!feeder) {
    return (
      <aside className="absolute top-3 right-3 z-10 w-72 rounded-md border bg-background/95 p-3 text-sm text-muted-foreground">
        {isLoading
          ? "Loading feeders…"
          : "No replay data. Run python3 demo/fixtures/make_fixtures.py."}
      </aside>
    );
  }

  return (
    <aside className="absolute top-3 right-3 bottom-3 z-10 flex w-[440px] max-w-[calc(100%-1.5rem)] flex-col gap-3 overflow-y-auto rounded-md border bg-background/95 p-3 backdrop-blur">
      <header className="flex items-start justify-between gap-2">
        <div>
          <h2 className="font-semibold">
            <select
              aria-label="Feeder"
              value={feederId}
              onChange={(event) => setSelected(event.target.value)}
              className="mr-1 rounded border bg-background px-1 text-sm"
            >
              {feeders?.map((f) => (
                <option key={f.id} value={f.id}>
                  {f.id}
                </option>
              ))}
            </select>
            {feeder.name}
          </h2>
          <p className="text-xs text-muted-foreground">
            {feeder.county} · limit {Math.round(feeder.limitKw)} kW · GIS trust{" "}
            {feeder.gisTrust ?? "unknown"} ·{" "}
            <span className="font-medium text-amber-400">{label ?? "no run"}</span>
          </p>
        </div>
      </header>

      <div className="grid grid-cols-2 gap-2 rounded border p-2">
        <Stat
          label="Baseline, minutes over limit today"
          value={`${baselineOver} min`}
          tone={baselineOver > 0 ? LIMIT_COLOR : undefined}
        />
        <Stat
          label="Sentinel, minutes over limit today"
          value={`${sentinelOver} min`}
          tone={sentinelOver > 0 ? LIMIT_COLOR : SENTINEL_COLOR}
        />
        <Stat
          label={`At ${timeLabel(step)}, ${policy} export`}
          value={now ? `${Math.round(nowExport ?? 0)} kW` : "—"}
          tone={now && (nowExport ?? 0) > now.limit ? LIMIT_COLOR : undefined}
        />
        <Stat label="Limit" value={`${Math.round(feeder.limitKw)} kW`} />
        <Stat
          label="Baseline, peak export"
          value={`${Math.round(statFor("baseline")?.peakExportKw ?? 0)} kW`}
        />
        <Stat
          label="Sentinel, peak export"
          value={`${Math.round(statFor("sentinel")?.peakExportKw ?? 0)} kW`}
        />
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
        <h3 className="text-sm font-medium">Battery SoC fan, Sentinel (%)</h3>
        {socData.length > 0 ? <LineChart data={socData} aspectRatio="5 / 2" margin={CHART_MARGIN} animationDuration={0}>
          <Grid />
          <Band upperKey="soc_q90" lowerKey="soc_q10" label="P10–P90" fill={SOC_COLOR} curve={curveMonotoneX} />
          <Line dataKey="soc_q50" stroke={SOC_COLOR} curve={curveMonotoneX} fadeEdges={false} />
          <XAxis numTicks={5} labels={socLabels} />
          <ChartTooltip />
        </LineChart> : <p className="text-xs text-muted-foreground">No schedule rows for this feeder.</p>}
      </section>

      <section className="space-y-1">
        <h3 className="text-sm font-medium">Sites that carried the absorption</h3>
        {(sites ?? []).length === 0 ? (
          <p className="text-xs text-muted-foreground">No absorption needed on this feeder today.</p>
        ) : (
          <ul className="space-y-1 text-sm">
            {sites?.map((site) => (
              <li key={site.assetId} className="flex justify-between">
                <span>Site {site.assetId}</span>
                <span className="text-muted-foreground">{site.absorbedKwh.toFixed(1)} kWh</span>
              </li>
            ))}
          </ul>
        )}
      </section>
    </aside>
  );
}
