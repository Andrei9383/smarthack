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

import { BASELINE_SERIES, LIMIT_SERIES, SENTINEL_SERIES, SOC_SERIES } from "./colors";
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

const CHART_MARGIN = { left: 8, right: 8, top: 8, bottom: 24 };

// Status text classes: Sentinel green, over-limit red (team/contracts/ui-style.md §2).
const OVER_TONE = "text-status-critical";
const SENTINEL_TONE = "text-brand-700";

const SECTION_LABEL = "text-[11px] font-medium tracking-[0.04em] text-muted-foreground uppercase";

function Stat({ label, value, tone }: { label: string; value: string; tone?: string }) {
  // "45 min" → big number, unit at 13px muted beside it. "—" has no unit.
  const gap = value.lastIndexOf(" ");
  const number = gap > 0 ? value.slice(0, gap) : value;
  const unit = gap > 0 ? value.slice(gap + 1) : "";
  return (
    <div className="flex flex-col gap-1">
      <div className="text-[11px] leading-snug text-muted-foreground">{label}</div>
      <div className="flex items-baseline gap-1">
        <span
          className={`text-2xl leading-none font-semibold tracking-tight tabular-nums ${tone ?? "text-foreground"}`}
        >
          {number}
        </span>
        {unit && <span className="text-[13px] text-muted-foreground">{unit}</span>}
      </div>
    </div>
  );
}

function ChartLegend({ items }: { items: { label: string; color: string; dashed?: boolean }[] }) {
  return (
    <div className="flex flex-wrap gap-x-4 gap-y-1 text-xs text-text-secondary">
      {items.map(({ label, color, dashed }) => (
        <span key={label} className="flex items-center gap-1.5">
          <span
            className="inline-block w-4 border-t-[2.5px]"
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
      <aside className="absolute top-3 right-3 z-10 w-72 rounded-lg border border-border bg-card p-4 text-sm text-muted-foreground shadow-card">
        {isLoading
          ? "Loading feeders…"
          : "No replay data. Run python3 demo/fixtures/make_fixtures.py."}
      </aside>
    );
  }

  return (
    <aside className="absolute top-3 right-3 bottom-3 z-10 flex w-[440px] max-w-[calc(100%-1.5rem)] flex-col gap-5 overflow-y-auto rounded-lg border border-border bg-card p-5 text-card-foreground shadow-card">
      <header className="flex items-start justify-between gap-2">
        <div className="space-y-1.5">
          <h2 className="flex items-center gap-2 text-base font-semibold">
            <select
              aria-label="Feeder"
              value={feederId}
              onChange={(event) => setSelected(event.target.value)}
              className="h-8 rounded-sm border border-border bg-muted px-2 text-sm font-semibold text-foreground outline-none focus-visible:ring-2 focus-visible:ring-ring"
            >
              {feeders?.map((f) => (
                <option key={f.id} value={f.id}>
                  {f.id}
                </option>
              ))}
            </select>
            {feeder.name}
          </h2>
          <p className="text-xs leading-relaxed text-text-secondary">
            {feeder.county} · limit <span className="tabular-nums">{Math.round(feeder.limitKw)}</span> kW · GIS trust{" "}
            {feeder.gisTrust ?? "unknown"} ·{" "}
            <span className="inline-flex rounded-full bg-status-warn/10 px-2 py-0.5 text-[11px] font-medium text-status-warn">
              {label ?? "no run"}
            </span>
          </p>
        </div>
      </header>

      <div className="grid grid-cols-2 gap-x-6 gap-y-5 rounded-lg bg-sidebar p-4">
        <Stat
          label="Baseline, minutes over limit today"
          value={`${baselineOver} min`}
          tone={baselineOver > 0 ? OVER_TONE : undefined}
        />
        <Stat
          label="Sentinel, minutes over limit today"
          value={`${sentinelOver} min`}
          tone={sentinelOver > 0 ? OVER_TONE : SENTINEL_TONE}
        />
        <Stat
          label={`At ${timeLabel(step)}, ${policy} export`}
          value={now ? `${Math.round(nowExport ?? 0)} kW` : "—"}
          tone={now && (nowExport ?? 0) > now.limit ? OVER_TONE : undefined}
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

      <section className="space-y-2">
        <h3 className={SECTION_LABEL}>Export vs limit (kW)</h3>
        <ChartLegend
          items={[
            { label: "Baseline", color: BASELINE_SERIES },
            { label: "Sentinel", color: SENTINEL_SERIES },
            { label: "Limit", color: LIMIT_SERIES, dashed: true },
          ]}
        />
        <LineChart data={exportData} aspectRatio="5 / 2" margin={CHART_MARGIN} animationDuration={0}>
          <Grid />
          <Line dataKey="baseline_export" stroke={BASELINE_SERIES} strokeWidth={2} curve={curveMonotoneX} fadeEdges={false} />
          <Line dataKey="sentinel_export" stroke={SENTINEL_SERIES} curve={curveMonotoneX} fadeEdges={false} />
          <Line dataKey="limit" stroke={LIMIT_SERIES} strokeWidth={2} fadeEdges={false} dashFromIndex={0} />
          <XAxis numTicks={5} labels={labels} />
          <ChartTooltip />
        </LineChart>
      </section>

      <section className="space-y-2">
        <h3 className={SECTION_LABEL}>Evening import (kW)</h3>
        <ChartLegend
          items={[
            { label: "Baseline", color: BASELINE_SERIES },
            { label: "Sentinel", color: SENTINEL_SERIES },
          ]}
        />
        <LineChart data={exportData} aspectRatio="5 / 2" margin={CHART_MARGIN} animationDuration={0}>
          <Grid />
          <Line dataKey="baseline_import" stroke={BASELINE_SERIES} strokeWidth={2} curve={curveMonotoneX} fadeEdges={false} />
          <Line dataKey="sentinel_import" stroke={SENTINEL_SERIES} curve={curveMonotoneX} fadeEdges={false} />
          <XAxis numTicks={5} labels={labels} />
          <ChartTooltip />
        </LineChart>
      </section>

      <section className="space-y-2">
        <h3 className={SECTION_LABEL}>Battery SoC fan, Sentinel (%)</h3>
        {socData.length > 0 ? <LineChart data={socData} aspectRatio="5 / 2" margin={CHART_MARGIN} animationDuration={0}>
          <Grid />
          <Band upperKey="soc_q90" lowerKey="soc_q10" label="P10–P90" fill={SOC_SERIES} fillOpacity={0.14} strokeOpacity={0.35} curve={curveMonotoneX} />
          <Line dataKey="soc_q50" stroke={SOC_SERIES} curve={curveMonotoneX} fadeEdges={false} />
          <XAxis numTicks={5} labels={socLabels} />
          <ChartTooltip />
        </LineChart> : <p className="text-xs text-muted-foreground">No schedule rows for this feeder.</p>}
      </section>

      <section className="space-y-2">
        <h3 className={SECTION_LABEL}>Sites that carried the absorption</h3>
        {(sites ?? []).length === 0 ? (
          <p className="text-xs text-muted-foreground">No absorption needed on this feeder today.</p>
        ) : (
          <ul className="divide-y divide-border text-[13px]">
            {sites?.map((site) => (
              <li key={site.assetId} className="flex justify-between py-1.5">
                <span>Site {site.assetId}</span>
                <span className="text-text-secondary tabular-nums">{site.absorbedKwh.toFixed(1)} kWh</span>
              </li>
            ))}
          </ul>
        )}
      </section>
    </aside>
  );
}
