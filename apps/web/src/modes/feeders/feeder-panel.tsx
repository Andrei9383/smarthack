"use client";

import { useQuery } from "@tanstack/react-query";
import { curveMonotoneX } from "d3-shape";
import { BatteryCharging, ChevronDown, Sun } from "lucide-react";
import { useMemo } from "react";

import { Band } from "@/components/charts/band";
import { Grid } from "@/components/charts/grid";
import { Line, LineChart } from "@/components/charts/line-chart";
import { ChartTooltip } from "@/components/charts/tooltip";
import { useShowOwnPill } from "@/components/shell/inspector-embedded";
import { stepToTs, timeLabel } from "@/demo/replay";
import { usePolicy, useReplayStep } from "@/demo/use-replay";
import { orpc } from "@/utils/orpc";

import { BASELINE_SERIES, LIMIT_SERIES, SENTINEL_SERIES, SOC_SERIES } from "./colors";
import {
  AXIS_MARGIN,
  ChartAxes,
  ChartLegend,
  EmptyNote,
  ProvenancePill,
  SectionHeader,
  yAxisFor,
} from "./ui";
import { NoticeSection } from "./notice-section";
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

// Status text classes (team/contracts/ui-style.md §2). Red only for a value over its limit now.
const OVER_TONE = "text-status-critical";
const SENTINEL_TONE = "text-brand-700";

function Value({ value, unit, className }: { value: string; unit?: string; className?: string }) {
  return (
    <span className={`inline-flex items-baseline justify-end gap-0.5 whitespace-nowrap tabular-nums ${className ?? ""}`}>
      <span>{value}</span>
      {unit && <span className="text-[11px] font-normal text-muted-foreground">{unit}</span>}
    </span>
  );
}

function PolicyHeader({ name, active }: { name: string; active: boolean }) {
  return (
    <th
      scope="col"
      className={`pb-2 text-right text-[11px] font-medium tracking-[0.04em] uppercase ${
        active ? "text-foreground" : "text-muted-foreground"
      }`}
    >
      <span className="inline-flex items-center gap-1.5">
        {active && <span className="size-1.5 rounded-full bg-primary" aria-label="shown on map" />}
        {name}
      </span>
    </th>
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
  const showPill = useShowOwnPill(label);
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

  const { exportData } = useMemo(() => {
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
    return { exportData: exportRows };
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

  const dates = useMemo(() => exportData.map((row) => row.date), [exportData]);
  const socDates = useMemo(() => socData.map((row) => row.date), [socData]);
  const exportAxis = useMemo(
    () =>
      yAxisFor(exportData.flatMap((row) => [row.baseline_export, row.sentinel_export, row.limit])),
    [exportData],
  );
  const importAxis = useMemo(
    () => yAxisFor(exportData.flatMap((row) => [row.baseline_import, row.sentinel_import])),
    [exportData],
  );
  const socAxis = useMemo(() => yAxisFor(socData.map((row) => row.soc_q90)), [socData]);

  const statFor = (name: string) =>
    overLimit?.find((row) => row.feederId === feederId && row.policy === name);
  const baselineOver = statFor("baseline")?.minutesOver ?? 0;
  const sentinelOver = statFor("sentinel")?.minutesOver ?? 0;
  const now = exportData.find((row) => row.ts === stepToTs(step));

  const limitKw = Math.round(feeder?.limitKw ?? 0);
  const cut = baselineOver > 0 ? Math.round((1 - sentinelOver / baselineOver) * 100) : 0;
  const overNow = (value?: number) => (now && (value ?? 0) > now.limit ? OVER_TONE : undefined);
  const rows = [
    {
      label: "Minutes over limit today",
      baseline: { value: String(baselineOver), unit: "min" },
      sentinel: {
        value: String(sentinelOver),
        unit: "min",
        className: sentinelOver < baselineOver ? SENTINEL_TONE : undefined,
      },
    },
    {
      label: "Peak export",
      baseline: { value: String(Math.round(statFor("baseline")?.peakExportKw ?? 0)), unit: "kW" },
      sentinel: { value: String(Math.round(statFor("sentinel")?.peakExportKw ?? 0)), unit: "kW" },
    },
    {
      label: `Export at ${timeLabel(step)}`,
      baseline: {
        value: now ? String(Math.round(now.baseline_export)) : "—",
        unit: now ? "kW" : undefined,
        className: overNow(now?.baseline_export),
      },
      sentinel: {
        value: now ? String(Math.round(now.sentinel_export)) : "—",
        unit: now ? "kW" : undefined,
        className: overNow(now?.sentinel_export),
      },
    },
  ];

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
    <aside className="absolute top-3 right-3 bottom-3 z-10 flex w-[440px] max-w-[calc(100%-1.5rem)] flex-col overflow-y-auto overscroll-contain rounded-lg border border-border bg-card text-card-foreground shadow-card">
      <header className="sticky top-0 z-10 space-y-2 border-b border-border bg-card/95 px-5 pt-4 pb-3 backdrop-blur-sm">
        <h2 className="text-base font-semibold">
          <span className="relative inline-flex items-center">
            <select
              aria-label="Feeder"
              value={feederId}
              onChange={(event) => setSelected(event.target.value)}
              className="h-9 cursor-pointer appearance-none rounded-sm border border-border bg-card py-1 pr-9 pl-3 text-base font-semibold text-foreground shadow-xs transition-colors outline-none hover:bg-muted focus-visible:ring-2 focus-visible:ring-ring"
            >
              {feeders?.map((f) => (
                <option key={f.id} value={f.id}>
                  {f.id} · {f.name.startsWith(`${f.id} `) ? f.name.slice(f.id.length + 1) : f.name}
                </option>
              ))}
            </select>
            <ChevronDown
              aria-hidden
              className="pointer-events-none absolute right-3 size-4 text-muted-foreground"
            />
          </span>
        </h2>
        <div className="flex flex-wrap items-center gap-x-1.5 gap-y-1 text-xs text-text-secondary">
          <span>{feeder.county}</span>
          <span aria-hidden>·</span>
          <span>
            limit <span className="tabular-nums">{limitKw}</span> kW
          </span>
          <span aria-hidden>·</span>
          {feeder.gisTrust === "low" ? (
            <span className="inline-flex rounded-full bg-status-warn/10 px-2 py-0.5 text-[11px] font-medium text-status-warn">
              GIS trust low
            </span>
          ) : (
            <span>GIS trust {feeder.gisTrust ?? "unknown"}</span>
          )}
          {showPill ? (
            <span className="ml-auto">
              <ProvenancePill label={label ?? "no run"} />
            </span>
          ) : null}
        </div>
      </header>

      <div className="flex flex-col divide-y divide-border px-5">
        <section className="space-y-4 py-5">
          <p className="flex flex-wrap items-baseline gap-x-2 gap-y-1 text-[15px] leading-snug font-semibold text-foreground">
            <span>{cut > 0 ? "Sentinel cuts minutes over limit" : "Minutes over limit"}</span>
            <span className="whitespace-nowrap tabular-nums">
              <span className="text-text-secondary">{baselineOver}</span>
              <span className="mx-1 text-muted-foreground" aria-label="to">
                →
              </span>
              <span className={cut > 0 ? SENTINEL_TONE : undefined}>{sentinelOver}</span>
            </span>
            {cut > 0 && (
              <span className="rounded-full bg-brand-50 px-2 py-0.5 text-xs font-semibold text-brand-700 tabular-nums">
                −{cut}%
              </span>
            )}
          </p>
          <table className="w-full text-[13px]">
            <thead>
              <tr>
                <th scope="col" className="pb-2 text-left">
                  <span className="sr-only">Metric</span>
                </th>
                <PolicyHeader name="Baseline" active={policy === "baseline"} />
                <PolicyHeader name="Sentinel" active={policy === "sentinel"} />
              </tr>
            </thead>
            <tbody className="divide-y divide-border">
              {rows.map((row) => (
                <tr key={row.label}>
                  <th scope="row" className="py-2 pr-3 text-left font-normal whitespace-nowrap text-text-secondary">
                    {row.label}
                  </th>
                  <td className="w-24 py-2 text-right text-text-secondary">
                    <Value {...row.baseline} />
                  </td>
                  <td className="w-24 py-2 text-right font-semibold text-foreground">
                    <Value {...row.sentinel} />
                  </td>
                </tr>
              ))}
            </tbody>
          </table>
        </section>

        <NoticeSection feederId={feederId} from={window.from} to={window.to} />

        <section className="space-y-3 py-6">
          <SectionHeader title="Export vs limit (kW)">
            <ChartLegend
              items={[
                { label: "Baseline", color: BASELINE_SERIES },
                { label: "Sentinel", color: SENTINEL_SERIES },
                { label: "Limit", color: LIMIT_SERIES, dashed: true },
              ]}
            />
          </SectionHeader>
          <div className="relative">
            <LineChart
              data={exportData}
              aspectRatio="2 / 1"
              margin={AXIS_MARGIN}
              animationDuration={0}
              yScaleDomainMax={exportAxis.domainMax}
            >
              <Grid rowTickValues={exportAxis.ticks} fadeHorizontal={false} />
              <Line dataKey="baseline_export" stroke={BASELINE_SERIES} strokeWidth={2} curve={curveMonotoneX} fadeEdges={false} />
              <Line dataKey="sentinel_export" stroke={SENTINEL_SERIES} curve={curveMonotoneX} fadeEdges={false} />
              <Line dataKey="limit" stroke={LIMIT_SERIES} strokeWidth={2} fadeEdges={false} dashFromIndex={0} />
              <ChartTooltip />
            </LineChart>
            <ChartAxes
              dates={dates}
              top={exportAxis.top}
              ticks={exportAxis.ticks}
              unit="kW"
              marks={[{ value: feeder.limitKw, label: `${limitKw} kW limit`, className: "text-status-critical" }]}
            />
          </div>
        </section>

        <section className="space-y-3 py-6">
          <SectionHeader title="Evening import (kW)">
            <ChartLegend
              items={[
                { label: "Baseline", color: BASELINE_SERIES },
                { label: "Sentinel", color: SENTINEL_SERIES },
              ]}
            />
          </SectionHeader>
          <div className="relative">
            <LineChart
              data={exportData}
              aspectRatio="2 / 1"
              margin={AXIS_MARGIN}
              animationDuration={0}
              yScaleDomainMax={importAxis.domainMax}
            >
              <Grid rowTickValues={importAxis.ticks} fadeHorizontal={false} />
              <Line dataKey="baseline_import" stroke={BASELINE_SERIES} strokeWidth={2} curve={curveMonotoneX} fadeEdges={false} />
              <Line dataKey="sentinel_import" stroke={SENTINEL_SERIES} curve={curveMonotoneX} fadeEdges={false} />
              <ChartTooltip />
            </LineChart>
            <ChartAxes dates={dates} top={importAxis.top} ticks={importAxis.ticks} unit="kW" />
          </div>
        </section>

        <section className="space-y-3 py-6">
          <SectionHeader title="Battery SoC fan, Sentinel (%)">
            {socData.length > 0 && (
              <ChartLegend
                items={[
                  { label: "P50", color: SOC_SERIES },
                  { label: "P10–P90", color: SOC_SERIES, dashed: true },
                ]}
              />
            )}
          </SectionHeader>
          {socData.length > 0 ? (
            <div className="relative">
              <LineChart
                data={socData}
                aspectRatio="2 / 1"
                margin={AXIS_MARGIN}
                animationDuration={0}
                yScaleDomainMax={socAxis.domainMax}
              >
                <Grid rowTickValues={socAxis.ticks} fadeHorizontal={false} />
                <Band upperKey="soc_q90" lowerKey="soc_q10" label="P10–P90" fill={SOC_SERIES} fillOpacity={0.14} strokeOpacity={0.35} curve={curveMonotoneX} />
                <Line dataKey="soc_q50" stroke={SOC_SERIES} curve={curveMonotoneX} fadeEdges={false} />
                <ChartTooltip />
              </LineChart>
              <ChartAxes dates={socDates} top={socAxis.top} ticks={socAxis.ticks} unit="%" />
            </div>
          ) : (
            <EmptyNote icon={BatteryCharging}>No schedule rows for this feeder.</EmptyNote>
          )}
        </section>

        <section className="space-y-3 py-6">
          <SectionHeader title="Sites that carried the absorption" />
          {(sites ?? []).length === 0 ? (
            <EmptyNote icon={Sun}>No absorption needed on this feeder today.</EmptyNote>
          ) : (
            <ul className="divide-y divide-border text-[13px]">
              {sites?.map((site) => (
                <li key={site.assetId} className="flex justify-between py-2">
                  <span>Site {site.assetId}</span>
                  <span className="text-text-secondary tabular-nums">
                    {site.absorbedKwh.toFixed(1)} <span className="text-[11px] text-muted-foreground">kWh</span>
                  </span>
                </li>
              ))}
            </ul>
          )}
        </section>
      </div>
    </aside>
  );
}
