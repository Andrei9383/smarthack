"use client";

import { useQuery } from "@tanstack/react-query";
import { curveMonotoneX } from "d3-shape";
import { BatteryCharging, ChevronDown, Sun } from "lucide-react";
import { useMemo, useState } from "react";

import { Band } from "@/components/charts/band";
import { Grid } from "@/components/charts/grid";
import { Line, LineChart } from "@/components/charts/line-chart";
import { ChartTooltip } from "@/components/charts/tooltip";
import { StatusMark } from "@/components/brand/status-mark";
import { Term } from "@/components/help";
import { useShowOwnPill } from "@/components/shell/inspector-embedded";
import { stepToTs, timeLabel } from "@/demo/replay";
import { usePolicy, useReplayStep } from "@/demo/use-replay";
import { DetailsSheet, RowButton, SheetSection } from "@/panels/disclosure";
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
import { NoticeRow } from "./notice-section";
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
      className={`pb-1.5 text-right text-[11px] font-medium tracking-[0.04em] uppercase ${
        active ? "text-foreground" : "text-muted-foreground"
      }`}
    >
      <span className="inline-flex items-center gap-1.5">
        {active && <StatusMark kind="live" size="xs" label="shown on map" />}
        {name}
      </span>
    </th>
  );
}

/** Short feeder name: "F7 Oșorhei" from id "F7" and name "F7 Oșorhei" or "Oșorhei". */
function shortName(id: string, name: string) {
  return name.startsWith(`${id} `) ? name : `${id} ${name}`;
}

/** One plain sentence: what is happening on this feeder at the replay clock. */
function nowSentence({
  name,
  limit,
  at,
  baselineNow,
  sentinelNow,
  baselineOver,
  sentinelOver,
}: {
  name: string;
  limit: number;
  at: string;
  baselineNow?: number;
  sentinelNow?: number;
  baselineOver: number;
  sentinelOver: number;
}) {
  const today =
    baselineOver > 0
      ? `Today Sentinel cut time over the limit from ${baselineOver} to ${sentinelOver} min.`
      : "It never went over its limit today.";
  if (baselineNow == null || sentinelNow == null) return `No reading for ${name} at ${at}. ${today}`;
  const b = Math.round(baselineNow);
  const s = Math.round(sentinelNow);
  if (s > limit) return `${name} is over its ${limit} kW limit at ${at}: ${s} kW even with Sentinel.`;
  if (b > limit)
    return `Sentinel holds ${name} at ${s} kW, under its ${limit} kW limit. Without it, solar would push it to ${b} kW.`;
  if (b < 1) return `${name} is not exporting at ${at}: no solar surplus. ${today}`;
  return `${name} is under its ${limit} kW limit at ${at} (${s} kW). ${today}`;
}

export function FeederPanel() {
  const [policy] = usePolicy();
  const [step] = useReplayStep();
  const [selected, setSelected] = useSelectedFeeder();
  const [detailsOpen, setDetailsOpen] = useState(false);
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
  const nowTs = stepToTs(step);
  const now = exportData.find((row) => row.ts === nowTs);
  const nowDate = bucharestWallClock(nowTs);

  const limitKw = Math.round(feeder?.limitKw ?? 0);
  const cut = baselineOver > 0 ? Math.round((1 - sentinelOver / baselineOver) * 100) : 0;
  const overNow = (value?: number) => (now && (value ?? 0) > now.limit ? OVER_TONE : undefined);
  const rows = [
    {
      label: (
        <>
          Minutes over <Term id="feeder-limit">limit</Term>
        </>
      ),
      key: "minutes",
      baseline: { value: String(baselineOver), unit: "min" },
      sentinel: {
        value: String(sentinelOver),
        unit: "min",
        className: sentinelOver < baselineOver ? SENTINEL_TONE : undefined,
      },
      delta: cut > 0 ? `−${cut}%` : null,
    },
    {
      label: "Peak export today",
      key: "peak",
      baseline: { value: String(Math.round(statFor("baseline")?.peakExportKw ?? 0)), unit: "kW" },
      sentinel: { value: String(Math.round(statFor("sentinel")?.peakExportKw ?? 0)), unit: "kW" },
      delta: null,
    },
    {
      label: `Export at ${timeLabel(step)}`,
      key: "now",
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
      delta: null,
    },
  ];

  if (!feeder) {
    return (
      <aside className="absolute top-3 right-3 z-10 w-72 rounded-xl bg-card p-4 text-sm text-muted-foreground shadow-floating">
        {isLoading
          ? "Loading feeders…"
          : "No replay data. Run python3 demo/fixtures/make_fixtures.py."}
      </aside>
    );
  }

  const name = shortName(feeder.id, feeder.name);
  const sentence = nowSentence({
    name,
    limit: limitKw,
    at: timeLabel(step),
    baselineNow: now?.baseline_export,
    sentinelNow: now?.sentinel_export,
    baselineOver,
    sentinelOver,
  });

  return (
    <aside className="absolute top-3 right-3 bottom-3 z-10 flex w-[440px] max-w-[calc(100%-1.5rem)] flex-col overflow-y-auto overscroll-contain rounded-xl bg-card text-card-foreground shadow-floating">
      <header className="space-y-1.5 px-5 pt-4">
        <h2 className="text-base font-semibold">
          <span className="relative inline-flex items-center">
            <select
              aria-label="Feeder"
              value={feederId}
              onChange={(event) => setSelected(event.target.value)}
              className="h-9 cursor-pointer appearance-none rounded-lg bg-card py-1 pr-9 pl-3 text-base font-semibold text-foreground shadow-raised transition-[background-color,box-shadow] duration-150 outline-none hover:bg-surface-raised focus-visible:ring-2 focus-visible:ring-ring"
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
          <span className="tabular-nums">{limitKw} kW limit</span>
          <span aria-hidden>·</span>
          {feeder.gisTrust === "low" ? (
            <span className="inline-flex items-center gap-1 text-status-warn">
              <StatusMark kind="warn" size="xs" />
              <Term id="gis-trust">GIS trust low</Term>
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

      <div className="flex flex-col px-5">
        <p className="pt-3.5 pb-3 text-[14px] leading-snug text-foreground">{sentence}</p>

        <table className="w-full text-[13px]">
          <thead>
            <tr>
              <th scope="col" className="pb-1.5 text-left">
                <span className="sr-only">Metric</span>
              </th>
              <PolicyHeader name="Baseline" active={policy === "baseline"} />
              <PolicyHeader name="Sentinel" active={policy === "sentinel"} />
            </tr>
          </thead>
          <tbody className="divide-y divide-border border-t border-border">
            {rows.map((row) => (
              <tr key={row.key}>
                <th scope="row" className="py-1.5 pr-2 text-left font-normal whitespace-nowrap text-text-secondary">
                  {row.label}
                </th>
                <td className="w-[4.5rem] py-1.5 text-right text-text-secondary">
                  <Value {...row.baseline} />
                </td>
                <td className="w-[5.5rem] py-1.5 text-right font-semibold text-foreground">
                  <span className="inline-flex items-baseline gap-1.5">
                    {row.delta ? (
                      <span className="rounded-full bg-brand-50 px-1.5 text-[10px] leading-4 font-semibold text-brand-700 tabular-nums">
                        {row.delta}
                      </span>
                    ) : null}
                    <Value {...row.sentinel} />
                  </span>
                </td>
              </tr>
            ))}
          </tbody>
        </table>

        <section className="space-y-2 pt-4 pb-1">
          <div className="flex flex-wrap items-center justify-between gap-x-3 gap-y-1">
            <SectionHeader title="Export today, kW" />
            <ChartLegend
              items={[
                { label: "Baseline", color: BASELINE_SERIES },
                { label: "Sentinel", color: SENTINEL_SERIES },
                { label: "Limit", color: LIMIT_SERIES, dashed: true },
              ]}
            />
          </div>
          <div className="relative">
            <LineChart
              data={exportData}
              aspectRatio="3 / 1"
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
              now={nowDate}
              marks={[{ value: feeder.limitKw, label: `${limitKw} kW limit`, className: "text-status-critical" }]}
            />
          </div>
        </section>

        <div className="mt-2 border-t border-border pb-2">
          <NoticeRow feederId={feederId} feederName={name} from={window.from} to={window.to} />
          <RowButton
            onClick={() => setDetailsOpen(true)}
            className="border-t border-border"
          >
            <span className="font-medium">Details</span>
            <span className="text-text-secondary"> · import, batteries, homes</span>
          </RowButton>
        </div>
      </div>

      <DetailsSheet
        open={detailsOpen}
        onOpenChange={setDetailsOpen}
        title={`${name}: the rest of the day`}
        description={`${feeder.county} · ${limitKw} kW limit · Baseline = what happens without Sentinel`}
      >
        <SheetSection
          title="Evening import (kW)"
          aside={
            <ChartLegend
              items={[
                { label: "Baseline", color: BASELINE_SERIES },
                { label: "Sentinel", color: SENTINEL_SERIES },
              ]}
            />
          }
        >
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
            <ChartAxes dates={dates} top={importAxis.top} ticks={importAxis.ticks} unit="kW" now={nowDate} />
          </div>
        </SheetSection>

        <SheetSection
          title={
            <>
              Battery <Term id="soc">charge</Term>, Sentinel (%)
            </>
          }
          aside={
            socData.length > 0 ? (
              <ChartLegend
                items={[
                  { label: "P50", color: SOC_SERIES },
                  { label: "P10–P90", color: SOC_SERIES, dashed: true },
                ]}
              />
            ) : null
          }
        >
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
              <ChartAxes dates={socDates} top={socAxis.top} ticks={socAxis.ticks} unit="%" now={nowDate} />
            </div>
          ) : (
            <EmptyNote icon={BatteryCharging}>No schedule rows for this feeder.</EmptyNote>
          )}
        </SheetSection>

        <SheetSection title="Homes that absorbed the surplus" aside={<span className="text-xs text-muted-foreground tabular-nums">{(sites ?? []).length} homes</span>}>
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
        </SheetSection>
      </DetailsSheet>
    </aside>
  );
}
