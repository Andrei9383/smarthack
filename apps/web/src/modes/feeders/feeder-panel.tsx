"use client";

import { useQuery } from "@tanstack/react-query";
import { curveMonotoneX } from "d3-shape";
import { BatteryCharging, ChevronDown, Sun } from "lucide-react";
import { useMemo, useState } from "react";

import { Band } from "@/components/charts/band";
import { Grid } from "@/components/charts/grid";
import { Line, LineChart } from "@/components/charts/line-chart";
import { ChartTooltip } from "@/components/charts/tooltip";
import { Term } from "@/components/help";
import { stepToTs, timeLabel } from "@/lib/replay";
import { useReplayStep } from "@/lib/use-replay";
import {
  type Delta,
  DetailsSheet,
  Fold,
  Metric,
  MetricRow,
  RowButton,
  SectionHeader,
  SheetSection,
  StatusPill,
} from "@/components/panels/disclosure";
import { useVerdicts } from "@/components/panels/judge/use-judge";
import { orpc } from "@/lib/orpc";
import { VerdictCard } from "@/components/panels/judge/verdict-card";

import { BASELINE_SERIES, LIMIT_SERIES, WATTSENTRY_SERIES, SOC_SERIES } from "./colors";
import { FeederRanking } from "./feeder-ranking";
import { GridRow } from "./grid-section";
import { LiveRow } from "./live-section";
import { LEVEL, NoticeRow, useNotice } from "./notice-section";
import { AXIS_MARGIN, ChartAxes, ChartLegend, EmptyNote, yAxisFor } from "./ui";
import { useDayWindow, useFeederList, useOverLimit } from "./use-feeder-data";
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

// Charts format dates in the browser's zone, so hand them Bucharest wall-clock time.
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

// Status text class. Red only for a value over its limit now.
const OVER_TONE = "text-status-critical";

/**
 * WattSentry against the baseline as a delta chip: "−40%" in green when WattSentry is lower (less time
 * over the limit, a lower peak), neutral when equal; none without a baseline to compare with.
 * `neutral` keeps the chip grey (export now: more export is only bad over the limit).
 */
function deltaVs(baseline: number | undefined, sentinel: number | undefined, neutral = false): Delta | undefined {
  if (baseline == null || sentinel == null || baseline < 1) return undefined;
  const change = Math.round((sentinel / baseline - 1) * 100);
  if (change === 0) return { text: "same", good: null };
  return { text: `${change > 0 ? "+" : "−"}${Math.abs(change)}%`, good: neutral ? null : change < 0 };
}

/** Judge verdicts for this feeder, folded on the glance; nothing when the DB has no Judge rows. */
function JudgeFold({ feederId }: { feederId: string }) {
  const { data } = useVerdicts(feederId);
  if (!data?.length) return null;
  return (
    <Fold title="Can we trust the models?" aside={`${data.length} ${data.length === 1 ? "model" : "models"}`}>
      <VerdictCard assetId={feederId} className="border-0 bg-transparent p-0 shadow-none" />
    </Fold>
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
      ? `Today WattSentry cut time over the limit from ${baselineOver} to ${sentinelOver} min.`
      : "It never went over its limit today.";
  if (baselineNow == null || sentinelNow == null) return `No reading for ${name} at ${at}. ${today}`;
  const b = Math.round(baselineNow);
  const s = Math.round(sentinelNow);
  if (s > limit) return `${name} is over its ${limit} kW limit at ${at}: ${s} kW even with WattSentry.`;
  if (b > limit)
    return `WattSentry holds ${name} at ${s} kW, under its ${limit} kW limit. Without it, solar would push it to ${b} kW.`;
  if (b < 1) return `${name} is not exporting at ${at}: no solar surplus. ${today}`;
  return `${name} is under its ${limit} kW limit at ${at} (${s} kW). ${today}`;
}

export function FeederPanel() {
  const [step] = useReplayStep();
  const [selected, setSelected] = useSelectedFeeder();
  const [detailsOpen, setDetailsOpen] = useState(false);
  const window = useDayWindow();
  const { data: feeders, isLoading } = useFeederList();
  const { data: overLimit } = useOverLimit();
  const feeder = feeders?.find((f) => f.id === selected) ?? feeders?.[0];
  const feederId = feeder?.id ?? "";
  const input = { feederId, ...window };

  const { data: curves } = useQuery(
    orpc.feeders.getCurves.queryOptions({ input, enabled: Boolean(feederId) }),
  );
  const { data: soc } = useQuery(
    orpc.feeders.getSocFan.queryOptions({
      input: { ...input, policy: "sentinel" }, // replay.py stores schedules for WattSentry only
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
  const overNow = (value?: number) => (now && (value ?? 0) > now.limit ? OVER_TONE : undefined);
  const baselinePeak = statFor("baseline")?.peakExportKw;
  const sentinelPeak = statFor("sentinel")?.peakExportKw;
  const { data: notice } = useNotice(feederId, window.from, window.to);
  const level = notice ? LEVEL[notice.level] : null;

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
        <div className="flex flex-wrap items-center gap-2">
          <h2 className="text-title">
            <span className="relative inline-flex items-center">
              <select
                aria-label="Feeder"
                value={feederId}
                onChange={(event) => setSelected(event.target.value)}
                className="h-9 cursor-pointer appearance-none rounded-lg bg-card py-1 pr-9 pl-3 text-title text-foreground shadow-raised transition-[background-color,box-shadow] duration-150 outline-none hover:bg-surface-raised focus-visible:ring-2 focus-visible:ring-ring"
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
          {level ? (
            <StatusPill status={level.pill} className="shrink-0">
              <span className="sr-only">Congestion level: </span>
              {level.label}
            </StatusPill>
          ) : null}
        </div>
        <div className="flex flex-wrap items-center gap-x-1.5 gap-y-1 text-caption text-text-secondary">
          <span>{feeder.county}</span>
          <span aria-hidden>·</span>
          <span className="tabular-nums">{limitKw} kW limit</span>
        </div>
      </header>

      <div className="flex flex-col px-5">
        <p className="pt-3 pb-4 leading-snug text-foreground">{sentence}</p>

        {/* Three tiles: WattSentry's value, its change against the baseline, the baseline underneath. */}
        <MetricRow className="border-y border-border py-3">
          <Metric
            label={
              <>
                Minutes over <Term id="feeder-limit">limit</Term>
              </>
            }
            value={String(sentinelOver)}
            unit="min"
            delta={deltaVs(baselineOver, sentinelOver)}
            sub={`Baseline ${baselineOver} min`}
          />
          <Metric
            label="Peak export today"
            value={String(Math.round(sentinelPeak ?? 0))}
            unit="kW"
            delta={deltaVs(baselinePeak, sentinelPeak)}
            sub={`Baseline ${Math.round(baselinePeak ?? 0)} kW`}
          />
          <Metric
            label={`Export at ${timeLabel(step)}`}
            value={now ? String(Math.round(now.sentinel_export)) : "—"}
            unit={now ? "kW" : undefined}
            tone={overNow(now?.sentinel_export)}
            delta={deltaVs(
              now?.baseline_export,
              now?.sentinel_export,
              !(now && now.baseline_export > now.limit),
            )}
            sub={
              now ? (
                <span className={overNow(now.baseline_export)}>Baseline {Math.round(now.baseline_export)} kW</span>
              ) : undefined
            }
          />
        </MetricRow>

        <section className="space-y-2 pt-5">
          <SectionHeader
            as="h3"
            title="Export today, kW"
            aside={
              <ChartLegend
                items={[
                  { label: "Baseline", color: BASELINE_SERIES },
                  { label: "WattSentry", color: WATTSENTRY_SERIES },
                  { label: "Limit", color: LIMIT_SERIES, dashed: true },
                ]}
              />
            }
          />
          <div className="relative">
            <LineChart
              data={exportData}
              aspectRatio="3 / 1"
              margin={AXIS_MARGIN}
             
              yScaleDomainMax={exportAxis.domainMax}
            >
              <Grid rowTickValues={exportAxis.ticks} />
              <Line dataKey="baseline_export" stroke={BASELINE_SERIES} strokeWidth={2} curve={curveMonotoneX} />
              <Line dataKey="sentinel_export" stroke={WATTSENTRY_SERIES} curve={curveMonotoneX} />
              <Line dataKey="limit" stroke={LIMIT_SERIES} strokeWidth={2} dashFromIndex={0} />
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

        <FeederRanking selected={feederId} onSelect={(id) => void setSelected(id)} />

        <div className="mt-4 pb-2">
          {feederId ? <JudgeFold feederId={feederId} /> : null}
          <div className="border-t border-border">
            <NoticeRow feederId={feederId} feederName={name} from={window.from} to={window.to} />
          </div>
          <LiveRow feederId={feederId} feederName={name} from={window.from} to={window.to} />
          <GridRow feederId={feederId} feederName={name} />
          <RowButton
            onClick={() => setDetailsOpen(true)}
            className="border-t border-border"
          >
            <span className="font-medium">Details</span>
          </RowButton>
        </div>
      </div>

      <DetailsSheet
        open={detailsOpen}
        onOpenChange={setDetailsOpen}
        title={`${name}: the rest of the day`}
        description={`${feeder.county} · ${limitKw} kW limit · Baseline = what happens without WattSentry`}
      >
        <SheetSection
          title="Evening import (kW)"
          aside={
            <ChartLegend
              items={[
                { label: "Baseline", color: BASELINE_SERIES },
                { label: "WattSentry", color: WATTSENTRY_SERIES },
              ]}
            />
          }
        >
          <div className="relative">
            <LineChart
              data={exportData}
              aspectRatio="2 / 1"
              margin={AXIS_MARGIN}
             
              yScaleDomainMax={importAxis.domainMax}
            >
              <Grid rowTickValues={importAxis.ticks} />
              <Line dataKey="baseline_import" stroke={BASELINE_SERIES} strokeWidth={2} curve={curveMonotoneX} />
              <Line dataKey="sentinel_import" stroke={WATTSENTRY_SERIES} curve={curveMonotoneX} />
              <ChartTooltip />
            </LineChart>
            <ChartAxes dates={dates} top={importAxis.top} ticks={importAxis.ticks} unit="kW" now={nowDate} />
          </div>
        </SheetSection>

        <SheetSection
          title={
            <>
              Battery <Term id="soc">charge</Term>, WattSentry (%)
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
               
                yScaleDomainMax={socAxis.domainMax}
              >
                <Grid rowTickValues={socAxis.ticks} />
                <Band upperKey="soc_q90" lowerKey="soc_q10" label="P10–P90" fill={SOC_SERIES} fillOpacity={0.14} strokeOpacity={0.35} curve={curveMonotoneX} />
                <Line dataKey="soc_q50" stroke={SOC_SERIES} curve={curveMonotoneX} />
                <ChartTooltip />
              </LineChart>
              <ChartAxes dates={socDates} top={socAxis.top} ticks={socAxis.ticks} unit="%" now={nowDate} />
            </div>
          ) : (
            <EmptyNote icon={BatteryCharging}>No schedule rows for this feeder.</EmptyNote>
          )}
        </SheetSection>

        <SheetSection title="Homes that absorbed the surplus" aside={<span className="text-caption text-muted-foreground tabular-nums">{(sites ?? []).length} homes</span>}>
          {(sites ?? []).length === 0 ? (
            <EmptyNote icon={Sun}>No absorption needed on this feeder today.</EmptyNote>
          ) : (
            <ul className="divide-y divide-border text-dense">
              {sites?.map((site) => (
                <li key={site.assetId} className="flex justify-between py-2">
                  <span>Site {site.assetId}</span>
                  <span className="text-text-secondary tabular-nums">
                    {site.absorbedKwh.toFixed(1)} <span className="text-caption text-muted-foreground">kWh</span>
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
