"use client";

// The last 24 h as settled: P10–P90 band, what WattSentry sold day-ahead (W3) and what the park
// made, with marks where W2 took turbines out of the sale; then the park battery (W5) on the same hours.

import { keepPreviousData, useQuery } from "@tanstack/react-query";
import { curveMonotoneX, curveStepAfter } from "d3-shape";
import { useMemo } from "react";
import { createPortal } from "react-dom";

import { Band } from "@/components/charts/band";
import { useChartPortalContainer, useChartStable } from "@/components/charts/chart-context";
import { Grid } from "@/components/charts/grid";
import { Line, LineChart } from "@/components/charts/line-chart";
import { ChartTooltip } from "@/components/charts/tooltip";
import { Metric, MetricRow } from "@/components/panels/disclosure";
import { orpc } from "@/lib/orpc";

import { lei } from "@/lib/utils";
import { FanHourTicks, FanYAxis } from "./fan-chart-overlays";

const SECTION_LABEL = "section-title";
const MARGIN = { left: 32, right: 8, top: 26, bottom: 24 };
const HOUR_MS = 3_600_000;
const BATTERY_VARIANTS = ["lp", "rule"];
const iso = (ms: number) => new Date(ms).toISOString().replace(".000Z", "Z");

type Hour = { date: Date; ts: string; p10: number; p90: number; sold: number; actual: number; removed: number };

// Sum the 15-min MWh into hours (= mean MW over the hour).
function byHour<T extends { ts: string }>(rows: T[], keys: (keyof T)[]) {
  const hours = new Map<string, Record<string, number>>();
  for (const row of rows) {
    const h = `${row.ts.slice(0, 13)}:00:00Z`;
    const acc = hours.get(h) ?? {};
    for (const k of keys) acc[k as string] = (acc[k as string] ?? 0) + (Number(row[k]) || 0);
    hours.set(h, acc);
  }
  return [...hours.entries()].sort(([a], [b]) => a.localeCompare(b));
}

/** A tick and "−N" where the sale dropped turbines W2 expected to be stopped. */
function RemovedMarks({ hours }: { hours: Hour[] }) {
  const { containerRef, xScale, margin, innerHeight } = useChartStable();
  const container = useChartPortalContainer(containerRef);
  if (!container) return null;
  const marks = hours.filter((h, i) => h.removed > 0 && h.removed !== hours[i - 1]?.removed);
  return createPortal(
    <div className="pointer-events-none absolute inset-0">
      {marks.map((h) => (
        <span key={h.ts} style={{ left: margin.left + xScale(h.date), top: margin.top, height: innerHeight }} className="absolute w-px bg-status-critical/60">
          <span className="absolute -top-4 -translate-x-1/2 font-mono text-[10px] leading-none text-status-critical">−{h.removed}</span>
        </span>
      ))}
    </div>,
    container,
  );
}

export function WindActions({ parkId, clock, wallClock }: { parkId?: string; clock: Date; wallClock: (ts: string) => Date }) {
  const end = Math.floor(clock.getTime() / HOUR_MS) * HOUR_MS;
  const input = { parkId, from: iso(end - 24 * HOUR_MS), to: iso(end) };
  const opts = { enabled: !!parkId, placeholderData: keepPreviousData, staleTime: Number.POSITIVE_INFINITY };
  const positions = useQuery({ ...orpc.production.getWindPositions.queryOptions({ input }), ...opts });
  const battery = useQuery({ ...orpc.production.getParkBattery.queryOptions({ input }), ...opts });

  const hours = useMemo<Hour[]>(() => {
    const rows = positions.data ?? [];
    const removed = new Map(rows.map((r) => [`${r.ts.slice(0, 13)}:00:00Z`, r.removedTurbines ?? 0]));
    return byHour(rows, ["fcP10Mwh", "fcP90Mwh", "soldDaMwh", "actualMwh"]).map(([ts, v]) => ({
      ts,
      date: wallClock(ts),
      p10: v.fcP10Mwh ?? 0,
      p90: v.fcP90Mwh ?? 0,
      sold: v.soldDaMwh ?? 0,
      actual: v.actualMwh ?? 0,
      removed: removed.get(ts) ?? 0,
    }));
  }, [positions.data, wallClock]);

  const bat = useMemo(() => {
    const rows = battery.data ?? [];
    const variant = BATTERY_VARIANTS.find((v) => rows.some((r) => r.variant === v));
    const mine = rows.filter((r) => r.variant === variant);
    const sum = (pick: (r: (typeof mine)[number]) => number | null) => mine.reduce((s, r) => s + Math.abs(pick(r) ?? 0), 0);
    const soc = byHour(mine, ["socMwh"]).map(([ts, v]) => ({ ts, date: wallClock(ts), soc: (v.socMwh ?? 0) / 4 }));
    return {
      variant,
      soc,
      savedLei: sum((r) => r.avoidableNoneLei) - sum((r) => r.avoidableLei),
      coveredMwh: sum((r) => r.devNoneMwh) - sum((r) => r.devMwh),
      cycledMwh: sum((r) => r.dischargeMwh),
    };
  }, [battery.data, wallClock]);

  if (!hours.length && !bat.soc.length) return null;
  const sold = hours.reduce((s, h) => s + h.sold, 0);
  const made = hours.reduce((s, h) => s + h.actual, 0);
  const cost = (positions.data ?? []).reduce((s, r) => s + (r.avoidableLei ?? 0), 0);

  return (
    <>
      {hours.length ? (
        <section className="space-y-1.5 pt-1">
          <div className="flex items-center justify-between gap-3">
            <h3 className={SECTION_LABEL}>Last 24 h: sold vs made (MWh/h)</h3>
            <ul className="flex items-center gap-3 text-[11px] text-text-secondary">
              <li className="flex items-center gap-1.5">
                <span className="h-0.5 w-3.5 rounded-full bg-sentinel" aria-hidden />
                Sold
              </li>
              <li className="flex items-center gap-1.5">
                <span className="h-0.5 w-3.5 rounded-full bg-status-actual" aria-hidden />
                Made
              </li>
              <li className="flex items-center gap-1.5">
                <span className="h-2.5 w-px bg-status-critical" aria-hidden />
                Turbines out
              </li>
            </ul>
          </div>
          <LineChart data={hours} aspectRatio="5 / 2" margin={MARGIN} animationDuration={0}>
            <Grid numTicksRows={4} fadeHorizontal={false} strokeDasharray="none" />
            <Band upperKey="p90" lowerKey="p10" label="P10–P90" fill="var(--brand-500)" fillOpacity={0.12} strokeOpacity={0.4} curve={curveMonotoneX} />
            <Line dataKey="sold" stroke="var(--sentinel)" curve={curveStepAfter} fadeEdges={false} />
            <Line dataKey="actual" stroke="var(--status-actual)" strokeWidth={1.5} curve={curveMonotoneX} fadeEdges={false} />
            <FanYAxis numTicks={4} unit="MWh" />
            <FanHourTicks everyHours={6} />
            <RemovedMarks hours={hours} />
            <ChartTooltip />
          </LineChart>
          <p className="text-xs text-text-secondary tabular-nums">
            Sold <span className="font-mono text-foreground">{Math.round(sold)}</span> MWh, made{" "}
            <span className="font-mono text-foreground">{Math.round(made)}</span> MWh; imbalance cost{" "}
            <span className="font-mono text-foreground">{lei(cost)}</span> lei.
          </p>
        </section>
      ) : null}

      {bat.soc.length ? (
        <section className="mt-5 space-y-1.5 border-t border-border pt-5">
          <h3 className={SECTION_LABEL}>Park battery, {bat.variant === "lp" ? "hourly plan" : "rule"} (45.7 MW / 91.4 MWh)</h3>
          <MetricRow className="py-1">
            <Metric label="Saved" value={lei(bat.savedLei)} unit="lei" sub="vs no battery" />
            <Metric label="Covered" value={Math.round(bat.coveredMwh)} unit="MWh" sub="forecast error" />
            <Metric label="Discharged" value={Math.round(bat.cycledMwh)} unit="MWh" />
          </MetricRow>
          <LineChart data={bat.soc} aspectRatio="5 / 1.6" margin={MARGIN} animationDuration={0}>
            <Grid numTicksRows={3} fadeHorizontal={false} strokeDasharray="none" />
            <Line dataKey="soc" stroke="var(--status-soc)" curve={curveMonotoneX} fadeEdges={false} />
            <FanYAxis numTicks={3} unit="SoC MWh" />
            <FanHourTicks everyHours={6} />
            <ChartTooltip />
          </LineChart>
        </section>
      ) : null}
    </>
  );
}
