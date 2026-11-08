"use client";

import { curveMonotoneX, curveStepAfter } from "d3-shape";
import { useState } from "react";
import { createPortal } from "react-dom";

import { Band } from "@/components/charts/band";
import { useChartPortalContainer, useChartStable } from "@/components/charts/chart-context";
import { Grid } from "@/components/charts/grid";
import { Line, LineChart } from "@/components/charts/line-chart";
import { ChartTooltip } from "@/components/charts/tooltip";
import { FanHourTicks, FanYAxis } from "@/modes/production/fan-chart-overlays";
import { DetailsLink, DetailsSheet, Metric, MetricRow, SheetSection, StatusPill } from "@/components/panels/disclosure";

import { type Step, useImbalanceData } from "./use-imbalance-data";

// Feature I (#177, spec I step 9c): what PPC buys for its prosumers vs what they really take, today's standard
// profile against WattSentry. The dock's Baseline/WattSentry switch flips every number on the page.
// Calm (#201): the glance is the title + policy pill, three KPIs (WattSentry's change as a delta chip) and the
// bought-vs-actual chart; the waterfall by policy and "What WattSentry learned" sit in the Details sheet.

const MARGIN = { left: 36, right: 8, top: 22, bottom: 24 };
const lei = (v: number) => Math.round(v).toLocaleString("en-GB");
const mwh = (v: number) => (Math.abs(v) < 10 ? v.toFixed(1) : Math.round(v).toLocaleString("en-GB"));
const DAY = new Intl.DateTimeFormat("en-GB", { timeZone: "Europe/Bucharest", weekday: "short", day: "numeric", month: "short", year: "numeric" });

const WATERFALL: [string, string][] = [
  ["profile", "PPC today: standard profile"],
  ["forecast_only", "+ M1/M2 forecast"],
  ["sentinel", "+ battery model"],
  ["sentinel_id", "+ 15:00 correction (PI)"],
];
const STRATEGY: Record<string, string> = {
  none: "No battery",
  self_consumption: "Self-consumption",
  price_driven: "Follows the price (MaxControl)",
  enrolled: "Enrolled with WattSentry",
  battery: "Battery, strategy not classified yet",
  unknown: "Unknown",
};

const CARD = "rounded-xl bg-card p-5 shadow-card";

const pct = (from?: number, to?: number) => (from && to !== undefined ? Math.round(((to - from) / Math.abs(from)) * 100) : null);
const signed = (p: number | null) => (p === null ? "" : `${p < 0 ? "−" : "+"}${Math.abs(p)}%`);

/** The 18:00-21:00 window (the "18:00 story"): PPC bought for solar that had already set. */
function EveningWindow({ steps }: { steps: Step[] }) {
  const { containerRef, xScale, margin, innerHeight } = useChartStable();
  const container = useChartPortalContainer(containerRef);
  const start = steps.find((s) => s.date.getHours() === 18);
  const end = steps.find((s) => s.date.getHours() === 21) ?? steps.at(-1);
  if (!container || !start || !end) return null;
  const left = margin.left + xScale(start.date);
  return createPortal(
    <div
      className="pointer-events-none absolute border-x border-status-warn/40 bg-status-warn/10"
      style={{ left, width: margin.left + xScale(end.date) - left, top: margin.top, height: innerHeight }}
    >
      <span className="absolute -top-4 left-1 text-caption leading-none font-medium text-status-warn tabular-nums">18–21</span>
    </div>,
    container,
  );
}

function Legend({ items }: { items: [string, string][] }) {
  return (
    <ul className="flex flex-wrap items-center gap-3 text-caption text-text-secondary">
      {items.map(([label, color]) => (
        <li key={label} className="flex items-center gap-1.5">
          <span className="h-0.5 w-3.5 rounded-full" style={{ background: color }} aria-hidden />
          {label}
        </li>
      ))}
    </ul>
  );
}

export function ImbalancePage() {
  const { policy, shown, steps, totals, window, comparison, strategies, isLoading } = useImbalanceData();
  const [detailsOpen, setDetailsOpen] = useState(false);
  const sentinel = policy === "sentinel";
  const now = totals.get(shown);
  const other = totals.get(sentinel ? "profile" : "sentinel");
  const base = totals.get("profile");
  const sen = totals.get("sentinel");
  const band = steps.some((s) => s.hasBand);
  const otherName = sentinel ? "today's method" : "WattSentry";
  const day = window?.from ? DAY.format(new Date(Date.parse(window.from) + 3_600_000)) : "—";
  const kpis: [string, string, keyof NonNullable<typeof now>, (v: number) => string][] = [
    ["Avoidable cost today", "lei", "avoidableLei", lei],
    ["Per MWh injected", "lei/MWh", "leiPerMwhInjected", lei],
    ["Bought for nothing 18–21", "MWh", "doubleCover1821Mwh", mwh],
  ];

  const byPolicy = new Map(comparison.map((c) => [c.policy, c]));
  const fall = WATERFALL.filter(([p]) => byPolicy.has(p)).map(([p, label]) => ({ p, label, v: byPolicy.get(p)!.avoidableLei }));
  const max = Math.max(1, ...fall.map((f) => f.v));
  const homes = (strategies?.rows ?? []).reduce((s, r) => s + r.homes, 0);

  if (!isLoading && !steps.length)
    return (
      <p className={`max-w-[640px] ${CARD} text-dense text-text-secondary`}>
        No imbalance replay in this database yet. Run <span className="font-mono text-foreground">python replay.py --mode imbalance</span> (#176)
        into the demo DB to fill <span className="font-mono">positions</span> and <span className="font-mono">portfolio_forecast</span>.
      </p>
    );

  return (
    <div className="@container w-full max-w-[1200px] space-y-4">
      <header className="flex flex-wrap items-start justify-between gap-3 px-1 pt-1">
        <div className="min-w-0">
          <div className="flex flex-wrap items-center gap-x-2.5 gap-y-1">
            <h1 className="text-page text-foreground">Prosumer portfolio: bought vs actual</h1>
            <StatusPill status={sentinel ? "ok" : "unknown"}>{sentinel ? "WattSentry" : "Today's method"}</StatusPill>
          </div>
          <p className="mt-1 text-body text-muted-foreground">
            <span className="text-text-secondary tabular-nums">{day}</span>
            {" · "}
            {sentinel ? "WattSentry: calibrated band, battery behaviour, buys its own quantile" : "Today's method: the standard consumption profile"}
          </p>
        </div>
        <DetailsLink onClick={() => setDetailsOpen(true)} className="mt-1" />
      </header>

      <section className={CARD} aria-label="Today's imbalance">
        <MetricRow>
          {kpis.map(([label, unit, key, fmt]) => {
            const v = now?.[key] as number | null | undefined;
            const o = other?.[key] as number | null | undefined;
            const d = sentinel ? pct(o ?? undefined, v ?? undefined) : pct(v ?? undefined, o ?? undefined);
            return (
              <Metric
                key={label}
                size="lg"
                label={label}
                value={v == null ? "—" : fmt(v)}
                unit={unit}
                tone={sentinel ? "text-status-ok" : undefined}
                // Lower is better for all three: a cut is good, a rise is bad.
                delta={d === null ? undefined : { text: `WattSentry ${signed(d)}`, good: d < 0 ? true : d > 0 ? false : null }}
                sub={`${otherName} ${o == null ? "—" : fmt(o)}`}
              />
            );
          })}
        </MetricRow>
      </section>

      <section className={CARD}>
        <div className="mb-2 flex flex-wrap items-center justify-between gap-x-4 gap-y-1">
          <p className="text-caption text-muted-foreground">MWh per 15 min, + = the portfolio takes from the grid</p>
          <Legend
            items={[
              ...(band ? ([["P10–P90", "var(--chart-band)"]] as [string, string][]) : []),
              ["Bought", sentinel ? "var(--sentinel)" : "var(--chart-baseline)"],
              ["Actual", "var(--status-actual)"],
            ]}
          />
        </div>
        {steps.length ? (
          <LineChart data={steps} aspectRatio="5 / 1.7" margin={MARGIN} animationDuration={0}>
            <Grid numTicksRows={4} fadeHorizontal={false} strokeDasharray="none" />
            {band ? <Band upperKey="p90" lowerKey="p10" label="P10–P90" fill="var(--chart-band)" fillOpacity={1} strokeOpacity={0} curve={curveMonotoneX} /> : null}
            <Line dataKey="bought" stroke={sentinel ? "var(--sentinel)" : "var(--chart-baseline)"} curve={curveStepAfter} fadeEdges={false} />
            <Line dataKey="actual" stroke="var(--status-actual)" strokeWidth={1.5} curve={curveMonotoneX} fadeEdges={false} />
            <EveningWindow steps={steps} />
            <FanYAxis numTicks={4} unit="MWh" />
            <FanHourTicks everyHours={3} />
            <ChartTooltip />
          </LineChart>
        ) : (
          <div className="h-[260px] animate-pulse rounded-lg bg-surface-raised" />
        )}
      </section>

      <DetailsSheet
        open={detailsOpen}
        onOpenChange={setDetailsOpen}
        title="Prosumer portfolio: details"
        description={`${day} · ${sentinel ? "WattSentry" : "Today's method"}`}
      >
        <SheetSection
          title="What each step brings · whole replay"
          aside={<span className="text-caption text-text-secondary tabular-nums">{signed(pct(base?.avoidableLei, sen?.avoidableLei))} today</span>}
        >
          <ol className="space-y-3">
            {fall.map((f, i) => {
              const prev = i ? fall[i - 1]!.v : f.v;
              return (
                <li key={f.p}>
                  <div className="mb-1 flex items-baseline justify-between gap-3 text-dense">
                    <span className={f.p === shown ? "font-semibold text-foreground" : "text-text-secondary"}>{f.label}</span>
                    <span className="font-medium whitespace-nowrap text-foreground tabular-nums">
                      {lei(f.v)} lei
                      {i ? <span className="ml-2 font-normal text-text-secondary">{signed(pct(prev, f.v))}</span> : null}
                    </span>
                  </div>
                  <div className="relative h-3 rounded-sm bg-surface-raised">
                    <span
                      className="absolute inset-y-0 left-0 rounded-sm"
                      style={{ width: `${(f.v / max) * 100}%`, background: i ? "var(--sentinel)" : "var(--chart-baseline)" }}
                    />
                    {i && prev > f.v ? (
                      <span
                        className="absolute inset-y-0 rounded-sm border border-dashed border-border-strong"
                        style={{ left: `${(f.v / max) * 100}%`, width: `${((prev - f.v) / max) * 100}%` }}
                      />
                    ) : null}
                  </div>
                </li>
              );
            })}
            {!fall.length ? <li className="text-dense text-text-secondary">No policy totals yet.</li> : null}
          </ol>
        </SheetSection>

        <SheetSection title="What WattSentry learned">
          <ul className="space-y-2.5">
            {(strategies?.rows ?? []).map((r) => (
              <li key={r.strategy} className="text-dense">
                <div className="flex items-baseline justify-between gap-3">
                  <span className="text-foreground">{STRATEGY[r.strategy] ?? r.strategy}</span>
                  <span className="whitespace-nowrap text-text-secondary tabular-nums">
                    {r.homes} homes{r.confidence == null ? "" : ` · ${Math.round(r.confidence * 100)}% sure`}
                  </span>
                </div>
                <div className="mt-1 h-1.5 rounded-full bg-surface-raised">
                  <span className="block h-full rounded-full bg-status-soc" style={{ width: `${(r.homes / Math.max(1, homes)) * 100}%` }} />
                </div>
              </li>
            ))}
          </ul>
          <p className="border-t border-border pt-2.5 text-caption leading-snug text-muted-foreground">
            {strategies?.source === "site_strategy"
              ? "Battery strategy per home, read from the meter data by the I4 classifier (mean confidence)."
              : "Not classified yet: homes split by declared battery only (site_strategy is empty)."}
          </p>
        </SheetSection>
      </DetailsSheet>
    </div>
  );
}
