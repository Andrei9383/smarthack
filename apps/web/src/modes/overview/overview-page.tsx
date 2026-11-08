"use client";

import { cn } from "@sentinel-app/ui/lib/utils";
import { useQuery } from "@tanstack/react-query";
import { curveMonotoneX, curveStepAfter } from "d3-shape";
import { ChevronRight } from "lucide-react";
import { parseAsString, useQueryStates } from "nuqs";
import { useMemo } from "react";
import { createPortal } from "react-dom";

import {
  useChartPortalContainer,
  useChartStable,
} from "@/components/charts/chart-context";
import { Grid } from "@/components/charts/grid";
import { Line, LineChart } from "@/components/charts/line-chart";
import { ChartTooltip } from "@/components/charts/tooltip";
import { dayWindow } from "@/demo/replay";
import { useReplayStep } from "@/demo/use-replay";
import type { ModePageProps } from "@/modes/registry";
import { type Delta, DeltaChip, StatusPill } from "@/panels/disclosure";
import { wallClock } from "@/panels/site/time";
import { orpc } from "@/utils/orpc";

import { useOverviewData } from "./use-overview-data";

// Overview for a supplier opening the app for the first time. It tells the pitch's story in order:
// 1. the problem and what it costs (avoidable imbalance today, standard profile vs Sentinel),
// 2. one chart that shows the gap (what PPC bought vs what its prosumers really used),
// 3. how Sentinel closes it (what each step saves over the full replay year),
// 4. the same engine on two more problems (grid cables, wind farms), one line each with a link.
// Detail lives in each mode; nothing here needs the replay controls to make sense.

const CARD = "rounded-xl bg-card shadow-card";
const FOREVER = Number.POSITIVE_INFINITY;
const LINK =
  "inline-flex shrink-0 cursor-pointer items-center gap-1 rounded-md px-1.5 py-0.5 text-dense font-medium whitespace-nowrap text-foreground underline-offset-2 transition-colors duration-150 hover:bg-surface-raised hover:underline focus-visible:outline-2 focus-visible:outline-ring";

const lei = (v: number | undefined) =>
  v === undefined ? "—" : Math.round(v).toLocaleString("en-GB");
const pctChange = (from?: number, to?: number) =>
  from === undefined || to === undefined || from === 0
    ? null
    : Math.round(((to - from) / Math.abs(from)) * 100);
const chip = (p: number | null): Delta | null =>
  p === null
    ? null
    : {
        text: `${p < 0 ? "−" : "+"}${Math.abs(p)}%`,
        good: p < 0 ? true : null,
      };
const DAY = new Intl.DateTimeFormat("en-GB", {
  timeZone: "Europe/Bucharest",
  weekday: "long",
  day: "numeric",
  month: "long",
});

const NAV = {
  mode: parseAsString,
  feeder: parseAsString,
  park: parseAsString,
  turbine: parseAsString,
  site: parseAsString,
};
function useGo() {
  const [, set] = useQueryStates(NAV);
  return (mode: string) =>
    set({ mode, feeder: null, park: null, turbine: null, site: null });
}

const BOUGHT_PROFILE = "Bought, standard profile";
const BOUGHT_SENTINEL = "Bought, Sentinel";
const ACTUAL = "Actually used";

/** Today's imbalance for the day on the replay clock, plus the full-year totals per policy. */
function useImbalance() {
  const [step] = useReplayStep();
  const window = dayWindow(step);
  const day = useQuery({
    ...orpc.imbalance.getImbalanceDay.queryOptions({ input: window }),
    staleTime: FOREVER,
  });
  const year = useQuery({
    ...orpc.imbalance.getPolicyComparison.queryOptions({ input: {} }),
    staleTime: FOREVER,
  });

  const rows = useMemo(() => {
    const byTs = new Map<string, Record<string, number | Date>>();
    for (const s of day.data?.steps ?? []) {
      if (s.policy !== "profile" && s.policy !== "sentinel") continue;
      const row = byTs.get(s.ts) ?? { date: wallClock(s.ts) };
      row[s.policy === "profile" ? BOUGHT_PROFILE : BOUGHT_SENTINEL] =
        s.bought ?? 0;
      row[ACTUAL] = s.actual ?? 0;
      byTs.set(s.ts, row);
    }
    return [...byTs.entries()]
      .sort(([a], [b]) => a.localeCompare(b))
      .map(([, r]) => r);
  }, [day.data]);

  const today = (policy: string) =>
    day.data?.totals.find((t) => t.policy === policy);
  const ofYear = (policy: string) =>
    year.data?.find((r) => r.policy === policy);
  return {
    dayName: DAY.format(new Date(Date.parse(window.from) + 3_600_000)),
    rows,
    profile: today("profile"),
    sentinel: today("sentinel"),
    year: {
      profile: ofYear("profile"),
      forecast: ofYear("forecast_only"),
      sentinel: ofYear("sentinel"),
      corrected: ofYear("sentinel_id"),
    },
    loading: day.isPending,
  };
}

/* ---------- 1. The problem and what it costs today ---------- */

function Problem() {
  const go = useGo();
  const { dayName, profile, sentinel, year } = useImbalance();
  const today = pctChange(profile?.avoidableLei, sentinel?.avoidableLei);
  const yearly = pctChange(
    year.profile?.avoidableLei,
    year.sentinel?.avoidableLei,
  );

  return (
    <section aria-labelledby="ov-problem" className={cn(CARD, "p-6")}>
      <div className="flex flex-wrap items-center justify-between gap-2">
        <p className="text-caption font-medium tracking-wide text-muted-foreground uppercase">
          Prosumer imbalance · {dayName}
        </p>
        <button type="button" onClick={() => go("imbalance")} className={LINK}>
          Open imbalance
          <ChevronRight aria-hidden className="size-3.5" />
        </button>
      </div>

      <h1
        id="ov-problem"
        className="mt-3 max-w-[30ch] text-page text-balance text-foreground"
      >
        {profile && sentinel ? (
          <>
            Today PPC’s prosumers cost{" "}
            <span className="whitespace-nowrap">
              {lei(profile.avoidableLei)} lei
            </span>{" "}
            in avoidable imbalance. Sentinel cuts that to{" "}
            <span className="whitespace-nowrap text-status-ok">
              {lei(sentinel.avoidableLei)} lei
            </span>
            .
          </>
        ) : (
          "Reading today’s imbalance…"
        )}
      </h1>

      <p className="mt-3 max-w-[78ch] text-body text-text-secondary">
        PPC buys tomorrow’s energy using a standard “typical home” profile. That
        profile can’t see rooftop solar or home batteries, so in the evening PPC
        ends up buying too much or too little, and the difference is settled at
        a worse price. Sentinel forecasts every home, battery included, and buys
        that instead.
      </p>

      <div className="mt-5 grid grid-cols-1 gap-3 sm:grid-cols-3">
        <Stat
          label="Standard profile, today"
          value={lei(profile?.avoidableLei)}
          unit="lei"
        />
        <Stat
          label="With Sentinel, today"
          value={lei(sentinel?.avoidableLei)}
          unit="lei"
          tone="text-status-ok"
          delta={chip(today)}
        />
        <Stat
          label="Over a full year (2025 replay)"
          value={lei(year.sentinel?.avoidableLei)}
          unit="lei"
          tone="text-status-ok"
          delta={chip(yearly)}
          sub={`instead of ${lei(year.profile?.avoidableLei)} lei with the standard profile`}
        />
      </div>
    </section>
  );
}

function Stat({
  label,
  value,
  unit,
  tone,
  delta,
  sub,
}: {
  label: string;
  value: string;
  unit: string;
  tone?: string;
  delta?: Delta | null;
  sub?: string;
}) {
  return (
    <div className="rounded-lg bg-surface-raised px-4 py-3">
      <p className="text-caption font-medium text-muted-foreground">{label}</p>
      <p className="mt-1 flex flex-wrap items-baseline gap-x-1.5 gap-y-1">
        <span
          className={cn(
            "text-[28px] leading-none font-semibold tracking-[-0.02em] tabular-nums",
            tone ?? "text-foreground",
          )}
        >
          {value}
        </span>
        <span className="text-dense text-muted-foreground">{unit}</span>
        {delta ? <DeltaChip delta={delta} /> : null}
      </p>
      {sub ? (
        <p className="mt-1.5 text-caption text-muted-foreground">{sub}</p>
      ) : null}
    </div>
  );
}

/* ---------- 2. The gap, in one chart ---------- */

/** Shades 18:00-21:00, when home batteries discharge and the standard profile is most wrong. */
function EveningWindow() {
  const { containerRef, xScale, margin, innerHeight, data } = useChartStable();
  const container = useChartPortalContainer(containerRef);
  const dates = (data as { date: Date }[]).map((d) => d.date);
  const start = dates.find((d) => d.getHours() === 18);
  const end = dates.find((d) => d.getHours() === 21) ?? dates.at(-1);
  if (!container || !start || !end) return null;
  const left = margin.left + xScale(start);
  return createPortal(
    <div
      className="pointer-events-none absolute border-x border-status-warn/40 bg-status-warn/10"
      style={{
        left,
        width: margin.left + xScale(end) - left,
        top: margin.top,
        height: innerHeight,
      }}
    >
      <span className="absolute -top-4 left-1 text-caption leading-none font-medium text-status-warn">
        evening peak
      </span>
    </div>,
    container,
  );
}

const AXIS =
  "pointer-events-none absolute text-[11px] leading-none text-muted-foreground tabular-nums";

/** One label per 3 hours (on the hour only) and MWh on the left with one decimal: values stay under 1 MWh. */
function Axes() {
  const { containerRef, xScale, yScale, margin, innerHeight, data } =
    useChartStable();
  const container = useChartPortalContainer(containerRef);
  if (!container) return null;
  const hours = (data as { date: Date }[])
    .map((d) => d.date)
    .filter((d) => d.getMinutes() === 0 && d.getHours() % 3 === 0);
  return createPortal(
    <div className="pointer-events-none absolute inset-0">
      <span className={AXIS} style={{ left: 0, top: 2 }}>
        MWh
      </span>
      {yScale.ticks(4).map((tick) => (
        <span
          key={tick}
          className={`${AXIS} -translate-y-1/2 text-right`}
          style={{
            left: 0,
            width: margin.left - 8,
            top: margin.top + yScale(tick),
          }}
        >
          {tick.toFixed(1)}
        </span>
      ))}
      {hours.map((d) => (
        <span
          key={d.getTime()}
          className={`${AXIS} -translate-x-1/2`}
          style={{
            left: margin.left + xScale(d),
            top: margin.top + innerHeight + 8,
          }}
        >
          {`${String(d.getHours()).padStart(2, "0")}:00`}
        </span>
      ))}
    </div>,
    container,
  );
}

function GapChart() {
  const { dayName, rows, loading } = useImbalance();
  const legend: [string, string][] = [
    [BOUGHT_PROFILE, "var(--chart-baseline)"],
    [BOUGHT_SENTINEL, "var(--sentinel)"],
    [ACTUAL, "var(--status-actual)"],
  ];
  return (
    <section aria-labelledby="ov-gap" className={cn(CARD, "p-6")}>
      <h2 id="ov-gap" className="section-title">
        What PPC bought vs what its prosumers actually used
      </h2>
      <p className="mt-1 text-dense text-text-secondary">
        {dayName}, every 15 minutes. The further a “bought” line is from the
        dark line, the more PPC pays to settle the difference.
      </p>
      <ul className="mt-3 flex flex-wrap items-center gap-4 text-caption text-text-secondary">
        {legend.map(([label, color]) => (
          <li key={label} className="flex items-center gap-1.5">
            <span
              className="h-0.5 w-4 rounded-full"
              style={{ background: color }}
              aria-hidden
            />
            {label}
          </li>
        ))}
        <li className="flex items-center gap-1.5">
          <span className="h-3 w-4 rounded-sm bg-status-warn/15" aria-hidden />
          18:00–21:00
        </li>
      </ul>
      <div className="mt-3">
        {rows.length ? (
          <LineChart
            data={rows}
            aspectRatio="5 / 1.6"
            margin={{ left: 40, right: 8, top: 22, bottom: 24 }}
            animationDuration={0}
          >
            <Grid
              numTicksRows={4}
              fadeHorizontal={false}
              strokeDasharray="none"
            />
            <Line
              dataKey={BOUGHT_PROFILE}
              stroke="var(--chart-baseline)"
              curve={curveStepAfter}
              fadeEdges={false}
            />
            <Line
              dataKey={BOUGHT_SENTINEL}
              stroke="var(--sentinel)"
              curve={curveStepAfter}
              fadeEdges={false}
            />
            <Line
              dataKey={ACTUAL}
              stroke="var(--status-actual)"
              strokeWidth={1.75}
              curve={curveMonotoneX}
              fadeEdges={false}
            />
            <EveningWindow />
            <Axes />
            <ChartTooltip />
          </LineChart>
        ) : (
          <div
            className={cn(
              "h-[260px] rounded-lg bg-surface-raised",
              loading && "animate-pulse",
            )}
          />
        )}
      </div>
      <p className="mt-2 text-caption text-muted-foreground">
        Above zero: customers take energy from the grid. Below zero: their solar
        panels push energy into it.
      </p>
    </section>
  );
}

/* ---------- 3. How Sentinel closes the gap ---------- */

function Steps() {
  const { year } = useImbalance();
  const steps = [
    { label: "PPC today: standard profile", v: year.profile?.avoidableLei },
    {
      label: "1 · Predict each home’s solar and consumption",
      v: year.forecast?.avoidableLei,
    },
    {
      label: "2 · Predict what home batteries will do",
      v: year.sentinel?.avoidableLei,
    },
    {
      label: "3 · Correct at 15:00 on the intraday market",
      v: year.corrected?.avoidableLei,
    },
  ].filter((s): s is { label: string; v: number } => s.v !== undefined);
  const max = Math.max(1, ...steps.map((s) => s.v));

  return (
    <section
      aria-labelledby="ov-steps"
      className={cn(CARD, "flex h-full flex-col p-6")}
    >
      <h2 id="ov-steps" className="section-title">
        How Sentinel closes the gap
      </h2>
      <p className="mt-1 text-dense text-text-secondary">
        Avoidable imbalance cost over a full year, after each step.
      </p>
      <ol className="mt-4 space-y-3.5">
        {steps.map((s, i) => {
          const prev = i ? steps[i - 1]!.v : s.v;
          const change = i ? pctChange(prev, s.v) : null;
          return (
            <li key={s.label}>
              <div className="mb-1 flex items-baseline justify-between gap-3 text-dense">
                <span className={i ? "text-foreground" : "text-text-secondary"}>
                  {s.label}
                </span>
                <span className="font-medium whitespace-nowrap text-foreground tabular-nums">
                  {lei(s.v)} lei
                  {change !== null ? (
                    <span className="ml-2 font-normal text-status-ok">
                      {chip(change)?.text}
                    </span>
                  ) : null}
                </span>
              </div>
              <div className="h-2.5 rounded-sm bg-surface-raised">
                <span
                  className="block h-full rounded-sm"
                  style={{
                    width: `${(s.v / max) * 100}%`,
                    background: i ? "var(--sentinel)" : "var(--chart-baseline)",
                  }}
                />
              </div>
            </li>
          );
        })}
        {!steps.length ? (
          <li className="text-dense text-text-secondary">
            No imbalance replay in this database yet.
          </li>
        ) : null}
      </ol>
      <p className="mt-auto pt-4 text-caption text-muted-foreground">
        Synthetic Romanian homes, real OPCOM day-ahead prices. Step 2 needs
        nothing from customers: it reads their meters.
      </p>
    </section>
  );
}

/* ---------- 4. Same engine, two more problems ---------- */

function MoreProblems() {
  const go = useGo();
  const data = useOverviewData();
  const over = data.feeders.filter((f) => f.status === "over");
  const near = data.feeders.filter((f) => f.status === "near");
  const worst = [...over].sort(
    (a, b) => (b.minutesOver ?? 0) - (a.minutesOver ?? 0),
  )[0];
  const turbineOrders = data.openOrders.filter(
    (o) => o.assetKind === "turbine",
  ).length;
  const total = data.feeders.length;

  const grid = !data.ready
    ? "Reading the grid…"
    : worst
      ? `${over.length} of ${total} neighbourhood cables went over their limit today (${worst.name}, ${worst.minutesOver} min). Home batteries soak up the midday solar peak to prevent it.`
      : `All ${total} neighbourhood cables stayed within their limit today${near.length ? `, ${near.length} came close` : ""}.`;
  const wind = `${data.stopped.length ? `${data.stopped.length} turbine${data.stopped.length === 1 ? "" : "s"} stopped, ` : ""}${turbineOrders} repair job${turbineOrders === 1 ? "" : "s"} open. Sentinel tells a real fault from a planned stop, ice or a broken sensor, so crews go out only when needed.`;

  return (
    <section aria-labelledby="ov-more" className="space-y-3">
      <h2 id="ov-more" className="section-title px-1">
        Same engine, two more problems
      </h2>
      <div className="grid grid-cols-1 gap-5 @min-[880px]:grid-cols-2">
        <MoreCard
          title="Grid cables"
          who="Rețele Electrice"
          status={over.length ? "critical" : near.length ? "warn" : "ok"}
          statusText={
            over.length
              ? "over limit"
              : near.length
                ? "close to limit"
                : "within limit"
          }
          text={grid}
          action="Open Feeders"
          onClick={() => go("feeders")}
        />
        <MoreCard
          title="Wind farms"
          who="PPC Renewables"
          status={
            data.stopped.length ? "critical" : turbineOrders ? "warn" : "ok"
          }
          statusText={
            data.stopped.length
              ? "turbines stopped"
              : turbineOrders
                ? "repairs open"
                : "running"
          }
          text={wind}
          action="Open O&M"
          onClick={() => go("om")}
        />
      </div>
    </section>
  );
}

function MoreCard({
  title,
  who,
  status,
  statusText,
  text,
  action,
  onClick,
}: {
  title: string;
  who: string;
  status: "ok" | "warn" | "critical";
  statusText: string;
  text: string;
  action: string;
  onClick: () => void;
}) {
  return (
    <div className={cn(CARD, "flex flex-col p-5")}>
      <div className="flex flex-wrap items-center gap-2">
        <h3 className="text-body font-semibold text-foreground">{title}</h3>
        <span className="text-caption text-muted-foreground">{who}</span>
        <StatusPill status={status} className="ml-auto">
          {statusText}
        </StatusPill>
      </div>
      <p className="mt-2 text-dense text-text-secondary">{text}</p>
      <div className="mt-auto pt-3">
        <button type="button" onClick={onClick} className={cn(LINK, "-ml-1.5")}>
          {action}
          <ChevronRight aria-hidden className="size-3.5" />
        </button>
      </div>
    </div>
  );
}

/* ---------- The page ---------- */

export function OverviewPage(_props: ModePageProps) {
  return (
    <div className="@container mx-auto flex w-full max-w-[1400px] flex-col gap-5">
      <Problem />
      <div className="grid grid-cols-1 gap-5 @min-[1100px]:grid-cols-[1.6fr_1fr]">
        <GapChart />
        <Steps />
      </div>
      <MoreProblems />
    </div>
  );
}
