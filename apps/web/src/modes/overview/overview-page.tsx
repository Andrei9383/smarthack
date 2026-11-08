"use client";

import { cn } from "@sentinel-app/ui/lib/utils";
import { useQuery } from "@tanstack/react-query";
import { curveMonotoneX, curveStepAfter } from "d3-shape";
import {
  ChevronRight,
  House,
  Pause,
  Play,
  UserRound,
  UtilityPole,
  Wind,
} from "lucide-react";
import type { LucideIcon } from "lucide-react";
import { parseAsString, useQueryStates } from "nuqs";
import { useMemo, useRef, useState } from "react";
import { createPortal } from "react-dom";

import {
  useChartPortalContainer,
  useChartStable,
} from "@/components/charts/chart-context";
import { Grid } from "@/components/charts/grid";
import { Line, LineChart } from "@/components/charts/line-chart";
import { ChartTooltip } from "@/components/charts/tooltip";
import {
  ACTION_LABEL,
  useAgentActions,
  useExplainAction,
} from "@/components/shell/agent-actions";
import { KIND_STYLE } from "@/components/shell/agent-toast";
import { useStoryNarration } from "@/components/voice/story-narration";
import { setPlaying, useReplayPlaying } from "@/demo/replay-bar";
import {
  clampStep,
  dayOfStep,
  dayWindow,
  HERO_DAY,
  STEPS_PER_DAY,
  stepToTs,
  timeLabel,
} from "@/demo/replay";
import { usePolicy, useReplayStep } from "@/demo/use-replay";
import type { ModePageProps } from "@/modes/registry";
import type { Delta } from "@/panels/disclosure";
import { wallClock } from "@/panels/site/time";
import { orpc } from "@/utils/orpc";

import { useOmAssets } from "@/modes/om/use-om-data";

import { MiniMap } from "./mini-map";
import { useOverviewData } from "./use-overview-data";

// Overview: what a supplier needs when it lands on the app, on a calm day as much as on a bad one.
// 1. Status strip: Prosumers, Grid cables, Wind farms: one number and one line each; the card opens that area.
// 2. Needs your decision: only what a person must do (review cases, an intraday purchase when PPC is
//    short). Empty, it says so and counts what the agent handled.
// 3. The day's bought-vs-used chart and this month's imbalance cost with and without Sentinel.

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

  // MWh Sentinel left PPC short today: customers used more than it bought (dev > 0 is the costly side).
  const shortMwh = useMemo(
    () =>
      (day.data?.steps ?? [])
        .filter((s) => s.policy === "sentinel")
        .reduce(
          (sum, s) => sum + Math.max(0, (s.actual ?? 0) - (s.bought ?? 0)),
          0,
        ),
    [day.data],
  );

  const today = (policy: string) =>
    day.data?.totals.find((t) => t.policy === policy);
  const ofYear = (policy: string) =>
    year.data?.find((r) => r.policy === policy);
  return {
    dayName: DAY.format(new Date(Date.parse(window.from) + 3_600_000)),
    rows,
    shortMwh,
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

/* ---------- 1. The problem, what it costs today, and the demo ---------- */

const MORNING_STEP = HERO_DAY * STEPS_PER_DAY + 24; // 06:00 on the story day

/** Starts the story day at 06:00 with Sentinel on; pressing again pauses, then resumes. */
function usePlayDemo() {
  const [rawStep, setStep] = useReplayStep();
  const [, setPolicy] = usePolicy();
  const playing = useReplayPlaying();
  const started = useRef(false);
  // #207: each fresh start from 06:00 is a new narrated run (voice only, when the server has a key).
  const [storyRun, setStoryRun] = useState(0);
  useStoryNarration(storyRun);
  const step = clampStep(rawStep);
  const toggle = () => {
    if (playing) {
      setPlaying(false);
      return;
    }
    const inStory =
      step >= MORNING_STEP && step < HERO_DAY * STEPS_PER_DAY + 80;
    setPolicy("sentinel");
    if (!started.current || !inStory) {
      setStep(MORNING_STEP);
      setStoryRun((run) => run + 1);
    }
    started.current = true;
    setPlaying(true);
  };
  return { playing, toggle };
}

type Level = "ok" | "watch" | "action";
const SHORT_ACTION_MWH = 0.2;
const mwh = (v: number) => (v < 10 ? v.toFixed(2) : Math.round(v).toString());
const plural = (n: number, one: string, many = `${one}s`) =>
  `${n} ${n === 1 ? one : many}`;

function useStatus() {
  const imb = useImbalance();
  const data = useOverviewData();
  const { data: assets } = useOmAssets();

  const saved =
    imb.profile && imb.sentinel
      ? imb.profile.avoidableLei - imb.sentinel.avoidableLei
      : undefined;
  const prosumers: Level =
    imb.shortMwh >= SHORT_ACTION_MWH
      ? "action"
      : imb.shortMwh > 0.005
        ? "watch"
        : "ok";

  const over = data.feeders
    .filter((f) => f.status === "over")
    .sort((a, b) => (b.minutesOver ?? 0) - (a.minutesOver ?? 0));
  const near = data.feeders.filter((f) => f.status === "near");
  const grid: Level = over.length ? "action" : near.length ? "watch" : "ok";

  const turbines = (assets ?? []).filter(
    (a) => a.assetKind === "turbine",
  ).length;
  const repairs = data.openOrders.filter(
    (o) => o.assetKind === "turbine",
  ).length;
  const stopped = data.stopped.length;
  const wind: Level = stopped ? "action" : repairs ? "watch" : "ok";

  return {
    imb,
    data,
    saved,
    prosumers,
    over,
    near,
    grid,
    turbines,
    repairs,
    stopped,
    wind,
  };
}

/* ---------- 1. Status strip ---------- */

function StatusStrip() {
  const go = useGo();
  const s = useStatus();
  const total = s.data.feeders.length;
  const within = total - s.over.length;

  const prosumerLine = {
    ok: "Today's purchase covers what customers use",
    watch: `Short ${mwh(s.imb.shortMwh)} MWh at times today`,
    action: `Short ${mwh(s.imb.shortMwh)} MWh today: buy the gap on intraday`,
  }[s.prosumers];
  const gridLine = {
    ok: `All ${total} cables within their limit`,
    watch: `${s.near[0]?.name ?? "A cable"} close to its limit`,
    action: `${s.over[0]?.name ?? "A cable"} over its limit for ${s.over[0]?.minutesOver ?? 0} min`,
  }[s.grid];
  const windLine = {
    ok: "All turbines running",
    watch: `${plural(s.repairs, "repair")} open, crews booked`,
    action: `${plural(s.stopped, "turbine")} stopped · ${plural(s.repairs, "repair")} open`,
  }[s.wind];

  return (
    <div className="grid grid-cols-1 gap-4 @min-[880px]:grid-cols-3">
      <StatusCard
        icon={House}
        title="Prosumers"
        level={s.imb.profile ? s.prosumers : undefined}
        value={lei(s.saved)}
        unit="lei saved today"
        line={prosumerLine}
        onClick={() => go("imbalance")}
      />
      <StatusCard
        icon={UtilityPole}
        title="Grid cables"
        level={s.data.ready ? s.grid : undefined}
        value={s.data.ready ? `${within}/${total}` : "—"}
        unit="within limit"
        line={gridLine}
        onClick={() => go("feeders")}
      />
      <StatusCard
        icon={Wind}
        title="Wind farms"
        level={s.turbines ? s.wind : undefined}
        value={s.turbines ? `${s.turbines - s.stopped}/${s.turbines}` : "—"}
        unit="turbines running"
        line={windLine}
        onClick={() => go("om")}
      />
    </div>
  );
}

function StatusCard({
  icon: Icon,
  title,
  level,
  value,
  unit,
  line,
  onClick,
}: {
  icon: LucideIcon;
  title: string;
  level: Level | undefined;
  value: string;
  unit: string;
  line: string;
  onClick: () => void;
}) {
  return (
    <button
      type="button"
      onClick={onClick}
      className={cn(
        CARD,
        "group flex cursor-pointer flex-col p-5 text-left transition-colors duration-150 hover:bg-surface-raised focus-visible:outline-2 focus-visible:outline-ring",
      )}
    >
      <span className="flex items-center gap-2.5">
        <span className="flex size-9 shrink-0 items-center justify-center rounded-full bg-surface-raised text-foreground">
          <Icon aria-hidden className="size-[18px]" />
        </span>
        <span className="text-body font-semibold text-foreground">{title}</span>
      </span>
      <span className="mt-4 flex items-baseline gap-1.5">
        <span className="text-[28px] leading-none font-semibold tracking-[-0.02em] text-foreground tabular-nums">
          {value}
        </span>
        <span className="text-dense text-muted-foreground">{unit}</span>
      </span>
      <span className="mt-2 flex items-center gap-1 text-dense text-text-secondary">
        <span className="min-w-0 flex-1">{level ? line : "Loading…"}</span>
        <ChevronRight
          aria-hidden
          className="size-4 shrink-0 text-muted-foreground transition-transform duration-150 group-hover:translate-x-0.5"
        />
      </span>
    </button>
  );
}

/* ---------- 2. Needs your decision ---------- */

const REVIEW_WHY: Record<string, string> = {
  ood: "readings unlike anything the model has seen",
  wide_interval: "forecast too uncertain to act on",
};
const TIME = new Intl.DateTimeFormat("en-GB", {
  timeZone: "Europe/Bucharest",
  weekday: "short",
  hour: "2-digit",
  minute: "2-digit",
});
const TOP_DECISIONS = 4;

type Decision = {
  key: string;
  icon: LucideIcon;
  title: string;
  why: string;
  when?: string;
  action: string;
  go: () => void;
};

function Decisions() {
  const go = useGo();
  const [, setNav] = useQueryStates({ ...NAV, omTab: parseAsString });
  const s = useStatus();
  const [rawStep] = useReplayStep();
  const step = clampStep(rawStep);
  const { actions } = useAgentActions();
  const handled = actions.filter(
    (a) => dayOfStep(a.step) === dayOfStep(step) && a.step <= step,
  ).length;

  const items: Decision[] = [];
  if (s.prosumers === "action")
    items.push({
      key: "intraday",
      icon: House,
      title: `Buy ${mwh(s.imb.shortMwh)} MWh on the intraday market`,
      why: "Customers are using more than PPC bought for today",
      action: "Open imbalance",
      go: () => go("imbalance"),
    });
  // One row per asset: its latest case (the replay never resolves cases, so they pile up).
  const latest = new Map<string, (typeof s.data.waiting)[number]>();
  for (const r of [...s.data.waiting].sort((a, b) => a.ts.localeCompare(b.ts)))
    latest.set(r.assetId, r);
  for (const r of [...latest.values()].sort((a, b) =>
    b.ts.localeCompare(a.ts),
  )) {
    const turbine = r.assetId.includes(".");
    items.push({
      key: r.actionId,
      icon: UserRound,
      title: `Check ${turbine ? `turbine ${r.assetId}` : `home ${r.assetId}`}`,
      why: `The agent is not sure: ${REVIEW_WHY[r.reason] ?? r.reason.replaceAll("_", " ")}`,
      when: TIME.format(new Date(r.ts)),
      action: "Review",
      go: () =>
        setNav({
          mode: "om",
          omTab: "review",
          feeder: null,
          park: null,
          turbine: null,
          site: null,
        }),
    });
  }
  const shown = items.slice(0, TOP_DECISIONS);

  return (
    <section
      aria-labelledby="ov-decide"
      className={cn(CARD, "flex h-full flex-col p-5")}
    >
      <div className="flex items-center justify-between gap-3">
        <h2 id="ov-decide" className="section-title">
          Needs your decision
        </h2>
        {items.length ? (
          <span className="rounded-full bg-status-critical/10 px-2 py-0.5 text-caption font-semibold text-status-critical tabular-nums">
            {items.length}
          </span>
        ) : null}
      </div>
      {shown.length ? (
        <ul className="mt-3 divide-y divide-border">
          {shown.map((d) => (
            <li key={d.key} className="flex items-center gap-3 py-2.5">
              <span className="flex size-8 shrink-0 items-center justify-center rounded-full bg-status-review/10 text-status-review">
                <d.icon aria-hidden className="size-4" />
              </span>
              <span className="min-w-0 flex-1">
                <span className="block text-dense font-medium text-foreground">
                  {d.title}
                </span>
                <span className="block truncate text-caption text-muted-foreground">
                  {d.when ? (
                    <span className="tabular-nums">{d.when} · </span>
                  ) : null}
                  {d.why}
                </span>
              </span>
              <button
                type="button"
                onClick={d.go}
                className="shrink-0 cursor-pointer rounded-lg bg-card px-3 py-1.5 text-dense font-medium text-foreground shadow-raised transition-colors duration-150 hover:bg-surface-raised focus-visible:outline-2 focus-visible:outline-ring"
              >
                {d.action}
              </button>
            </li>
          ))}
        </ul>
      ) : (
        <div className="flex flex-1 flex-col items-center justify-center gap-1 py-8 text-center">
          <span className="text-body font-medium text-foreground">
            Nothing needs you right now
          </span>
          <span className="text-dense text-muted-foreground">
            Sentinel handled {plural(handled, "action")} today on its own.
          </span>
        </div>
      )}
      {items.length > shown.length ? (
        <button
          type="button"
          onClick={shown.at(-1)?.go}
          className={cn(LINK, "mt-auto -ml-1.5 self-start pt-2")}
        >
          See all {items.length}
          <ChevronRight aria-hidden className="size-3.5" />
        </button>
      ) : null}
      {items.length ? (
        <p className="mt-auto pt-3 text-caption text-muted-foreground">
          Sentinel handled {plural(handled, "other action")} today on its own.
        </p>
      ) : null}
    </section>
  );
}

/* ---------- 4. Activity log: what the agent did today ---------- */

const LOG_FIRST = 6;

function ActivityLog() {
  const [rawStep] = useReplayStep();
  const step = clampStep(rawStep);
  const { actions } = useAgentActions();
  const explain = useExplainAction();
  const [all, setAll] = useState(false);
  const today = actions
    .filter((a) => dayOfStep(a.step) === dayOfStep(step) && a.step <= step)
    .sort((a, b) => b.step - a.step);
  const shown = all ? today : today.slice(0, LOG_FIRST);

  return (
    <section aria-labelledby="ov-log" className={cn(CARD, "p-5")}>
      <div className="flex items-center justify-between gap-3">
        <h2 id="ov-log" className="section-title">
          Activity log
        </h2>
        <span className="text-caption text-muted-foreground tabular-nums">
          {plural(today.length, "action")} today
        </span>
      </div>
      {shown.length ? (
        <ul className="mt-2 divide-y divide-border">
          {shown.map((a) => {
            const { icon: Icon, tile } = KIND_STYLE[a.kind];
            return (
              <li key={a.id} className="flex items-center gap-3 py-2">
                <span className="w-11 shrink-0 font-mono text-caption text-muted-foreground tabular-nums">
                  {timeLabel(a.step)}
                </span>
                <span
                  className={cn(
                    "grid size-6 shrink-0 place-items-center rounded-md text-primary-foreground",
                    tile,
                  )}
                >
                  <Icon aria-hidden className="size-3.5" />
                </span>
                <span className="w-24 shrink-0 text-caption font-medium text-text-secondary">
                  {ACTION_LABEL[a.kind]}
                </span>
                <span className="min-w-0 flex-1 truncate text-dense text-foreground">
                  {a.title}
                </span>
                <button
                  type="button"
                  onClick={() => explain(a)}
                  className={cn(LINK, "text-caption")}
                >
                  Why?
                </button>
              </li>
            );
          })}
        </ul>
      ) : (
        <p className="mt-2 py-4 text-dense text-muted-foreground">
          No agent actions yet today.
        </p>
      )}
      {today.length > LOG_FIRST ? (
        <button
          type="button"
          onClick={() => setAll((v) => !v)}
          className={cn(LINK, "mt-1 -ml-1.5")}
        >
          {all ? "Show less" : `Show all ${today.length}`}
        </button>
      ) : null}
    </section>
  );
}

/** Small demo control: starts the story day for presentations, out of the way for daily use. */
function DemoButton() {
  const demo = usePlayDemo();
  return (
    <button
      type="button"
      onClick={demo.toggle}
      className="inline-flex cursor-pointer items-center gap-1.5 rounded-lg px-2.5 py-1 text-caption font-medium text-muted-foreground transition-colors duration-150 hover:bg-surface-raised hover:text-foreground focus-visible:outline-2 focus-visible:outline-ring"
    >
      {demo.playing ? (
        <Pause aria-hidden className="size-3.5 fill-current" />
      ) : (
        <Play aria-hidden className="size-3.5 fill-current" />
      )}
      {demo.playing ? "Pause demo" : "Play demo"}
    </button>
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
        {dayName}. The closer to the dark line, the lower the cost.
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
    </section>
  );
}

/* ---------- 3. This month so far ---------- */

const WITHOUT = "Without Sentinel";
const WITH = "With Sentinel";
const MONTH = new Intl.DateTimeFormat("en-GB", { timeZone: "Europe/Bucharest", month: "long" });
const DAY_NUM = new Intl.DateTimeFormat("en-GB", { timeZone: "Europe/Bucharest", day: "numeric" });

/** Imbalance cost from the 1st of the month to the replay clock, with and without Sentinel. */
function useMonthSoFar() {
  const [rawStep] = useReplayStep();
  const step = clampStep(rawStep);
  const { from: dayFrom } = dayWindow(step);
  // Local midnight on the 1st: the day window starts at local midnight, so keep its UTC hour.
  const d = new Date(dayFrom);
  const local = new Date(d.getTime() + 12 * 3_600_000); // safely inside the local day
  const monthStart = new Date(
    Date.UTC(local.getUTCFullYear(), local.getUTCMonth(), 1, d.getUTCHours()) -
      (d.getUTCHours() > 12 ? 86_400_000 : 0),
  ).toISOString().replace(".000Z", "Z");
  const to = stepToTs(step + 1);
  const q = useQuery({
    ...orpc.imbalance.getImbalanceDay.queryOptions({ input: { from: monthStart, to } }),
    staleTime: FOREVER,
  });
  return useMemo(() => {
    const steps = (q.data?.steps ?? [])
      .filter((s) => s.policy === "profile" || s.policy === "sentinel")
      .sort((a, b) => a.ts.localeCompare(b.ts));
    const byHour = new Map<string, { date: Date; [WITHOUT]: number; [WITH]: number }>();
    let without = 0;
    let withS = 0;
    for (const s of steps) {
      if (s.policy === "profile") without += s.avoidableLei ?? 0;
      else withS += s.avoidableLei ?? 0;
      const hour = s.ts.slice(0, 13);
      byHour.set(hour, { date: wallClock(s.ts), [WITHOUT]: without, [WITH]: withS });
    }
    const first = steps[0]?.ts;
    return {
      rows: [...byHour.values()],
      without,
      with: withS,
      range: first
        ? `${DAY_NUM.format(new Date(first))}–${DAY_NUM.format(new Date(to))} ${MONTH.format(new Date(to))}`
        : "",
      loading: q.isPending,
    };
  }, [q.data, q.isPending, to]);
}

function MonthSoFar() {
  const go = useGo();
  const m = useMonthSoFar();
  const saved = m.without - m.with;
  const cut = pctChange(m.without, m.with);
  return (
    <section aria-labelledby="ov-month" className={cn(CARD, "flex h-full flex-col p-6")}>
      <div className="flex items-baseline justify-between gap-3">
        <h2 id="ov-month" className="section-title">
          This month so far
        </h2>
        <span className="text-dense text-text-secondary tabular-nums">{m.range}</span>
      </div>
      <p className="mt-3 flex items-baseline gap-2">
        <span className="text-[32px] leading-none font-semibold tracking-tight text-foreground tabular-nums">
          {m.rows.length ? lei(saved) : "—"}
        </span>
        <span className="text-dense text-text-secondary">lei saved</span>
        {cut !== null && m.rows.length ? (
          <span className="text-dense font-medium text-status-ok">{chip(cut)?.text}</span>
        ) : null}
      </p>
      <p className="mt-2 text-dense text-text-secondary">
        Imbalance cost: <b className="font-medium text-foreground tabular-nums">{lei(m.with)} lei</b> with
        Sentinel, <span className="tabular-nums">{lei(m.without)} lei</span> without.
      </p>
      <ul className="mt-4 flex items-center gap-4 text-caption text-text-secondary">
        <li className="flex items-center gap-1.5">
          <span className="h-0.5 w-4 rounded-full bg-[var(--chart-baseline)]" aria-hidden />
          {WITHOUT}
        </li>
        <li className="flex items-center gap-1.5">
          <span className="h-0.5 w-4 rounded-full bg-[var(--sentinel)]" aria-hidden />
          {WITH}
        </li>
      </ul>
      <div className="mt-2 flex-1">
        {m.rows.length > 1 ? (
          <LineChart
            data={m.rows}
            aspectRatio="5 / 2"
            margin={{ left: 4, right: 4, top: 8, bottom: 4 }}
            animationDuration={0}
          >
            <Grid numTicksRows={3} fadeHorizontal={false} strokeDasharray="none" />
            <Line dataKey={WITHOUT} stroke="var(--chart-baseline)" fadeEdges={false} />
            <Line dataKey={WITH} stroke="var(--sentinel)" strokeWidth={1.75} fadeEdges={false} />
            <ChartTooltip />
          </LineChart>
        ) : (
          <div className={cn("h-[140px] rounded-lg bg-surface-raised", m.loading && "animate-pulse")} />
        )}
      </div>
      <button type="button" onClick={() => go("imbalance")} className={cn(LINK, "mt-3 self-start")}>
        Details
        <ChevronRight aria-hidden className="size-3.5" />
      </button>
    </section>
  );
}

/* ---------- The page ---------- */

export function OverviewPage(_props: ModePageProps) {
  const go = useGo();
  const { dayName } = useImbalance();
  return (
    <div className="@container mx-auto flex w-full max-w-[1400px] flex-col gap-4">
      <div className="flex items-center justify-between gap-3 px-1">
        <p className="text-caption font-medium tracking-wide text-muted-foreground uppercase">
          {dayName}
        </p>
        <DemoButton />
      </div>
      <StatusStrip />
      <div className="grid grid-cols-1 gap-4 @min-[880px]:grid-cols-[1.25fr_1fr]">
        <Decisions />
        <MiniMap
          onOpen={() => go("feeders")}
          className="h-[300px] @min-[880px]:h-auto @min-[880px]:min-h-[300px]"
        />
      </div>
      <div className="grid grid-cols-1 gap-4 @min-[1100px]:grid-cols-[1.6fr_1fr]">
        <GapChart />
        <MonthSoFar />
      </div>
      <ActivityLog />
    </div>
  );
}
