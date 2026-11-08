"use client";

import { cn } from "@sentinel-app/ui/lib/utils";
import { ChevronRight, CircleHelp, Pause, Play } from "lucide-react";
import { useQuery } from "@tanstack/react-query";
import { parseAsString, useQueryStates } from "nuqs";
import { type ReactNode, useEffect, useMemo, useRef, useState } from "react";

import { StatusMark, type StatusKind } from "@/components/brand/status-mark";
import { Term } from "@/components/help";
import {
  ACTION_LABEL,
  ACTION_MARK,
  type ActionMarker,
  groupMarkers,
  useAgentActions,
  useExplainAction,
} from "@/components/shell/agent-actions";
import { CountUp } from "@/components/shell/count-up";
import { Popover, PopoverContent, PopoverTitle, PopoverTrigger } from "@/components/shell/popover";
import { useOverview } from "@/demo/kpi-strip";
import { setPlaying, useReplayPlaying } from "@/demo/replay-bar";
import { clampStep, clockLabel, dayLabel, dayOfStep, HERO_DAY, STEPS_PER_DAY, timeLabel } from "@/demo/replay";
import { usePolicy, useReplayStep } from "@/demo/use-replay";
import type { ModePageProps } from "@/modes/registry";
import { useGrid } from "@/modes/feeders/grid-section";
import {
  type Delta,
  DeltaChip,
  DetailsLink,
  DetailsSheet,
  Metric,
  Num,
  PopList,
  SectionHeader,
  SheetSection,
  StatusPill,
} from "@/panels/disclosure";
import { TrustBoard } from "@/panels/judge/trust-board";
import { useTrustBoard } from "@/panels/judge/use-judge";
import { orpc } from "@/utils/orpc";

import { MiniMap } from "./mini-map";
import { type FeederState, useOverviewData } from "./use-overview-data";

// Overview, Calm (#197, #206): a briefing card (one sentence, one primary action) and a map card, a row
// of four KPI tiles, then "Needs attention" (top 5) beside the last 5 agent actions, and one line for
// imbalance cost. Everything else (the full attention list, the per-feeder table with baseline vs
// Sentinel, all agent activity and model trust) sits in one Details sheet; the week's Sentinel vs
// baseline totals are in Summary.

const NAV_PARSERS = {
  mode: parseAsString,
  feeder: parseAsString,
  park: parseAsString,
  turbine: parseAsString,
  site: parseAsString,
  zone: parseAsString,
  omTab: parseAsString,
};

function useGo() {
  const [, set] = useQueryStates(NAV_PARSERS);
  return (state: Partial<Record<keyof typeof NAV_PARSERS, string | null>>) => set(state);
}

const plural = (n: number, noun: string, many = `${noun}s`) => `${n} ${n === 1 ? noun : many}`;
const turbineName = (id: string) => id.split(".")[1]?.replace("TID", "T") ?? id;
const listOf = (parts: string[]) => parts.join(", ").replace(/, ([^,]*)$/, " and $1");
const MORNING_STEP = HERO_DAY * STEPS_PER_DAY + 24; // 06:00 on the story day
/** Rows on the glance; the rest sit in the Details sheet. */
const TOP_ATTENTION = 5;
const TOP_ACTIVITY = 5;

/** Calm card: white on the pale page, 12px radius, hairline + whisper shadow (ui-style §3). */
const CARD = "rounded-xl bg-card shadow-card";
/** A quiet ink text control: "See all", "Breakdown", "Open imbalance". */
const QUIET_LINK =
  "inline-flex shrink-0 cursor-pointer items-center gap-1 rounded-md px-1.5 py-0.5 text-caption font-medium whitespace-nowrap text-foreground underline-offset-2 transition-colors duration-150 hover:bg-surface-raised hover:underline focus-visible:outline-2 focus-visible:outline-ring data-[popup-open]:bg-surface-raised";

/** Sentinel against a reference as a percentage chip; green when Sentinel is better, neutral otherwise. */
function deltaOf(from?: number, to?: number, up?: boolean): Delta | null {
  if (from === undefined || to === undefined || from === 0) return null;
  const pct = Math.round(((to - from) / Math.abs(from)) * 100);
  if (pct === 0) return null;
  const better = up ? pct > 0 : pct < 0;
  return { text: `${pct < 0 ? "−" : "+"}${Math.abs(pct)}%`, good: better ? true : null };
}

function GoButton({ onClick, children }: { onClick: () => void; children: ReactNode }) {
  return (
    <button
      type="button"
      onClick={onClick}
      className="mt-3 inline-flex h-8 w-full cursor-pointer items-center justify-center gap-1 rounded-lg bg-primary px-3 text-dense font-semibold text-primary-foreground transition-[background-color,transform] duration-150 outline-none hover:bg-primary/85 active:scale-[0.98] focus-visible:ring-2 focus-visible:ring-ring focus-visible:ring-offset-2 focus-visible:ring-offset-card"
    >
      {children}
      <ChevronRight aria-hidden className="size-3.5" />
    </button>
  );
}

type DetailsTarget = "numbers" | "attention" | "feeders" | "activity" | "trust";
type OpenDetails = (target: DetailsTarget) => void;
const detailsId = (target: DetailsTarget) => `ov-details-${target}`;

/* ---------- 1. The shift briefing: one sentence, one primary action ---------- */

function Briefing({ onTour }: { onTour: () => void }) {
  const data = useOverviewData();
  const [rawStep, setStep] = useReplayStep();
  const [, setPolicy] = usePolicy();
  const playing = useReplayPlaying();
  const step = clampStep(rawStep);

  const over = data.feeders.filter((f) => f.status === "over");
  const near = data.feeders.filter((f) => f.status === "near");
  const extras = [
    near.length > 0 ? `${plural(near.length, "feeder")} near ${near.length === 1 ? "its" : "their"} limit` : null,
    data.stopped.length > 0 ? `${plural(data.stopped.length, "turbine")} stopped` : null,
    data.waiting.length > 0 ? `${plural(data.waiting.length, "case")} waiting for a person` : null,
  ].filter((part): part is string => Boolean(part));
  const lead = over.length === 0 ? "Grid is calm." : `${plural(over.length, "feeder")} went over the limit today.`;
  const sentence = data.ready ? `${lead} ${extras.length ? `${listOf(extras)}.` : "Nothing is waiting."}` : "Reading the grid…";

  // First press starts the story day at 06:00 with Sentinel on; after a pause it resumes, unless the
  // clock has left the story day's daylight hours.
  const started = useRef(false);
  const playStory = () => {
    if (playing) {
      setPlaying(false);
      return;
    }
    const inStory = step >= MORNING_STEP && step < HERO_DAY * STEPS_PER_DAY + 80;
    setPolicy("sentinel");
    if (!started.current || !inStory) setStep(MORNING_STEP);
    started.current = true;
    setPlaying(true);
  };

  return (
    <section aria-labelledby="ov-briefing" className={cn(CARD, "flex h-full min-h-[240px] flex-col p-6")}>
      <div className="flex items-center gap-2.5">
        <span className="font-mono text-dense font-semibold tracking-tight text-foreground">{clockLabel(step)}</span>
        <StatusPill status={playing ? "forecast" : "ok"}>
          <StatusMark kind={playing ? "live" : "ok"} size="xs" pulse={playing} />
          {playing ? "Replaying the story" : "Shift briefing"}
        </StatusPill>
      </div>

      <p id="ov-briefing" aria-live="polite" className="mt-4 max-w-[36ch] text-page text-balance text-foreground">
        {sentence}
      </p>

      <div className="mt-auto flex flex-wrap items-center gap-2 pt-6">
        <button
          type="button"
          onClick={playStory}
          className="inline-flex h-10 cursor-pointer items-center gap-2 rounded-lg bg-primary pr-4 pl-3.5 text-body font-semibold text-primary-foreground transition-[background-color,transform] duration-150 hover:bg-primary/85 active:scale-[0.98] focus-visible:outline-2 focus-visible:outline-offset-2 focus-visible:outline-ring"
        >
          {playing ? <Pause aria-hidden className="size-4 fill-current" /> : <Play aria-hidden className="size-4 fill-current" />}
          {playing ? "Pause the story" : "Play today's story"}
        </button>
        <button
          type="button"
          onClick={onTour}
          className="inline-flex h-10 cursor-pointer items-center gap-1.5 rounded-lg bg-card px-3.5 text-body font-medium text-foreground shadow-raised transition-[background-color,box-shadow] duration-150 hover:bg-surface-raised focus-visible:outline-2 focus-visible:outline-offset-2 focus-visible:outline-ring"
        >
          <CircleHelp aria-hidden className="size-4 text-muted-foreground" />
          How it works
        </button>
        <span className="ml-1 hidden text-caption text-muted-foreground @min-[1300px]:inline">
          from {clockLabel(MORNING_STEP)}, with Sentinel
        </span>
      </div>
    </section>
  );
}

/* ---------- 2. KPI row: four tiles ---------- */

// Two by two on a narrow page, one row of four from 880px. Each tile sits in its own cell, and the
// hairline divider belongs to the cell, so it follows the layout (#206).
const KPI_CELLS = [
  "",
  "border-l border-border pl-4",
  "@min-[880px]:border-l @min-[880px]:border-border @min-[880px]:pl-4",
  "border-l border-border pl-4",
];
function KpiGrid({ children }: { children: ReactNode[] }) {
  return (
    <div className="mt-2 grid grid-cols-2 gap-y-4 @min-[880px]:grid-cols-4 [&_span]:whitespace-normal">
      {children.map((child, index) => (
        <div key={index} className={cn("min-w-0", KPI_CELLS[index])}>
          {child}
        </div>
      ))}
    </div>
  );
}

function KpiRow({ onDetails }: { onDetails: OpenDetails }) {
  const data = useOverviewData();
  const [rawStep] = useReplayStep();
  const { actions } = useAgentActions();
  const { data: day } = useOverview();
  const step = clampStep(rawStep);

  const within = data.feeders.filter((f) => f.status === "ok" || f.status === "near").length;
  const today = dayOfStep(step);
  const homes = actions
    .filter((a) => a.kind === "dispatch" && dayOfStep(a.step) === today && a.step <= step)
    .reduce((sum, a) => sum + (a.homes ?? 0), 0);
  const wind = data.parks.reduce((sum, park) => sum + park.mwh, 0);
  const known = (v: number | null | undefined): v is number => v !== null && v !== undefined;

  return (
    <section aria-labelledby="ov-kpis" className={cn(CARD, "px-5 pt-3 pb-4")}>
      <SectionHeader
        title={<span id="ov-kpis">Today</span>}
        aside={<DetailsLink onClick={() => onDetails("numbers")} />}
      />
      <KpiGrid>
        <Metric
          size="lg"
          label={
            <>
              <Term id="feeder">Feeders</Term> within their limit
            </>
          }
          value={<CountUp value={data.ready ? within : 0} />}
          unit={`/ ${data.feeders.length || "–"}`}
        />
        <Metric size="lg" label="Batteries in today's plan" value={<CountUp value={homes} />} />
        <Metric
          size="lg"
          label="Wind expected, next 24 h"
          value={<CountUp value={wind} />}
          unit="MWh"
          sub={known(day?.productionMwh) ? `${day.productionMwh.toFixed(0)} MWh produced today` : undefined}
        />
        <Metric
          size="lg"
          label="Park availability"
          value={known(day?.availabilityPct) ? <CountUp value={day.availabilityPct} digits={1} /> : "—"}
          unit={known(day?.availabilityPct) ? "%" : undefined}
          sub={known(day?.activeIssues) ? plural(day.activeIssues, "open work order") : undefined}
        />
      </KpiGrid>
    </section>
  );
}

/* ---------- 3. Needs attention ---------- */

type Attention = {
  key: string;
  severity: number;
  mark: StatusKind;
  pulse?: boolean;
  text: ReactNode;
  title: string;
  facts: [string, ReactNode][];
  goLabel: string;
  go: () => void;
};

function kw(value: number | undefined) {
  return value === undefined ? "—" : `${Math.round(value)} kW`;
}

function useAttention(): Attention[] {
  const data = useOverviewData();
  const go = useGo();
  return useMemo(() => {
    const rows: Attention[] = [];
    const over = data.feeders.filter((f) => f.status === "over");
    const near = data.feeders.filter((f) => f.status === "near");
    const openFeeder = (f: FeederState) => () => go({ mode: "feeders", feeder: f.id, site: null, turbine: null });

    for (const f of [...over].sort((a, b) => (b.minutesOver ?? 0) - (a.minutesOver ?? 0))) {
      rows.push({
        key: `f-${f.id}`,
        severity: 3,
        mark: "critical",
        pulse: true,
        text: (
          <>
            <b>{f.name}</b> was over its {Math.round(f.limitKw)} kW limit for {f.minutesOver} min today
          </>
        ),
        title: `${f.name}, ${f.county}`,
        facts: [
          ["Limit", kw(f.limitKw)],
          ["Highest export today", kw(f.peakKw)],
          ["Minutes over the limit", `${f.minutesOver ?? "—"} min`],
        ],
        goLabel: `Open ${f.id} in Feeders`,
        go: openFeeder(f),
      });
    }
    if (data.stopped.length > 0) {
      const first = data.stopped[0];
      const one = data.stopped.length === 1 && first;
      rows.push({
        key: "stopped",
        severity: 2.5,
        mark: "critical",
        text: one ? (
          <>
            <b>Turbine {turbineName(first.id)}</b> at {first.parkId} is stopped
          </>
        ) : (
          <>
            <b>{plural(data.stopped.length, "turbine")}</b> stopped at {first?.parkId ?? "WP01"}
          </>
        ),
        title: one ? `Turbine ${turbineName(first.id)} · ${first.parkId}` : `${first?.parkId ?? "WP01"} wind park`,
        facts: [
          ["Stopped", plural(data.stopped.length, "turbine")],
          ["First", first ? `${turbineName(first.id)} · ${first.reason.replaceAll("_", " ") || "no reason given"}` : "—"],
          ["Park", first?.parkId ?? "WP01"],
        ],
        goLabel: one ? `Open ${turbineName(first.id)} in O&M` : "Open the park in O&M",
        go: () =>
          go({ mode: "om", park: first?.parkId ?? "WP01", turbine: one ? first.id : null, site: null }),
      });
    }
    if (data.waiting.length > 0) {
      const oldest = [...data.waiting].sort((a, b) => a.ts.localeCompare(b.ts))[0];
      rows.push({
        key: "review",
        severity: 2,
        mark: "review",
        text: (
          <>
            <b>{plural(data.waiting.length, "case")}</b> waiting for a person to check
          </>
        ),
        title: "Human review queue",
        facts: [
          ["Waiting", plural(data.waiting.length, "case")],
          ["Oldest", oldest ? `${oldest.assetId.includes(".") ? turbineName(oldest.assetId) : `Home ${oldest.assetId}`}` : "—"],
          ["Why a person", "Agent not sure enough"],
        ],
        goLabel: "Open the review queue",
        go: () => go({ mode: "om", omTab: "review", site: null, turbine: null }),
      });
    }
    for (const f of near) {
      rows.push({
        key: `f-${f.id}`,
        severity: 1.5,
        mark: "warn",
        text: (
          <>
            <b>{f.name}</b> peaks at {Math.round(f.peakKw ?? 0)} kW, close to its {Math.round(f.limitKw)} kW limit
          </>
        ),
        title: `${f.name}, ${f.county}`,
        facts: [
          ["Limit", kw(f.limitKw)],
          ["Highest export today", kw(f.peakKw)],
          ["Room left", kw(f.limitKw - (f.peakKw ?? 0))],
        ],
        goLabel: `Open ${f.id} in Feeders`,
        go: openFeeder(f),
      });
    }
    if (data.topOrder) {
      const o = data.topOrder;
      const turbine = o.assetKind === "turbine";
      const who = turbine ? `Turbine ${turbineName(o.assetId)}` : `Home ${o.assetId}`;
      rows.push({
        key: "order",
        severity: 1,
        mark: "warn",
        text: (
          <>
            <b>{who}</b> tops the repair list: {(o.fault ?? "fault").replaceAll("_", " ")}
            {o.lostLeiDay ? `, ${Math.round(o.lostLeiDay)} lei a day` : ""}
          </>
        ),
        title: `${who} · first on the repair list`,
        facts: [
          ["Fault", (o.fault ?? "fault").replaceAll("_", " ")],
          ["Lost a day", o.lostLeiDay ? `${Math.round(o.lostLeiDay)} lei` : "—"],
          ["Crew", o.crewType.replaceAll("_", " ")],
        ],
        goLabel: `Open ${who} in O&M`,
        go: () =>
          turbine
            ? go({ mode: "om", park: "WP01", turbine: o.assetId, site: null })
            : go({ mode: "om", site: o.assetId, turbine: null }),
      });
    }
    return rows.sort((a, b) => b.severity - a.severity);
  }, [data.feeders, data.stopped, data.waiting, data.topOrder, go]);
}

const factRows = (facts: [string, ReactNode][]) => facts.map(([label, value]) => ({ key: label, label, value }));

function NeedsAttention({ onDetails }: { onDetails: OpenDetails }) {
  const all = useAttention();
  const data = useOverviewData();
  const rows = all.slice(0, TOP_ATTENTION);
  const hidden = all.length - rows.length;
  return (
    <section aria-labelledby="ov-attention" className={cn(CARD, "h-full px-5 pt-3 pb-4")}>
      <SectionHeader
        title={<span id="ov-attention">Needs attention</span>}
        aside={
          hidden > 0 ? (
            <button type="button" onClick={() => onDetails("attention")} className={QUIET_LINK}>
              See all {all.length}
              <ChevronRight aria-hidden className="size-3.5" />
            </button>
          ) : all.length > 0 ? (
            <span className="text-caption text-muted-foreground">{plural(all.length, "item")}</span>
          ) : null
        }
      />
      {!data.ready ? (
        <p className="mt-2 text-dense text-text-secondary">Loading…</p>
      ) : rows.length === 0 ? (
        <p className="mt-2 flex items-center gap-2 py-2 text-body text-text-secondary">
          <StatusMark kind="ok" /> Nothing needs a person right now.
        </p>
      ) : (
        <ul className="-mx-2 mt-1 flex flex-col">
          {rows.map((row) => (
            <li key={row.key}>
              <Popover>
                <PopoverTrigger className="group flex min-h-11 w-full cursor-pointer items-center gap-3 rounded-lg px-2 py-2 text-left transition-colors duration-150 hover:bg-surface-raised focus-visible:outline-2 focus-visible:outline-ring data-[popup-open]:bg-surface-raised">
                  <StatusMark kind={row.mark} size="md" pulse={row.pulse} />
                  <span className="min-w-0 flex-1 text-body leading-snug text-text-secondary [&_b]:font-semibold [&_b]:text-foreground">
                    {row.text}
                  </span>
                  <ChevronRight
                    aria-hidden
                    className="size-4 shrink-0 text-muted-foreground transition-transform duration-150 group-hover:translate-x-0.5 group-hover:text-foreground"
                  />
                </PopoverTrigger>
                <PopoverContent side="right" align="center" sideOffset={10}>
                  <div className="mb-2.5 flex items-center gap-2">
                    <StatusMark kind={row.mark} size="sm" />
                    <PopoverTitle className="section-title">{row.title}</PopoverTitle>
                  </div>
                  <div className="text-caption">
                    <PopList rows={factRows(row.facts)} />
                  </div>
                  <GoButton onClick={row.go}>{row.goLabel}</GoButton>
                </PopoverContent>
              </Popover>
            </li>
          ))}
        </ul>
      )}
    </section>
  );
}

/* ---------- Imbalance glance (#177, #192) ---------- */

function ImbalanceGlance() {
  const go = useGo();
  const { data } = useQuery({ ...orpc.imbalance.getPolicyComparison.queryOptions({ input: {} }), staleTime: Number.POSITIVE_INFINITY });
  const of = (policy: string) => data?.find((r) => r.policy === policy);
  const sentinel = of("sentinel");
  const profile = of("profile");
  const lei = (v?: number) => (v === undefined ? "—" : Math.round(v).toLocaleString("en-US"));
  const delta = deltaOf(profile?.avoidableLei, sentinel?.avoidableLei);
  // One full-width line: title and context left, the number and its change right (#206).
  return (
    <section aria-labelledby="ov-imbalance" className={cn(CARD, "flex flex-wrap items-center gap-x-6 gap-y-3 px-5 py-4")}>
      <div className="min-w-[240px] flex-1">
        <h2 id="ov-imbalance" className="section-title">
          Imbalance cost
        </h2>
        <p className="mt-0.5 text-caption text-muted-foreground">
          {sentinel ? (
            <>
              Avoidable, {Math.round(sentinel.days)} days across the portfolio · a fixed profile costs{" "}
              <span className="font-medium text-text-secondary">{lei(profile?.avoidableLei)} lei</span>
            </>
          ) : (
            "No imbalance replay yet."
          )}
        </p>
      </div>
      <div className="flex items-center gap-2.5">
        {sentinel ? (
          <Num
            value={lei(sentinel.avoidableLei)}
            unit="lei"
            className="text-[24px] leading-none font-semibold tracking-[-0.02em] text-foreground"
          />
        ) : (
          <span className="text-[24px] leading-none font-semibold text-muted-foreground">—</span>
        )}
        {delta ? <DeltaChip delta={delta} /> : null}
      </div>
      <button
        type="button"
        onClick={() => go({ mode: "imbalance", feeder: null, park: null, turbine: null, site: null })}
        className={QUIET_LINK}
      >
        Open imbalance
        <ChevronRight aria-hidden className="size-3.5" />
      </button>
    </section>
  );
}

/* ---------- 5. Agent activity: the last five ---------- */

function when(markerStep: number, now: number) {
  return dayOfStep(markerStep) === dayOfStep(now)
    ? timeLabel(markerStep)
    : `${dayLabel(dayOfStep(markerStep)).split(" ")[0]} ${timeLabel(markerStep)}`;
}

function ActivityRow({ group, now, wide = false }: { group: ActionMarker; now: number; wide?: boolean }) {
  const explain = useExplainAction();
  const lead = group.items[0];
  if (!lead) return null;
  const more = group.items.length - 1;
  return (
    <li
      className={cn(
        "grid items-start gap-x-3 rounded-lg px-2 py-2 text-dense leading-snug transition-colors duration-150 hover:bg-surface-raised",
        wide ? "grid-cols-[44px_12px_1fr]" : "grid-cols-[64px_12px_1fr]",
      )}
    >
      <span className="pt-px font-mono text-caption whitespace-nowrap text-muted-foreground">
        {wide ? timeLabel(group.step) : when(group.step, now)}
      </span>
      <StatusMark kind={ACTION_MARK[group.kind]} size="sm" className="mt-[3px]" label={ACTION_LABEL[group.kind]} />
      <span className="min-w-0 text-text-secondary">
        <span className="text-foreground">{lead.title}</span>
        {more > 0 ? <span className="text-muted-foreground"> (+{more} more)</span> : null}{" "}
        <button
          type="button"
          onClick={() => explain(lead)}
          aria-label={`Why: ${lead.title}`}
          className="cursor-pointer font-medium whitespace-nowrap text-foreground underline decoration-border-strong underline-offset-2 hover:decoration-foreground focus-visible:decoration-foreground focus-visible:outline-none"
        >
          why?
        </button>
      </span>
    </li>
  );
}

function useActivity() {
  const [rawStep] = useReplayStep();
  const step = clampStep(rawStep);
  const { actions, isLoading } = useAgentActions();
  const past = useMemo(() => groupMarkers(actions.filter((a) => a.step <= step)), [actions, step]);
  const byDay = useMemo(() => {
    const days = new Map<number, ActionMarker[]>();
    for (const group of [...past].reverse()) {
      const day = dayOfStep(group.step);
      days.set(day, [...(days.get(day) ?? []), group]);
    }
    return [...days.entries()];
  }, [past]);
  const later = groupMarkers(actions).length - past.length;
  return { step, past, byDay, later, isLoading };
}

function AgentActivity({ onDetails }: { onDetails: OpenDetails }) {
  const [policy] = usePolicy();
  const { step, past, isLoading } = useActivity();
  const latest = past.slice(-TOP_ACTIVITY).reverse();

  return (
    <section aria-labelledby="ov-activity" className={cn(CARD, "h-full px-5 pt-3 pb-3")}>
      <SectionHeader
        title={<span id="ov-activity">Agent activity</span>}
        aside={
          past.length > 0 ? (
            <button type="button" onClick={() => onDetails("activity")} className={QUIET_LINK}>
              See all {past.length}
              <ChevronRight aria-hidden className="size-3.5" />
            </button>
          ) : null
        }
      />
      {policy === "baseline" ? (
        <p className="pb-1 text-caption text-muted-foreground">Showing what Sentinel did; Baseline has no agent.</p>
      ) : null}
      {isLoading ? (
        <p className="mt-1 text-dense text-text-secondary">Loading…</p>
      ) : latest.length === 0 ? (
        <p className="mt-1 text-dense text-text-secondary">No actions yet at this point in the week.</p>
      ) : (
        <ul className="-mx-2 flex flex-col">
          {latest.map((group) => (
            <ActivityRow key={`${group.kind}-${group.step}`} group={group} now={step} />
          ))}
        </ul>
      )}
    </section>
  );
}

/* ---------- Details: everything that is not on the glance ---------- */

function OverviewDetails({ target, onClose }: { target: DetailsTarget | null; onClose: () => void }) {
  // Hooks run while the sheet is closed too, so the page fetches exactly what it did before (#124).
  const data = useOverviewData();
  const attention = useAttention();
  const { step, byDay, later } = useActivity();
  const { data: day } = useOverview();
  const grid = useGrid().data?.hosting;
  const { data: trust } = useTrustBoard();

  // Scroll the requested section into view once the sheet has opened.
  useEffect(() => {
    if (!target) return;
    const timer = window.setTimeout(
      () => document.getElementById(detailsId(target))?.scrollIntoView({ block: "start" }),
      60,
    );
    return () => window.clearTimeout(timer);
  }, [target]);

  const fixed = (v: number | null | undefined, digits: number, unit = "") =>
    v === null || v === undefined ? "—" : `${v.toFixed(digits)}${unit}`;
  const prosumers = grid ? deltaOf(grid.totalWithout, grid.totalWith, true) : null;

  return (
    <DetailsSheet
      open={target !== null}
      onOpenChange={(open) => {
        if (!open) onClose();
      }}
      title="Overview details"
      description={
        <>
          Everything behind the glance, up to <span className="font-mono">{clockLabel(step)}</span>.
        </>
      }
      wide
    >
      <SheetSection title={<span id={detailsId("numbers")} className="scroll-mt-4">Today&apos;s numbers</span>}>
        <div className="text-dense">
          <PopList
            rows={[
              { key: "mwh", label: "Production today", value: fixed(day?.productionMwh, 0, " MWh") },
              { key: "wind", label: "Avg wind today", value: fixed(day?.avgWindKmh, 1, " km/h") },
              { key: "avail", label: "Park availability", value: fixed(day?.availabilityPct, 1, "%") },
              { key: "orders", label: "Open work orders", value: fixed(day?.activeIssues, 0) },
            ]}
          />
        </div>
      </SheetSection>

      <SheetSection
        title={<span id={detailsId("attention")} className="scroll-mt-4">Needs attention</span>}
        aside={<span className="text-caption text-muted-foreground">{plural(attention.length, "item")}</span>}
      >
        {attention.length === 0 ? (
          <p className="text-text-secondary">Nothing needs a person right now.</p>
        ) : (
          <ul className="divide-y divide-border">
            {attention.map((row) => (
              <li key={row.key} className="py-3 first:pt-0">
                <div className="flex items-start gap-2.5">
                  <StatusMark kind={row.mark} size="sm" className="mt-1" />
                  <p className="min-w-0 flex-1 leading-snug text-text-secondary [&_b]:font-semibold [&_b]:text-foreground">
                    {row.text}
                  </p>
                </div>
                <div className="mt-2 pl-5 text-caption">
                  <PopList rows={factRows(row.facts)} />
                </div>
                <button type="button" onClick={row.go} className={cn(QUIET_LINK, "mt-1.5 ml-3.5")}>
                  {row.goLabel}
                  <ChevronRight aria-hidden className="size-3.5" />
                </button>
              </li>
            ))}
          </ul>
        )}
      </SheetSection>

      <SheetSection
        title={
          <span id={detailsId("feeders")} className="scroll-mt-4">
            Minutes over the limit, per feeder
          </span>
        }
        aside={<span className="text-caption text-muted-foreground">{dayLabel(dayOfStep(step))}</span>}
      >
        <table className="w-full text-dense">
          <thead>
            <tr className="text-caption text-muted-foreground">
              <th className="pb-1.5 text-left font-medium">Feeder</th>
              <th className="pb-1.5 text-right font-medium">Limit kW</th>
              <th className="pb-1.5 text-right font-medium">Baseline</th>
              <th className="pb-1.5 text-right font-medium text-foreground">Sentinel</th>
            </tr>
          </thead>
          <tbody className="divide-y divide-border border-t border-border">
            {data.feeders.map((f) => (
              <tr key={f.id}>
                <td className="py-1.5 pr-2 font-medium whitespace-nowrap text-foreground">{f.name}</td>
                <td className="py-1.5 text-right text-text-secondary">{Math.round(f.limitKw)}</td>
                <td className="py-1.5 text-right text-text-secondary">{f.baselineMinutes ?? "—"}</td>
                <td className="py-1.5 text-right font-semibold text-foreground">{f.sentinelMinutes ?? "—"}</td>
              </tr>
            ))}
          </tbody>
        </table>
        {grid ? (
          <div className="flex items-center gap-3 rounded-lg bg-surface-raised px-3 py-2.5">
            <div className="min-w-0 flex-1">
              <p className="text-text-secondary">
                Prosumers the cables can still take{" "}
                <span className="text-muted-foreground">· all feeders, {grid.label}</span>
              </p>
              <p className="mt-0.5 text-caption text-muted-foreground">
                without Sentinel{" "}
                <span className="font-medium text-text-secondary">{Math.round(grid.totalWithout)} homes</span>
              </p>
            </div>
            <Num value={Math.round(grid.totalWith)} unit="homes" className="text-title text-foreground" />
            {prosumers ? <DeltaChip delta={prosumers} /> : null}
          </div>
        ) : null}
        <dl className="space-y-1.5 text-caption leading-snug text-text-secondary">
          <div>
            <dt className="inline font-semibold text-foreground">Baseline</dt>{" "}
            <dd className="inline">is the same week without Sentinel: batteries only store their own solar.</dd>
          </div>
          <div>
            <dt className="inline font-semibold text-foreground">Minutes over</dt>{" "}
            <dd className="inline">count 15-minute steps where export crossed the feeder's limit.</dd>
          </div>
          <div>
            <dt className="inline font-semibold text-foreground">A comfort breach</dt>{" "}
            <dd className="inline">is a home battery dropping below the reserve its owner keeps.</dd>
          </div>
        </dl>
      </SheetSection>

      <SheetSection title={<span id={detailsId("activity")} className="scroll-mt-4">Agent activity</span>}>
        <p className="text-caption text-text-secondary">
          Everything Sentinel did up to <span className="font-mono">{clockLabel(step)}</span>, newest first.{" "}
          {later > 0 ? `${later} more later in the week: play the replay to see them.` : ""}
        </p>
        {byDay.length === 0 ? (
          <p className="text-text-secondary">No actions yet at this point in the week.</p>
        ) : (
          byDay.map(([d, groups]) => (
            <div key={d} className="pt-2">
              <h4 className="sticky top-0 z-[1] bg-card/95 py-1 section-label backdrop-blur-sm">
                {dayLabel(d)} · {plural(groups.length, "action")}
              </h4>
              <ul className="-mx-2 flex flex-col">
                {groups.map((group) => (
                  <ActivityRow key={`${group.kind}-${group.step}`} group={group} now={step} wide />
                ))}
              </ul>
            </div>
          ))
        )}
      </SheetSection>

      {trust?.witnesses.length ? (
        <section id={detailsId("trust")} className="scroll-mt-4 border-t border-border py-5">
          <TrustBoard />
        </section>
      ) : null}
    </DetailsSheet>
  );
}

/* ---------- The page ---------- */

export function OverviewPage({ onTour }: ModePageProps) {
  const go = useGo();
  const [details, setDetails] = useState<DetailsTarget | null>(null);
  return (
    <>
      {/* Two even columns from 880px (one below). Each row is its own grid, so cards in a row share a
          height and nothing leaves a gap. Centred with a 1600px cap; the dock uses the same cap (#206). */}
      <div className="@container mx-auto flex w-full max-w-[1600px] flex-col gap-5">
        <div className="grid grid-cols-1 gap-5 @min-[880px]:grid-cols-2">
          <Briefing onTour={onTour} />
          <MiniMap
            onOpen={() => go({ mode: "feeders", site: null, turbine: null })}
            className="h-[280px] @min-[880px]:h-auto @min-[880px]:min-h-[260px]"
          />
        </div>
        <KpiRow onDetails={setDetails} />
        <div className="grid grid-cols-1 gap-5 @min-[880px]:grid-cols-2">
          <NeedsAttention onDetails={setDetails} />
          <AgentActivity onDetails={setDetails} />
        </div>
        <ImbalanceGlance />
      </div>
      <OverviewDetails target={details} onClose={() => setDetails(null)} />
    </>
  );
}
