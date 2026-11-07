"use client";

import { Sheet, SheetContent, SheetDescription, SheetHeader, SheetTitle } from "@sentinel-app/ui/components/sheet";
import { cn } from "@sentinel-app/ui/lib/utils";
import { ChevronRight, CircleHelp, Info, Pause, Play } from "lucide-react";
import { parseAsString, useQueryStates } from "nuqs";
import { type ReactNode, useMemo, useRef, useState } from "react";

import { ShaderBanner } from "@/components/brand/shader-banner";
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
import { setPlaying, useReplayPlaying } from "@/demo/replay-bar";
import { clampStep, clockLabel, dayLabel, dayOfStep, HERO_DAY, STEPS_PER_DAY, timeLabel } from "@/demo/replay";
import { usePolicy, useReplayStep } from "@/demo/use-replay";
import type { ModePageProps } from "@/modes/registry";

import { MiniMap } from "./mini-map";
import { type FeederState, useOverviewData } from "./use-overview-data";

// Overview v3 (#124): a page, not a full-screen map. One briefing on the shader banner, a small
// map, then three glance sections whose details open in popovers or a sheet. Same data as before:
// every number from the old Overview inspector is still here, at most two clicks away.

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

function Unit({ children }: { children: ReactNode }) {
  return <span className="ml-1 text-[0.45em] font-medium tracking-normal text-muted-foreground">{children}</span>;
}

function SectionHeading({ children, aside }: { children: ReactNode; aside?: ReactNode }) {
  return (
    <div className="mb-2 flex min-h-6 items-center justify-between gap-3 px-1">
      <h2 className="text-[11px] font-semibold tracking-[0.06em] whitespace-nowrap text-muted-foreground uppercase">
        {children}
      </h2>
      {aside}
    </div>
  );
}

const QUIET_LINK =
  "inline-flex shrink-0 items-center gap-1 rounded-md px-1.5 py-0.5 text-[12px] font-medium whitespace-nowrap text-brand-700 transition-colors duration-150 hover:bg-brand-50 hover:text-brand-900 focus-visible:outline-2 focus-visible:outline-ring data-[popup-open]:bg-brand-50";

function Delta({ from, to, className }: { from?: number; to?: number; className?: string }) {
  if (from === undefined || to === undefined || from === 0) return null;
  const pct = Math.round(((to - from) / Math.abs(from)) * 100);
  if (pct === 0) return null;
  const better = pct < 0;
  return (
    <span
      className={cn(
        "inline-flex h-6 items-center rounded-full px-2 text-[12px] font-semibold whitespace-nowrap tabular-nums shadow-raised",
        better ? "bg-brand-50 text-brand-700" : "bg-card text-text-secondary",
        className,
      )}
    >
      {pct < 0 ? "−" : "+"}
      {Math.abs(pct)}%
    </span>
  );
}

function Facts({ rows }: { rows: [string, ReactNode][] }) {
  return (
    <dl className="mt-2.5 divide-y divide-border border-y border-border">
      {rows.map(([label, value]) => (
        <div key={label} className="flex items-baseline justify-between gap-4 py-1.5">
          <dt className="text-[12px] text-text-secondary">{label}</dt>
          <dd className="text-right text-[13px] font-semibold whitespace-nowrap text-foreground tabular-nums">{value}</dd>
        </div>
      ))}
    </dl>
  );
}

function GoButton({ onClick, children }: { onClick: () => void; children: ReactNode }) {
  return (
    <button
      type="button"
      onClick={onClick}
      className="mt-3 inline-flex h-8 w-full items-center justify-center gap-1 rounded-lg bg-foreground px-3 text-[13px] font-medium text-background shadow-inset-hi transition-[background-color,transform] duration-150 hover:bg-foreground/85 active:scale-[0.98] focus-visible:outline-2 focus-visible:outline-offset-2 focus-visible:outline-ring"
    >
      {children}
      <ChevronRight aria-hidden className="size-3.5" />
    </button>
  );
}

/* ---------- 1. The shift briefing ---------- */

function Briefing({ onTour }: { onTour: () => void }) {
  const data = useOverviewData();
  const [rawStep, setStep] = useReplayStep();
  const [, setPolicy] = usePolicy();
  const playing = useReplayPlaying();
  const { actions } = useAgentActions();
  const step = clampStep(rawStep);

  const over = data.feeders.filter((f) => f.status === "over");
  const near = data.feeders.filter((f) => f.status === "near");
  const within = data.feeders.filter((f) => f.status === "ok" || f.status === "near").length;
  const extras = [
    near.length > 0 ? `${plural(near.length, "feeder")} near ${near.length === 1 ? "its" : "their"} limit` : null,
    data.stopped.length > 0 ? `${plural(data.stopped.length, "turbine")} stopped` : null,
    data.waiting.length > 0 ? `${plural(data.waiting.length, "case")} waiting for a person` : null,
  ].filter((part): part is string => Boolean(part));
  const lead = over.length === 0 ? "Grid is calm." : `${plural(over.length, "feeder")} went over the limit today.`;
  const sentence = data.ready ? `${lead} ${extras.length ? `${listOf(extras)}.` : "Nothing is waiting."}` : "Reading the grid…";

  const today = dayOfStep(step);
  const homes = actions
    .filter((a) => a.kind === "dispatch" && dayOfStep(a.step) === today && a.step <= step)
    .reduce((sum, a) => sum + (a.homes ?? 0), 0);
  const wind = data.parks.reduce((sum, park) => sum + park.mwh, 0);

  const numbers: { value: number; unit: ReactNode; label: ReactNode }[] = [
    {
      value: data.ready ? within : 0,
      unit: `/ ${data.feeders.length || "–"}`,
      label: (
        <>
          <Term id="feeder">feeders</Term> within their limit
        </>
      ),
    },
    { value: homes, unit: "", label: "home batteries in today's plan" },
    { value: wind, unit: "MWh", label: "wind expected, next 24 h" },
  ];

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
    <ShaderBanner
      tone={over.length > 0 ? "alert" : "emerald"}
      className="flex h-full min-h-[300px] flex-col rounded-2xl shadow-floating"
    >
      <div className="flex h-full flex-col px-7 pt-6 pb-6 min-[1440px]:px-8">
        <p className="flex items-center gap-2 text-[12px] font-medium text-foreground/70">
          <StatusMark kind={playing ? "live" : "ok"} size="sm" pulse={playing} />
          <span className="font-mono text-[13px] font-semibold tracking-tight text-foreground tabular-nums">
            {clockLabel(step)}
          </span>
          <span aria-hidden className="text-foreground/30">
            /
          </span>
          {playing ? "Replaying the story" : "Shift briefing"}
        </p>

        <p
          aria-live="polite"
          className="mt-3 max-w-[30ch] text-[26px] leading-[1.15] font-semibold tracking-[-0.02em] text-balance text-foreground min-[1440px]:text-[28px]"
        >
          {sentence}
        </p>

        <dl className="mt-auto flex flex-wrap gap-x-9 gap-y-3 pt-5">
          {numbers.map((n, index) => (
            <div key={index} className="min-w-0">
              <dd className="text-[34px] leading-none font-semibold tracking-[-0.03em] text-foreground tabular-nums">
                <CountUp value={n.value} />
                {n.unit ? <Unit>{n.unit}</Unit> : null}
              </dd>
              <dt className="mt-1.5 text-[12px] whitespace-nowrap text-text-secondary">{n.label}</dt>
            </div>
          ))}
        </dl>

        <div className="mt-5 flex flex-wrap items-center gap-2">
          <button
            type="button"
            onClick={playStory}
            className="inline-flex h-10 items-center gap-2 rounded-xl bg-primary pr-4 pl-3.5 text-[14px] font-semibold text-primary-foreground shadow-brand transition-[background-color,transform] duration-150 hover:bg-brand-700 active:scale-[0.98] focus-visible:outline-2 focus-visible:outline-offset-2 focus-visible:outline-ring"
          >
            {playing ? (
              <Pause aria-hidden className="size-4 fill-current" />
            ) : (
              <Play aria-hidden className="size-4 fill-current" />
            )}
            {playing ? "Pause the story" : "Play today's story"}
          </button>
          <button
            type="button"
            onClick={onTour}
            className="inline-flex h-10 items-center gap-1.5 rounded-xl bg-card/80 px-3.5 text-[14px] font-medium text-foreground shadow-raised backdrop-blur-sm transition-[background-color,box-shadow] duration-150 hover:bg-card hover:shadow-floating focus-visible:outline-2 focus-visible:outline-offset-2 focus-visible:outline-ring"
          >
            <CircleHelp aria-hidden className="size-4 text-text-secondary" />
            How it works
          </button>
          <span className="ml-1 hidden text-[12px] text-text-secondary min-[1440px]:inline">
            from {clockLabel(MORNING_STEP)}, with Sentinel
          </span>
        </div>
      </div>
    </ShaderBanner>
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

function NeedsAttention() {
  const all = useAttention();
  const data = useOverviewData();
  const rows = all.slice(0, 4);
  const hidden = all.length - rows.length;
  return (
    <section aria-labelledby="ov-attention">
      <SectionHeading
        aside={
          <span className="text-[12px] text-muted-foreground tabular-nums">
            {all.length === 0 ? "" : hidden > 0 ? `top 4 of ${all.length}` : plural(all.length, "item")}
          </span>
        }
      >
        <span id="ov-attention">Needs attention</span>
      </SectionHeading>
      {!data.ready ? (
        <p className="px-1 text-[13px] text-text-secondary">Loading…</p>
      ) : rows.length === 0 ? (
        <p className="flex items-center gap-2 px-1 py-2 text-[14px] text-text-secondary">
          <StatusMark kind="ok" /> Nothing needs a person right now.
        </p>
      ) : (
        <ul className="flex flex-col gap-1">
          {rows.map((row) => (
            <li key={row.key}>
              <Popover>
                <PopoverTrigger className="group flex min-h-12 w-full items-center gap-3 rounded-xl px-3 py-2.5 text-left transition-[background-color,box-shadow] duration-150 hover:bg-card hover:shadow-raised focus-visible:outline-2 focus-visible:outline-ring data-[popup-open]:bg-card data-[popup-open]:shadow-raised">
                  <StatusMark kind={row.mark} size="md" pulse={row.pulse} />
                  <span className="min-w-0 flex-1 text-[14px] leading-snug text-text-secondary [&_b]:font-semibold [&_b]:text-foreground">
                    {row.text}
                  </span>
                  <ChevronRight
                    aria-hidden
                    className="size-4 shrink-0 text-muted-foreground transition-transform duration-150 group-hover:translate-x-0.5 group-hover:text-foreground"
                  />
                </PopoverTrigger>
                <PopoverContent side="right" align="center" sideOffset={10}>
                  <div className="flex items-center gap-2">
                    <StatusMark kind={row.mark} size="sm" />
                    <PopoverTitle className="text-[14px] font-semibold">{row.title}</PopoverTitle>
                  </div>
                  <Facts rows={row.facts} />
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

/* ---------- 4. Today vs. without Sentinel ---------- */

function TodayVsWithout() {
  const data = useOverviewData();
  const [rawStep] = useReplayStep();
  const step = clampStep(rawStep);
  const sum = (pick: (f: FeederState) => number | undefined) =>
    data.feeders.every((f) => pick(f) === undefined) ? undefined : data.feeders.reduce((s, f) => s + (pick(f) ?? 0), 0);
  const max = (pick: (f: FeederState) => number | undefined) =>
    data.feeders.every((f) => pick(f) === undefined) ? undefined : Math.max(...data.feeders.map((f) => pick(f) ?? 0));

  const rows: { label: ReactNode; unit: string; sentinel?: number; baseline?: number; note: string }[] = [
    {
      label: (
        <>
          Minutes over the <Term id="feeder-limit">limit</Term>
        </>
      ),
      unit: "min",
      sentinel: sum((f) => f.sentinelMinutes),
      baseline: sum((f) => f.baselineMinutes),
      note: "all feeders, today",
    },
    {
      label: "Highest feeder peak",
      unit: "kW",
      sentinel: max((f) => f.sentinelPeakKw),
      baseline: max((f) => f.baselinePeakKw),
      note: "today",
    },
    {
      label: (
        <>
          <Term id="comfort-reserve">Comfort breaches</Term>
        </>
      ),
      unit: "",
      sentinel: data.comfort.sentinel,
      baseline: data.comfort.baseline,
      note: "this week",
    },
  ];

  return (
    <section aria-labelledby="ov-today">
      <SectionHeading
        aside={
          <Popover>
            <PopoverTrigger className={QUIET_LINK}>
              <Info aria-hidden className="size-3.5" />
              Breakdown
            </PopoverTrigger>
            <PopoverContent side="left" align="start" className="w-[360px]">
              <PopoverTitle className="text-[14px] font-semibold">
                Minutes over the limit, per feeder · {dayLabel(dayOfStep(step))}
              </PopoverTitle>
              <table className="mt-2.5 w-full text-[12px] tabular-nums">
                <thead>
                  <tr className="text-[10px] tracking-[0.05em] text-muted-foreground uppercase">
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
              <dl className="mt-3 space-y-1.5 border-t border-border pt-2.5 text-[12px] leading-snug text-text-secondary">
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
            </PopoverContent>
          </Popover>
        }
      >
        <span id="ov-today">Today vs. without Sentinel</span>
      </SectionHeading>
      <ul className="flex flex-col divide-y divide-border rounded-2xl bg-card px-5 shadow-floating">
        {rows.map((row, index) => (
          <li key={index} className="flex items-center gap-3 py-4">
            <div className="min-w-0 flex-1">
              <p className="text-[12px] text-text-secondary">
                {row.label} <span className="text-muted-foreground">· {row.note}</span>
              </p>
              <p className="mt-1 text-[32px] leading-none font-semibold tracking-[-0.03em] text-foreground tabular-nums">
                {row.sentinel === undefined ? (
                  <span className="text-muted-foreground">—</span>
                ) : (
                  <>
                    <CountUp value={data.ready ? row.sentinel : 0} />
                    {row.unit ? <Unit>{row.unit}</Unit> : null}
                  </>
                )}
              </p>
              <p className="mt-1 text-[12px] text-muted-foreground tabular-nums">
                without Sentinel{" "}
                <span className="font-medium text-text-secondary">
                  {row.baseline === undefined ? "—" : Math.round(row.baseline)}
                  {row.unit ? ` ${row.unit}` : ""}
                </span>
              </p>
            </div>
            <Delta from={row.baseline} to={row.sentinel} />
          </li>
        ))}
      </ul>
    </section>
  );
}

/* ---------- 5. Agent activity ---------- */

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
        "grid items-start gap-x-3 rounded-xl px-3 py-2 text-[13px] leading-snug transition-[background-color,box-shadow] duration-150 hover:bg-card hover:shadow-raised",
        wide ? "grid-cols-[44px_12px_1fr]" : "grid-cols-[64px_12px_1fr]",
      )}
    >
      <span className="pt-px font-mono text-[12px] whitespace-nowrap text-muted-foreground tabular-nums">
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
          className="font-medium whitespace-nowrap text-brand-700 underline-offset-2 hover:underline focus-visible:underline focus-visible:outline-none"
        >
          why?
        </button>
      </span>
    </li>
  );
}

function AgentActivity() {
  const [rawStep] = useReplayStep();
  const [policy] = usePolicy();
  const step = clampStep(rawStep);
  const { actions, isLoading } = useAgentActions();
  const [open, setOpen] = useState(false);
  const past = useMemo(() => groupMarkers(actions.filter((a) => a.step <= step)), [actions, step]);
  const latest = past.slice(-3).reverse();
  const byDay = useMemo(() => {
    const days = new Map<number, ActionMarker[]>();
    for (const group of [...past].reverse()) {
      const day = dayOfStep(group.step);
      days.set(day, [...(days.get(day) ?? []), group]);
    }
    return [...days.entries()];
  }, [past]);
  const later = groupMarkers(actions).length - past.length;

  return (
    <section aria-labelledby="ov-activity">
      <SectionHeading
        aside={
          past.length > 0 ? (
            <button type="button" onClick={() => setOpen(true)} className={QUIET_LINK}>
              See all {past.length}
              <ChevronRight aria-hidden className="size-3.5" />
            </button>
          ) : null
        }
      >
        <span id="ov-activity">Agent activity</span>
      </SectionHeading>
      {policy === "baseline" ? (
        <p className="px-1 pb-1 text-[12px] text-muted-foreground">Showing what Sentinel did; Baseline has no agent.</p>
      ) : null}
      {isLoading ? (
        <p className="px-1 text-[13px] text-text-secondary">Loading…</p>
      ) : latest.length === 0 ? (
        <p className="px-1 text-[13px] text-text-secondary">No actions yet at this point in the week.</p>
      ) : (
        <ul className="flex flex-col">
          {latest.map((group) => (
            <ActivityRow key={`${group.kind}-${group.step}`} group={group} now={step} />
          ))}
        </ul>
      )}

      <Sheet open={open} onOpenChange={setOpen}>
        <SheetContent side="right" className="w-[480px] border-0 bg-card p-0 shadow-overlay sm:max-w-[480px]">
          <SheetHeader className="border-b border-border px-5 pt-5 pb-4">
            <SheetTitle className="text-[16px] font-semibold">Agent activity</SheetTitle>
            <SheetDescription className="text-[12px] text-text-secondary">
              Everything Sentinel did up to <span className="font-mono tabular-nums">{clockLabel(step)}</span>, newest
              first. {later > 0 ? `${later} more later in the week: play the replay to see them.` : ""}
            </SheetDescription>
          </SheetHeader>
          <div className="min-h-0 flex-1 overflow-y-auto overscroll-contain px-2 pb-6 [scrollbar-width:thin]">
            {byDay.map(([day, groups]) => (
              <div key={day} className="pt-4">
                <h3 className="sticky top-0 z-[1] bg-card/95 px-3 py-1 font-mono text-[11px] font-semibold tracking-[0.04em] text-muted-foreground uppercase backdrop-blur-sm">
                  {dayLabel(day)} · {plural(groups.length, "action")}
                </h3>
                <ul className="flex flex-col">
                  {groups.map((group) => (
                    <ActivityRow key={`${group.kind}-${group.step}`} group={group} now={step} wide />
                  ))}
                </ul>
              </div>
            ))}
          </div>
        </SheetContent>
      </Sheet>
    </section>
  );
}

/* ---------- The page ---------- */

export function OverviewPage({ onTour }: ModePageProps) {
  const go = useGo();
  return (
    <div className="@container w-full max-w-[1200px]">
      <div
        className={cn(
          "grid grid-cols-1 gap-x-6 gap-y-7",
          "@min-[880px]:grid-cols-[minmax(0,1fr)_400px] @min-[1100px]:grid-cols-[minmax(0,1fr)_420px]",
        )}
      >
        <div className="@min-[880px]:col-start-1 @min-[880px]:row-start-1">
          <Briefing onTour={onTour} />
        </div>
        <MiniMap
          onOpen={() => go({ mode: "feeders", site: null, turbine: null })}
          className="h-[300px] @min-[880px]:col-start-2 @min-[880px]:row-start-1 @min-[880px]:h-auto @min-[880px]:min-h-[300px]"
        />
        <div className="@min-[880px]:col-start-1 @min-[880px]:row-start-2">
          <NeedsAttention />
        </div>
        <div className="@min-[880px]:col-start-2 @min-[880px]:row-span-2 @min-[880px]:row-start-2">
          <TodayVsWithout />
        </div>
        <div className="@min-[880px]:col-start-1 @min-[880px]:row-start-3">
          <AgentActivity />
        </div>
      </div>
    </div>
  );
}
