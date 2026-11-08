"use client";

import { cn } from "@wattsentry/ui/lib/utils";
import { useQuery } from "@tanstack/react-query";
import { type KeyboardEvent, type PointerEvent, useEffect, useMemo, useRef, useState } from "react";

import { StatusMark, type StatusKind } from "@/components/status-mark";
import { setPlaying, useReplayPlaying, useReplayTicker } from "@/lib/replay-playing";
import {
  clampStep,
  clockLabel,
  DAYS,
  dayLabel,
  dayOfStep,
  DEFAULT_STEP,
  STEPS_IN_WEEK,
  STEPS_PER_DAY,
  stepToTs,
  tsToStep,
} from "@/lib/replay";
import { type Policy, usePolicy, useReplayStep } from "@/lib/use-replay";
import { orpc } from "@/lib/orpc";

import { groupMarkers, useAgentActions } from "./agent-actions";
import { showAgentToast } from "./agent-toast";

// The bottom dock: the week as seven day cards (each says whether a cable went over its limit that day),
// and below them the chosen day as a 24-hour strip with the over-limit hours shaded.

// "Now": where the replay starts and "Back to live" returns.
const NOW_STEP = DEFAULT_STEP;
const NOW_DAY = dayOfStep(NOW_STEP);
const DAY_LAST = STEPS_PER_DAY - 1;
const WARN_HOURS = 3; // up to this many hours over the limit in a day is "watch", beyond it a problem
const HOUR_TICKS = [0, 3, 6, 9, 12, 15, 18, 21, 24];

const isTyping = (target: EventTarget | null) =>
  target instanceof HTMLElement &&
  (target.isContentEditable || ["INPUT", "TEXTAREA", "SELECT", "BUTTON"].includes(target.tagName));

const pctOfDay = (s: number) => `${(s / STEPS_PER_DAY) * 100}%`;
const hoursLabel = (h: number) => (h < 1 ? `${Math.round(h * 60)} min` : `${Math.round(h * 10) / 10} h`);

type Range = { start: number; end: number; feeders: number };

/** Consecutive 15-min steps where at least one cable was over its limit, as [start, end) on the week's step axis. */
function useOverLimitRanges(policy: Policy) {
  const { data } = useQuery({
    ...orpc.feeders.getOverLimitSteps.queryOptions({
      input: { policy, from: stepToTs(0), to: stepToTs(STEPS_IN_WEEK) },
    }),
    staleTime: Number.POSITIVE_INFINITY,
  });
  return useMemo(() => {
    const ranges: Range[] = [];
    for (const row of data ?? []) {
      const step = tsToStep(row.ts);
      const last = ranges.at(-1);
      if (last && last.end === step) {
        last.end = step + 1;
        last.feeders = Math.max(last.feeders, row.feeders);
      } else ranges.push({ start: step, end: step + 1, feeders: row.feeders });
    }
    return ranges;
  }, [data]);
}

const overlap = (r: Range, from: number, to: number) => Math.max(0, Math.min(r.end, to) - Math.max(r.start, from));

function dayStatus(hours: number): { kind: StatusKind; text: string } {
  if (hours === 0) return { kind: "ok", text: "ok" };
  return { kind: hours <= WARN_HOURS ? "warn" : "critical", text: hoursLabel(hours) };
}

function PolicySwitch({ policy, onChange }: { policy: Policy; onChange: (p: Policy) => void }) {
  const options: { value: Policy; label: string }[] = [
    { value: "baseline", label: "Baseline" },
    { value: "sentinel", label: "WattSentry" },
  ];
  return (
    <div role="radiogroup" aria-label="Policy shown" data-tour="policy" className="relative flex rounded-lg bg-surface-raised p-[3px]">
      <span
        aria-hidden
        className={cn(
          "absolute top-[3px] bottom-[3px] w-[calc(50%-3px)] rounded-md bg-card shadow-card transition-[left] duration-250 ease-out motion-reduce:transition-none",
          policy === "baseline" ? "left-[3px]" : "left-1/2",
        )}
      />
      {options.map((option) => (
        <button
          key={option.value}
          type="button"
          role="radio"
          aria-checked={policy === option.value}
          onClick={() => onChange(option.value)}
          className={cn(
            "relative z-[1] flex h-8 flex-1 items-center justify-center gap-1.5 rounded-md px-2 text-dense font-medium whitespace-nowrap transition-colors duration-200 min-[1440px]:px-3",
            policy === option.value ? "text-foreground" : "text-text-secondary hover:text-foreground",
          )}
        >
          <StatusMark kind={option.value === "sentinel" ? "ok" : "unknown"} size="xs" />
          {option.label}
        </button>
      ))}
    </div>
  );
}

/** Seven day buttons: day name and how long a cable was over its limit; days after today read as forecast. */
function DayCards({ ranges, day, onPick }: { ranges: Range[]; day: number; onPick: (day: number) => void }) {
  return (
    <div role="tablist" aria-label="Day" className="grid grid-cols-7 gap-1.5">
      {Array.from({ length: DAYS }, (_, d) => {
        const from = d * STEPS_PER_DAY;
        const hours = ranges.reduce((sum, r) => sum + overlap(r, from, from + STEPS_PER_DAY), 0) / 4;
        const status = dayStatus(hours);
        const forecast = d > NOW_DAY;
        const selected = d === day;
        return (
          <button
            key={d}
            type="button"
            role="tab"
            aria-selected={selected}
            aria-label={`${dayLabel(d)}: ${hours ? `cables over limit for ${hoursLabel(hours)}` : "all cables within limit"}${forecast ? " (forecast)" : ""}`}
            onClick={() => onPick(d)}
            className={cn(
              "flex min-w-0 cursor-pointer items-center justify-between gap-1.5 rounded-lg border px-2 py-1 text-left transition-colors duration-150 focus-visible:outline-2 focus-visible:outline-ring",
              forecast ? "border-dashed" : "",
              selected
                ? "border-foreground bg-card"
                : "border-border bg-surface-raised/40 hover:bg-surface-raised",
            )}
          >
            <span className={cn("truncate text-[12px] leading-tight", selected ? "font-semibold text-foreground" : "text-text-secondary")}>
              <span className="min-[1440px]:hidden">{dayLabel(d).split(" ")[0]}</span>
              <span className="hidden min-[1440px]:inline">{dayLabel(d)}</span>
            </span>
            <span
              className={cn(
                "flex shrink-0 items-center gap-1 text-[11px] leading-tight tabular-nums",
                forecast ? "text-muted-foreground" : "text-text-secondary",
              )}
            >
              <StatusMark kind={status.kind} size="xs" />
              <span className="hidden @min-[640px]:inline">{status.text}</span>
            </span>
          </button>
        );
      })}
    </div>
  );
}

/** The chosen day, 00:00-24:00: over-limit hours shaded red, the future hatched, a line at the time shown. */
function DayStrip({
  ranges,
  step,
  setStep,
}: {
  ranges: Range[];
  step: number;
  setStep: (step: number) => void;
}) {
  const day = dayOfStep(step);
  const from = day * STEPS_PER_DAY;
  const inDay = step - from;
  const strip = useRef<HTMLDivElement>(null);
  const [dragging, setDragging] = useState(false);
  const futureFrom = NOW_STEP - from; // < 0: whole day is future, > 96: whole day is past

  const stepAt = (clientX: number) => {
    const box = strip.current?.getBoundingClientRect();
    if (!box) return step;
    return from + Math.round(Math.min(1, Math.max(0, (clientX - box.left) / box.width)) * DAY_LAST);
  };
  const onPointerDown = (event: PointerEvent<HTMLDivElement>) => {
    event.currentTarget.setPointerCapture(event.pointerId);
    setDragging(true);
    setPlaying(false);
    setStep(stepAt(event.clientX));
  };
  const onPointerMove = (event: PointerEvent<HTMLDivElement>) => {
    if (dragging) setStep(stepAt(event.clientX));
  };
  const onKeyDown = (event: KeyboardEvent<HTMLDivElement>) => {
    const moves: Record<string, number> = {
      ArrowLeft: -1,
      ArrowRight: 1,
      ArrowDown: -1,
      ArrowUp: 1,
      PageDown: -STEPS_PER_DAY,
      PageUp: STEPS_PER_DAY,
    };
    if (event.key in moves) {
      event.preventDefault();
      setStep(clampStep(step + (moves[event.key] ?? 0) * (event.shiftKey ? 4 : 1)));
    } else if (event.key === "Home") setStep(from);
    else if (event.key === "End") setStep(from + DAY_LAST);
  };

  const dayRanges = ranges
    .filter((r) => overlap(r, from, from + STEPS_PER_DAY) > 0)
    .map((r) => ({ ...r, start: Math.max(r.start, from) - from, end: Math.min(r.end, from + STEPS_PER_DAY) - from }));

  return (
    <div className="relative">
      <div
        ref={strip}
        role="slider"
        tabIndex={0}
        aria-label="Time of day"
        aria-valuemin={from}
        aria-valuemax={from + DAY_LAST}
        aria-valuenow={step}
        aria-valuetext={clockLabel(step)}
        onPointerDown={onPointerDown}
        onPointerMove={onPointerMove}
        onPointerUp={() => setDragging(false)}
        onPointerCancel={() => setDragging(false)}
        onKeyDown={onKeyDown}
        className="group relative mt-1.5 h-3 cursor-pointer touch-none rounded-sm outline-none focus-visible:ring-2 focus-visible:ring-ring/40"
      >
        <div className="absolute inset-0 overflow-hidden rounded-sm bg-surface-raised">
          {dayRanges.map((r) => (
            <span
              key={r.start}
              title={`${clockLabel(from + r.start)}: ${r.feeders} ${r.feeders === 1 ? "cable" : "cables"} over limit`}
              className="absolute inset-y-0 bg-status-critical/80"
              style={{ left: pctOfDay(r.start), width: `max(2px, ${pctOfDay(r.end - r.start)})` }}
            />
          ))}
          {futureFrom < STEPS_PER_DAY ? (
            <span
              aria-hidden
              className="absolute inset-y-0 right-0 bg-card/60 [background-image:repeating-linear-gradient(135deg,transparent_0_3px,var(--border)_3px_4px)]"
              style={{ left: pctOfDay(Math.max(0, futureFrom)) }}
            />
          ) : null}
        </div>
        <span
          aria-hidden
          className={cn(
            "absolute -top-1 -bottom-1 -ml-[1.5px] w-[3px] rounded-full bg-foreground transition-transform duration-150 group-hover:scale-x-125 motion-reduce:transition-none",
            dragging && "scale-x-125",
          )}
          style={{ left: pctOfDay(inDay) }}
        />
      </div>
      <div aria-hidden className="relative mt-1 h-3">
        {HOUR_TICKS.map((h) => (
          <span
            key={h}
            className={cn(
              "absolute text-[10px] leading-none text-muted-foreground tabular-nums",
              h === 0 ? "" : h === 24 ? "-translate-x-full" : "-translate-x-1/2",
            )}
            style={{ left: `${(h / 24) * 100}%` }}
          >
            {String(h).padStart(2, "0")}:00
          </span>
        ))}
      </div>
    </div>
  );
}

export function TimelineDock({ className, compact = false }: { className?: string; compact?: boolean }) {
  useReplayTicker();
  const [rawStep, setStep] = useReplayStep();
  const [policy, setPolicy] = usePolicy();
  const playing = useReplayPlaying();
  const step = clampStep(rawStep);
  const { actions } = useAgentActions();
  const markers = useMemo(() => groupMarkers(actions), [actions]);
  const ranges = useOverLimitRanges(policy);
  const agentOn = policy === "sentinel";

  // Crossing an action while the demo plays raises its toast (the last one crossed, if several).
  const previous = useRef(step);
  useEffect(() => {
    const before = previous.current;
    previous.current = step;
    if (!playing || !agentOn || step <= before || step - before > 8) return;
    const crossed = markers.filter((m) => m.step > before && m.step <= step);
    const last = crossed.at(-1);
    const lead = last?.items[0];
    if (lead) showAgentToast(lead, { others: last.items.length - 1, fast: true });
  }, [step, playing, agentOn, markers]);

  // Space plays/pauses the demo anywhere outside a text field or button.
  useEffect(() => {
    const onKey = (event: globalThis.KeyboardEvent) => {
      if (event.code !== "Space" || isTyping(event.target)) return;
      event.preventDefault();
      setPlaying((value) => !value);
    };
    window.addEventListener("keydown", onKey);
    return () => window.removeEventListener("keydown", onKey);
  }, []);

  const live = step === NOW_STEP;
  const status = playing ? "Playing demo" : live ? "Live" : step < NOW_STEP ? "Viewing the past" : "Viewing the forecast";

  return (
    <div
      data-tour="timeline"
      className={cn(
        "flex items-center gap-3 rounded-xl bg-card px-3 py-2 shadow-floating min-[1440px]:gap-4 min-[1440px]:px-4",
        compact && "py-2",
        className,
      )}
    >
      <div className="w-[118px] shrink-0 min-[1440px]:w-[140px]">
        <b className="block font-mono text-[13px] leading-tight font-semibold tracking-tighter whitespace-nowrap text-foreground tabular-nums min-[1440px]:text-[15px]">
          {clockLabel(step)}
        </b>
        <span
          className={cn(
            "flex items-center gap-1.5 text-caption whitespace-nowrap",
            live || playing ? "font-medium text-status-ok" : "text-muted-foreground",
          )}
        >
          <StatusMark kind={live || playing ? "ok" : "unknown"} size="xs" pulse={live || playing} />
          {status}
        </span>
        {!live && !playing ? (
          <button
            type="button"
            onClick={() => setStep(NOW_STEP)}
            className="mt-1 h-6 cursor-pointer rounded-md bg-primary px-2 text-[11px] font-medium whitespace-nowrap text-primary-foreground transition-colors duration-150 hover:bg-primary/85 focus-visible:outline-2 focus-visible:outline-ring"
          >
            Back to live
          </button>
        ) : null}
      </div>

      <div className="@container min-w-0 flex-1 select-none">
        <DayCards
          ranges={ranges}
          day={dayOfStep(step)}
          onPick={(d) => {
            setPlaying(false);
            setStep(d * STEPS_PER_DAY + (step % STEPS_PER_DAY));
          }}
        />
        <DayStrip ranges={ranges} step={step} setStep={setStep} />
      </div>

      <div className="flex shrink-0 flex-col items-end gap-1">
        <PolicySwitch policy={policy} onChange={setPolicy} />
      </div>
    </div>
  );
}
