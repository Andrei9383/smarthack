"use client";

import { cn } from "@sentinel-app/ui/lib/utils";
import { ChevronRight, Play } from "lucide-react";
import { parseAsBoolean, parseAsInteger, parseAsString, useQueryStates } from "nuqs";
import { useCallback, useEffect, useRef, useState } from "react";

import { setPlaying } from "./replay-bar";
import { DEFAULT_STEP, FAULT_DAY, NOON_STEP, STEPS_PER_DAY } from "./replay";

const EVENING_STEP = DEFAULT_STEP - NOON_STEP + 76; // 19:00 on the hero day
const FAULT_STEP = FAULT_DAY * STEPS_PER_DAY + NOON_STEP;

// Demo script, plan §7. The story days come from #30. Step 6 opens the summary instead of a jump.
const JUMPS = [
  {
    title: "Production",
    note: "Wind over Romania and the 24 h forecast for WP01",
    state: { mode: "production", park: "WP01", t: DEFAULT_STEP },
  },
  {
    title: "Demand",
    note: "Where use peaks this evening, 19:00",
    state: { mode: "demand", t: EVENING_STEP },
  },
  {
    title: "Feeders F7",
    note: "Sentinel keeps F7 Oșorhei under its limit at noon",
    state: { mode: "feeders", feeder: "F7", policy: "sentinel", t: DEFAULT_STEP },
  },
  {
    title: "String failure, 1042",
    note: "A broken solar string becomes a repair job",
    state: { mode: "feeders", feeder: "F7", site: "1042", t: FAULT_STEP },
  },
  {
    title: "O&M park",
    note: "WP01 turbines, crews and the review queue",
    state: { mode: "om", park: "WP01", t: FAULT_STEP + STEPS_PER_DAY },
  },
] as const;

const SUMMARY_STEP = { title: "Summary", note: "The week, Sentinel against the baseline" };
const STEP_COUNT = JUMPS.length + 1;

const PARSERS = {
  mode: parseAsString,
  feeder: parseAsString,
  park: parseAsString,
  site: parseAsString,
  zone: parseAsString,
  turbine: parseAsString,
  policy: parseAsString,
  t: parseAsInteger,
  summary: parseAsBoolean,
};

export function useSummaryOpen() {
  const [{ summary }, set] = useQueryStates({ summary: PARSERS.summary });
  return [Boolean(summary), (open: boolean) => set({ summary: open || null })] as const;
}

const isTyping = (target: EventTarget | null) =>
  target instanceof HTMLElement &&
  (target.isContentEditable || ["INPUT", "TEXTAREA", "SELECT"].includes(target.tagName));

/** Runs demo step `index` (0-based) and returns the next step to suggest. */
function useRunStep() {
  const [, setState] = useQueryStates(PARSERS);
  const [summaryOpen, setSummaryOpen] = useSummaryOpen();
  return useCallback(
    (index: number) => {
      setPlaying(false);
      const jump = JUMPS[index];
      // Tudor's ?zone clearing stays: a jump must never inherit a selected demand zone.
      if (jump) setState({ site: null, turbine: null, summary: null, zone: null, ...jump.state });
      else setSummaryOpen(!summaryOpen);
    },
    [setState, setSummaryOpen, summaryOpen],
  );
}

/** "Demo 1/6" in the top bar, opening the numbered script (keys 1–6 work anywhere). */
export function DemoPopover() {
  const [open, setOpen] = useState(false);
  const [next, setNext] = useState(0);
  const [summaryOpen] = useSummaryOpen();
  const runStep = useRunStep();
  const root = useRef<HTMLDivElement>(null);

  const run = useCallback(
    (index: number) => {
      runStep(index);
      setNext(Math.min(STEP_COUNT - 1, index + 1));
      setOpen(false);
    },
    [runStep],
  );

  useEffect(() => {
    const onKey = (event: KeyboardEvent) => {
      if (event.metaKey || event.ctrlKey || event.altKey || isTyping(event.target)) return;
      const digit = Number(event.key);
      if (Number.isInteger(digit) && digit >= 1 && digit <= STEP_COUNT) {
        event.preventDefault();
        run(digit - 1);
      } else if (event.key === "Escape") setOpen(false);
    };
    const onPointer = (event: PointerEvent) => {
      if (root.current && !root.current.contains(event.target as Node)) setOpen(false);
    };
    window.addEventListener("keydown", onKey);
    window.addEventListener("pointerdown", onPointer);
    return () => {
      window.removeEventListener("keydown", onKey);
      window.removeEventListener("pointerdown", onPointer);
    };
  }, [run]);

  const steps = [...JUMPS.map(({ title, note }) => ({ title, note })), SUMMARY_STEP];

  return (
    <div ref={root} data-tour="demo" className="relative flex">
      <button
        type="button"
        aria-expanded={open}
        aria-haspopup="dialog"
        onClick={() => setOpen((value) => !value)}
        className={cn(
          "flex items-center gap-2 rounded-xl bg-card px-3.5 text-dense font-semibold text-foreground shadow-floating transition-colors duration-150 hover:bg-surface-raised focus-visible:outline-2 focus-visible:outline-offset-2 focus-visible:outline-ring",
          open && "bg-surface-raised",
        )}
      >
        <Play aria-hidden className="size-4" />
        Demo
        <span className="font-mono text-[11px] font-medium text-muted-foreground tabular-nums">
          {next + 1}/{STEP_COUNT}
        </span>
      </button>

      <div
        role="dialog"
        aria-label="Demo script"
        className={cn(
          "absolute top-[calc(100%+8px)] right-0 z-50 w-[340px] origin-top-right rounded-xl border border-border bg-card p-2 shadow-overlay transition-[opacity,transform] duration-200 ease-out motion-reduce:transition-none",
          open ? "translate-y-0 opacity-100" : "pointer-events-none -translate-y-1.5 opacity-0",
        )}
      >
        <header className="flex items-center justify-between px-2.5 pt-2 pb-1.5">
          <b className="text-[13px] font-semibold">Demo script</b>
          <span className="text-[11px] font-semibold tracking-[0.06em] text-muted-foreground uppercase">keys 1–6</span>
        </header>
        {steps.map((step, index) => {
          const current = index === next;
          const done = index < next || (index === STEP_COUNT - 1 && summaryOpen);
          return (
            <button
              key={step.title}
              type="button"
              tabIndex={open ? 0 : -1}
              onClick={() => run(index)}
              className={cn(
                "flex w-full items-center gap-3 rounded-lg px-2.5 py-2 text-left transition-colors duration-150",
                current ? "bg-surface-raised" : "hover:bg-surface-raised",
              )}
            >
              <span
                className={cn(
                  "grid size-6 shrink-0 place-items-center rounded-md border font-mono text-xs font-semibold",
                  current
                    ? "border-primary bg-primary text-primary-foreground"
                    : done
                      ? "border-status-ok/30 bg-card text-status-ok"
                      : "border-border bg-card text-foreground",
                )}
              >
                {index + 1}
              </span>
              <span className="min-w-0">
                <b className="block text-[13px] font-semibold text-foreground">{step.title}</b>
                <span className="block truncate text-xs text-muted-foreground">{step.note}</span>
              </span>
            </button>
          );
        })}
        <div className="mt-1.5 flex items-center justify-between border-t border-border px-2.5 pt-2 pb-1">
          <span className="text-xs text-muted-foreground">Press 1–6 anywhere</span>
          <button
            type="button"
            tabIndex={open ? 0 : -1}
            onClick={() => run(next)}
            className="inline-flex h-8 items-center gap-1 rounded-lg bg-primary px-3 text-[13px] font-medium text-primary-foreground transition-colors duration-150 hover:bg-primary/85"
          >
            Run step {next + 1}
            <ChevronRight aria-hidden className="size-4" />
          </button>
        </div>
      </div>
    </div>
  );
}
