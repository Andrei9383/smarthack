"use client";

import { parseAsBoolean, parseAsInteger, parseAsString, useQueryStates } from "nuqs";
import { useCallback, useEffect } from "react";

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
    note: "WattSentry keeps F7 Oșorhei under its limit at noon",
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

const STEP_COUNT = JUMPS.length + 1; // the six jumps, then the summary

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

/**
 * Keys 1–6 run the demo script anywhere (the command palette sends the same keys). No button: the
 * top bar stays clean (#206); the steps are listed in the command palette under "Demo script".
 */
export function DemoKeys() {
  const runStep = useRunStep();
  useEffect(() => {
    const onKey = (event: KeyboardEvent) => {
      if (event.metaKey || event.ctrlKey || event.altKey || isTyping(event.target)) return;
      const digit = Number(event.key);
      if (Number.isInteger(digit) && digit >= 1 && digit <= STEP_COUNT) {
        event.preventDefault();
        runStep(digit - 1);
      }
    };
    window.addEventListener("keydown", onKey);
    return () => window.removeEventListener("keydown", onKey);
  }, [runStep]);
  return null;
}
