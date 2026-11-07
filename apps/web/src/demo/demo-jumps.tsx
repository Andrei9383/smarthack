"use client";

import { Button } from "@sentinel-app/ui/components/button";
import { parseAsBoolean, parseAsInteger, parseAsString, useQueryStates } from "nuqs";

import { NOON_STEP, STEPS_PER_DAY } from "./replay";

const EVENING_STEP = 76; // 19:00

// Demo script, plan §7. String failure at site 1042 is on day 4 (#31).
const JUMPS = [
  { label: "1 Production", state: { mode: "production", park: "WP01", t: NOON_STEP } },
  { label: "2 Demand", state: { mode: "demand", t: EVENING_STEP } },
  { label: "3 Feeders F7", state: { mode: "feeders", feeder: "F7", policy: "sentinel", t: NOON_STEP } },
  {
    label: "4 String failure, 1042",
    state: { mode: "feeders", feeder: "F7", site: "1042", t: 3 * STEPS_PER_DAY + NOON_STEP },
  },
  { label: "5 O&M park", state: { mode: "om", park: "WP01", t: 4 * STEPS_PER_DAY + NOON_STEP } },
] as const;

const PARSERS = {
  mode: parseAsString,
  feeder: parseAsString,
  park: parseAsString,
  site: parseAsString,
  policy: parseAsString,
  t: parseAsInteger,
  summary: parseAsBoolean,
};

export function useSummaryOpen() {
  const [{ summary }, set] = useQueryStates({ summary: PARSERS.summary });
  return [Boolean(summary), (open: boolean) => set({ summary: open || null })] as const;
}

export function DemoJumps() {
  const [, setState] = useQueryStates(PARSERS);
  const [summaryOpen, setSummaryOpen] = useSummaryOpen();

  return (
    <div className="flex flex-wrap items-center gap-1 border-t px-2 py-1">
      <span className="mr-1 text-xs text-muted-foreground">Demo</span>
      {JUMPS.map(({ label, state }) => (
        <Button
          key={label}
          variant="ghost"
          size="sm"
          onClick={() => setState({ site: null, summary: null, ...state })}
        >
          {label}
        </Button>
      ))}
      <Button
        variant={summaryOpen ? "default" : "ghost"}
        size="sm"
        onClick={() => setSummaryOpen(!summaryOpen)}
      >
        6 Summary
      </Button>
    </div>
  );
}
