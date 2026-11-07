"use client";

import { Button } from "@sentinel-app/ui/components/button";
import { parseAsBoolean, parseAsInteger, parseAsString, useQueryStates } from "nuqs";

import { DEFAULT_STEP, FAULT_DAY, NOON_STEP, STEPS_PER_DAY } from "./replay";

const EVENING_STEP = DEFAULT_STEP - NOON_STEP + 76; // 19:00 on the hero day
const FAULT_STEP = FAULT_DAY * STEPS_PER_DAY + NOON_STEP;

// Demo script, plan §7. The story days come from #30.
const JUMPS = [
  { label: "1 Production", state: { mode: "production", park: "WP01", t: DEFAULT_STEP } },
  { label: "2 Demand", state: { mode: "demand", t: EVENING_STEP } },
  { label: "3 Feeders F7", state: { mode: "feeders", feeder: "F7", policy: "sentinel", t: DEFAULT_STEP } },
  {
    label: "4 String failure, 1042",
    state: { mode: "feeders", feeder: "F7", site: "1042", t: FAULT_STEP },
  },
  { label: "5 O&M park", state: { mode: "om", park: "WP01", t: FAULT_STEP + STEPS_PER_DAY } },
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
    <div className="flex shrink-0 flex-wrap items-center gap-2 border-t border-border bg-sidebar px-4 py-2">
      <span className="mr-2 text-[11px] font-medium tracking-[0.04em] text-muted-foreground uppercase">Demo</span>
      {JUMPS.map(({ label, state }) => (
        <Button
          key={label}
          variant="outline"
          size="sm"
          onClick={() => setState({ site: null, summary: null, ...state })}
          className={JUMP_CLASS}
        >
          <JumpLabel label={label} />
        </Button>
      ))}
      <Button
        variant="outline"
        size="sm"
        aria-pressed={summaryOpen}
        onClick={() => setSummaryOpen(!summaryOpen)}
        className={
          summaryOpen
            ? `${JUMP_CLASS} border-brand-100 bg-brand-50 text-brand-900 hover:bg-brand-100 hover:text-brand-900 [&>span:first-child]:text-brand-700`
            : JUMP_CLASS
        }
      >
        <JumpLabel label="6 Summary" />
      </Button>
    </div>
  );
}

const JUMP_CLASS =
  "h-8 gap-1.5 rounded-md bg-background px-3 text-[13px] text-text-secondary hover:bg-muted hover:text-foreground";

/** "1 Production" → a muted step number followed by the label; the text itself is unchanged. */
function JumpLabel({ label }: { label: string }) {
  const space = label.indexOf(" ");
  return (
    <>
      <span className="font-semibold text-muted-foreground tabular-nums">{label.slice(0, space)}</span>{" "}
      <span>{label.slice(space + 1)}</span>
    </>
  );
}
