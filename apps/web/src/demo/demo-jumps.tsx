"use client";

import {
  DropdownMenu,
  DropdownMenuContent,
  DropdownMenuGroup,
  DropdownMenuItem,
  DropdownMenuLabel,
  DropdownMenuSeparator,
  DropdownMenuTrigger,
} from "@sentinel-app/ui/components/dropdown-menu";
import { Check, ChevronUp, Presentation } from "lucide-react";
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

  // A compact menu at the right end of the floating bottom dock (app-shell).
  return (
    <DropdownMenu>
      <DropdownMenuTrigger
        className={`inline-flex h-8 shrink-0 items-center gap-1.5 rounded-md border px-3 text-[13px] font-medium transition-colors duration-150 outline-none focus-visible:ring-2 focus-visible:ring-ring/50 data-popup-open:bg-muted ${
          summaryOpen
            ? "border-brand-100 bg-brand-50 text-brand-900 hover:bg-brand-100"
            : "border-border bg-background text-text-secondary hover:bg-muted hover:text-foreground"
        }`}
      >
        <Presentation aria-hidden className="size-4" />
        Demo
        <ChevronUp aria-hidden className="size-3.5 text-muted-foreground" />
      </DropdownMenuTrigger>
      <DropdownMenuContent
        side="top"
        align="end"
        sideOffset={12}
        className="w-60 rounded-lg border border-border p-1 shadow-card ring-0"
      >
        <DropdownMenuGroup>
          <DropdownMenuLabel className="px-2 pt-1.5 pb-1 text-[11px] font-medium tracking-[0.04em] text-muted-foreground uppercase">
            Demo
          </DropdownMenuLabel>
          {JUMPS.map(({ label, state }) => (
            <DropdownMenuItem
              key={label}
              onClick={() => setState({ site: null, summary: null, ...state })}
              className={ITEM_CLASS}
            >
              <JumpLabel label={label} />
            </DropdownMenuItem>
          ))}
        </DropdownMenuGroup>
        <DropdownMenuSeparator className="my-1 bg-border" />
        <DropdownMenuItem onClick={() => setSummaryOpen(!summaryOpen)} className={ITEM_CLASS}>
          <JumpLabel label="6 Summary" />
          {summaryOpen ? <Check aria-hidden className="ml-auto size-4 text-primary" /> : null}
        </DropdownMenuItem>
      </DropdownMenuContent>
    </DropdownMenu>
  );
}

const ITEM_CLASS =
  "gap-2.5 rounded-md px-2 py-2 text-[13px] text-foreground focus:bg-brand-50 focus:text-brand-900 focus:**:text-brand-900";

/** "1 Production" → a muted step number followed by the label; the text itself is unchanged. */
function JumpLabel({ label }: { label: string }) {
  const space = label.indexOf(" ");
  return (
    <>
      <span className="w-3 text-center font-semibold text-muted-foreground tabular-nums">{label.slice(0, space)}</span>{" "}
      <span>{label.slice(space + 1)}</span>
    </>
  );
}
