"use client";

import { cn } from "@sentinel-app/ui/lib/utils";
import { Search } from "lucide-react";

import { DemoPopover } from "@/demo/demo-jumps";
import { KpiChips } from "@/demo/kpi-strip";
import { MODES, type ModeId } from "@/modes/registry";

import { useShortcutLabel } from "./command-palette";
import { MODE_COPY } from "./mode-copy";
import { ProvenanceBadge } from "./provenance-badge";

const FLOAT = "rounded-xl bg-card/95 shadow-floating backdrop-blur-sm";
// The Demo button lives in demo/; lift it onto the same elevation as the rest of the bar.
const DEMO_ELEVATION =
  "[&>[data-tour=demo]>button]:border-transparent [&>[data-tour=demo]>button]:shadow-floating [&>[data-tour=demo]>[role=dialog]]:border-0 [&>[data-tour=demo]>[role=dialog]]:shadow-overlay";

/** Floating top bar: mode title + plain descriptor, KPI chips, Jump to (⌘K), provenance, the demo script. */
export function TopBar({ mode, onSearch, className }: { mode: ModeId; onSearch: () => void; className?: string }) {
  const copy = MODE_COPY[mode];
  const shortcut = useShortcutLabel();
  return (
    <header className={cn("pointer-events-none flex items-stretch gap-2.5 [&>*]:pointer-events-auto", className)}>
      <div className={cn(FLOAT, "flex min-w-0 shrink flex-col justify-center px-4 py-2")}>
        <h1 className="text-base leading-tight font-semibold whitespace-nowrap text-foreground">{MODES[mode].label}</h1>
        <p className="truncate text-xs text-text-secondary">{copy.descriptor}</p>
      </div>
      <div data-tour="kpis" className={cn(FLOAT, "flex min-w-0 shrink")}>
        <KpiChips />
      </div>
      <div className={cn("ml-auto flex shrink-0 items-stretch gap-2.5", DEMO_ELEVATION)}>
        <button
          type="button"
          onClick={onSearch}
          aria-keyshortcuts="Meta+K Control+K"
          aria-label="Jump to a feeder, turbine, county or view"
          className={cn(
            FLOAT,
            "group flex items-center gap-2 pr-2 pl-3 text-[13px] text-text-secondary transition-[color,box-shadow] duration-150 hover:text-foreground hover:shadow-overlay focus-visible:outline-2 focus-visible:outline-offset-2 focus-visible:outline-ring",
          )}
        >
          <Search aria-hidden className="size-4" />
          <span className="hidden min-[1440px]:inline">Search</span>
          <kbd className="ml-1 rounded-md bg-muted px-1.5 py-0.5 font-mono text-[11px] font-medium text-muted-foreground shadow-inset-hi">
            {shortcut}
          </kbd>
        </button>
        <ProvenanceBadge sources={copy.sources} />
        <DemoPopover />
      </div>
    </header>
  );
}
