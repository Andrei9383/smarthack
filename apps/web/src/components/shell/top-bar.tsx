"use client";

import { cn } from "@wattsentry/ui/lib/utils";
import { Search } from "lucide-react";

import { DemoKeys } from "@/components/shell/demo-jumps";
import { MODES, type ModeId } from "@/modes/registry";

import { useShortcutLabel } from "./command-palette";
import { MODE_COPY } from "./mode-copy";
import { ThemeToggle } from "./theme-toggle";

const FLOAT = "rounded-xl bg-card shadow-floating";

/**
 * Floating top bar (Calm): the page title and one plain line under it, then search (⌘K) and the theme
 * switch. No numbers, no demo button, no data-source pill (#206): each view shows its own few KPIs and
 * the inspector keeps its provenance pill. Keys 1–6 still run the demo script.
 */
export function TopBar({ mode, onSearch, className }: { mode: ModeId; onSearch: () => void; className?: string }) {
  const copy = MODE_COPY[mode];
  const shortcut = useShortcutLabel();
  return (
    <header className={cn("pointer-events-none flex items-stretch gap-2.5 [&>*]:pointer-events-auto", className)}>
      <div className={cn(FLOAT, "flex min-w-0 shrink flex-col justify-center px-5 py-2")}>
        <h1 className="text-title leading-tight tracking-tight whitespace-nowrap text-foreground">{MODES[mode].label}</h1>
        <p className="truncate text-caption text-muted-foreground">{copy.descriptor}</p>
      </div>
      <div className="ml-auto flex shrink-0 items-stretch gap-2.5">
        <button
          type="button"
          onClick={onSearch}
          aria-keyshortcuts="Meta+K Control+K"
          aria-label="Jump to a feeder, turbine, county or view"
          className={cn(
            FLOAT,
            "group flex min-w-[220px] items-center gap-2 pr-2 pl-3.5 text-dense text-muted-foreground transition-[color,box-shadow,background-color] duration-fast hover:bg-surface-raised hover:text-foreground focus-visible:outline-2 focus-visible:outline-offset-2 focus-visible:outline-ring",
          )}
        >
          <Search aria-hidden className="size-4" />
          <span>Search feeders, turbines, counties</span>
          <kbd className="ml-auto rounded-md border border-border bg-surface-raised px-1.5 py-0.5 font-mono text-[11px] font-medium text-muted-foreground">
            {shortcut}
          </kbd>
        </button>
        <ThemeToggle className={FLOAT} />
        <DemoKeys />
      </div>
    </header>
  );
}
