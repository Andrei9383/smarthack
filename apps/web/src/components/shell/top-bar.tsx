"use client";

import { cn } from "@sentinel-app/ui/lib/utils";

import { DemoPopover } from "@/demo/demo-jumps";
import { KpiChips } from "@/demo/kpi-strip";
import { MODES, type ModeId } from "@/modes/registry";

import { MODE_COPY } from "./mode-copy";
import { ProvenanceBadge } from "./provenance-badge";

const FLOAT = "rounded-xl border border-border bg-card/95 shadow-card";

/** Floating top bar: mode title + plain descriptor, KPI chips, provenance, the demo script. */
export function TopBar({ mode, className }: { mode: ModeId; className?: string }) {
  const copy = MODE_COPY[mode];
  return (
    <header className={cn("pointer-events-none flex items-stretch gap-2.5 [&>*]:pointer-events-auto", className)}>
      <div className={cn(FLOAT, "flex min-w-0 shrink flex-col justify-center px-4 py-2")}>
        <h1 className="text-base leading-tight font-semibold whitespace-nowrap text-foreground">{MODES[mode].label}</h1>
        <p className="truncate text-xs text-text-secondary">{copy.descriptor}</p>
      </div>
      <div className={cn(FLOAT, "flex min-w-0 shrink")}>
        <KpiChips />
      </div>
      <div className="ml-auto flex shrink-0 items-stretch gap-2.5">
        <ProvenanceBadge sources={copy.sources} />
        <DemoPopover />
      </div>
    </header>
  );
}
