"use client";

import { cn } from "@sentinel-app/ui/lib/utils";
import { ChevronRight } from "lucide-react";
import type { ReactNode } from "react";

import { HowItDecides } from "@/components/help";

import { ProvenancePill } from "@/components/provenance-pill";
import { useOverview } from "@/demo/kpi-strip";
import { MODES, type ModeId } from "@/modes/registry";

import { InspectorEmbedded } from "./inspector-embedded";
import { MODE_COPY } from "./mode-copy";

const plural = (n: number, noun: string, many = `${noun}s`) => `${n} ${n === 1 ? noun : many}`;

function useSubline(mode: ModeId) {
  const { data } = useOverview();
  if (!data) return null;
  switch (mode) {
    case "overview":
      return `Romania · ${plural(data.feeders, "feeder")} · ${plural(data.homes, "home")} · ${plural(data.parks, "wind park")}`;
    case "feeders":
      return `${plural(data.feeders, "feeder")} · ${plural(data.homes, "home")} with solar and a battery`;
    case "om":
      return `${plural(data.turbines, "turbine")} · ${plural(data.homes, "home solar site")}`;
    case "production":
      return `${plural(data.parks, "wind park")} · rooftop solar by county`;
    case "demand":
      return "Forecast per county and feeder, today";
  }
}

// Lifts a mode panel's own floating <aside> into the inspector: no second card, no own position.
const PANEL_IN_INSPECTOR =
  "[&>aside]:static [&>aside]:inset-auto [&>aside]:z-auto [&>aside]:h-auto [&>aside]:w-full [&>aside]:max-w-none [&>aside]:overflow-visible [&>aside]:rounded-none [&>aside]:border-0 [&>aside]:shadow-none";

/** The one floating right inspector every mode renders into (prototype .insp). */
export function Inspector({ mode, children, className }: { mode: ModeId; children: ReactNode; className?: string }) {
  const { label: runLabel } = useOverview();
  const copy = MODE_COPY[mode];
  const { icon: Icon } = MODES[mode];
  const subline = useSubline(mode);

  return (
    <aside
      aria-label={`${MODES[mode].label} details`}
      data-tour="inspector"
      className={cn(
        "flex flex-col overflow-hidden rounded-xl border border-border bg-card shadow-[0_1px_2px_rgb(13_13_13/0.05),0_16px_40px_rgb(13_13_13/0.10)]",
        className,
      )}
    >
      <div
        key={mode}
        className="min-h-0 flex-1 overflow-x-hidden overflow-y-auto overscroll-contain [scrollbar-width:thin] motion-safe:animate-in motion-safe:fade-in-0 motion-safe:slide-in-from-right-3 motion-safe:duration-250"
      >
        <header className="flex items-start gap-3 border-b border-border px-5 pt-[18px] pb-3.5">
          <span className="grid size-10 shrink-0 place-items-center rounded-[10px] bg-primary text-primary-foreground">
            <Icon aria-hidden className="size-5" />
          </span>
          <div className="min-w-0 flex-1">
            <h2 className="truncate text-base leading-tight font-semibold text-foreground">{copy.heading}</h2>
            {subline ? <p className="mt-0.5 text-xs leading-snug text-pretty text-text-secondary tabular-nums">{subline}</p> : null}
            {runLabel ? <ProvenancePill label={runLabel} className="mt-1.5" /> : null}
          </div>
        </header>

        {copy.what ? (
          <section className="border-b border-border px-5 pt-3.5 pb-3">
            <p className="text-[13px] leading-snug text-foreground">{copy.what}</p>
            <dl className="mt-2 grid grid-cols-[auto_1fr] gap-x-3 gap-y-1 text-xs leading-snug">
              <dt className="text-muted-foreground">Agent now</dt>
              <dd className="text-text-secondary">{copy.doing}</dd>
              <dt className="text-muted-foreground">Look at</dt>
              <dd className="text-text-secondary">{copy.look}</dd>
            </dl>
          </section>
        ) : null}

        {/* #114: the agent loop for this mode, one click away so the live content stays on top. */}
        <details className="group/how border-b border-border">
          <summary className="flex cursor-pointer list-none items-center gap-2 px-5 py-2.5 text-[13px] font-medium text-foreground transition-colors duration-150 select-none hover:bg-sidebar focus-visible:outline-2 focus-visible:-outline-offset-2 focus-visible:outline-ring [&::-webkit-details-marker]:hidden">
            <ChevronRight
              aria-hidden
              className="size-4 text-muted-foreground transition-transform duration-150 group-open/how:rotate-90"
            />
            How Sentinel decides
            <span className="ml-auto text-[11px] font-normal whitespace-nowrap text-muted-foreground">4 steps</span>
          </summary>
          <HowItDecides mode={mode} title="The agent's loop" className="px-5 pt-1 pb-4" />
        </details>

        <div className={PANEL_IN_INSPECTOR}>
          <InspectorEmbedded runLabel={runLabel ?? null}>{children}</InspectorEmbedded>
        </div>
      </div>
    </aside>
  );
}
