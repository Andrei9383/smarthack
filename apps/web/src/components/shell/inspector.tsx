"use client";

import { cn } from "@sentinel-app/ui/lib/utils";
import { ChevronRight, Info } from "lucide-react";
import type { ReactNode } from "react";

import { HowItDecides } from "@/components/help";

import { useOverview } from "@/demo/kpi-strip";
import { MODES, type ModeId } from "@/modes/registry";

import { InspectorEmbedded } from "./inspector-embedded";
import { MODE_COPY } from "./mode-copy";
import { Popover, PopoverContent, PopoverTitle, PopoverTrigger } from "./popover";

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
  const copy = MODE_COPY[mode];
  const subline = useSubline(mode);

  return (
    <aside
      aria-label={`${MODES[mode].label} details`}
      data-tour="inspector"
      className={cn(
        "flex flex-col overflow-hidden rounded-xl bg-card shadow-floating",
        className,
      )}
    >
      <div
        key={mode}
        className="min-h-0 flex-1 overflow-x-hidden overflow-y-auto overscroll-contain [scrollbar-width:thin] motion-safe:animate-in motion-safe:fade-in-0 motion-safe:slide-in-from-right-3 motion-safe:duration-250"
      >
        {/* Calm header: the view's title and its data-source pill. The plain-language "what", the
            counts and the agent's loop are one click away, under "About this view". */}
        <header className="flex items-start gap-3 px-5 pt-5 pb-3">
          <div className="min-w-0 flex-1">
            <h2 className="truncate text-title tracking-tight text-foreground">{copy.heading}</h2>
            {copy.what || subline ? (
              <Popover>
                <PopoverTrigger className="mt-1 -ml-1.5 inline-flex items-center gap-1 rounded-md px-1.5 py-0.5 text-caption font-medium text-muted-foreground transition-colors duration-150 hover:bg-surface-raised hover:text-foreground focus-visible:outline-2 focus-visible:outline-ring data-[popup-open]:bg-surface-raised">
                  <Info aria-hidden className="size-3.5" />
                  About this view
                </PopoverTrigger>
                <PopoverContent side="left" align="start" sideOffset={14} className="w-[300px]">
                  <PopoverTitle className="text-dense font-semibold">{copy.heading}</PopoverTitle>
                  {copy.what ? <p className="mt-1.5 text-dense leading-snug text-text-secondary">{copy.what}</p> : null}
                  {subline ? <p className="mt-1.5 text-caption text-muted-foreground tabular-nums">{subline}</p> : null}
                  {copy.doing || copy.look ? (
                    <dl className="mt-3 space-y-2.5 border-t border-border pt-3 text-dense leading-snug">
                      {copy.doing ? (
                        <div>
                          <dt className="section-label">The agent now</dt>
                          <dd className="mt-0.5 text-text-secondary">{copy.doing}</dd>
                        </div>
                      ) : null}
                      {copy.look ? (
                        <div>
                          <dt className="section-label">Look at</dt>
                          <dd className="mt-0.5 text-text-secondary">{copy.look}</dd>
                        </div>
                      ) : null}
                    </dl>
                  ) : null}
                </PopoverContent>
              </Popover>
            ) : null}
          </div>
        </header>

        <div className={PANEL_IN_INSPECTOR}>
          <InspectorEmbedded>{children}</InspectorEmbedded>
        </div>

        {/* #114: the agent loop for this mode, folded under the live content. */}
        <details className="group/how mx-5 mt-2 mb-4 border-t border-border">
          <summary className="-mx-2 flex cursor-pointer list-none items-center gap-2 rounded-lg px-2 py-3 text-dense font-medium text-foreground transition-colors duration-150 select-none hover:bg-surface-raised focus-visible:outline-2 focus-visible:-outline-offset-2 focus-visible:outline-ring [&::-webkit-details-marker]:hidden">
            <ChevronRight
              aria-hidden
              className="size-4 text-muted-foreground transition-transform duration-150 group-open/how:rotate-90"
            />
            How Sentinel decides
            <span className="ml-auto text-caption font-normal whitespace-nowrap text-muted-foreground">4 steps</span>
          </summary>
          <HowItDecides mode={mode} title="The agent's loop" className="pt-1 pb-2" />
        </details>
      </div>
    </aside>
  );
}
