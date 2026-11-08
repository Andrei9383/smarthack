"use client";

import { cn } from "@sentinel-app/ui/lib/utils";
import type { ReactNode } from "react";

import { MODES, type ModeId } from "@/modes/registry";

import { InspectorEmbedded } from "./inspector-embedded";

// Lifts a mode panel's own floating <aside> into the inspector: no second card, no own position.
const PANEL_IN_INSPECTOR =
  "[&>aside]:static [&>aside]:inset-auto [&>aside]:z-auto [&>aside]:h-auto [&>aside]:w-full [&>aside]:max-w-none [&>aside]:overflow-visible [&>aside]:rounded-none [&>aside]:border-0 [&>aside]:shadow-none";

/** The one floating right inspector every mode renders into (prototype .insp). */
export function Inspector({ mode, children, className }: { mode: ModeId; children: ReactNode; className?: string }) {
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
        className="min-h-0 flex-1 overflow-x-hidden overflow-y-auto overscroll-contain pt-2 pb-3 [scrollbar-width:thin] motion-safe:animate-in motion-safe:fade-in-0 motion-safe:slide-in-from-right-3 motion-safe:duration-250"
      >
        {/* The panel starts with the mode's own content: no title, "About this view" or agent loop. */}
        <div className={PANEL_IN_INSPECTOR}>
          <InspectorEmbedded>{children}</InspectorEmbedded>
        </div>
      </div>
    </aside>
  );
}
