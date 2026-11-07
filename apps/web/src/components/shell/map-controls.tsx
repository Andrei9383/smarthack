"use client";

import { useMap } from "@sentinel-app/ui/components/map";
import { cn } from "@sentinel-app/ui/lib/utils";
import { Maximize, Minus, Plus } from "lucide-react";
import type { Map as MapLibreMap } from "maplibre-gl";
import type { ReactNode } from "react";

export const ROMANIA_BOUNDS: [[number, number], [number, number]] = [
  [20.26, 43.62],
  [29.7, 48.27],
];

/** Fits all of Romania into the map area the floating surfaces leave free. */
export function fitRomania(map: MapLibreMap, duration = 400) {
  const width = map.getContainer().clientWidth;
  const want = { top: 92, bottom: 132, left: width >= 1000 ? 196 : 24, right: width >= 1000 ? 452 : 24 };
  // fitBounds adds its padding on top of the map's own (modes keep a right padding for the
  // inspector), so pass only what is still missing.
  const has = map.getPadding();
  const missing = (side: keyof typeof want) => Math.max(0, want[side] - (has[side] ?? 0));
  map.fitBounds(ROMANIA_BOUNDS, {
    padding: { top: missing("top"), bottom: missing("bottom"), left: missing("left"), right: missing("right") },
    duration: window.matchMedia("(prefers-reduced-motion: reduce)").matches ? 0 : duration,
  });
}

function ControlButton({ label, onClick, children }: { label: string; onClick: () => void; children: ReactNode }) {
  return (
    <button
      type="button"
      aria-label={label}
      title={label}
      onClick={onClick}
      className="grid size-[34px] place-items-center border-b border-border text-text-secondary transition-colors duration-150 last:border-b-0 hover:bg-sidebar hover:text-foreground focus-visible:outline-2 focus-visible:-outline-offset-2 focus-visible:outline-ring"
    >
      {children}
    </button>
  );
}

/** Zoom in, zoom out, fit Romania: a stacked card beside the inspector, above the dock. */
export function ShellMapControls({ className }: { className?: string }) {
  const { map } = useMap();
  return (
    <div
      className={cn(
        "flex flex-col overflow-hidden rounded-xl border border-border bg-card/95 shadow-card",
        className,
      )}
    >
      <ControlButton label="Zoom in" onClick={() => map?.zoomTo(map.getZoom() + 1, { duration: 300 })}>
        <Plus aria-hidden className="size-4" />
      </ControlButton>
      <ControlButton label="Zoom out" onClick={() => map?.zoomTo(map.getZoom() - 1, { duration: 300 })}>
        <Minus aria-hidden className="size-4" />
      </ControlButton>
      <ControlButton label="Show all of Romania" onClick={() => map && fitRomania(map)}>
        <Maximize aria-hidden className="size-4" />
      </ControlButton>
    </div>
  );
}
