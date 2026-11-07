"use client";

import { Map } from "@sentinel-app/ui/components/map";
import { cn } from "@sentinel-app/ui/lib/utils";
import { ChevronRight } from "lucide-react";

import { StatusMark } from "@/components/brand/status-mark";
import { useBasemapStyles, useRomaniaFocus } from "@/demo/use-basemap";

import { OverviewLayers } from "./overview-layers";

const ROMANIA_CENTER: [number, number] = [25, 45.9];

function RomaniaFocus() {
  useRomaniaFocus();
  return null;
}

/**
 * The Overview's map, kept small on purpose: all of Romania with every asset, fixed in place (no
 * drag or zoom, so the page scrolls normally). Clicking a feeder chip, the park or a county opens
 * that mode with the asset selected; "Open map" opens Feeders, the busiest map.
 */
export function MiniMap({ onOpen, className }: { onOpen: () => void; className?: string }) {
  const { offline, styles } = useBasemapStyles();
  return (
    <section
      aria-label="Map of Romania with every asset"
      data-tour="map"
      className={cn("relative flex flex-col overflow-hidden rounded-2xl bg-card shadow-floating", className)}
    >
      <header className="flex items-baseline justify-between gap-3 px-4 pt-3 pb-2">
        <h2 className="text-[13px] font-semibold text-foreground">Romania, every asset</h2>
        <button
          type="button"
          onClick={onOpen}
          className="inline-flex items-center gap-0.5 rounded-md text-[13px] font-medium text-brand-700 transition-colors duration-150 hover:text-brand-900 hover:underline focus-visible:outline-2 focus-visible:outline-ring"
        >
          Open map
          <ChevronRight aria-hidden className="size-3.5" />
        </button>
      </header>
      <div className="relative min-h-0 flex-1 [&_.maplibregl-ctrl-attrib]:scale-90">
        <Map
          key={offline ? "offline" : "online"}
          styles={styles}
          theme="light"
          center={ROMANIA_CENTER}
          zoom={5}
          minZoom={4}
          dragPan={false}
          scrollZoom={false}
          boxZoom={false}
          dragRotate={false}
          doubleClickZoom={false}
          touchZoomRotate={false}
          keyboard={false}
        >
          <RomaniaFocus />
          <OverviewLayers compact />
        </Map>
      </div>
      <footer className="flex items-center gap-3 border-t border-border px-4 py-2 text-[11px] whitespace-nowrap text-text-secondary">
        <span className="flex items-center gap-1">
          <StatusMark kind="ok" size="xs" />
          within limit
        </span>
        <span className="flex items-center gap-1">
          <StatusMark kind="warn" size="xs" />
          near
        </span>
        <span className="flex items-center gap-1">
          <StatusMark kind="critical" size="xs" />
          over
        </span>
        <span className="ml-auto text-muted-foreground">{offline ? "Offline basemap · " : ""}tint = evening demand</span>
      </footer>
    </section>
  );
}
