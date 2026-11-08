"use client";

import { Map } from "@wattsentry/ui/components/map";
import { cn } from "@wattsentry/ui/lib/utils";
import { ChevronRight } from "lucide-react";

import { StatusMark } from "@/components/status-mark";
import { useBasemapStyles, useRomaniaFocus } from "@/lib/use-basemap";
import { useMapTheme } from "@/lib/palette";

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
  const mapTheme = useMapTheme();
  return (
    <section
      aria-label="Map of Romania with every asset"
      data-tour="map"
      className={cn("relative isolate overflow-hidden rounded-xl bg-card shadow-card", className)}
    >
      {/* Carvo-style map card: the map fills the card, title, action and legend float on top. */}
      <div className="absolute inset-0 [&_.maplibregl-ctrl-attrib]:scale-90">
        <Map
          key={`${offline ? "offline" : "online"}-${mapTheme}`}
          styles={styles}
          theme={mapTheme}
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
      <header className="pointer-events-none absolute inset-x-3 top-3 z-10 flex items-start justify-between gap-2">
        <h2 className="pointer-events-auto rounded-lg bg-card/95 px-3 py-1.5 section-title shadow-floating backdrop-blur-sm">
          Romania, every asset
        </h2>
        <button
          type="button"
          onClick={onOpen}
          className="pointer-events-auto inline-flex h-8 cursor-pointer items-center gap-0.5 rounded-lg bg-card px-3 text-dense font-medium text-foreground shadow-floating transition-colors duration-150 hover:bg-surface-raised focus-visible:outline-2 focus-visible:outline-ring"
        >
          Open map
          <ChevronRight aria-hidden className="size-3.5" />
        </button>
      </header>
      <footer className="pointer-events-none absolute bottom-3 left-3 z-10 flex max-w-[calc(100%-4.5rem)] flex-wrap items-center gap-x-3 gap-y-0.5 rounded-lg bg-card/95 px-3 py-1.5 text-caption whitespace-nowrap text-text-secondary shadow-floating backdrop-blur-sm">
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
      </footer>
    </section>
  );
}
