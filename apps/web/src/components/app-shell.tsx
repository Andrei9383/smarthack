"use client";

import { Map, useMap } from "@wattsentry/ui/components/map";
import { cn } from "@wattsentry/ui/lib/utils";
import { parseAsStringLiteral, useQueryState } from "nuqs";
import { useCallback, useEffect, useState } from "react";

import { defaultTourSteps, Tour, useTour } from "@/components/help";
import { AgentToast } from "@/components/shell/agent-toast";
import { CommandPalette } from "@/components/shell/command-palette";
import { Inspector } from "@/components/shell/inspector";
import { LegendCard } from "@/components/shell/legend-card";
import { ShellMapControls } from "@/components/shell/map-controls";
import { Rail } from "@/components/shell/rail";
import { TimelineDock } from "@/components/shell/timeline-dock";
import { TopBar } from "@/components/shell/top-bar";
import { VoiceAssistantProvider, VoicePanel } from "@/components/voice/voice-assistant";
import { SummaryCard } from "@/components/shell/summary-card";
import { useBasemapStyles, useRomaniaFocus } from "@/lib/use-basemap";
import { useMapTheme } from "@/lib/palette";
import { TurbineSheet } from "@/modes/om/turbine-sheet";
import { MODE_IDS, MODES, type ModeId } from "@/modes/registry";
import { FlexDock, useFlexOpen } from "@/components/panels/flex/flex-panel";
import { SiteSheet } from "@/components/panels/site/site-sheet";

const ROMANIA_CENTER: [number, number] = [25, 45.9];
const RAIL_KEY = "sentinel.rail";

// Feeders and Demand draw their own legend inside their layers; the shell moves it into the
// prototype's legend slot (top-left, under the top bar). The other modes use the shell's LegendCard.
const OWN_LEGEND = new Set<ModeId>(["feeders", "demand"]);
// Written out in full: Tailwind only generates classes it can read literally in the source.
const MOVE_MODE_LEGEND =
  "[&_.maplibregl-map>div.pointer-events-none.bg-card]:top-[80px] [&_.maplibregl-map>div.pointer-events-none.bg-card]:left-4 [&_.maplibregl-map>div.pointer-events-none.bg-card]:bottom-auto [&_.maplibregl-map>div.pointer-events-none.bg-card]:rounded-xl [&_.maplibregl-map>div.pointer-events-none.bg-card]:bg-card";
const HIDE_MODE_LEGEND = "[&_.maplibregl-map>div.pointer-events-none.bg-card]:hidden";

// Inspector width and the gap the map-side surfaces (dock, map controls) keep from it.
const INSPECTOR_W = "w-[352px] min-[1440px]:w-[392px]";
const BESIDE_INSPECTOR = "right-[384px] min-[1440px]:right-[424px]";

// Renders inside <Map> so it can style the basemap once it has loaded.
// West, south / east, north, with room for the panels floating over the map's edges.
const ROMANIA_BOUNDS: [[number, number], [number, number]] = [[16.5, 41], [34.5, 50.5]];

function RomaniaFocus() {
  useRomaniaFocus();
  return null;
}

// The map lives right of the rail, so collapsing the rail resizes it.
function ResizeWith({ value }: { value: unknown }) {
  const { map } = useMap();
  useEffect(() => {
    if (!map) return;
    const timers = [0, 120, 240].map((ms) => setTimeout(() => map.resize(), ms));
    return () => {
      for (const timer of timers) clearTimeout(timer);
    };
  }, [map, value]);
  return null;
}

function useRailExpanded() {
  const [expanded, setExpanded] = useState(true);
  useEffect(() => {
    let stored: string | null = null;
    try {
      stored = window.localStorage.getItem(RAIL_KEY);
    } catch {
      // Private mode or blocked storage: fall back to the width rule.
    }
    setExpanded(stored ? stored === "expanded" : window.matchMedia("(min-width: 768px)").matches);
  }, []);
  const toggle = useCallback(() => {
    setExpanded((value) => {
      try {
        window.localStorage.setItem(RAIL_KEY, value ? "collapsed" : "expanded");
      } catch {
        // Not persisted; the toggle still works for this visit.
      }
      return !value;
    });
  }, []);
  return [expanded, toggle] as const;
}

export function AppShell() {
  const [mode, setMode] = useQueryState("mode", parseAsStringLiteral(MODE_IDS).withDefault("overview"));
  const active = MODES[mode];
  const { Layers, Panel, Page } = active;
  const { offline, styles } = useBasemapStyles();
  const mapTheme = useMapTheme();
  const [railExpanded, toggleRail] = useRailExpanded();
  const [flexOpen] = useFlexOpen();
  const hasFlex = mode === "feeders" || mode === "om";
  const flexShown = hasFlex && flexOpen;
  const tour = useTour();
  const [paletteOpen, setPaletteOpen] = useState(false);

  return (
    <VoiceAssistantProvider>
    <div className="flex h-svh w-full overflow-hidden bg-sidebar text-foreground">
      <Rail
        mode={mode}
        onMode={(id) => setMode(id)}
        expanded={railExpanded}
        onToggle={toggleRail}
        onHelp={tour.start}
      />

      <main
        className={cn(
          // Inset layer: the content sits on the frame (--sidebar) as a bordered panel, one level up.
          "relative my-2 mr-2 min-w-0 flex-1 overflow-hidden rounded-lg border border-border bg-background",
          flexShown ? HIDE_MODE_LEGEND : MOVE_MODE_LEGEND,
        )}
      >
        {Page ? (
          <>
            {/* #124: Overview is a page. The rail, top bar and a compact dock stay; the map and
                inspector give way to the briefing, the mini map and the glance sections. */}
            <div
              key={mode}
              className="absolute inset-x-0 top-[72px] bottom-0 overflow-x-hidden overflow-y-auto overscroll-contain px-4 pt-3 pb-[100px] [scrollbar-gutter:stable_both-edges] [scrollbar-width:thin] motion-safe:animate-in motion-safe:fade-in-0 motion-safe:duration-250"
            >
              <Page onTour={tour.start} />
            </div>
          </>
        ) : (
          <>
            {/* The tour's "map" target: the free map area between the floating surfaces. */}
            <div data-tour="map" aria-hidden className="pointer-events-none absolute top-[76px] bottom-[116px] left-[208px] right-[384px] min-[1440px]:right-[424px]" />
            {/* Remount on a basemap or theme switch so mode layers are re-added with the new colours. */}
            <Map
              key={`${offline ? "offline" : "online"}-${mapTheme}`}
              styles={styles}
              theme={mapTheme}
              center={ROMANIA_CENTER}
              zoom={5.8}
              // Romania always fills the view: no zooming out past the whole country or panning away from it.
              minZoom={5.3}
              maxBounds={ROMANIA_BOUNDS}
            >
              <RomaniaFocus />
              <ResizeWith value={railExpanded} />
              <ShellMapControls className={cn("absolute bottom-[124px] z-10", BESIDE_INSPECTOR)} />
              {Layers ? <Layers /> : null}
            </Map>

            {!OWN_LEGEND.has(mode) && !flexShown ? (
              <LegendCard mode={mode} className="absolute top-[80px] left-4 z-[15]" />
            ) : null}

            {/* #31/#32: the flexibility panel opens from Feeders and O&M. Closed, its button sits beside
                the inspector; open, the panel takes the legend's column on the left. */}
            {hasFlex ? (
              flexShown ? (
                <div className="absolute top-[76px] bottom-[116px] left-4 z-[16] w-[380px] max-w-[calc(100%-2rem)] [&>aside]:inset-0">
                  <FlexDock />
                </div>
              ) : (
                <div
                  className={cn(
                    "absolute top-[80px] z-[16] [&>button]:static [&>button]:h-9 [&>button]:rounded-xl",
                    BESIDE_INSPECTOR,
                  )}
                >
                  <FlexDock />
                </div>
              )
            ) : null}

            <Inspector mode={mode} className={cn("absolute top-[76px] right-4 bottom-4 z-20", INSPECTOR_W)}>
              {Panel ? (
                <Panel />
              ) : (
                <p className="px-5 py-4 text-sm leading-relaxed text-text-secondary">{active.blurb}</p>
              )}
            </Inspector>

          </>
        )}

        {/* One instance each across modes: the dock owns the replay ticker (unmounting it pauses). */}
        <TopBar mode={mode} onSearch={() => setPaletteOpen(true)} className="absolute inset-x-4 top-3 z-30 h-14" />
        <AgentToast className={cn("absolute left-4 z-[25]", Page ? "bottom-[92px]" : "bottom-[116px]")} />
        {/* #207: the "Ask WattSentry" conversation sits in the toast's slot, above it. */}
        <VoicePanel className={cn("absolute left-4 z-[26]", Page ? "bottom-[92px]" : "bottom-[116px]")} />
        <TimelineDock
          compact={Boolean(Page)}
          className={cn("absolute bottom-4 left-4 z-20", Page ? "right-4 mx-auto max-w-[1600px]" : BESIDE_INSPECTOR)}
        />
        <SummaryCard />
      </main>
      <SiteSheet />
      <TurbineSheet />
      <Tour steps={defaultTourSteps} open={tour.open} onOpenChange={tour.onOpenChange} />
      <CommandPalette open={paletteOpen} onOpenChange={setPaletteOpen} onTour={tour.start} />
    </div>
    </VoiceAssistantProvider>
  );
}
