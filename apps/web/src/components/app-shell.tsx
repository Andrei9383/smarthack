"use client";

import { Map, MapControls } from "@sentinel-app/ui/components/map";
import {
  Sidebar,
  SidebarContent,
  SidebarFooter,
  SidebarGroup,
  SidebarGroupLabel,
  SidebarHeader,
  SidebarInset,
  SidebarMenu,
  SidebarMenuButton,
  SidebarMenuItem,
  SidebarProvider,
  SidebarTrigger,
} from "@sentinel-app/ui/components/sidebar";
import { cn } from "@sentinel-app/ui/lib/utils";
import { Zap } from "lucide-react";
import { parseAsStringLiteral, useQueryState } from "nuqs";

import { PROVENANCE_PILL_CLASS } from "@/components/provenance-pill";
import { DemoJumps } from "@/demo/demo-jumps";
import { KpiStrip, RunContext } from "@/demo/kpi-strip";
import { ReplayBar, ReplayStatus } from "@/demo/replay-bar";
import { SummaryCard } from "@/demo/summary-card";
import { useBasemapStyles, useRomaniaFocus } from "@/demo/use-basemap";
import { MODE_IDS, MODES, type ModeId } from "@/modes/registry";
import { TurbineSheet } from "@/modes/om/turbine-sheet";
import { FlexDock } from "@/panels/flex/flex-panel";
import { SiteSheet } from "@/panels/site/site-sheet";

const ROMANIA_CENTER: [number, number] = [25, 45.9];

// One line under each mode in the sidebar and in the page header.
const DESCRIPTORS: Record<ModeId, string> = {
  feeders: "Peak shaving on LV feeders",
  om: "Alarms → work orders",
  production: "Wind & solar forecast",
  demand: "Where demand will peak",
};

// Renders inside <Map> so it can style the basemap once it has loaded.
function RomaniaFocus() {
  useRomaniaFocus();
  return null;
}

export function AppShell() {
  const [mode, setMode] = useQueryState(
    "mode",
    parseAsStringLiteral(MODE_IDS).withDefault("feeders"),
  );
  const active = MODES[mode];
  const { Layers, Panel } = active;
  const { offline, styles } = useBasemapStyles();

  return (
    <SidebarProvider>
      <Sidebar collapsible="icon">
        <SidebarHeader className="h-16 flex-row items-center gap-3 border-b border-sidebar-border px-4 py-0 group-data-[collapsible=icon]:px-2">
          <span className="flex size-8 shrink-0 items-center justify-center rounded-lg bg-primary text-primary-foreground shadow-sm">
            <Zap className="size-4" />
          </span>
          <div className="min-w-0 group-data-[collapsible=icon]:hidden">
            <div className="truncate text-sm leading-5 font-semibold tracking-tight text-foreground">
              Prosumer Sentinel
            </div>
            <div className="truncate text-[11px] leading-4 text-muted-foreground">Grid &amp; O&amp;M agent · PPC</div>
          </div>
        </SidebarHeader>
        <SidebarContent>
          <SidebarGroup className="px-3 py-4">
            <SidebarGroupLabel className="h-6 px-2 text-[11px] font-medium tracking-[0.04em] text-muted-foreground uppercase">
              Modes
            </SidebarGroupLabel>
            <SidebarMenu className="gap-1">
              {MODE_IDS.map((id) => {
                const { label, icon: Icon } = MODES[id];
                return (
                  <SidebarMenuItem key={id}>
                    <SidebarMenuButton
                      size="lg"
                      isActive={mode === id}
                      onClick={() => setMode(id)}
                      className="group/mode h-auto items-start gap-3 rounded-lg px-2.5 py-2 text-foreground transition-colors duration-150 hover:bg-muted hover:text-foreground data-active:bg-brand-50 data-active:font-normal data-active:text-brand-900 [&>span:last-child]:whitespace-normal"
                    >
                      <Icon className="mt-0.5 text-text-secondary group-data-active/mode:text-primary" />
                      <span className="flex min-w-0 flex-col">
                        <span className="truncate text-[13px] leading-5 font-medium group-data-active/mode:font-semibold">
                          {label}
                        </span>
                        <span className="truncate text-[11px] leading-4 text-muted-foreground group-data-active/mode:text-brand-700">
                          {DESCRIPTORS[id]}
                        </span>
                      </span>
                    </SidebarMenuButton>
                  </SidebarMenuItem>
                );
              })}
            </SidebarMenu>
          </SidebarGroup>
        </SidebarContent>
        <SidebarFooter className="p-3 group-data-[collapsible=icon]:hidden">
          <ReplayStatus />
        </SidebarFooter>
      </Sidebar>

      <SidebarInset className="h-svh overflow-hidden">
        <header className="flex h-16 shrink-0 items-center gap-4 border-b border-border bg-background px-4">
          <SidebarTrigger className="rounded-md text-text-secondary transition-colors duration-150 hover:bg-muted hover:text-foreground" />
          <span aria-hidden className="h-6 w-px bg-border" />
          <div className="min-w-0 flex-1">
            <h1 className="truncate text-base leading-6 font-semibold text-foreground">{active.label}</h1>
            <p className="truncate text-xs leading-4 text-text-secondary">{DESCRIPTORS[mode]}</p>
          </div>
          <RunContext />
        </header>

        <KpiStrip />

        <main className="relative min-h-0 flex-1">
          {/* Remount on a basemap switch so mode layers are re-added to the new style. */}
          <Map key={offline ? "offline" : "online"} styles={styles} theme="light" center={ROMANIA_CENTER} zoom={5.8} minZoom={4}>
            <RomaniaFocus />
            <MapControls
              position="bottom-left"
              className="bottom-[96px] left-4 [&>div]:rounded-lg [&>div]:border-border [&>div]:shadow-card"
            />
            {Layers ? <Layers /> : null}
          </Map>

          {/* Panels float in the area above the bottom dock, so their bottom edge clears it. */}
          <div className="pointer-events-none absolute inset-x-0 top-0 bottom-[84px] z-10 [&>*]:pointer-events-auto">
            {Panel ? (
              <Panel />
            ) : (
              <aside className="absolute top-4 left-4 z-10 w-80 max-w-[calc(100%-2rem)] rounded-xl border border-border bg-card p-5 shadow-card">
                <h2 className="text-base font-semibold text-foreground">{active.label}</h2>
                <p className="mt-2 text-sm leading-relaxed text-text-secondary">{active.blurb}</p>
                <p className="mt-3 text-xs text-muted-foreground">Placeholder panel, see {active.card}</p>
              </aside>
            )}
            {/* #31/#32: the flexibility panel opens from Feeders and O&M. */}
            {mode === "feeders" || mode === "om" ? <FlexDock /> : null}
            {offline ? (
              <span className={cn(PROVENANCE_PILL_CLASS, "absolute bottom-3 left-16 z-10 bg-card shadow-card")}>
                Offline basemap
              </span>
            ) : null}
          </div>

          {/* One floating dock: replay controls, then the demo script menu. */}
          <div className="absolute inset-x-4 bottom-4 z-20 flex items-center gap-4 rounded-xl border border-border bg-card px-3 py-2 shadow-card">
            <ReplayBar />
            <span aria-hidden className="h-6 w-px shrink-0 bg-border" />
            <DemoJumps />
          </div>
          <SummaryCard />
        </main>
        <SiteSheet />
        <TurbineSheet />
      </SidebarInset>
    </SidebarProvider>
  );
}
