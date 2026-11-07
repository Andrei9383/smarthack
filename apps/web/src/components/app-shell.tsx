"use client";

import { Map, MapControls } from "@sentinel-app/ui/components/map";
import {
  Sidebar,
  SidebarContent,
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
import { Tabs, TabsList, TabsTrigger } from "@sentinel-app/ui/components/tabs";
import { Zap } from "lucide-react";
import { parseAsStringLiteral, useQueryState } from "nuqs";

import { DemoJumps } from "@/demo/demo-jumps";
import { KpiStrip } from "@/demo/kpi-strip";
import { ReplayBar } from "@/demo/replay-bar";
import { SummaryCard } from "@/demo/summary-card";
import { useBasemapStyles } from "@/demo/use-basemap";
import { MODE_IDS, MODES, type ModeId } from "@/modes/registry";
import { TurbineSheet } from "@/modes/om/turbine-sheet";
import { FlexDock } from "@/panels/flex/flex-panel";
import { SiteSheet } from "@/panels/site/site-sheet";

const ROMANIA_CENTER: [number, number] = [25, 45.9];

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
        <SidebarHeader className="h-14 flex-row items-center gap-2.5 border-b border-sidebar-border px-3 py-0 group-data-[collapsible=icon]:px-2">
          <span className="flex size-8 shrink-0 items-center justify-center rounded-md bg-primary text-primary-foreground">
            <Zap className="size-4" />
          </span>
          <span className="truncate text-sm font-semibold tracking-tight text-foreground group-data-[collapsible=icon]:hidden">
            Prosumer Sentinel
          </span>
        </SidebarHeader>
        <SidebarContent>
          <SidebarGroup className="px-2 py-4">
            <SidebarGroupLabel className="h-6 px-2 text-[11px] font-medium tracking-[0.04em] text-muted-foreground uppercase">
              Modes
            </SidebarGroupLabel>
            <SidebarMenu className="gap-1">
              {MODE_IDS.map((id) => {
                const { label, icon: Icon } = MODES[id];
                return (
                  <SidebarMenuItem key={id}>
                    <SidebarMenuButton
                      isActive={mode === id}
                      onClick={() => setMode(id)}
                      className="h-9 rounded-md px-2.5 text-[13px] text-text-secondary hover:bg-muted hover:text-foreground data-active:bg-brand-50 data-active:font-semibold data-active:text-brand-900 data-active:[&_svg]:text-primary"
                    >
                      <Icon />
                      <span>{label}</span>
                    </SidebarMenuButton>
                  </SidebarMenuItem>
                );
              })}
            </SidebarMenu>
          </SidebarGroup>
        </SidebarContent>
      </Sidebar>

      <SidebarInset className="h-svh overflow-hidden">
        <header className="flex h-14 shrink-0 items-stretch gap-4 border-b border-border bg-background px-4">
          <div className="flex items-center">
            <SidebarTrigger className="rounded-md text-text-secondary hover:bg-muted hover:text-foreground" />
          </div>
          <span aria-hidden className="my-4 w-px bg-border" />
          <Tabs value={mode} onValueChange={(value) => setMode(value as ModeId)} className="h-full">
            <TabsList
              variant="line"
              className="gap-6 p-0 group-data-horizontal/tabs:h-full"
            >
              {MODE_IDS.map((id) => (
                <TabsTrigger
                  key={id}
                  value={id}
                  className="h-full flex-none px-0.5 text-sm font-medium text-text-secondary after:h-0.5 after:rounded-full after:bg-primary hover:text-foreground data-active:text-brand-700 group-data-horizontal/tabs:after:bottom-0"
                >
                  {MODES[id].label}
                </TabsTrigger>
              ))}
            </TabsList>
          </Tabs>
        </header>

        <KpiStrip />

        <main className="relative min-h-0 flex-1">
          {/* Remount on a basemap switch so mode layers are re-added to the new style. */}
          <Map key={offline ? "offline" : "online"} styles={styles} theme="light" center={ROMANIA_CENTER} zoom={5.8} minZoom={4}>
            <MapControls position="bottom-left" />
            {Layers ? <Layers /> : null}
          </Map>
          {Panel ? (
            <Panel />
          ) : (
            <aside className="absolute top-4 left-4 z-10 w-80 max-w-[calc(100%-2rem)] rounded-lg border border-border bg-card p-5 shadow-card">
              <h2 className="text-base font-semibold text-foreground">{active.label}</h2>
              <p className="mt-2 text-sm leading-relaxed text-text-secondary">{active.blurb}</p>
              <p className="mt-3 text-xs text-muted-foreground">Placeholder panel, see {active.card}</p>
            </aside>
          )}
          {/* #31/#32: the flexibility panel opens from Feeders and O&M. */}
          {mode === "feeders" || mode === "om" ? <FlexDock /> : null}
          <SummaryCard />
          {offline ? (
            <span className="absolute bottom-4 left-16 z-10 rounded-full border border-border bg-card px-3 py-1 text-xs font-medium text-text-secondary shadow-card">
              Offline basemap
            </span>
          ) : null}
        </main>
        <ReplayBar />
        <DemoJumps />
        <SiteSheet />
        <TurbineSheet />
      </SidebarInset>
    </SidebarProvider>
  );
}
