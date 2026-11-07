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
        <SidebarHeader className="flex-row items-center gap-2 p-3 font-semibold">
          <Zap className="size-4 shrink-0 text-primary" />
          <span className="truncate group-data-[collapsible=icon]:hidden">Prosumer Sentinel</span>
        </SidebarHeader>
        <SidebarContent>
          <SidebarGroup>
            <SidebarGroupLabel>Modes</SidebarGroupLabel>
            <SidebarMenu>
              {MODE_IDS.map((id) => {
                const { label, icon: Icon } = MODES[id];
                return (
                  <SidebarMenuItem key={id}>
                    <SidebarMenuButton isActive={mode === id} onClick={() => setMode(id)}>
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
        <header className="flex items-center gap-3 border-b p-2">
          <SidebarTrigger />
          <Tabs value={mode} onValueChange={(value) => setMode(value as ModeId)}>
            <TabsList>
              {MODE_IDS.map((id) => (
                <TabsTrigger key={id} value={id}>
                  {MODES[id].label}
                </TabsTrigger>
              ))}
            </TabsList>
          </Tabs>
        </header>

        <KpiStrip />

        <main className="relative min-h-0 flex-1">
          {/* Remount on a basemap switch so mode layers are re-added to the new style. */}
          <Map key={offline ? "offline" : "online"} styles={styles} theme="dark" center={ROMANIA_CENTER} zoom={5.8} minZoom={4}>
            <MapControls position="bottom-left" />
            {Layers ? <Layers /> : null}
          </Map>
          {Panel ? (
            <Panel />
          ) : (
            <aside className="absolute top-3 left-3 z-10 w-72 max-w-[calc(100%-1.5rem)] rounded-md border bg-background/90 p-3 backdrop-blur">
              <h2 className="font-medium">{active.label}</h2>
              <p className="mt-1 text-sm text-muted-foreground">{active.blurb}</p>
              <p className="mt-2 text-xs text-muted-foreground">Placeholder panel, see {active.card}</p>
            </aside>
          )}
          {/* #31/#32: the flexibility panel opens from Feeders and O&M. */}
          {mode === "feeders" || mode === "om" ? <FlexDock /> : null}
          <SummaryCard />
          {offline ? (
            <span className="absolute bottom-3 left-14 z-10 rounded border bg-background/90 px-2 py-1 text-xs text-muted-foreground">
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
