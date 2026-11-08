import {
  Activity,
  Factory,
  LayoutDashboard,
  Network,
  Scale,
  Wrench,
  type LucideIcon,
} from "lucide-react";
import type { ComponentType } from "react";

import { DemandLayers } from "./demand/demand-layers";
import { DemandPanel } from "./demand/demand-panel";
import { FeederPanel } from "./feeders/feeder-panel";
import { FeedersLayers } from "./feeders/feeders-layers";
import { ImbalancePage } from "./imbalance/imbalance-page";
import { OmLayers } from "./om/om-layers";
import { OmPanel } from "./om/om-panel";
import { OverviewPage } from "./overview/overview-page";
import { ProductionLayers } from "./production/production-layers";
import { ProductionPanel } from "./production/production-panel";

// Overview comes first and is the default when `?mode` is missing (#113). Since #124 it is a page
// (briefing, mini map, glance sections), not a full-screen map.
export const MODE_IDS = [
  "overview",
  "feeders",
  "om",
  "production",
  "demand",
  "imbalance",
] as const;
export type ModeId = (typeof MODE_IDS)[number];

// Align with ModeConfig in ./types.ts when #38 merges. Layers render inside <Map>, so switching
// modes keeps the basemap and never refetches it.
/** What the shell hands a full-page mode (one without the map and inspector). */
export type ModePageProps = {
  /** Opens the guided tour (#114). */
  onTour: () => void;
};

export type ModeEntry = {
  label: string;
  icon: LucideIcon;
  blurb: string;
  card: string;
  Layers?: ComponentType;
  Panel?: ComponentType;
  /**
   * A full page instead of map + inspector (#124: Overview). The shell keeps the rail, the top bar
   * and a compact dock, and renders this in the map area.
   */
  Page?: ComponentType<ModePageProps>;
};

export const MODES: Record<ModeId, ModeEntry> = {
  overview: {
    label: "Overview",
    icon: LayoutDashboard,
    blurb:
      "What prosumers cost PPC in imbalance today, and how WattSentry cuts it.",
    card: "#124",
    Page: OverviewPage,
  },
  feeders: {
    label: "Feeders",
    icon: Network,
    blurb: "Forecast export vs limit per feeder, baseline vs WattSentry.",
    card: "#19",
    Layers: FeedersLayers,
    Panel: FeederPanel,
  },
  om: {
    label: "O&M",
    icon: Wrench,
    blurb: "Sites and turbines by status, alert and work-order queues.",
    card: "#32",
    Layers: OmLayers,
    Panel: OmPanel,
  },
  production: {
    label: "Production",
    icon: Factory,
    blurb: "Wind field, parks and solar-dense counties by expected MWh.",
    card: "#39",
    Layers: ProductionLayers,
    Panel: ProductionPanel,
  },
  demand: {
    label: "Demand",
    icon: Activity,
    blurb: "Zones by peak-stress score, peak hour and MW.",
    card: "#40",
    Layers: DemandLayers,
    Panel: DemandPanel,
  },
  imbalance: {
    label: "Imbalance",
    icon: Scale,
    blurb:
      "What PPC buys for its prosumers vs what they take: band, 18–21 window, avoidable lei.",
    card: "#177",
    Page: ImbalancePage,
  },
};
