import { Activity, Factory, Network, Wrench, type LucideIcon } from "lucide-react";
import type { ComponentType } from "react";

import { FeederPanel } from "./feeders/feeder-panel";
import { FeedersLayers } from "./feeders/feeders-layers";
import { ProductionLayers } from "./production/production-layers";
import { ProductionPanel } from "./production/production-panel";

export const MODE_IDS = ["feeders", "om", "production", "demand"] as const;
export type ModeId = (typeof MODE_IDS)[number];

// Align with ModeConfig in ./types.ts when #38 merges. Layers render inside <Map>, so switching
// modes keeps the basemap and never refetches it.
export type ModeEntry = {
  label: string;
  icon: LucideIcon;
  blurb: string;
  card: string;
  Layers?: ComponentType;
  Panel?: ComponentType;
};

export const MODES: Record<ModeId, ModeEntry> = {
  feeders: {
    label: "Feeders",
    icon: Network,
    blurb: "Forecast export vs limit per feeder, baseline vs Sentinel.",
    card: "#19",
    Layers: FeedersLayers,
    Panel: FeederPanel,
  },
  om: {
    label: "O&M",
    icon: Wrench,
    blurb: "Sites and turbines by status, alert and work-order queues.",
    card: "#32",
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
  },
};
