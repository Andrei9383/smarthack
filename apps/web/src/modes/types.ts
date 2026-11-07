import type { LucideIcon } from "lucide-react";

export const MODE_IDS = ["overview", "feeders", "om", "production", "demand"] as const;
export type ModeId = (typeof MODE_IDS)[number];

export interface MapLayerConfig {
  id: string;
  name: string;
  type: "fill" | "line" | "circle" | "symbol" | "raster" | "deckgl";
  visible: boolean;
  opacity?: number;
}

export interface PanelConfig {
  id: string;
  title: string;
  position: "left" | "right" | "bottom" | "drawer";
  defaultOpen?: boolean;
}

export interface LegendItem {
  label: string;
  color: string;
  description?: string;
}

export interface LegendConfig {
  title: string;
  items: LegendItem[];
}

export interface ModeConfig {
  id: ModeId;
  label: string;
  icon?: LucideIcon;
  model: string; // e.g. "LP + P10", "M4 SiteHealth + M6 WindNBM", "M1 PVNet + M5 WindPower", "M3 DemandNet"
  tier: "MUST" | "SHOULD";
  blurb: string;
  layers: MapLayerConfig[];
  panels: PanelConfig[];
  queries: string[]; // oRPC query procedure paths
  legend: LegendConfig;
}
