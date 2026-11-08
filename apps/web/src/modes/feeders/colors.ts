// Raw colour values for MapLibre paint and SVG charts, which can't read Tailwind classes.
// Mirrors team/contracts/ui-style.md (Calm, light + charcoal); the RGB source is apps/web/src/lib/palette.ts.
// Components must not hold hex literals of their own.
import { hex, MAP_SAFE, onPaletteTheme, rgba } from "@/lib/palette";

// Map: feeder status. Map-safe mid-tones that read on both basemaps (Positron and dark-matter).
export const FEEDER_OK = hex(MAP_SAFE.ok);
export const FEEDER_WARN = hex(MAP_SAFE.warn);
export const FEEDER_OVER = hex(MAP_SAFE.critical);
export const FEEDER_UNKNOWN = hex(MAP_SAFE.unknown);

// Theme-dependent: the halo that lifts lines off the basemap, and the gaps cut into low-trust lines
// (reads as dashed). Both are the land colour of the current theme; the map remounts on a theme change.
export let FEEDER_CASING = "";
export let LOW_TRUST_GAP = "";
onPaletteTheme((palette) => {
  FEEDER_CASING = hex(palette.mapCasing);
  LOW_TRUST_GAP = hex(palette.mapCasing);
});

// Charts: WattSentry is always --sentinel, the baseline always --chart-baseline (ink at 45%).
export const SENTINEL_SERIES = "var(--sentinel)";
export const BASELINE_SERIES = "var(--chart-baseline)";
export const LIMIT_SERIES = "var(--status-critical)";
export const SOC_SERIES = "var(--status-soc)";
// Live chart (#130): measured readings in ink, the forecast in the forecast blue.
export const ACTUAL_SERIES = "var(--foreground)";
export const FORECAST_SERIES = "var(--status-forecast)";

// Map motion (#115): emerald charging ripples and the light flow dashes on an over-limit (red) line.
export const CHARGE_RIPPLE = hex(MAP_SAFE.brand);
export const FLOW_DASH = rgba([255, 255, 255], 0.95);
export const FLOW_GAP = rgba([255, 255, 255], 0);
