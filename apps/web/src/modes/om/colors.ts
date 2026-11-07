// Raw colour values for code that cannot read CSS variables: MapLibre paint expressions, the canvas
// status marks and the SVG/visx charts in this mode. The RGB source is apps/web/src/lib/palette.ts
// (team/contracts/ui-style.md §2, §3); components use the Tailwind tokens (`text-status-critical`,
// `bg-brand-50`, …) instead.
import { activePaletteTheme, hex, MAP_SAFE, paletteFor, rgba } from "@/lib/palette";

const current = () => paletteFor(activePaletteTheme());

export const OM_COLORS = {
  // Status hues for the map and charts: map-safe mid-tones that read on both basemaps and both themes.
  ok: hex(MAP_SAFE.ok),
  warn: hex(MAP_SAFE.warn),
  critical: hex(MAP_SAFE.critical),
  forecast: hex(MAP_SAFE.forecast),
  review: hex(MAP_SAFE.review),
  unknown: hex(MAP_SAFE.unknown),
  soc: hex(MAP_SAFE.soc),
  /** measured series in SVG charts (follows the theme through the CSS variable) */
  actual: "var(--status-actual)",
  /** chart grid lines (§3), SVG only */
  grid: "var(--chart-grid)",
  /** ring around map markers so they lift off either basemap */
  ring: hex([255, 255, 255]),
  /** hairline border around neutral cluster discs (--border of the current theme) */
  get border() {
    return hex(current().border);
  },
  /** cluster disc fill (--card of the current theme) */
  get card() {
    return hex(current().surface);
  },
  /** cluster count text on the disc (--foreground of the current theme) */
  get label() {
    return hex(current().foreground);
  },
  /** contact shadow under a map status mark (foreground at 28%) */
  markShadow: rgba([13, 20, 28], 0.28),
  /** specular light on a map status mark's dome */
  markLight: rgba([255, 255, 255], 0.5),
  markLightEnd: rgba([255, 255, 255], 0),
};

/** Site / turbine status → colour, same keys as the shared STATUS_COLORS, remapped to the map-safe set. */
export const OM_STATUS_COLORS: Record<string, string> = {
  eligible: OM_COLORS.ok,
  derated: OM_COLORS.warn,
  excluded: OM_COLORS.unknown,
  human_review: OM_COLORS.review,
  // a turbine that is out of service is stopped, which reads red
  stopped: OM_COLORS.critical,
};
