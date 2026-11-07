// Raw colour values for code that cannot read CSS variables: MapLibre paint expressions and the
// SVG/visx charts in this mode. They mirror the hex values of team/contracts/ui-style.md (§2, §3);
// components use the Tailwind tokens (`text-status-critical`, `bg-brand-50`, …) instead.
// Keep this file in sync with packages/ui/src/styles/globals.css.

export const OM_COLORS = {
  ok: "#0E9F5A",
  warn: "#D97706",
  critical: "#E5484D",
  forecast: "#0284C7",
  review: "#8B5CF6",
  unknown: "#94A3B8",
  actual: "#0D0D0D",
  soc: "#6366F1",
  /** chart grid lines (§3) */
  grid: "#EEF0F2",
  /** ring around map markers so they lift off the light basemap */
  ring: "#FFFFFF",
  /** hairline border (--border) around neutral cluster discs */
  border: "#E6E8EB",
  /** cluster disc fill (--card) */
  card: "#FFFFFF",
  /** contact shadow under a map status mark (foreground at 28%) */
  markShadow: "rgba(13, 13, 13, 0.28)",
  /** specular light on a map status mark's dome */
  markLight: "rgba(255, 255, 255, 0.5)",
  markLightEnd: "rgba(255, 255, 255, 0)",
} as const;

/** Site / turbine status → colour, same keys as the shared STATUS_COLORS, remapped to the light palette. */
export const OM_STATUS_COLORS: Record<string, string> = {
  eligible: OM_COLORS.ok,
  derated: OM_COLORS.warn,
  excluded: OM_COLORS.unknown,
  human_review: OM_COLORS.review,
  // a turbine that is out of service is stopped, which reads red
  stopped: OM_COLORS.critical,
};
