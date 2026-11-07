// Raw colour values for MapLibre paint and SVG charts, which can't read Tailwind classes.
// Mirrors team/contracts/ui-style.md (PPC light + emerald); keep in sync with the tokens in
// packages/ui/src/styles/globals.css. Components must not hold hex literals of their own.

// Map: feeder status, tuned for the light basemap (land #F7F8FA).
export const FEEDER_OK = "#0E9F5A"; // --status-ok
export const FEEDER_WARN = "#D97706"; // --status-warn
export const FEEDER_OVER = "#E5484D"; // --status-critical
export const FEEDER_UNKNOWN = "#94A3B8"; // --status-unknown
export const FEEDER_CASING = "#FFFFFF"; // white halo that lifts lines off the basemap
export const LOW_TRUST_GAP = "#FFFFFF"; // gaps cut into low-trust lines (reads as dashed)

// Charts: Sentinel is always --primary, the baseline always #0D0D0D at 55%.
export const SENTINEL_SERIES = "var(--primary)";
export const BASELINE_SERIES = "rgb(13 13 13 / 0.55)";
export const LIMIT_SERIES = "var(--status-critical)";
export const SOC_SERIES = "var(--status-soc)";
