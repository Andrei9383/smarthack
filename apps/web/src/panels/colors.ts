// Colour values for chart props and inline swatches in the site and flex panels. They are CSS variable
// references to the status/brand tokens in packages/ui/src/styles/globals.css (team/contracts/ui-style.md
// §2, §3), so SVG strokes, fills and inline styles follow the light/dark theme. Components should still use
// the Tailwind token classes (text-status-ok, bg-status-warn/12, …) wherever they can.

export const OK = "var(--status-ok)"; // ok, eligible, Sentinel (= --primary)
export const WARN = "var(--status-warn)"; // derated, warning
export const CRITICAL = "var(--status-critical)"; // over limit, fault
export const FORECAST = "var(--status-forecast)"; // forecast / expected (dashed)
export const SOC = "var(--status-soc)"; // battery state of charge
export const REVIEW = "var(--status-review)"; // handed to a human
export const UNKNOWN = "var(--status-unknown)"; // excluded, no data
export const ACTUAL = "var(--status-actual)"; // measured series
/** Measured actual at 85%, for bars and legend swatches where a solid line reads too heavy. */
export const ACTUAL_SOFT = "color-mix(in oklab, var(--status-actual) 85%, transparent)";
/** Baseline / naive policy: foreground at 55%, so green vs grey always reads "ours vs baseline". */
export const BASELINE = "var(--chart-baseline)";

/** Site status → colour, for inline swatches. (modes/om has its own map-safe hex set.) */
export const STATUS_COLORS: Record<string, string> = {
  eligible: OK,
  derated: WARN,
  excluded: UNKNOWN,
  human_review: REVIEW,
};
