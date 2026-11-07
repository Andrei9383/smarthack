// Raw colour values for chart props and inline swatches in the site and flex panels.
// These mirror the status/brand tokens in team/contracts/ui-style.md (§2, §3) and
// packages/ui/src/styles/globals.css. Keep them in sync; components should use the
// Tailwind token classes (text-status-ok, bg-status-warn/10, …) wherever they can.

export const OK = "#0E9F5A"; // --status-ok / --primary: ok, eligible, Sentinel
export const WARN = "#D97706"; // --status-warn: derated, warning
export const CRITICAL = "#E5484D"; // --status-critical: over limit, fault
export const FORECAST = "#0284C7"; // --status-forecast: forecast / expected (dashed)
export const SOC = "#6366F1"; // --status-soc: battery state of charge
export const REVIEW = "#8B5CF6"; // --status-review: handed to a human
export const UNKNOWN = "#94A3B8"; // --status-unknown: excluded, no data
export const ACTUAL = "#0D0D0D"; // --status-actual: measured series
/** Baseline / naive policy: #0D0D0D at 55%, so green vs grey always reads "ours vs baseline". */
export const BASELINE = "rgba(13, 13, 13, 0.55)";

/** Site status → hex. Also used by modes/om for map layers, so values stay hex strings. */
export const STATUS_COLORS: Record<string, string> = {
  eligible: OK,
  derated: WARN,
  excluded: UNKNOWN,
  human_review: REVIEW,
};
