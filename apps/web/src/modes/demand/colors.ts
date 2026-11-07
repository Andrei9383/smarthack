// Raw colour values for MapLibre paint and SVG charts, which can't read Tailwind classes.
// Mirrors team/contracts/ui-style.md (PPC light + emerald); keep in sync with the tokens in
// packages/ui/src/styles/globals.css. Components must not hold hex literals of their own.

// Stress ramp, light → strong (§3 "Map colour ramps"). Stress 1 = P90 demand at capacity.
export const STRESS_RAMP: readonly (readonly [stress: number, color: string])[] = [
  [0, "#ECFBF2"], // --brand-50
  [0.5, "#FDE68A"],
  [0.75, "#F59E0B"],
  [1, "#E5484D"], // --status-critical: red zones
];
export const STRESS_FILL_OPACITY = 0.78;

// Feeder lines inside a selected county sit on top of the county fill, so they use the
// stronger status colours plus a white casing instead of the pale end of the ramp.
export const FEEDER_RAMP: readonly (readonly [stress: number, color: string])[] = [
  [0, "#0E9F5A"], // --status-ok
  [0.6, "#D97706"], // --status-warn
  [1, "#E5484D"], // --status-critical
];
export const FEEDER_CASING = "#FFFFFF";

export const COUNTY_OUTLINE = "#B8BFC7"; // a step darker than basemap borders (#C9CED4)
export const SELECTED_OUTLINE = "#0D0D0D"; // --foreground

// Chart: zone demand is a forecast (P50 with P10–P90 band).
export const DEMAND_SERIES = "var(--status-forecast)";
