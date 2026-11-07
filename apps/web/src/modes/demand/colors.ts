// Raw colour values for MapLibre paint and SVG charts, which can't read Tailwind classes.
// Mirrors team/contracts/ui-style.md (PPC light + emerald); keep in sync with the tokens in
// packages/ui/src/styles/globals.css. Components must not hold hex literals of their own.

// Stress ramp, light → strong (§3 "Map colour ramps"). Stress 1 = P90 demand at capacity.
export const STRESS_RAMP: readonly (readonly [stress: number, color: string])[] = [
  [0, "#D1F5E0"], // --brand-100: low stress is still a visible tint, unlike "no data"
  [0.5, "#FDE68A"],
  [0.75, "#F59E0B"],
  [1, "#E5484D"], // --status-critical: red zones
];
export const STRESS_FILL_OPACITY = 0.85;
export const STRESS_HOVER_OPACITY = 1; // hover: the same colour, slightly darker

// Counties without rows: light grey diagonal hatch, never confused with low stress.
export const NO_DATA_FILL = "#F2F3F5"; // --muted
export const NO_DATA_HATCH = "#D3D8DE"; // between --border and the basemap borders

// Feeder lines inside a selected county sit on top of the county fill, so they use the
// stronger status colours plus a white casing instead of the pale end of the ramp. The panel's
// stress meters use the same ramp, since the pale end would vanish on a grey track.
export const FEEDER_RAMP: readonly (readonly [stress: number, color: string])[] = [
  [0, "#0E9F5A"], // --status-ok
  [0.6, "#D97706"], // --status-warn
  [1, "#E5484D"], // --status-critical
];
export const FEEDER_CASING = "#FFFFFF";

export const COUNTY_OUTLINE = "#B8BFC7"; // a step darker than basemap borders (#C9CED4)
export const SELECTED_OUTLINE = "#0B7F48"; // --brand-700
export const SELECTED_SHADOW = "#0D0D0D"; // drawn blurred at low opacity below the outline

// Chart: zone demand is a forecast (P50 with P10–P90 band).
export const DEMAND_SERIES = "var(--status-forecast)";

const rgbOf = (hex: string) => [1, 3, 5].map((index) => Number.parseInt(hex.slice(index, index + 2), 16));

/** Colour at `stress` along a ramp (clamped), as an `rgb()` string for inline styles. */
export function rampColor(stress: number, ramp: typeof STRESS_RAMP = FEEDER_RAMP) {
  const stops = ramp.map(([at, color]) => [at, rgbOf(color)] as const);
  const first = stops[0];
  const last = stops.at(-1);
  if (!first || !last) return "currentColor";
  let lower = first;
  let upper = last;
  for (let index = 1; index < stops.length; index += 1) {
    const stop = stops[index];
    const previous = stops[index - 1];
    if (stop && previous && stress <= stop[0]) {
      lower = previous;
      upper = stop;
      break;
    }
  }
  const span = upper[0] - lower[0];
  const t = span > 0 ? Math.min(1, Math.max(0, (stress - lower[0]) / span)) : 1;
  const channel = (index: number) =>
    Math.round((lower[1][index] ?? 0) + ((upper[1][index] ?? 0) - (lower[1][index] ?? 0)) * t);
  return `rgb(${channel(0)} ${channel(1)} ${channel(2)})`;
}
