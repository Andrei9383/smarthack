// Raw colour values for MapLibre paint and SVG charts, which can't read Tailwind classes.
// Mirrors team/contracts/ui-style.md (Cockpit, dark + light); the RGB source is apps/web/src/lib/palette.ts.
// Components must not hold hex literals of their own.
import { hex, MAP_SAFE, onPaletteTheme, rgba } from "@/lib/palette";

// Stress ramp, light → strong (§3 "Map colour ramps"). Stress 1 = P90 demand at capacity. One ramp for
// both themes (the legend reads it at import): the low end is a mint that is still a visible tint on the
// light land and on the dark land, unlike "no data".
export const STRESS_RAMP: readonly (readonly [stress: number, color: string])[] = [
  [0, "#A7E8C5"],
  [0.5, "#F5D76E"],
  [0.75, "#F59E0B"],
  [1, hex(MAP_SAFE.critical)], // red zones
];
export const STRESS_HOVER_OPACITY = 1; // hover: the same colour, slightly darker

// Feeder lines inside a selected county sit on top of the county fill, so they use the
// stronger status colours plus a casing instead of the pale end of the ramp. The panel's
// stress meters use the same ramp, since the pale end would vanish on a grey track.
export const FEEDER_RAMP: readonly (readonly [stress: number, color: string])[] = [
  [0, hex(MAP_SAFE.ok)],
  [0.6, hex(MAP_SAFE.warn)],
  [1, hex(MAP_SAFE.critical)],
];

// Theme-dependent values; the map remounts on a theme change, so the layers re-read them.
// Most counties have no rows: keep them flat and quiet (basemap land) so the few with data pop.
export let STRESS_FILL_OPACITY = 0.9;
export let NO_DATA_FILL = "";
export let NO_DATA_BORDER = "";
export let DATA_BORDER = ""; // foreground at 15%
export let FEEDER_CASING = "";
export let SELECTED_OUTLINE = ""; // brand green for the current basemap
export let SELECTED_SHADOW = ""; // drawn blurred at low opacity below the outline
onPaletteTheme((palette, theme) => {
  // The pale end of the ramp would glare on the dark land at full strength.
  STRESS_FILL_OPACITY = theme === "dark" ? 0.78 : 0.9;
  NO_DATA_FILL = hex(palette.mapLand);
  NO_DATA_BORDER = hex(palette.mapCounty);
  DATA_BORDER = rgba(palette.foreground, 0.15);
  FEEDER_CASING = hex(palette.mapCasing);
  SELECTED_OUTLINE = hex(palette.mapBrand);
  SELECTED_SHADOW = theme === "dark" ? rgba([0, 0, 0], 1) : hex(palette.foreground);
});

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
