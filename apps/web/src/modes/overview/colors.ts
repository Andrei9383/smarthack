// Raw colour values for MapLibre paint, which can't read Tailwind classes. They re-use the mode
// colour modules (team/contracts/ui-style.md §2–3) so Overview reads exactly like each mode.
import { onPaletteTheme, rgba } from "@/lib/palette";

export { FEEDER_CASING, FEEDER_OK, FEEDER_OVER, FEEDER_UNKNOWN, FEEDER_WARN } from "../feeders/colors";
export { STRESS_RAMP } from "../demand/colors";

// Counties are tinted quietly on Overview: the stress ramp at low opacity, borders barely there.
export const COUNTY_TINT_OPACITY = 0.38;
export const COUNTY_HOVER_OPACITY = 0.6;
/** Foreground at 8% of the current theme (the map remounts on a theme change). */
export let COUNTY_BORDER = "";
onPaletteTheme((palette) => {
  COUNTY_BORDER = rgba(palette.foreground, 0.08);
});
