// Raw basemap colours for MapLibre paint properties, which can't read CSS variables.
// Mirrors team/contracts/ui-style.md §1: Romania reads as the subject, neighbours recede. One set per theme
// (CARTO Positron in light, dark-matter in dark); the RGB source is apps/web/src/lib/palette.ts.
import { hex, paletteFor, type ThemeName } from "@/lib/palette";

export function basemapFor(theme: ThemeName) {
  const palette = paletteFor(theme);
  return {
    sea: hex(palette.mapSea),
    land: hex(palette.mapLand), // Romania
    neighbour: hex(palette.mapNeighbour),
    border: hex(palette.mapBorder), // other country borders
    focusBorder: hex(palette.mapFocusBorder), // Romania's border
    county: hex(palette.mapCounty),
  } as const;
}

/** Light-theme basemap colours (kept for older imports). */
export const BASEMAP = basemapFor("light");
