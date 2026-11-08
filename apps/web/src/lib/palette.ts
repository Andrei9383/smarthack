// Theme-aware RGB mirrors of the tokens in packages/ui/src/styles/globals.css, for deck.gl, MapLibre and
// canvas code that can't read CSS variables. Keep them in step with the CSS.
//
// - `MAP_SAFE` holds the status hues drawn on the map. They read on both basemaps (≥ 3:1 against
//   Positron's white land and our dark land), so layer modules may capture them at import.
// - `onPaletteTheme()` lets a per-mode colors.ts keep its theme-dependent exports (casings, borders,
//   no-data fills, basemap tints) in step; app-shell remounts the map when the theme changes, so the
//   layers re-read them.
import { useTheme } from "next-themes";

export type Rgb = [number, number, number];
export type Rgba = [number, number, number, number];
export type ThemeName = "light" | "dark";

/** The app opens in the light theme (providers.tsx). */
export const DEFAULT_THEME: ThemeName = "light";

const LIGHT = {
  surface: [255, 255, 255],
  border: [231, 232, 235],
  foreground: [17, 17, 19],
  // Basemap tints painted over CARTO Positron by lib/use-basemap.ts: neutral greys, Romania in white.
  mapSea: [234, 237, 240],
  mapLand: [255, 255, 255],
  mapNeighbour: [243, 244, 246],
  mapBorder: [206, 208, 213],
  mapFocusBorder: [186, 188, 194],
  mapCounty: [226, 227, 231],
  /** Halo under map lines and gaps cut into them: the land colour, so lines lift off the basemap. */
  mapCasing: [255, 255, 255],
  /** Map line or particle colour for brand green on this basemap. */
  mapBrand: [11, 127, 72],
} as const satisfies Record<string, Rgb>;

const DARK = {
  surface: [31, 32, 35],
  border: [46, 47, 51],
  foreground: [237, 237, 239],
  // Basemap tints painted over CARTO dark-matter: Romania sits a step above the frame, like a card.
  mapSea: [17, 18, 20],
  mapLand: [36, 37, 41],
  mapNeighbour: [26, 27, 30],
  mapBorder: [56, 57, 62],
  mapFocusBorder: [82, 83, 90],
  mapCounty: [48, 49, 54],
  mapCasing: [24, 25, 27],
  mapBrand: [61, 214, 140],
} as const satisfies Record<string, Rgb>;

type Palette = { readonly [K in keyof typeof LIGHT]: Rgb };

const PALETTES: Record<ThemeName, Palette> = { light: LIGHT, dark: DARK };

/** Narrows next-themes' `resolvedTheme` (or anything else) to a theme name; unknown means the default. */
export function themeName(theme: string | null | undefined): ThemeName {
  return theme === "light" || theme === "dark" ? theme : DEFAULT_THEME;
}

/** The RGB palette for a theme. */
export function paletteFor(theme: string | null | undefined): Palette {
  return PALETTES[themeName(theme)];
}

/** The current theme's name, from next-themes (client components only). */
function useThemeName(): ThemeName {
  const { resolvedTheme } = useTheme();
  return themeName(resolvedTheme);
}

/** `#rrggbb` for MapLibre paint and canvas fills. */
export function hex([r, g, b]: Rgb): string {
  return `#${[r, g, b].map((channel) => channel.toString(16).padStart(2, "0")).join("")}`;
}

/** `rgba(…)` string with an alpha, for MapLibre paint and canvas strokes. */
export function rgba([r, g, b]: Rgb, alpha: number): string {
  return `rgba(${r}, ${g}, ${b}, ${alpha})`;
}

/** deck.gl colour: RGB plus a 0–255 alpha. */
export function rgbaArray([r, g, b]: Rgb, alpha = 1): Rgba {
  return [r, g, b, Math.round(alpha * 255)];
}

/**
 * Status hues for map overlays. Mid-tones on purpose: each keeps ≥ 3:1 against both Positron's white land
 * and the dark land above, so a feeder line or turbine mark reads the same in either theme. UI text and
 * badges use the per-theme --status-* tokens instead.
 */
export const MAP_SAFE = {
  ok: [14, 159, 90],
  warn: [217, 119, 6],
  critical: [229, 72, 77],
  forecast: [2, 132, 199],
  soc: [99, 102, 241],
  review: [139, 92, 246],
  unknown: [122, 136, 153],
  brand: [18, 183, 106],
} as const satisfies Record<string, Rgb>;

// The theme the map layers are drawn for. app-shell / mini-map set it before they (re)mount the map.
let activeTheme: ThemeName = DEFAULT_THEME;
const listeners = new Set<(palette: Palette, theme: ThemeName) => void>();

/** The theme the map layers are currently drawn for. */
export function activePaletteTheme(): ThemeName {
  return activeTheme;
}

/** Sets the theme for theme-dependent colour exports. Idempotent, so it's safe to call while rendering. */
function setPaletteTheme(theme: string | null | undefined) {
  const next = themeName(theme);
  if (next === activeTheme) return;
  activeTheme = next;
  for (const listener of listeners) listener(PALETTES[next], next);
}

/** Runs `listener` now and on every theme change. colors.ts modules use it to update `export let` values. */
export function onPaletteTheme(listener: (palette: Palette, theme: ThemeName) => void) {
  listeners.add(listener);
  listener(PALETTES[activeTheme], activeTheme);
  return () => listeners.delete(listener);
}

/**
 * Syncs the active palette theme with next-themes during render and returns the theme name, for use as
 * part of the map's `key`: a theme change remounts the map, so every layer re-reads its colours.
 */
export function useMapTheme(): ThemeName {
  const theme = useThemeName();
  setPaletteTheme(theme);
  return theme;
}
