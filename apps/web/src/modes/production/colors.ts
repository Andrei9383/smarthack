// Raw colour values for code that can't read CSS variables (deck.gl particle ramps, MapLibre paint).
// The RGB source is apps/web/src/lib/palette.ts (team/contracts/ui-style.md §3 particles).
import { onPaletteTheme, type Rgb, rgbaArray, type Rgba } from "@/lib/palette";

// Spec §3: production wind particles are brand green at 60%, light → strong by wind speed. The slow end is a
// mid green, not the palest tint, so it never vanishes into the land colour.
const PARTICLE_ALPHA = 0.6;
const RAMPS: Record<"light" | "dark", readonly [Rgb, Rgb, Rgb]> = {
  light: [
    [108, 217, 160], // --chart-4 (light)
    [18, 183, 106], // --brand-500 (light)
    [11, 127, 72], // --primary (light)
  ],
  // On the dark basemap the strong end is the brightest, so faster wind still reads as "more".
  dark: [
    [26, 127, 82], // --chart-4 (dark)
    [47, 191, 122], // --brand-500 (dark)
    [167, 243, 201], // --brand-900 (dark)
  ],
};

/** Theme-dependent; read when the particle layer is (re)built, and the map remounts on a theme change. */
export let PARTICLE_COLOR_RAMP: [number, Rgba][] = [];
onPaletteTheme((_palette, theme) => {
  const [slow, mid, fast] = RAMPS[theme];
  PARTICLE_COLOR_RAMP = [
    [0, rgbaArray(slow, PARTICLE_ALPHA)],
    [0.5, rgbaArray(mid, PARTICLE_ALPHA)],
    [1, rgbaArray(fast, PARTICLE_ALPHA)],
  ];
});
