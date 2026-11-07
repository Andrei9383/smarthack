// Raw colour values for code that can't read CSS variables (deck.gl particle ramps, MapLibre paint).
// These mirror the brand tokens in team/contracts/ui-style.md (§2 brand green, §3 particles) and
// packages/ui/src/styles/globals.css. Keep them in sync with the spec; components use the Tailwind tokens.

type Rgba = [number, number, number, number];

const BRAND_400 = "#6CD9A0"; // --chart-4
const BRAND_500 = "#12B76A"; // --brand-500
const BRAND_700 = "#0B7F48"; // --brand-700

// Spec §3: production wind particles are brand-700 at 60% on the light basemap.
const PARTICLE_ALPHA = 0.6;

function rgba(hex: string, alpha = 1): Rgba {
  const value = Number.parseInt(hex.slice(1), 16);
  return [(value >> 16) & 255, (value >> 8) & 255, value & 255, Math.round(alpha * 255)];
}

// Light → strong green by wind speed. The slow end starts at chart-4, not brand-100, because
// brand-100 vanishes against the #F7F8FA land.
export const PARTICLE_COLOR_RAMP: [number, Rgba][] = [
  [0, rgba(BRAND_400, PARTICLE_ALPHA)],
  [0.5, rgba(BRAND_500, PARTICLE_ALPHA)],
  [1, rgba(BRAND_700, PARTICLE_ALPHA)],
];
