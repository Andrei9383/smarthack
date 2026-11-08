// Hex mirrors of the brand tokens for the WebGPU shader banner, which can't read CSS variables.
// Keep in step with packages/ui/src/styles/globals.css and apps/web/src/lib/palette.ts (ui-style.md §2, §7).
export const SHADER_COLORS = {
  // Light theme: pale emerald, everything above ~85% lightness so dark text keeps AA contrast.
  white: "#ffffff",
  mist: "#f6f7f8",
  brand50: "#ecfbf2",
  brand100: "#d1f5e0",
  brand300: "#6cd9a0",
  brand500: "#12b76a",
  forecast: "#7cc4ea",
  warn: "#f5c27a",
  critical: "#f4a3a6",
  // Dark theme (Calm charcoal): charcoal and emerald, everything below ~25% lightness so light text keeps AA.
  night: "#18191b",
  nightSurface: "#1f2023",
  nightBrand50: "#15352a",
  nightBrand100: "#1c4535",
  nightBrand300: "#1f6b4a",
  nightForecast: "#174a5c",
  nightWarn: "#4a3a1a",
  nightCritical: "#4a2328",
} as const;
