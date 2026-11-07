// Hex mirrors of the brand tokens for the WebGPU shader banner, which can't read CSS variables.
// Keep in step with packages/ui/src/styles/globals.css (ui-style.md §2).
export const SHADER_COLORS = {
  white: "#ffffff",
  mist: "#f7f8fa",
  brand50: "#ecfbf2",
  brand100: "#d1f5e0",
  brand300: "#6cd9a0",
  brand500: "#12b76a",
  brand600: "#0e9f5a",
  brand700: "#0b7f48",
  forecast: "#7cc4ea",
  warn: "#f5c27a",
  critical: "#f4a3a6",
} as const;
