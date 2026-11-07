// Raw basemap colours for MapLibre paint properties, which can't read CSS variables.
// Mirrors team/contracts/ui-style.md §1: Romania reads as the subject, neighbours recede.
export const BASEMAP = {
  sea: "#e6eef2",
  land: "#ffffff", // Romania
  neighbour: "#f2f3f5",
  border: "#c9ced4", // other country borders
  focusBorder: "#b8bec6", // Romania's border
  county: "#dde1e5",
} as const;
