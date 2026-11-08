// Small helpers shared across the app.

/** Right map padding that keeps the focus clear of the inspector (panel width + gutter). */
export const PANEL_PADDING_PX = 452;

/** staleTime for replay queries: the replay database never changes while the app runs. */
export const FOREVER = Number.POSITIVE_INFINITY;

const BUCHAREST_HM = new Intl.DateTimeFormat("en-GB", {
  timeZone: "Europe/Bucharest",
  hour: "2-digit",
  minute: "2-digit",
  hourCycle: "h23",
});
/** "13:45", Bucharest wall-clock time, for an ISO timestamp, epoch ms or Date. */
export const hm = (t: string | number | Date) => BUCHAREST_HM.format(new Date(t));

export const lei = (v: number) => Math.round(v).toLocaleString("en-GB");

/** Coordinates of a stored GeoJSON LineString, or [] when missing or malformed. */
export function lineOf(geometryJson?: string | null): [number, number][] {
  if (!geometryJson) return [];
  try {
    return (JSON.parse(geometryJson) as { coordinates?: [number, number][] }).coordinates ?? [];
  } catch {
    return [];
  }
}

export function parseJson(text: string | null | undefined): Record<string, unknown> | null {
  if (!text) return null;
  try {
    return JSON.parse(text) as Record<string, unknown>;
  } catch {
    return null;
  }
}
