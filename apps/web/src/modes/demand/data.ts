import type { ZoneDemandItem } from "@sentinel-app/api/contract/demand";

export type Zone = {
  id: string;
  level: string;
  name: string;
  parentId: string | null;
  label: string;
  rows: ZoneDemandItem[];
  peakStress: number;
  peakTs: string;
  peakMw: number;
  stressByTs: Map<string, number>;
};

// Per-step stress = q90 / capacity. Capacity isn't stored, but peak stress is its value at the
// day's highest q90, so this holds whether peak_stress is per step (fixtures) or per day (replay).
function toZone(rows: ZoneDemandItem[]): Zone {
  const first = rows[0] as ZoneDemandItem;
  const peakStress = Math.max(...rows.map((row) => row.peakStress));
  const q90 = (row: ZoneDemandItem) => row.q90Mw ?? row.q50Mw;
  const maxQ90 = Math.max(...rows.map(q90));
  const peak = rows.reduce((best, row) => (row.q50Mw > best.q50Mw ? row : best), first);
  return {
    id: first.zoneId,
    level: first.level,
    name: first.zoneName,
    parentId: first.parentZoneId ?? null,
    label: first.label,
    rows,
    peakStress,
    peakTs: peak.ts,
    peakMw: peak.q50Mw,
    stressByTs: new Map(
      rows.map((row) => [row.ts, maxQ90 > 0 ? (peakStress * q90(row)) / maxQ90 : 0]),
    ),
  };
}

export function toZones(rows: ZoneDemandItem[]) {
  const grouped = new Map<string, ZoneDemandItem[]>();
  for (const row of rows) {
    const zoneRows = grouped.get(row.zoneId);
    if (zoneRows) zoneRows.push(row);
    else grouped.set(row.zoneId, [row]);
  }
  return new Map([...grouped].map(([id, zoneRows]) => [id, toZone(zoneRows)]));
}

export function byStress(zones: Zone[]) {
  return [...zones].sort((a, b) => b.peakStress - a.peakStress || a.name.localeCompare(b.name));
}

export function countyOf(zone: Zone | undefined, zones: Map<string, Zone>): Zone | undefined {
  let current = zone;
  while (current && current.level !== "county") {
    current = current.parentId ? zones.get(current.parentId) : undefined;
  }
  return current;
}

// Fixture zones are a few hundred homes, so most values read better in kW.
export function formatPower(mw: number) {
  return Math.abs(mw) < 1 ? `${Math.round(mw * 1000)} kW` : `${mw.toFixed(1)} MW`;
}

const normalized = (name: string) =>
  name
    .normalize("NFD")
    .replace(/\p{Diacritic}/gu, "")
    .toLowerCase();

// Fixtures identify counties by code ('CT'), replay by name ('Constanta'); the shapes carry both.
export function zonesByCountyCode(
  counties: Zone[],
  shapes: { code: string; name: string }[],
): Map<string, Zone> {
  const codeOf = new Map(
    shapes.flatMap(({ code, name }) => [
      [normalized(code), code],
      [normalized(name), code],
    ]),
  );
  return new Map(
    counties.flatMap((county) => {
      const code = codeOf.get(normalized(county.id));
      return code ? [[code, county] as const] : [];
    }),
  );
}
