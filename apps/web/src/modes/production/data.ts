// Row shapes follow park_forecasts (#38). Mock values only, until
// packages/api/src/routers/production.ts reads the SQLite. DATA_LABEL says so on screen.

export const DATA_LABEL = "mock data";

export type Region = "Dobrogea" | "Moldova";

export type Park = {
  id: string;
  name: string;
  county: string;
  region: Region;
  lat: number;
  lon: number;
  rated_mw: number;
};

export type ParkForecastRow = {
  park_id: string;
  ts: string;
  p10_mw: number;
  p50_mw: number;
  p90_mw: number;
};

export type SolarCounty = { county: string; lat: number; lon: number; expected_mwh: number };

const HOURS = 24;
const DAY_START_UTC = Date.UTC(2025, 5, 3, 0, 0);

export const PARKS: Park[] = [
  { id: "WP01", name: "Constanta nord", county: "CT", region: "Dobrogea", lat: 44.55, lon: 28.5, rated_mw: 120 },
  { id: "WP02", name: "Constanta vest", county: "CT", region: "Dobrogea", lat: 44.05, lon: 27.75, rated_mw: 80 },
  { id: "WP03", name: "Vaslui est", county: "VS", region: "Moldova", lat: 46.6, lon: 27.85, rated_mw: 60 },
];

export const SOLAR_COUNTIES: SolarCounty[] = [
  { county: "IF", lat: 44.5, lon: 26.1, expected_mwh: 420 },
  { county: "BV", lat: 45.7, lon: 25.5, expected_mwh: 260 },
  { county: "CJ", lat: 46.7, lon: 23.6, expected_mwh: 310 },
  { county: "TM", lat: 45.75, lon: 21.2, expected_mwh: 380 },
  { county: "DJ", lat: 44.3, lon: 23.8, expected_mwh: 340 },
];

const CAPACITY_FACTOR: Record<string, number> = { WP01: 0.62, WP02: 0.55, WP03: 0.34 };

export function parkForecast(park: Park): ParkForecastRow[] {
  const factor = CAPACITY_FACTOR[park.id] ?? 0.4;
  return Array.from({ length: HOURS }, (_, hour) => {
    const shape = 0.72 + 0.28 * Math.cos(((hour - 3) / 24) * 2 * Math.PI);
    const p50 = park.rated_mw * factor * shape;
    const spread = 0.12 + 0.012 * hour;
    return {
      park_id: park.id,
      ts: new Date(DAY_START_UTC + hour * 3_600_000).toISOString(),
      p10_mw: Math.round(p50 * (1 - spread) * 10) / 10,
      p50_mw: Math.round(p50 * 10) / 10,
      p90_mw: Math.round(Math.min(park.rated_mw, p50 * (1 + spread)) * 10) / 10,
    };
  });
}

export function expectedMwh(rows: ParkForecastRow[]): number {
  return rows.reduce((total, row) => total + row.p50_mw, 0); // hourly rows, so MW sums to MWh
}
