// Row shapes follow the agreed #38 draft (feeder_curves, schedules, feeders). Mock values only,
// until packages/api/src/routers/feeders.ts reads demo/fixtures.sqlite. DATA_LABEL says so on screen.

export const DATA_LABEL = "mock data";

export type Policy = "baseline" | "sentinel";

export type Feeder = {
  id: string;
  name: string;
  county: string;
  limit_kw: number;
  gis_trust: "ok" | "low";
  line: [number, number][];
};

export type FeederCurveRow = {
  feeder_id: string;
  policy: Policy;
  ts: string;
  export_kw: number;
  import_kw: number;
  limit_kw: number;
  slack_kw: number;
};

export type SocRow = { ts: string; soc_q10: number; soc_q50: number; soc_q90: number };
export type AbsorbingSite = { asset_id: string; absorbed_kwh: number };

const STEP_MIN = 15;
const STEPS = 96;
const DAY_START_UTC = Date.UTC(2025, 5, 2, 21, 0); // 00:00 Europe/Bucharest on 2025-06-03

export const FEEDERS: Feeder[] = [
  { id: "F7", name: "Cogealac rural", county: "CT", limit_kw: 400, gis_trust: "ok", line: [[28.42, 44.56], [28.5, 44.5], [28.56, 44.43]] },
  { id: "F3", name: "Medgidia est", county: "CT", limit_kw: 520, gis_trust: "low", line: [[28.27, 44.25], [28.36, 44.28], [28.45, 44.24]] },
  { id: "F12", name: "Barlad sud", county: "VS", limit_kw: 480, gis_trust: "ok", line: [[27.66, 46.23], [27.7, 46.17], [27.76, 46.12]] },
  { id: "F18", name: "Codlea", county: "BV", limit_kw: 650, gis_trust: "ok", line: [[25.42, 45.72], [25.5, 45.68], [25.58, 45.66]] },
  { id: "F24", name: "Popesti-Leordeni", county: "IF", limit_kw: 700, gis_trust: "low", line: [[26.13, 44.38], [26.2, 44.36], [26.28, 44.39]] },
  { id: "F31", name: "Turda vest", county: "CJ", limit_kw: 560, gis_trust: "ok", line: [[23.7, 46.58], [23.78, 46.56], [23.86, 46.52]] },
];

// Peak PV export per feeder, relative to its limit. Only F7 and F12 exceed it under the baseline.
const PEAK_RATIO: Record<string, number> = { F7: 1.32, F3: 0.9, F12: 1.12, F18: 0.74, F24: 0.82, F31: 0.68 };

const hourOf = (step: number) => (step * STEP_MIN) / 60;
const tsOf = (step: number) => new Date(DAY_START_UTC + step * STEP_MIN * 60_000).toISOString();
const bump = (hour: number, center: number, width: number) =>
  Math.exp(-((hour - center) ** 2) / (2 * width ** 2));
const round = (value: number) => Math.round(value * 10) / 10;

export function feederCurves(feederId: string): FeederCurveRow[] {
  const feeder = FEEDERS.find((f) => f.id === feederId);
  if (!feeder) return [];
  const peak = feeder.limit_kw * (PEAK_RATIO[feederId] ?? 0.8);
  const rows: FeederCurveRow[] = [];
  for (let step = 0; step < STEPS; step++) {
    const hour = hourOf(step);
    const baselineExport = peak * bump(hour, 13, 2.4);
    const baselineImport = 0.42 * feeder.limit_kw * bump(hour, 19.5, 1.6);
    const sentinelExport = Math.min(baselineExport, 0.93 * feeder.limit_kw);
    const sentinelImport = baselineImport * (1 - 0.45 * bump(hour, 19.5, 1.2));
    const common = { feeder_id: feederId, ts: tsOf(step), limit_kw: feeder.limit_kw };
    rows.push({
      ...common,
      policy: "baseline",
      export_kw: round(baselineExport),
      import_kw: round(baselineImport),
      slack_kw: round(Math.max(0, baselineExport - feeder.limit_kw)),
    });
    rows.push({
      ...common,
      policy: "sentinel",
      export_kw: round(sentinelExport),
      import_kw: round(sentinelImport),
      slack_kw: round(Math.max(0, sentinelExport - feeder.limit_kw)),
    });
  }
  return rows;
}

export function minutesOverLimit(rows: FeederCurveRow[], policy: Policy): number {
  return rows.filter((r) => r.policy === policy && r.export_kw > r.limit_kw).length * STEP_MIN;
}

export function peakExport(rows: FeederCurveRow[], policy: Policy): number {
  return Math.max(0, ...rows.filter((r) => r.policy === policy).map((r) => r.export_kw));
}

export function feederSoc(feederId: string): SocRow[] {
  const spread = feederId === "F7" ? 0.12 : 0.08;
  return Array.from({ length: STEPS }, (_, step) => {
    const hour = hourOf(step);
    const charged = Math.min(1, Math.max(0, (hour - 11) / 3));
    const discharged = Math.min(1, Math.max(0, (hour - 18) / 3));
    const q50 = 0.25 + 0.65 * charged - 0.55 * discharged;
    return {
      ts: tsOf(step),
      soc_q10: round(100 * Math.max(0.1, q50 - spread)),
      soc_q50: round(100 * q50),
      soc_q90: round(100 * Math.min(1, q50 + spread)),
    };
  });
}

export function absorbingSites(feederId: string): AbsorbingSite[] {
  if (feederId !== "F7") return [];
  return [
    { asset_id: "1042", absorbed_kwh: 18.4 },
    { asset_id: "1017", absorbed_kwh: 16.9 },
    { asset_id: "1088", absorbed_kwh: 14.2 },
    { asset_id: "1033", absorbed_kwh: 11.7 },
  ];
}
