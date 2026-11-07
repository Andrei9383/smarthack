export type Kpi = {
  runId: string;
  policy: string;
  name: string;
  value: number;
  unit?: string | null;
  label: string;
};

// replay.py stores these as a 0..1 fraction with no unit; fixtures store percent with a unit.
const FRACTIONS = new Set(["peak_reduction", "p10_hit_rate", "clip_rate"]);
const ALIASES: Record<string, string> = { peak_reduction_pct: "peak_reduction" };

export function canonicalName(name: string) {
  return ALIASES[name] ?? name;
}

/** Number and unit apart, so the unit can be rendered small and muted (and never wrap). */
export function formatKpiParts(kpi: Kpi): { number: string; unit: string } {
  const name = canonicalName(kpi.name);
  if (FRACTIONS.has(name)) {
    const percent = kpi.unit === "%" ? kpi.value : kpi.value * 100;
    return { number: percent.toFixed(1), unit: "%" };
  }
  const number =
    Number.isInteger(kpi.value) || Math.abs(kpi.value) >= 100 ? kpi.value.toFixed(0) : kpi.value.toFixed(2);
  const unit = kpi.unit && kpi.unit !== "count" ? kpi.unit : "";
  return { number, unit };
}

