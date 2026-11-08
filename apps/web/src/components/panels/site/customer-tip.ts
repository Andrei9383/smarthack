// The one sentence the homeowner sees. Plain words, no model names, and only
// facts that are in the run: if a number is missing, the sentence does without it.

export type TipInput = {
  status: string | undefined;
  reason: string | undefined;
  hasBattery: boolean;
  workOrder?: {
    fault?: string | null;
    daysToVisit?: number | null;
    lostKwhDay?: number | null;
    lostLeiDay?: number | null;
    action: string;
  };
  absorbedKwhToday: number;
  reserveKwh?: number | null;
  reserveTouched: boolean;
};

const FAULT_WORDS: Record<string, string> = {
  string_failure: "part of your panels stopped producing",
  inverter_fault: "your inverter is not converting all of its power",
  soiling: "your panels need cleaning",
  shading: "something new is shading your panels",
  degradation: "your panels produce less than they should for their age",
};

export function customerTip(t: TipInput): string {
  if (t.status === "human_review") {
    return "Your system behaves unlike anything we have seen before, so we have left your battery out of grid services until a person checks it. There is nothing you need to do.";
  }
  if (t.status === "excluded" && t.reason === "stale_telemetry") {
    return "We have not heard from your inverter for a while. Check that it is on and connected to your home Wi-Fi; until then your battery simply runs on its own.";
  }
  if (
    t.workOrder &&
    (t.status === "derated" || t.workOrder.action === "truck")
  ) {
    const what =
      FAULT_WORDS[t.workOrder.fault ?? ""] ??
      "your system is producing less than expected";
    const visit =
      t.workOrder.daysToVisit != null
        ? ` A technician is booked within ${t.workOrder.daysToVisit} day${t.workOrder.daysToVisit === 1 ? "" : "s"}.`
        : " We have booked a technician.";
    const loss =
      t.workOrder.lostKwhDay != null
        ? ` Until then you lose about ${t.workOrder.lostKwhDay.toFixed(1)} kWh a day${t.workOrder.lostLeiDay != null ? ` (≈ ${Math.round(t.workOrder.lostLeiDay)} lei)` : ""}.`
        : "";
    return `It looks like ${what}.${visit}${loss}`;
  }
  if (t.hasBattery && t.absorbedKwhToday > 0.05) {
    const reserve =
      t.reserveKwh != null && !t.reserveTouched
        ? ` Your comfort reserve of ${t.reserveKwh.toFixed(1)} kWh was never used.`
        : "";
    return `Today your battery soaked up ${t.absorbedKwhToday.toFixed(1)} kWh of midday surplus that the street could not take, and gave it back in the evening.${reserve}`;
  }
  return "Your system is working normally. Nothing to do.";
}
