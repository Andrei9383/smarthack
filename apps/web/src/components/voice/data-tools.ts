"use client";

import { DAYS, dayWindow, STEPS_PER_DAY, stepToDate, stepToTs } from "@/demo/replay";
import { client } from "@/utils/orpc";

// What "Ask WattSentry" can look up (card #207). The agent runs in ElevenLabs' cloud; these run in
// the browser as client tools and read the same API the screens use, so a spoken answer matches
// what is on screen. Each returns a short plain-text block the agent can read out.

const DATE = new Intl.DateTimeFormat("en-GB", {
  timeZone: "Europe/Bucharest",
  weekday: "long",
  day: "numeric",
  month: "long",
});
const ISO_DAY = new Intl.DateTimeFormat("en-CA", { timeZone: "Europe/Bucharest" }); // YYYY-MM-DD

const dayName = (day: number) => DATE.format(stepToDate(day * STEPS_PER_DAY));
const dayIso = (day: number) => ISO_DAY.format(stepToDate(day * STEPS_PER_DAY));
const WEEKDAYS = ["monday", "tuesday", "wednesday", "thursday", "friday", "saturday", "sunday"];

/**
 * The replay week is Monday 2 – Sunday 8 June 2025. Accepts "today", "yesterday", "tomorrow", a weekday,
 * "2025-06-04", "4 June", "June 4" or "day 3" (1 = Monday); anything else means the replay's current day.
 */
export function parseDay(input: unknown, currentDay: number): { day: number } | { error: string } {
  const text = String(input ?? "").trim().toLowerCase();
  if (!text || text === "today" || text === "now") return { day: currentDay };
  const clamp = (day: number) =>
    day >= 0 && day < DAYS ? { day } : { error: `The replay covers ${dayName(0)} to ${dayName(DAYS - 1)} 2025 only.` };
  if (text === "yesterday") return clamp(currentDay - 1);
  if (text === "tomorrow") return clamp(currentDay + 1);
  const weekday = WEEKDAYS.findIndex((name) => text.startsWith(name.slice(0, 3)));
  if (weekday >= 0) return { day: weekday };
  const iso = text.match(/2025-06-(\d{1,2})/);
  const dayOfMonth = iso?.[1] ?? text.match(/(?:^|\D)(\d{1,2})(?:st|nd|rd|th)?\s*(?:of\s*)?june|june\s*(\d{1,2})/)?.slice(1).find(Boolean);
  if (dayOfMonth) return clamp(Number(dayOfMonth) - 2);
  const index = text.match(/^(?:day\s*)?(\d)$/);
  if (index) return clamp(Number(index[1]) - 1);
  return { error: `I couldn't read "${input}" as a day. The replay week is ${dayName(0)} to ${dayName(DAYS - 1)} 2025.` };
}

const round = (value: number | null | undefined, digits = 0) =>
  value === null || value === undefined ? "unknown" : value.toFixed(digits);
const say = (lines: (string | null | false | undefined)[]) => lines.filter(Boolean).join("\n");

/** Feeders: where they are, their limits, and minutes over the limit on one day, with and without WattSentry. */
export async function getFeeders(dayInput: unknown, currentDay: number) {
  const parsed = parseDay(dayInput, currentDay);
  if ("error" in parsed) return parsed.error;
  const window = dayWindow(parsed.day * STEPS_PER_DAY);
  const [feeders, overLimit] = await Promise.all([client.feeders.list({}), client.feeders.getOverLimit(window)]);
  const row = (id: string, policy: string) => overLimit.find((r) => r.feederId === id && r.policy === policy);
  const within = (policy: string) => feeders.filter((f) => (row(f.id, policy)?.minutesOver ?? 0) === 0).length;
  return say([
    `Feeders on ${dayName(parsed.day)} 2025 (low-voltage feeders in Romania; limit = what the line can take back from rooftop solar).`,
    `Within their limit all day: ${within("sentinel")} of ${feeders.length} with WattSentry, ${within("baseline")} of ${feeders.length} without WattSentry.`,
    ...feeders.map((f) => {
      const withAgent = row(f.id, "sentinel");
      const without = row(f.id, "baseline");
      return `- ${f.name} (${f.locality}, ${f.county} county): limit ${round(f.limitKw)} kW. With WattSentry: ${round(withAgent?.minutesOver)} min over, peak ${round(withAgent?.peakExportKw)} kW. Without: ${round(without?.minutesOver)} min over, peak ${round(without?.peakExportKw)} kW.`;
    }),
  ]);
}

/** Wind parks and turbines: where they are, size, and which turbines are stopped right now. */
export async function getWindParks(currentStep: number) {
  const [parks, assets] = await Promise.all([
    client.production.listParks({}),
    client.om.listAssets({ assetKind: "turbine", at: stepToTs(currentStep) }),
  ]);
  const stopped = assets.filter((a) => a.status === "excluded");
  return say([
    "Wind parks (eolian parks). All are in Romania; the data is synthetic but placed at real Romanian sites.",
    ...parks.map((p) => {
      const down = stopped.filter((t) => t.parkId === p.id);
      return `- ${p.name}, ${p.county} county (lat ${p.lat.toFixed(2)}, lon ${p.lon.toFixed(2)}): ${p.turbineCount} turbines, ${round(p.ratedMw, 1)} MW rated. Stopped now: ${down.length}${down.length ? ` (${down.map((t) => t.name || t.id).slice(0, 8).join(", ")})` : ""}.`;
    }),
  ]);
}

/** Faults on one day: work orders, alerts and cases handed to a person. */
export async function getFaults(dayInput: unknown, currentDay: number) {
  const parsed = parseDay(dayInput, currentDay);
  if ("error" in parsed) return parsed.error;
  const { from, to } = dayWindow(parsed.day * STEPS_PER_DAY);
  const [orders, alerts, reviews] = await Promise.all([
    client.om.listWorkOrders({}),
    client.om.listAlerts({ until: to }),
    client.om.listReviewQueue({ until: to }),
  ]);
  const iso = dayIso(parsed.day);
  const dayOrders = orders.filter((o) => o.day === iso).sort((a, b) => b.score - a.score);
  const dayAlerts = alerts.filter((a) => a.ts >= from && a.ts < to);
  const dayReviews = reviews.filter((r) => r.ts >= from && r.ts < to);
  const fault = (value?: string | null) => (value ?? "unknown").replace(/^pv_fault:/, "").replaceAll("_", " ");
  const codes = new Map<string, number>();
  for (const alert of dayAlerts) codes.set(alert.code, (codes.get(alert.code) ?? 0) + 1);
  return say([
    `Faults on ${dayName(parsed.day)} 2025.`,
    `Work orders: ${dayOrders.length}.`,
    ...dayOrders.slice(0, 15).map(
      (o) =>
        `- ${o.assetKind === "turbine" ? "Turbine" : "Home"} ${o.assetId}: ${fault(o.fault)}, ${o.crewType.replaceAll("_", " ")}, confidence ${Math.round(o.confidence * 100)}%, status ${o.status}${o.lostLeiDay ? `, about ${round(o.lostLeiDay)} lei lost per day` : ""}.`,
    ),
    dayOrders.length > 15 && `(and ${dayOrders.length - 15} more)`,
    `Alerts: ${dayAlerts.length}${codes.size ? ` (${[...codes].map(([code, n]) => `${n} ${code.toLowerCase().replaceAll("_", " ")}`).join(", ")})` : ""}.`,
    `Cases handed to a person because the agent was not sure: ${dayReviews.length}.`,
  ]);
}

/** One day at a glance plus the week's results against no WattSentry. */
export async function getDaySummary(dayInput: unknown, currentDay: number, currentStep: number) {
  const parsed = parseDay(dayInput, currentDay);
  if ("error" in parsed) return parsed.error;
  const window = dayWindow(parsed.day * STEPS_PER_DAY);
  const now = parsed.day === currentDay ? stepToTs(currentStep) : window.to;
  const [overview, kpis] = await Promise.all([
    client.manifest.getOverview({ ...window, now }),
    client.manifest.getKpis({}),
  ]);
  // Week-level KPIs only (the per-feeder-day ones like minutes_over_limit_F7_d2 are left out).
  const KPI_NAMES: Record<string, string> = {
    minutes_over_limit: "Minutes over the feeder limits",
    energy_over_limit: "Energy over the feeder limits",
    peak_overshoot: "Average peak above the limit",
    comfort_breaches: "Comfort breaches (homes below their battery comfort reserve)",
    p10_hit_rate: "Share of flexibility promises delivered (P10 hit rate)",
    peak_reduction: "Evening peak reduction",
    peak_reduction_congested: "Evening peak reduction on congested feeders",
    lei_per_home: "Value per home",
    alerts_per_turbine_day: "Alerts per turbine per day",
    clip_rate: "Inverter clipping rate",
  };
  const FRACTION = new Set(["p10_hit_rate", "peak_reduction", "peak_reduction_congested", "clip_rate"]);
  const byName = new Map<string, { sentinel?: number; baseline?: number; unit?: string | null }>();
  for (const k of kpis) {
    if (!(k.name in KPI_NAMES)) continue;
    const entry = byName.get(k.name) ?? { unit: k.unit };
    if (k.policy === "sentinel") entry.sentinel = k.value;
    if (k.policy === "baseline") entry.baseline = k.value;
    byName.set(k.name, entry);
  }
  const value = (name: string, v: number | undefined, unit?: string | null) =>
    v === undefined ? "not measured" : FRACTION.has(name) ? `${round(v * 100, 1)}%` : `${round(v, 1)}${unit ? ` ${unit}` : ""}`;
  return say([
    `${dayName(parsed.day)} 2025 at a glance: ${overview.feeders} feeders, ${overview.homes} homes, ${overview.parks} wind park(s) with ${overview.turbines} turbines.`,
    `Wind production: ${round(overview.productionMwh)} MWh, average wind ${round(overview.avgWindKmh)} km/h, park availability ${round(overview.availabilityPct, 1)}%, open issues ${round(overview.activeIssues)}.`,
    "Results for the whole week, with WattSentry vs without:",
    ...[...byName].map(
      ([name, k]) =>
        `- ${KPI_NAMES[name]}: ${value(name, k.sentinel, k.unit)} with WattSentry, ${value(name, k.baseline, k.unit)} without.`,
    ),
  ]);
}
