"use client";

import { useQuery } from "@tanstack/react-query";
import { ClipboardList, Gauge, type LucideIcon, Wind, Zap } from "lucide-react";

import { ProvenancePill } from "@/components/provenance-pill";
import { orpc } from "@/utils/orpc";

import { dayWindow, stepToTs } from "./replay";
import { useReplayStep } from "./use-replay";

// Value and unit are rendered separately so the unit can sit small and muted beside the number.
const format = (value: number | null | undefined, digits: number) =>
  value === null || value === undefined ? null : value.toFixed(digits);

const plural = (n: number, noun: string) => `${n} ${noun}${n === 1 ? "" : "s"}`;

// Same query keys in KpiStrip and RunContext, so React Query fetches each once.
function useOverview() {
  const [step] = useReplayStep();
  const { data } = useQuery(
    orpc.manifest.getOverview.queryOptions({
      input: { ...dayWindow(step), now: stepToTs(step) },
      placeholderData: (previous) => previous,
    }),
  );
  const { data: runs } = useQuery(orpc.manifest.getRuns.queryOptions({ input: {} }));
  return { data, label: runs?.[0]?.label };
}

export function KpiStrip() {
  const { data } = useOverview();

  const tiles: { title: string; icon: LucideIcon; value: string | null; unit?: string }[] = [
    { title: "Production today", icon: Zap, value: format(data?.productionMwh, 0), unit: "MWh" },
    { title: "Avg wind today", icon: Wind, value: format(data?.avgWindKmh, 1), unit: "km/h" },
    { title: "Park availability", icon: Gauge, value: format(data?.availabilityPct, 1), unit: "%" },
    {
      title: "Open work orders",
      icon: ClipboardList,
      value: data?.activeIssues === null || data?.activeIssues === undefined ? null : String(data.activeIssues),
    },
  ];

  return (
    <section className="shrink-0 border-b border-border bg-background">
      <div className="grid grid-cols-2 divide-border md:grid-cols-4 md:divide-x">
        {tiles.map(({ title, icon: Icon, value, unit }) => (
          <div key={title} className="flex h-20 min-w-0 items-center gap-3.5 px-6">
            <span className="flex size-10 shrink-0 items-center justify-center rounded-lg bg-brand-50 text-brand-700">
              <Icon aria-hidden className="size-5" />
            </span>
            <div className="min-w-0">
              <div className="truncate text-xs font-medium text-text-secondary">{title}</div>
              <div className="flex items-baseline gap-1 whitespace-nowrap">
                <span className="text-2xl leading-8 font-semibold tracking-tight text-foreground tabular-nums">
                  {value ?? "—"}
                </span>
                {value !== null && unit ? (
                  <span className="text-[13px] font-medium text-muted-foreground">{unit}</span>
                ) : null}
              </div>
            </div>
          </div>
        ))}
      </div>
    </section>
  );
}

/** Run context for the page header: fleet size plus the provenance pill. */
export function RunContext() {
  const { data, label } = useOverview();
  if (!data || data.homes + data.turbines === 0) return null;
  return (
    <div className="flex min-w-0 items-center gap-3 text-[13px] text-text-secondary">
      <span className="truncate tabular-nums">
        {plural(data.homes, "home")} on {plural(data.feeders, "feeder")} ·{" "}
        {plural(data.turbines, "turbine")} in {plural(data.parks, "park")}
      </span>
      {label ? <ProvenancePill label={label} /> : null}
    </div>
  );
}
