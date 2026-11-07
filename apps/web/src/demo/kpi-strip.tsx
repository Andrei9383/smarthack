"use client";

import { useQuery } from "@tanstack/react-query";

import { orpc } from "@/utils/orpc";

import { dayWindow, stepToTs } from "./replay";
import { useReplayStep } from "./use-replay";

// Value and unit are rendered separately so the unit can sit small and muted beside the number.
const format = (value: number | null | undefined, digits: number) =>
  value === null || value === undefined ? null : value.toFixed(digits);

const plural = (n: number, noun: string) => `${n} ${noun}${n === 1 ? "" : "s"}`;

export function KpiStrip() {
  const [step] = useReplayStep();
  const { data } = useQuery(
    orpc.manifest.getOverview.queryOptions({
      input: { ...dayWindow(step), now: stepToTs(step) },
      placeholderData: (previous) => previous,
    }),
  );
  const { data: runs } = useQuery(orpc.manifest.getRuns.queryOptions({ input: {} }));
  const label = runs?.[0]?.label;

  const tiles: { title: string; value: string | null; unit?: string }[] = [
    { title: "Production today", value: format(data?.productionMwh, 0), unit: "MWh" },
    { title: "Avg wind today", value: format(data?.avgWindKmh, 1), unit: "km/h" },
    { title: "Park availability", value: format(data?.availabilityPct, 1), unit: "%" },
    {
      title: "Open work orders",
      value: data?.activeIssues === null || data?.activeIssues === undefined ? null : String(data.activeIssues),
    },
  ];

  return (
    <section className="shrink-0 border-b border-border bg-sidebar">
      <div className="grid grid-cols-2 gap-x-6 gap-y-4 px-6 py-4 md:grid-cols-4">
        {tiles.map(({ title, value, unit }) => (
          <div key={title} className="min-w-0">
            <div className="truncate text-[11px] font-medium tracking-[0.04em] text-muted-foreground uppercase">
              {title}
            </div>
            <div className="mt-1 flex items-baseline gap-1 whitespace-nowrap">
              <span className="text-[28px] leading-9 font-semibold tracking-tight text-foreground tabular-nums">
                {value ?? "—"}
              </span>
              {value !== null && unit ? (
                <span className="text-[13px] text-muted-foreground">{unit}</span>
              ) : null}
            </div>
          </div>
        ))}
      </div>
      {data && data.homes + data.turbines > 0 ? (
        <p className="flex flex-wrap items-center gap-x-2 gap-y-1 border-t border-border px-6 py-2 text-xs text-text-secondary">
          <span>
            {plural(data.homes, "home")} on {plural(data.feeders, "feeder")} ·{" "}
            {plural(data.turbines, "turbine")} in {plural(data.parks, "park")}
          </span>
          {label ? (
            <span className="inline-flex items-center gap-1.5 rounded-full border border-border bg-background px-2 py-0.5 font-medium text-foreground">
              <span aria-hidden className="size-1.5 rounded-full bg-status-warn" />
              {label}
            </span>
          ) : null}
        </p>
      ) : null}
    </section>
  );
}
