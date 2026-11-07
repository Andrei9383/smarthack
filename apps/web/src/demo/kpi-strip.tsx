"use client";

import { useQuery } from "@tanstack/react-query";

import { orpc } from "@/utils/orpc";

import { dayWindow, stepToTs } from "./replay";
import { useReplayStep } from "./use-replay";

const format = (value: number | null | undefined, digits: number, unit: string) =>
  value === null || value === undefined ? "—" : `${value.toFixed(digits)}${unit}`;

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

  const tiles = [
    { title: "Production today", value: format(data?.productionMwh, 0, " MWh") },
    { title: "Avg wind today", value: format(data?.avgWindKmh, 1, " km/h") },
    { title: "Park availability", value: format(data?.availabilityPct, 1, "%") },
    {
      title: "Open work orders",
      value: data?.activeIssues === null || data?.activeIssues === undefined ? "—" : String(data.activeIssues),
    },
  ];

  return (
    <section className="border-b">
      <div className="grid grid-cols-2 gap-px bg-border md:grid-cols-4">
        {tiles.map(({ title, value }) => (
          <div key={title} className="bg-background p-3">
            <div className="text-xs text-muted-foreground">{title}</div>
            <div className="text-xl font-semibold tabular-nums">{value}</div>
          </div>
        ))}
      </div>
      {data && data.homes + data.turbines > 0 ? (
        <p className="border-t px-3 py-1 text-xs text-muted-foreground">
          {data.homes} homes on {data.feeders} feeders · {data.turbines} turbines in {data.parks}{" "}
          parks{label ? (
            <>
              {" "}
              · <span className="font-medium text-amber-400">{label}</span>
            </>
          ) : null}
        </p>
      ) : null}
    </section>
  );
}
