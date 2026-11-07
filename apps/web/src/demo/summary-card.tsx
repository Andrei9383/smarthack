"use client";

import { Button } from "@sentinel-app/ui/components/button";
import { useQuery } from "@tanstack/react-query";
import { X } from "lucide-react";

import { orpc } from "@/utils/orpc";

import { useSummaryOpen } from "./demo-jumps";
import { canonicalName, formatKpi } from "./kpi-format";

// Plan §7 step 7. The WindMon comparison comes from the baseline row of the same metric.
const METRICS = [
  { name: "peak_reduction", label: "Peak reduction" },
  { name: "minutes_over_limit", label: "Minutes over limit" },
  { name: "p10_hit_rate", label: "P10 hit rate" },
  { name: "comfort_breaches", label: "Comfort breaches" },
  { name: "lei_per_home", label: "Lei per home" },
  { name: "alerts_per_turbine_day", label: "Alerts per turbine-day" },
];

export function SummaryCard() {
  const [open, setOpen] = useSummaryOpen();
  const { data: kpis } = useQuery(
    orpc.manifest.getKpis.queryOptions({ input: {}, enabled: open }),
  );
  const { data: runs } = useQuery(
    orpc.manifest.getRuns.queryOptions({ input: {}, enabled: open }),
  );
  if (!open) return null;

  const find = (name: string, policy: string) =>
    kpis?.find((kpi) => canonicalName(kpi.name) === name && kpi.policy === policy);
  const realRun = runs?.find((run) => run.kind === "replay");
  const anyKpi = (kpis?.length ?? 0) > 0;

  return (
    <div className="absolute inset-x-3 top-3 z-20 mx-auto max-w-xl rounded-md border bg-background p-4 shadow-lg">
      <div className="flex items-center justify-between">
        <h2 className="font-semibold">Results</h2>
        <Button variant="ghost" size="icon" aria-label="Close summary" onClick={() => setOpen(false)}>
          <X />
        </Button>
      </div>
      {anyKpi && !realRun ? (
        <p className="mt-1 rounded bg-amber-500/15 px-2 py-1 text-xs font-medium text-amber-300">
          Fixture data for UI development. Not a result.
        </p>
      ) : null}
      <table className="mt-2 w-full text-sm">
        <thead className="text-left text-xs text-muted-foreground">
          <tr>
            <th className="font-normal">Metric</th>
            <th className="font-normal">Baseline</th>
            <th className="font-normal">Sentinel</th>
            <th className="font-normal">Label</th>
            <th className="font-normal">Run ID</th>
          </tr>
        </thead>
        <tbody>
          {METRICS.map(({ name, label }) => {
            const baseline = find(name, "baseline");
            const sentinel = find(name, "sentinel");
            const shown = sentinel ?? baseline;
            return (
              <tr key={name}>
                <td className="py-1 pr-2">{label}</td>
                <td className="tabular-nums">{baseline ? formatKpi(baseline) : "—"}</td>
                <td className="tabular-nums">{sentinel ? formatKpi(sentinel) : "—"}</td>
                <td className="text-xs text-muted-foreground">{shown?.label ?? "—"}</td>
                <td className="text-xs text-muted-foreground">{shown?.runId ?? "—"}</td>
              </tr>
            );
          })}
        </tbody>
      </table>
      {anyKpi ? null : <p className="mt-2 text-xs text-muted-foreground">No logged run yet.</p>}
    </div>
  );
}
