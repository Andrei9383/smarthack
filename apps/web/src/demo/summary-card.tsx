"use client";

import { Button } from "@sentinel-app/ui/components/button";
import { useQuery } from "@tanstack/react-query";
import { TriangleAlert, X } from "lucide-react";

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
    <div className="absolute inset-x-4 top-4 z-20 mx-auto max-w-xl rounded-lg border border-border bg-card p-5 shadow-card">
      <div className="flex items-center justify-between gap-4">
        <h2 className="text-base font-semibold text-foreground">Results</h2>
        <Button
          variant="ghost"
          size="icon-sm"
          aria-label="Close summary"
          onClick={() => setOpen(false)}
          className="-mr-1.5 rounded-md text-muted-foreground hover:text-foreground"
        >
          <X />
        </Button>
      </div>
      {anyKpi && !realRun ? (
        <p className="mt-3 flex items-center gap-2 rounded-md bg-status-warn/10 px-3 py-2 text-xs font-medium text-foreground">
          <TriangleAlert aria-hidden className="size-3.5 shrink-0 text-status-warn" />
          Fixture data for UI development. Not a result.
        </p>
      ) : null}
      <table className="mt-4 w-full text-[13px]">
        <thead className="text-left text-[11px] tracking-[0.04em] text-muted-foreground uppercase">
          <tr className="border-b border-border">
            <th className="pb-2 font-medium">Metric</th>
            <th className="pb-2 text-right font-medium">Baseline</th>
            <th className="pb-2 text-right font-medium text-brand-700">Sentinel</th>
            <th className="pb-2 pl-4 font-medium">Label</th>
            <th className="pb-2 font-medium">Run ID</th>
          </tr>
        </thead>
        <tbody className="divide-y divide-border">
          {METRICS.map(({ name, label }) => {
            const baseline = find(name, "baseline");
            const sentinel = find(name, "sentinel");
            const shown = sentinel ?? baseline;
            return (
              <tr key={name}>
                <td className="py-2 pr-3 text-foreground">{label}</td>
                <td className="py-2 text-right text-text-secondary tabular-nums">
                  {baseline ? formatKpi(baseline) : "—"}
                </td>
                <td className="py-2 text-right font-semibold text-brand-700 tabular-nums">
                  {sentinel ? formatKpi(sentinel) : "—"}
                </td>
                <td className="py-2 pl-4 text-xs text-text-secondary">{shown?.label ?? "—"}</td>
                <td className="py-2 font-mono text-[11px] break-all text-muted-foreground">
                  {shown?.runId ?? "—"}
                </td>
              </tr>
            );
          })}
        </tbody>
      </table>
      {anyKpi ? null : <p className="mt-3 text-xs text-muted-foreground">No logged run yet.</p>}
    </div>
  );
}
