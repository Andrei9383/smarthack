"use client";

import { Button } from "@sentinel-app/ui/components/button";
import { useQuery } from "@tanstack/react-query";
import { TriangleAlert, X } from "lucide-react";

import { Fold } from "@/panels/disclosure";

import { orpc } from "@/utils/orpc";

import { useSummaryOpen } from "./demo-jumps";
import { canonicalName, formatKpiParts, type Kpi } from "./kpi-format";

// Plan §7 step 7. The WindMon comparison comes from the baseline row of the same metric.
const METRICS = [
  { name: "peak_reduction", label: "Peak reduction" },
  { name: "minutes_over_limit", label: "Minutes over limit" },
  // kWh and kW over the limit, next to minutes (team/insights.md, replay#506): minutes can rise while the overload falls.
  { name: "energy_over_limit", label: "Energy over limit" },
  { name: "peak_overshoot", label: "Peak over limit" },
  { name: "p10_hit_rate", label: "P10 hit rate" },
  { name: "comfort_breaches", label: "Comfort breaches" },
  { name: "lei_per_home", label: "Lei per home" },
  { name: "alerts_per_turbine_day", label: "Alerts per turbine-day" },
];

// The three numbers the summary leads with.
const HEROES = ["peak_reduction", "minutes_over_limit", "comfort_breaches"];

// Sentinel-only metrics that are already measured against the baseline; their hero tile says so.
const VS_BASELINE: Record<string, string> = { peak_reduction: "Feeder peak cut vs baseline" };

// Which way is better, for colouring the change chip. Metrics not listed get a neutral chip.
const BETTER: Record<string, "up" | "down"> = {
  peak_reduction: "up",
  minutes_over_limit: "down",
  energy_over_limit: "down",
  peak_overshoot: "down",
  comfort_breaches: "down",
  alerts_per_turbine_day: "down",
};

function change(name: string, baseline?: Kpi, sentinel?: Kpi) {
  if (!baseline || !sentinel || baseline.value === 0) return null;
  const pct = ((sentinel.value - baseline.value) / Math.abs(baseline.value)) * 100;
  const better = BETTER[name];
  const good = better === "down" ? pct < 0 : better === "up" ? pct > 0 : null;
  const text = `${pct < 0 ? "\u2212" : "+"}${Math.abs(pct).toFixed(0)}%`;
  return { text, good };
}

function ChangeChip({ delta }: { delta: ReturnType<typeof change> }) {
  if (!delta) return null;
  const tone =
    delta.good === true
      ? "bg-status-ok/10 text-status-ok"
      : delta.good === false
        ? "bg-status-critical/10 text-status-critical"
        : "bg-muted text-text-secondary";
  return (
    <span className={`inline-flex rounded-full px-2 py-0.5 text-[11px] font-semibold whitespace-nowrap tabular-nums ${tone}`}>
      {delta.text}
    </span>
  );
}

/** Number with its unit small and muted beside it; the pair never wraps. */
function Value({ kpi, className = "" }: { kpi?: Kpi; className?: string }) {
  if (!kpi) return <span className="text-muted-foreground">—</span>;
  const { number, unit } = formatKpiParts(kpi);
  return (
    <span className={`whitespace-nowrap tabular-nums ${className}`}>
      {number}
      {unit ? <span className="ml-0.5 text-[0.75em] font-normal text-muted-foreground">{unit}</span> : null}
    </span>
  );
}

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

  const runIds = [
    ...new Set(
      METRICS.flatMap(({ name }) => [find(name, "baseline")?.runId, find(name, "sentinel")?.runId]).filter(
        (id): id is string => Boolean(id),
      ),
    ),
  ];

  return (
    <>
      <div
        aria-hidden
        onClick={() => setOpen(false)}
        className="fixed inset-0 z-40 bg-scrim backdrop-blur-[2px] animate-in fade-in-0 duration-150"
      />
      <div
        role="dialog"
        aria-label="Results"
        className="fixed top-[10vh] left-1/2 z-50 max-h-[80vh] w-[720px] max-w-[calc(100vw-2rem)] -translate-x-1/2 overflow-y-auto rounded-2xl bg-card p-7 shadow-overlay animate-in fade-in-0 zoom-in-[0.98] duration-150"
      >
        <div className="flex items-center justify-between gap-4">
          <div className="flex items-center gap-3">
            <h2 className="text-page text-foreground">Results</h2>
          </div>
          <Button
            variant="ghost"
            size="icon-sm"
            aria-label="Close summary"
            onClick={() => setOpen(false)}
            className="-mr-1.5 rounded-md text-muted-foreground transition-colors duration-150 hover:text-foreground"
          >
            <X />
          </Button>
        </div>
        {anyKpi && !realRun ? (
          <p className="mt-4 flex items-center gap-2 rounded-md bg-status-warn/10 px-3 py-2 text-xs font-medium text-foreground">
            <TriangleAlert aria-hidden className="size-3.5 shrink-0 text-status-warn" />
            Fixture data for UI development. Not a result.
          </p>
        ) : null}

        {anyKpi ? (
          <div className="mt-6 grid grid-cols-1 gap-y-4 sm:grid-cols-3 sm:divide-x sm:divide-border">
            {HEROES.map((name) => {
              const metric = METRICS.find((m) => m.name === name);
              const baseline = find(name, "baseline");
              const sentinel = find(name, "sentinel");
              const shown = sentinel ?? baseline;
              return (
                <div key={name} className="min-w-0 sm:px-5 sm:first:pl-0">
                  <div className="truncate text-caption font-medium text-muted-foreground">{metric?.label}</div>
                  <div className="mt-1.5 flex items-center gap-2">
                    <Value kpi={shown} className="text-kpi text-foreground" />
                    <ChangeChip delta={change(name, baseline, sentinel)} />
                  </div>
                  <div className="mt-1.5 h-4 text-caption text-muted-foreground">
                    {baseline && sentinel ? (
                      <>
                        Baseline <Value kpi={baseline} className="text-text-secondary" />
                      </>
                    ) : (
                      VS_BASELINE[name] ?? null
                    )}
                  </div>
                </div>
              );
            })}
          </div>
        ) : null}

        <div className="mt-7">
          <Fold title="All metrics, baseline vs Sentinel" aside={`${METRICS.length} metrics`}>
            <table className="w-full text-dense">
              <thead className="text-left text-caption text-muted-foreground">
                <tr className="border-b border-border">
                  <th className="pb-2 font-medium">Metric</th>
                  <th className="pb-2 text-right font-medium">Baseline</th>
                  <th className="pb-2 text-right font-medium text-status-ok">Sentinel</th>
                  <th className="w-20 pb-2 text-right font-medium">Change</th>
                </tr>
              </thead>
              <tbody className="divide-y divide-border">
                {METRICS.map(({ name, label: metricLabel }) => {
                  const baseline = find(name, "baseline");
                  const sentinel = find(name, "sentinel");
                  return (
                    <tr key={name}>
                      <td className="py-2.5 pr-3 whitespace-nowrap text-foreground">{metricLabel}</td>
                      <td className="py-2.5 pl-3 text-right text-text-secondary">
                        <Value kpi={baseline} />
                      </td>
                      <td className="py-2.5 pl-3 text-right font-semibold text-foreground">
                        <Value kpi={sentinel} />
                      </td>
                      <td className="py-2.5 pl-3 text-right">
                        <ChangeChip delta={change(name, baseline, sentinel)} />
                      </td>
                    </tr>
                  );
                })}
              </tbody>
            </table>
          </Fold>
        </div>
        {anyKpi ? null : <p className="mt-3 text-xs text-muted-foreground">No logged run yet.</p>}
      </div>
    </>
  );
}
