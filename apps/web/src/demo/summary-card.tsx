"use client";

import { Button } from "@sentinel-app/ui/components/button";
import { X } from "lucide-react";

import { useSummaryOpen } from "./demo-jumps";

// Filled from kpis + runs once #38 and #29 land. Empty until then: never invent a result.
const METRICS = [
  "Peak reduction",
  "P10 hit rate",
  "Comfort breaches",
  "Lei per home",
  "Alerts per turbine-day vs WindMon rules",
];

export function SummaryCard() {
  const [open, setOpen] = useSummaryOpen();
  if (!open) return null;

  return (
    <div className="absolute inset-x-3 top-3 z-20 mx-auto max-w-md rounded-md border bg-background p-4 shadow-lg">
      <div className="flex items-center justify-between">
        <h2 className="font-semibold">Results</h2>
        <Button variant="ghost" size="icon" aria-label="Close summary" onClick={() => setOpen(false)}>
          <X />
        </Button>
      </div>
      <table className="mt-2 w-full text-sm">
        <thead className="text-left text-xs text-muted-foreground">
          <tr>
            <th className="font-normal">Metric</th>
            <th className="font-normal">Value</th>
            <th className="font-normal">Run ID</th>
          </tr>
        </thead>
        <tbody>
          {METRICS.map((metric) => (
            <tr key={metric}>
              <td className="py-1 pr-2">{metric}</td>
              <td>—</td>
              <td>—</td>
            </tr>
          ))}
        </tbody>
      </table>
      <p className="mt-2 text-xs text-muted-foreground">No logged run promoted yet.</p>
    </div>
  );
}
