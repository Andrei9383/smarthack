"use client";

import { useQuery } from "@tanstack/react-query";

import { CountUp } from "@/components/shell/count-up";
import { orpc } from "@/utils/orpc";

import { dayWindow, stepToTs } from "./replay";
import { useReplayStep } from "./use-replay";

// Same query keys everywhere (top bar, inspector, Overview), so React Query fetches each once.
export function useOverview() {
  const [step] = useReplayStep();
  const { data } = useQuery(
    orpc.manifest.getOverview.queryOptions({
      input: { ...dayWindow(step), now: stepToTs(step) },
      placeholderData: (previous) => previous,
    }),
  );
  const { data: runs } = useQuery(orpc.manifest.getRuns.queryOptions({ input: {} }));
  return { data, label: runs?.[0]?.label, runId: runs?.[0]?.id };
}

type Chip = { title: string; value: number | null | undefined; digits: number; unit?: string };

/** The KPI chips in the floating top bar: label above, number with a small muted unit. */
export function KpiChips() {
  const { data } = useOverview();
  const chips: Chip[] = [
    { title: "Production today", value: data?.productionMwh, digits: 0, unit: "MWh" },
    { title: "Avg wind today", value: data?.avgWindKmh, digits: 1, unit: "km/h" },
    { title: "Park availability", value: data?.availabilityPct, digits: 1, unit: "%" },
    { title: "Open work orders", value: data?.activeIssues, digits: 0 },
  ];

  return (
    <div className="flex min-w-0 items-stretch overflow-x-auto px-1 [scrollbar-width:none]">
      {chips.map(({ title, value, digits, unit }) => (
        <div
          key={title}
          className="flex shrink-0 flex-col justify-center border-r border-border px-2.5 py-1.5 min-[1440px]:px-3.5 whitespace-nowrap last:border-r-0"
        >
          <span className="text-[10px] font-semibold tracking-[0.06em] text-muted-foreground uppercase">{title}</span>
          <span className="font-mono text-base leading-snug font-semibold text-foreground tabular-nums">
            {value === null || value === undefined ? (
              <span className="text-muted-foreground">—</span>
            ) : (
              <>
                <CountUp value={value} digits={digits} />
                {unit ? <span className="ml-1 font-sans text-xs font-normal text-muted-foreground">{unit}</span> : null}
              </>
            )}
          </span>
        </div>
      ))}
    </div>
  );
}
