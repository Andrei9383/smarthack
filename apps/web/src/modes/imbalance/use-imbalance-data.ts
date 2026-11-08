import { keepPreviousData, useQuery } from "@tanstack/react-query";
import { useMemo } from "react";

import { dayWindow } from "@/lib/replay";
import { usePolicy, useReplayStep } from "@/lib/use-replay";
import { wallClock } from "@/components/panels/site/time";
import { orpc } from "@/lib/orpc";
import { FOREVER } from "@/lib/utils";

// The global Baseline/WattSentry switch picks between PPC's method today and WattSentry. Every policy of the day is in
// one query, so the switch only re-slices rows the browser already has.
const POLICY_OF = { baseline: "profile", sentinel: "sentinel" } as const;
export type Step = { date: Date; ts: string; p10: number; p90: number; bought: number; actual: number; hasBand: boolean };

export function useImbalanceData() {
  const [step] = useReplayStep();
  const [policy] = usePolicy();
  const day = useQuery({
    ...orpc.imbalance.getImbalanceDay.queryOptions({ input: dayWindow(step) }),
    placeholderData: keepPreviousData,
    staleTime: FOREVER,
  });
  const comparison = useQuery({ ...orpc.imbalance.getPolicyComparison.queryOptions({ input: {} }), staleTime: FOREVER });
  const strategies = useQuery({ ...orpc.imbalance.getStrategyBreakdown.queryOptions({ input: {} }), staleTime: FOREVER });

  const shown = POLICY_OF[policy];
  const steps = useMemo<Step[]>(
    () =>
      (day.data?.steps ?? [])
        .filter((s) => s.policy === shown)
        .map((s) => ({
          ts: s.ts,
          date: wallClock(s.ts),
          p10: s.p10 ?? 0,
          p90: s.p90 ?? 0,
          bought: s.bought ?? 0,
          actual: s.actual ?? 0,
          hasBand: s.p10 !== null && s.p90 !== null,
        })),
    [day.data, shown],
  );
  const totals = useMemo(() => new Map((day.data?.totals ?? []).map((t) => [t.policy, t])), [day.data]);

  return {
    policy,
    shown,
    steps,
    totals,
    window: day.data,
    comparison: comparison.data ?? [],
    strategies: strategies.data,
    isLoading: day.isPending,
  };
}
