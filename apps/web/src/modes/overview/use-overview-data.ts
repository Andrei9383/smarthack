import { useQuery } from "@tanstack/react-query";
import { useMemo } from "react";

import { usePolicy } from "@/demo/use-replay";
import { canonicalName } from "@/demo/kpi-format";
import { useDemandData } from "@/modes/demand/use-demand-data";
import { useFeederList, useOverLimit } from "@/modes/feeders/use-feeder-data";
import { useOmAssets, useReviewQueue, useWorkOrders } from "@/modes/om/use-om-data";
import { expectedMwh } from "@/modes/production/data";
import { useProductionData } from "@/modes/production/use-production-data";
import { orpc } from "@/utils/orpc";

// Overview reuses each mode's own hooks and queries (same query keys), so it adds no new API
// and switching to a mode afterwards hits a warm cache.

export type FeederState = {
  id: string;
  name: string;
  county: string;
  limitKw: number;
  line: [number, number][];
  minutesOver: number | undefined;
  peakKw: number | undefined;
  baselineMinutes: number | undefined;
  baselinePeakKw: number | undefined;
  status: "ok" | "near" | "over" | "unknown";
};

function lineOf(geometryJson?: string | null): [number, number][] {
  if (!geometryJson) return [];
  try {
    return (JSON.parse(geometryJson) as { coordinates?: [number, number][] }).coordinates ?? [];
  } catch {
    return [];
  }
}

const NEAR_SHARE = 0.9; // a peak within 10 % of the limit counts as "near"

export function useOverviewFeeders() {
  const [policy] = usePolicy();
  const { data: feeders } = useFeederList();
  const { data: overLimit } = useOverLimit();
  return useMemo<FeederState[]>(
    () =>
      (feeders ?? []).map((feeder) => {
        const row = (name: string) => overLimit?.find((r) => r.feederId === feeder.id && r.policy === name);
        const shown = row(policy);
        const baseline = row("baseline");
        const minutesOver = shown?.minutesOver;
        const peakKw = shown?.peakExportKw;
        const status: FeederState["status"] =
          minutesOver === undefined
            ? "unknown"
            : minutesOver > 0
              ? "over"
              : (peakKw ?? 0) >= feeder.limitKw * NEAR_SHARE
                ? "near"
                : "ok";
        return {
          id: feeder.id,
          name: feeder.name,
          county: feeder.county,
          limitKw: feeder.limitKw,
          line: lineOf(feeder.geometryJson),
          minutesOver,
          peakKw,
          baselineMinutes: baseline?.minutesOver,
          baselinePeakKw: baseline?.peakExportKw,
          status,
        };
      }),
    [feeders, overLimit, policy],
  );
}

export function useOverviewData() {
  const feeders = useOverviewFeeders();
  const { data: assets } = useOmAssets();
  const { data: review } = useReviewQueue();
  const { data: orders } = useWorkOrders();
  const production = useProductionData();
  const demand = useDemandData();
  const { data: kpis } = useQuery(orpc.manifest.getKpis.queryOptions({ input: {} }));

  return useMemo(() => {
    const turbines = (assets ?? []).filter((a) => a.assetKind === "turbine");
    // Same rule as the O&M map: an excluded turbine is out of service, i.e. stopped.
    const stopped = turbines.filter((t) => t.status === "excluded");
    const waiting = (review ?? []).filter((r) => !r.resolved);
    const openOrders = (orders ?? []).filter((o) => o.status === "open");
    const topOrder = [...openOrders].sort((a, b) => b.score - a.score)[0];
    const parks = production.parks.map((park) => ({
      ...park,
      mwh: expectedMwh(production.rows.filter((row) => row.parkId === park.id)),
      stopped: stopped.filter((t) => t.parkId === park.id).length,
    }));
    const kpi = (name: string, policy: string) =>
      kpis?.find((k) => canonicalName(k.name) === name && k.policy === policy)?.value;

    return {
      feeders,
      stopped,
      waiting,
      openOrders,
      topOrder,
      parks,
      counties: demand.counties,
      demandTs: demand.ts,
      comfort: { baseline: kpi("comfort_breaches", "baseline"), sentinel: kpi("comfort_breaches", "sentinel") },
      ready: feeders.length > 0,
    };
  }, [feeders, assets, review, orders, production.parks, production.rows, demand.counties, demand.ts, kpis]);
}
