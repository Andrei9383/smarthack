import { keepPreviousData, useQueries, useQuery } from "@tanstack/react-query";
import { useMemo } from "react";

import { dayWindow, stepToTs } from "@/demo/replay";
import { type Policy, useReplayStep } from "@/demo/use-replay";
import { orpc } from "@/utils/orpc";

export function useDayWindow() {
  const [step] = useReplayStep();
  return dayWindow(step);
}

export function useFeederList() {
  return useQuery(orpc.feeders.list.queryOptions({ input: {} }));
}

export function useOverLimit() {
  const window = useDayWindow();
  return useQuery(orpc.feeders.getOverLimit.queryOptions({ input: window }));
}

export function useRunLabel() {
  const { data } = useQuery(orpc.manifest.getRuns.queryOptions({ input: {} }));
  return data?.[0]?.label;
}

export type FeederNow = {
  exportKw: number;
  limitKw: number;
  /** Sentinel only: the feeder's batteries gained charge in this 15-min step. */
  charging: boolean;
};

// Stable reference, so useQueries memoises the combined array between renders.
const dataOf = <T,>(results: { data?: T }[]) => results.map((result) => result.data);

const SOC_RISE = 0.002; // median SoC gain per step that counts as charging (0..1 scale)

/**
 * Each feeder's export and battery state at the replay clock, for the map (#115). Uses the same
 * day-window getCurves / getSocFan queries as the feeder panel, so the selected feeder is cached.
 */
export function useFeedersNow(feederIds: string[], policy: Policy) {
  const [step] = useReplayStep();
  const window = dayWindow(step);
  const ts = stepToTs(step);
  const curveData = useQueries({
    queries: feederIds.map((feederId) => ({
      ...orpc.feeders.getCurves.queryOptions({ input: { feederId, ...window } }),
      placeholderData: keepPreviousData,
    })),
    combine: dataOf,
  });
  const socData = useQueries({
    queries: feederIds.map((feederId) => ({
      ...orpc.feeders.getSocFan.queryOptions({
        input: { feederId, ...window, policy: "sentinel" }, // replay stores schedules for Sentinel only
        enabled: policy === "sentinel",
      }),
      placeholderData: keepPreviousData,
    })),
    combine: dataOf,
  });

  return useMemo(() => {
    const now = new Map<string, FeederNow>();
    feederIds.forEach((feederId, index) => {
      const row = curveData[index]?.find((r) => r.ts === ts && r.policy === policy);
      if (!row) return;
      let charging = false;
      const fan = socData[index];
      if (policy === "sentinel" && fan) {
        const at = fan.findIndex((point) => point.ts === ts);
        const before = fan[at - 1] ?? fan[at];
        const after = at === 0 ? fan[1] : fan[at];
        if (at >= 0 && before && after) charging = after.socQ50 - before.socQ50 > SOC_RISE;
      }
      now.set(feederId, { exportKw: row.exportKw, limitKw: row.limitKw, charging });
    });
    return now;
  }, [feederIds, policy, ts, curveData, socData]);
}

/** Absorbing homes on one feeder with their positions, for ripples once the camera is close. */
export function useAbsorbingHomes(feederId: string | null) {
  const window = useDayWindow();
  const { data: sites } = useQuery(
    orpc.feeders.getAbsorption.queryOptions({
      input: { feederId: feederId ?? "", ...window },
      enabled: Boolean(feederId),
    }),
  );
  const { data: assets } = useQuery(
    orpc.om.listAssets.queryOptions({ input: { assetKind: "prosumer" }, enabled: Boolean(feederId) }),
  );
  return useMemo(() => {
    const ids = new Set((sites ?? []).map((site) => site.assetId));
    return (assets ?? []).flatMap((asset) =>
      ids.has(asset.id) && asset.lon != null && asset.lat != null
        ? [[asset.lon, asset.lat] as [number, number]]
        : [],
    );
  }, [sites, assets]);
}
