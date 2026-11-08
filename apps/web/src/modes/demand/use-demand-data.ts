import { useQuery } from "@tanstack/react-query";
import { parseAsString, useQueryState } from "nuqs";
import { useMemo } from "react";

import { dayWindow, stepToTs } from "@/lib/replay";
import { useReplayStep } from "@/lib/use-replay";
import { orpc } from "@/lib/orpc";

import { FOREVER } from "@/lib/utils";
import { byStress, toZones } from "./data";

export function useSelectedZone() {
  return useQueryState("zone", parseAsString);
}

// One query per day: scrubbing the replay bar only re-slices rows the browser already has.
export function useDemandData() {
  const [step] = useReplayStep();
  const window = dayWindow(step);
  const { data, isPending } = useQuery(
    orpc.demand.getZones.queryOptions({ input: window, staleTime: FOREVER }),
  );
  const zones = useMemo(() => toZones(data ?? []), [data]);
  const all = useMemo(() => [...zones.values()], [zones]);
  const counties = useMemo(() => byStress(all.filter((zone) => zone.level === "county")), [all]);
  const feeders = useMemo(() => byStress(all.filter((zone) => zone.level === "feeder")), [all]);

  return { zones, counties, feeders, all, ts: stepToTs(step), isLoading: isPending };
}
