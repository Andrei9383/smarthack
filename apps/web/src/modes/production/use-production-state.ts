import { parseAsString, useQueryState } from "nuqs";

import { stepToDate } from "@/demo/replay";
import { useReplayStep } from "@/demo/use-replay";

// Index into windfield/meta.json, whose hours are 00..23 UTC.
export function useHour() {
  const [step] = useReplayStep();
  return stepToDate(step).getUTCHours();
}

export function useSelectedPark() {
  return useQueryState("park", parseAsString.withDefault("WP01"));
}
