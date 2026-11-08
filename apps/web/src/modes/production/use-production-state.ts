import { parseAsString, useQueryState } from "nuqs";

import { STEP_MINUTES } from "@/lib/replay";
import { useReplayStep } from "@/lib/use-replay";

const STEPS_PER_HOUR = 60 / STEP_MINUTES;

// Index into windfield/meta.json, whose hours start at the replay clock's first step.
export function useHour() {
  const [step] = useReplayStep();
  return Math.floor(step / STEPS_PER_HOUR);
}

export function useSelectedPark() {
  return useQueryState("park", parseAsString.withDefault("WP01"));
}
