"use client";

import { useQuery } from "@tanstack/react-query";

import { orpc } from "@/utils/orpc";

import { dayWindow, stepToTs } from "./replay";
import { useReplayStep } from "./use-replay";

// Same query keys everywhere (rail, inspector, Overview), so React Query fetches each once.
// The top-bar KPI chips that lived here were removed in the Calm redesign (#196): the top bar shows no numbers.
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
