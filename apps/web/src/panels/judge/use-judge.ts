import { keepPreviousData, useQuery } from "@tanstack/react-query";

import { stepToDate } from "@/demo/replay";
import { useReplayStep } from "@/demo/use-replay";
import { orpc } from "@/utils/orpc";

// Judge rows are keyed by the local (Europe/Bucharest) calendar day, design §8.
const LOCAL_DAY = new Intl.DateTimeFormat("en-CA", { timeZone: "Europe/Bucharest" });
export const localDay = (step: number) => LOCAL_DAY.format(stepToDate(step));

export function useJudgeDay() {
  const [step] = useReplayStep();
  return localDay(step);
}

export function useVerdicts(assetId?: string) {
  const until = useJudgeDay();
  return useQuery({
    ...orpc.judge.verdicts.queryOptions({ input: { until, assetId } }),
    placeholderData: keepPreviousData,
  });
}

export function useTrustBoard() {
  const until = useJudgeDay();
  return useQuery({
    ...orpc.judge.trustBoard.queryOptions({ input: { until } }),
    placeholderData: keepPreviousData,
  });
}

export const WITNESS_LABEL: Record<string, string> = {
  pvnet: "PVNet",
  pvlib: "pvlib",
  loadnet: "LoadNet",
  sitehealth: "Site health",
  windfailure: "Wind failure",
};
export const witnessName = (model: string) => WITNESS_LABEL[model] ?? model;
