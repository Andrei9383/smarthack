import { keepPreviousData, useQuery } from "@tanstack/react-query";

import { stepToDate } from "@/lib/replay";
import { useReplayStep } from "@/lib/use-replay";
import { orpc } from "@/lib/orpc";

// Judge rows are keyed by the local (Europe/Bucharest) calendar day.
const LOCAL_DAY = new Intl.DateTimeFormat("en-CA", { timeZone: "Europe/Bucharest" });

export function useVerdicts(assetId?: string) {
  const [step] = useReplayStep();
  const until = LOCAL_DAY.format(stepToDate(step));
  return useQuery({
    ...orpc.judge.verdicts.queryOptions({ input: { until, assetId } }),
    placeholderData: keepPreviousData,
  });
}

const WITNESS_LABEL: Record<string, string> = {
  pvnet: "PVNet",
  pvlib: "pvlib",
  loadnet: "LoadNet",
  sitehealth: "Site health",
  windfailure: "Wind failure",
};
export const witnessName = (model: string) => WITNESS_LABEL[model] ?? model;
