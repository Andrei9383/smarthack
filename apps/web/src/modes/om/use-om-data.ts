import { keepPreviousData, useQuery } from "@tanstack/react-query";

import { stepToTs } from "@/lib/replay";
import { useReplayStep } from "@/lib/use-replay";
import { orpc } from "@/lib/orpc";

// Everything O&M shows is "as of" the replay clock: nothing appears before it happened.
export function useReplayTs() {
  const [step] = useReplayStep();
  return stepToTs(step);
}

export function useOmAssets() {
  const at = useReplayTs();
  return useQuery({
    ...orpc.om.listAssets.queryOptions({ input: { at } }),
    placeholderData: keepPreviousData,
  });
}

export function useWorkOrders() {
  const until = useReplayTs();
  return useQuery({
    ...orpc.om.listWorkOrders.queryOptions({ input: { until } }),
    placeholderData: keepPreviousData,
  });
}

export function useAlerts() {
  const until = useReplayTs();
  return useQuery({
    ...orpc.om.listAlerts.queryOptions({ input: { until } }),
    placeholderData: keepPreviousData,
  });
}

export function useReviewQueue() {
  const until = useReplayTs();
  return useQuery({
    ...orpc.om.listReviewQueue.queryOptions({ input: { until } }),
    placeholderData: keepPreviousData,
  });
}
