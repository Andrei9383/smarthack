import { useQuery } from "@tanstack/react-query";

import { dayWindow } from "@/demo/replay";
import { useReplayStep } from "@/demo/use-replay";
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
