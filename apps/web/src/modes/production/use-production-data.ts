import { keepPreviousData, useQuery } from "@tanstack/react-query";
import { useMemo } from "react";

import { useReplayStep } from "@/demo/use-replay";
import { orpc } from "@/utils/orpc";

import { forecastWindow, inWindow } from "./data";

const FOREVER = Number.POSITIVE_INFINITY;

// One replay week per DB, so runs, parks and forecasts are fetched once and sliced by the clock.
export function useProductionData() {
  const [step] = useReplayStep();
  const window = forecastWindow(step);
  const { data: run } = useQuery(orpc.production.getRun.queryOptions({ staleTime: FOREVER }));
  const parks = useQuery(orpc.production.listParks.queryOptions({ staleTime: FOREVER }));
  const forecasts = useQuery(orpc.production.getParkForecasts.queryOptions({ staleTime: FOREVER }));
  const { data: counties } = useQuery(
    orpc.production.getCountyTotals.queryOptions({
      input: window,
      staleTime: FOREVER,
      placeholderData: keepPreviousData,
    }),
  );
  const rows = useMemo(
    () => inWindow(forecasts.data ?? [], window),
    [forecasts.data, window.from, window.to],
  );

  return {
    label: run?.label,
    parks: parks.data ?? [],
    rows,
    counties: counties ?? [],
    isLoading: parks.isPending || forecasts.isPending,
  };
}
