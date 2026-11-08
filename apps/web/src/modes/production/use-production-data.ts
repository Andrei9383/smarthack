import { keepPreviousData, useQuery } from "@tanstack/react-query";
import { useMemo } from "react";

import { useReplayStep } from "@/lib/use-replay";
import { orpc } from "@/lib/orpc";

import { FOREVER } from "@/lib/utils";
import { forecastWindow, inWindow } from "./data";

// One replay week per DB, so parks and forecasts are fetched once and sliced by the clock.
export function useProductionData() {
  const [step] = useReplayStep();
  const window = forecastWindow(step);
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
    parks: parks.data ?? [],
    rows,
    counties: counties ?? [],
    isLoading: parks.isPending || forecasts.isPending,
  };
}
