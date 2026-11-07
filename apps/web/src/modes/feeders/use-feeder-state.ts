import { parseAsString, useQueryState } from "nuqs";

export function useSelectedFeeder() {
  return useQueryState("feeder", parseAsString.withDefault("F7"));
}
