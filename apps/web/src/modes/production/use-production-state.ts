import { parseAsInteger, parseAsString, useQueryState } from "nuqs";

export function useHour() {
  return useQueryState("hour", parseAsInteger.withDefault(12));
}

export function useSelectedPark() {
  return useQueryState("park", parseAsString.withDefault("WP01"));
}
