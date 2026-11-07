import { parseAsString, parseAsStringLiteral, useQueryState } from "nuqs";

// `?turbine=WP01.T04` opens the turbine drawer; `?park=WP01` (demo jump 5) focuses a park.
export function useSelectedTurbine() {
  return useQueryState("turbine", parseAsString);
}

export function useFocusedPark() {
  return useQueryState("park", parseAsString);
}

export const OM_TABS = ["orders", "review", "alerts", "audit"] as const;
export function useOmTab() {
  return useQueryState(
    "omTab",
    parseAsStringLiteral(OM_TABS).withDefault("orders"),
  );
}
