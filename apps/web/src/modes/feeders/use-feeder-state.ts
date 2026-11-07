import { parseAsString, parseAsStringLiteral, useQueryState } from "nuqs";

const POLICIES = ["baseline", "sentinel"] as const;

export function usePolicy() {
  return useQueryState("policy", parseAsStringLiteral(POLICIES).withDefault("sentinel"));
}

export function useSelectedFeeder() {
  return useQueryState("feeder", parseAsString.withDefault("F7"));
}
