import { parseAsInteger, parseAsStringLiteral, useQueryState } from "nuqs";

import { NOON_STEP } from "./replay";

const POLICIES = ["baseline", "sentinel"] as const;

// Shared by every mode, so a link reproduces exactly what is on screen.
export function usePolicy() {
  return useQueryState("policy", parseAsStringLiteral(POLICIES).withDefault("sentinel"));
}

export function useReplayStep() {
  return useQueryState("t", parseAsInteger.withDefault(NOON_STEP));
}
