import { parseAsInteger, parseAsStringLiteral, useQueryState } from "nuqs";

import { DEFAULT_STEP } from "./replay";

const POLICIES = ["baseline", "sentinel"] as const;
export type Policy = (typeof POLICIES)[number];

// Shared by every mode, so a link reproduces exactly what is on screen.
export function usePolicy() {
  return useQueryState("policy", parseAsStringLiteral(POLICIES).withDefault("sentinel"));
}

export function useReplayStep() {
  return useQueryState("t", parseAsInteger.withDefault(DEFAULT_STEP));
}
