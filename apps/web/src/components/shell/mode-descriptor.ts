import type { ModeId } from "@/modes/registry";

/** Rail, top-bar and command-palette descriptor: one plain line, no jargon. */
export const MODE_DESCRIPTOR: Record<ModeId, string> = {
  overview: "What prosumers cost PPC today",
  feeders: "Keep cables under their limit",
  om: "Alarms become repair jobs",
  production: "Wind and solar coming in",
  demand: "Where use peaks this evening",
  imbalance: "Stop paying for prosumer imbalance",
};
