"use client";

import type { ReactNode } from "react";

// SLOT for card #114: replace this with `<Term>` from `@/components/help/term` once it lands.
// Until then a dotted underline plus a native hover/focus title carries the plain definition,
// so jargon in the shell copy is never left unexplained.
const DEFINITIONS: Record<string, string> = {
  feeder: "A low-voltage street cable that connects a group of homes to the local transformer.",
  limit: "The most power a feeder cable can carry safely. Above it, the cable overheats or voltage rises.",
  p10: "P10 / P50 / P90: a likely range. 9 times in 10 the real value is above P10 and below P90; P50 is the middle guess.",
  baseline: "Baseline = what happens without Sentinel: home batteries just store their own solar power.",
  sentinel: "Sentinel = the agent: it plans home-battery charging so each feeder stays under its limit.",
  derated: "Derated: still running, but Sentinel counts on less power from it because something looks wrong.",
  stress: "Stress = forecast evening demand (P90) divided by what the network can carry. 1.0 means full.",
  review: "Human review: the agent is not sure enough to act, so it hands the case to a person.",
  synthetic: "Synthetic RO: simulated Romanian homes and feeders. Real RO: measured Romanian data. Non-RO: data from outside Romania.",
  workorder: "A repair job for a crew, ranked by how much energy and money the fault costs each day.",
  soc: "State of charge: how full a home battery is, in percent.",
  curtailment: "Curtailment: deliberately producing less than possible, usually because the grid cannot take it.",
};

export type TermKey = keyof typeof DEFINITIONS;

export function Term({ k, children }: { k: TermKey; children: ReactNode }) {
  return (
    <abbr
      title={DEFINITIONS[k]}
      tabIndex={0}
      className="cursor-help underline decoration-text-secondary/60 decoration-dotted underline-offset-[3px] outline-none focus-visible:rounded-sm focus-visible:ring-2 focus-visible:ring-ring/50"
    >
      {children}
    </abbr>
  );
}
