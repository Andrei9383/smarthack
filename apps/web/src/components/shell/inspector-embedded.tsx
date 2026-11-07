"use client";

import { createContext, type ReactNode, useContext } from "react";

// A mode panel rendered inside the shell inspector sits under the inspector's own header, which
// already shows the mode title and the run's provenance pill. Panels read this to drop their own
// copies, so each view has one title and one pill. Asset headers ("F7 · Oșorhei", "Bihor") stay.
type Embedded = { runLabel: string | null };

const InspectorEmbeddedContext = createContext<Embedded | null>(null);

export function InspectorEmbedded({ runLabel, children }: { runLabel: string | null; children: ReactNode }) {
  return <InspectorEmbeddedContext.Provider value={{ runLabel }}>{children}</InspectorEmbeddedContext.Provider>;
}

/** True when the panel is lifted into the shell inspector (which then owns the mode title). */
export function useInspectorEmbedded() {
  return useContext(InspectorEmbeddedContext) !== null;
}

const norm = (label: string) => label.trim().toLowerCase();
// Placeholders the panels show before their run loads; the shell pill covers them.
const PLACEHOLDER = /^(loading|no run( loaded)?)$/;

/**
 * Whether a panel should show its own provenance pill: always standalone; inside the inspector only
 * when it says something the shell pill does not (a different run, a different source).
 */
export function useShowOwnPill(label: string | null | undefined) {
  const embedded = useContext(InspectorEmbeddedContext);
  if (!embedded) return true;
  if (!label || PLACEHOLDER.test(norm(label))) return false;
  return !embedded.runLabel || norm(label) !== norm(embedded.runLabel);
}
