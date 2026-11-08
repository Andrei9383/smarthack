"use client";

import { createContext, type ReactNode, useContext } from "react";

// A mode panel rendered inside the shell inspector sits under the inspector's own header, which
// already shows the mode title. Panels read this to drop their own copy, so each view has one title.
// Asset headers ("F7 · Oșorhei", "Bihor") stay.
const InspectorEmbeddedContext = createContext(false);

export function InspectorEmbedded({ children }: { children: ReactNode }) {
  return <InspectorEmbeddedContext.Provider value={true}>{children}</InspectorEmbeddedContext.Provider>;
}

/** True when the panel is lifted into the shell inspector (which then owns the mode title). */
export function useInspectorEmbedded() {
  return useContext(InspectorEmbeddedContext);
}
