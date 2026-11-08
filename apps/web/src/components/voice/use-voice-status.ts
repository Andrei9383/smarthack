"use client";

import { useQuery } from "@tanstack/react-query";
import { useSyncExternalStore } from "react";

// What the server can do (card #207): both are false on a checkout without ELEVENLABS_API_KEY,
// and the voice controls stay hidden.
export type VoiceStatus = { narration: boolean; assistant: boolean };

const OFF: VoiceStatus = { narration: false, assistant: false };

export function useVoiceStatus(): VoiceStatus {
  const { data } = useQuery({
    queryKey: ["voice-status"],
    queryFn: async () => {
      const res = await fetch("/api/voice");
      return res.ok ? ((await res.json()) as VoiceStatus) : OFF;
    },
    staleTime: Number.POSITIVE_INFINITY,
    retry: false,
  });
  return data ?? OFF;
}

/* ---------- The listener's choice: narrate the story or not (per browser) ---------- */

const NARRATION_KEY = "sentinel.narration";
let narrationOn: boolean | undefined;
const listeners = new Set<() => void>();

function readNarration() {
  if (narrationOn === undefined) {
    try {
      narrationOn = window.localStorage.getItem(NARRATION_KEY) !== "off";
    } catch {
      narrationOn = true;
    }
  }
  return narrationOn;
}

export function setNarrationOn(on: boolean) {
  narrationOn = on;
  try {
    window.localStorage.setItem(NARRATION_KEY, on ? "on" : "off");
  } catch {
    // Not remembered; the switch still works for this visit.
  }
  for (const listener of listeners) listener();
}

export function useNarrationOn() {
  return useSyncExternalStore(
    (listener) => {
      listeners.add(listener);
      return () => {
        listeners.delete(listener);
      };
    },
    readNarration,
    () => false,
  );
}
