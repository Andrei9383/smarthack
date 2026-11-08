"use client";

import { useEffect, useSyncExternalStore } from "react";

import { clampStep, STEPS_IN_WEEK } from "./replay";
import { useReplayStep } from "./use-replay";

const TICK_MS = 250;

// Play/pause lives in a tiny module store (not URL state) so the rail footer, the dock and the
// agent-action toasts all read the same playing flag.
let playingNow = false;
const listeners = new Set<() => void>();
const subscribe = (listener: () => void) => {
  listeners.add(listener);
  return () => {
    listeners.delete(listener);
  };
};
export function setPlaying(next: boolean | ((value: boolean) => boolean)) {
  playingNow = typeof next === "function" ? next(playingNow) : next;
  for (const listener of listeners) listener();
}

export function useReplayPlaying() {
  return useSyncExternalStore(subscribe, () => playingNow, () => false);
}

// The story narrator holds the clock while it speaks, so the voice and the screen stay on
// the same moment. Playing stays true: the button still reads "Pause the story".
let heldNow = false;
export function setReplayHeld(held: boolean) {
  heldNow = held;
}

/** Advances the replay clock one 15-minute step per tick while playing. Mount once (the dock). */
export function useReplayTicker() {
  const [, setStep] = useReplayStep();
  const playing = useReplayPlaying();
  useEffect(() => {
    if (!playing) return;
    const timer = setInterval(() => {
      if (heldNow) return;
      setStep((value) => {
        const next = clampStep(value) + 1;
        if (next >= STEPS_IN_WEEK) {
          setPlaying(false);
          return STEPS_IN_WEEK - 1;
        }
        return next;
      });
    }, TICK_MS);
    return () => clearInterval(timer);
  }, [playing, setStep]);
  useEffect(() => () => setPlaying(false), []);
}
