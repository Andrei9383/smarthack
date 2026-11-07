"use client";

import { cn } from "@sentinel-app/ui/lib/utils";
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

/** Advances the replay clock one 15-minute step per tick while playing. Mount once (the dock). */
export function useReplayTicker() {
  const [, setStep] = useReplayStep();
  const playing = useReplayPlaying();
  useEffect(() => {
    if (!playing) return;
    const timer = setInterval(() => {
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

/** Live dot: pulses green while the replay plays, quiet grey when paused. */
export function LiveDot({ playing, className }: { playing: boolean; className?: string }) {
  return (
    <span aria-hidden className={cn("relative flex size-1.5 shrink-0", className)}>
      {playing ? (
        <>
          <span className="absolute inline-flex size-full rounded-full bg-live opacity-60 motion-safe:animate-ping" />
          <span className="relative inline-flex size-1.5 rounded-full bg-live" />
        </>
      ) : (
        <span className="relative inline-flex size-1.5 rounded-full bg-status-unknown" />
      )}
    </span>
  );
}

