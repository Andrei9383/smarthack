"use client";

import { Button } from "@sentinel-app/ui/components/button";
import { Tabs, TabsList, TabsTrigger } from "@sentinel-app/ui/components/tabs";
import { Pause, Play, SkipBack, SkipForward } from "lucide-react";
import { useEffect, useSyncExternalStore } from "react";

import {
  clampStep,
  DAYS,
  dayLabel,
  dayOfStep,
  STEPS_IN_WEEK,
  STEPS_PER_DAY,
  stepOfDay,
  timeLabel,
} from "./replay";
import { usePolicy, useReplayStep } from "./use-replay";

const TICK_MS = 250;

// Play/pause lives in a tiny module store (not URL state, same as before) so the sidebar's
// replay status can show the live dot while the bar plays.
let playingNow = false;
const listeners = new Set<() => void>();
const subscribe = (listener: () => void) => {
  listeners.add(listener);
  return () => {
    listeners.delete(listener);
  };
};
function setPlaying(next: boolean | ((value: boolean) => boolean)) {
  playingNow = typeof next === "function" ? next(playingNow) : next;
  for (const listener of listeners) listener();
}

export function useReplayPlaying() {
  return useSyncExternalStore(subscribe, () => playingNow, () => false);
}

/** Live dot: pulses green while the replay plays, quiet grey when paused. */
export function LiveDot({ playing }: { playing: boolean }) {
  return (
    <span aria-hidden className="relative flex size-2 shrink-0">
      {playing ? (
        <>
          <span className="absolute inline-flex size-full animate-ping rounded-full bg-live opacity-60" />
          <span className="relative inline-flex size-2 rounded-full bg-live" />
        </>
      ) : (
        <span className="relative inline-flex size-2 rounded-full bg-status-unknown" />
      )}
    </span>
  );
}

/** Quiet status line for the sidebar footer, driven by the same replay state as the bar. */
export function ReplayStatus() {
  const [step] = useReplayStep();
  const playing = useReplayPlaying();
  const current = clampStep(step);
  return (
    <div className="flex items-center gap-2.5 rounded-lg border border-sidebar-border bg-background px-3 py-2.5 text-xs">
      <LiveDot playing={playing} />
      <span className="truncate text-text-secondary">
        Replay · <span className="font-medium text-foreground tabular-nums">{dayLabel(dayOfStep(current))} {timeLabel(current)}</span>
      </span>
    </div>
  );
}

export function ReplayBar() {
  const [step, setStep] = useReplayStep();
  const [policy, setPolicy] = usePolicy();
  const playing = useReplayPlaying();
  const current = clampStep(step);
  const day = dayOfStep(current);

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

  const progress = (current / (STEPS_IN_WEEK - 1)) * 100;

  // Rendered inside the floating bottom dock (app-shell), so no surface of its own.
  return (
    <div className="flex min-w-0 flex-1 flex-wrap items-center gap-x-4 gap-y-2">
      <div className="flex items-center gap-1">
        <Button
          variant="ghost"
          size="icon-sm"
          aria-label="Previous 15 minutes"
          onClick={() => setStep(clampStep(current - 1))}
          className="rounded-md text-text-secondary transition-colors duration-150 hover:bg-muted hover:text-foreground"
        >
          <SkipBack />
        </Button>
        <Button
          variant="default"
          size="icon"
          aria-label={playing ? "Pause" : "Play"}
          onClick={() => setPlaying((value) => !value)}
          className="size-9 rounded-full shadow-sm transition-colors duration-150 hover:bg-brand-700"
        >
          {playing ? <Pause className="fill-current" /> : <Play className="translate-x-px fill-current" />}
        </Button>
        <Button
          variant="ghost"
          size="icon-sm"
          aria-label="Next 15 minutes"
          onClick={() => setStep(clampStep(current + 1))}
          className="rounded-md text-text-secondary transition-colors duration-150 hover:bg-muted hover:text-foreground"
        >
          <SkipForward />
        </Button>
      </div>

      <span className="flex w-[104px] items-center gap-2 text-sm font-semibold whitespace-nowrap text-foreground tabular-nums">
        <LiveDot playing={playing} />
        {dayLabel(day)} {timeLabel(current)}
      </span>

      <div className="flex min-w-72 flex-1 items-center gap-3">
        <input
          aria-label="Replay time"
          type="range"
          min={0}
          max={STEPS_IN_WEEK - 1}
          value={current}
          onChange={(event) => setStep(Number(event.target.value))}
          style={{
            background: `linear-gradient(to right, var(--primary) ${progress}%, var(--border) ${progress}%)`,
          }}
          className="h-1.5 min-w-24 flex-1 cursor-pointer appearance-none rounded-full accent-primary focus-visible:outline-2 focus-visible:outline-offset-4 focus-visible:outline-ring [&::-moz-range-thumb]:size-4 [&::-moz-range-thumb]:rounded-full [&::-moz-range-thumb]:border-2 [&::-moz-range-thumb]:border-background [&::-moz-range-thumb]:bg-primary [&::-moz-range-thumb]:shadow-sm [&::-webkit-slider-thumb]:size-4 [&::-webkit-slider-thumb]:appearance-none [&::-webkit-slider-thumb]:rounded-full [&::-webkit-slider-thumb]:border-2 [&::-webkit-slider-thumb]:border-background [&::-webkit-slider-thumb]:bg-primary [&::-webkit-slider-thumb]:shadow-sm"
        />

        <div className="flex shrink-0 gap-0.5 rounded-lg bg-muted p-0.5">
          {Array.from({ length: DAYS }, (_, index) => (
            <Button
              key={index}
              variant="ghost"
              size="xs"
              aria-pressed={index === day}
              onClick={() => setStep(index * STEPS_PER_DAY + stepOfDay(current))}
              className={
                index === day
                  ? "h-7 rounded-md bg-background px-2 text-[12px] text-foreground shadow-sm hover:bg-background"
                  : "h-7 rounded-md px-2 text-[12px] text-text-secondary transition-colors duration-150 hover:bg-background/60 hover:text-foreground"
              }
            >
              {dayLabel(index)}
            </Button>
          ))}
        </div>
      </div>

      <Tabs value={policy} onValueChange={(value) => setPolicy(value as typeof policy)}>
        <TabsList className="rounded-lg bg-muted p-0.5 group-data-horizontal/tabs:h-8">
          <TabsTrigger
            value="baseline"
            className="rounded-md px-3 text-[12px] text-text-secondary data-active:bg-background data-active:text-foreground data-active:shadow-sm"
          >
            Baseline
          </TabsTrigger>
          <TabsTrigger
            value="sentinel"
            className="rounded-md px-3 text-[12px] text-text-secondary data-active:bg-background data-active:text-brand-700 data-active:shadow-sm"
          >
            Sentinel
          </TabsTrigger>
        </TabsList>
      </Tabs>
    </div>
  );
}
