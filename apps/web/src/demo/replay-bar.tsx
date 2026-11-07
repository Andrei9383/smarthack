"use client";

import { Button } from "@sentinel-app/ui/components/button";
import { Tabs, TabsList, TabsTrigger } from "@sentinel-app/ui/components/tabs";
import { Pause, Play, SkipBack, SkipForward } from "lucide-react";
import { useEffect, useState } from "react";

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

export function ReplayBar() {
  const [step, setStep] = useReplayStep();
  const [policy, setPolicy] = usePolicy();
  const [playing, setPlaying] = useState(false);
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

  return (
    <div className="flex flex-wrap items-center gap-2 border-t p-2">
      <div className="flex items-center gap-1">
        <Button
          variant="outline"
          size="icon"
          aria-label="Previous 15 minutes"
          onClick={() => setStep(clampStep(current - 1))}
        >
          <SkipBack />
        </Button>
        <Button
          variant="outline"
          size="icon"
          aria-label={playing ? "Pause" : "Play"}
          onClick={() => setPlaying((value) => !value)}
        >
          {playing ? <Pause /> : <Play />}
        </Button>
        <Button
          variant="outline"
          size="icon"
          aria-label="Next 15 minutes"
          onClick={() => setStep(clampStep(current + 1))}
        >
          <SkipForward />
        </Button>
      </div>

      <span className="w-24 text-sm font-medium tabular-nums">
        {dayLabel(day)} {timeLabel(current)}
      </span>

      <input
        aria-label="Replay time"
        type="range"
        min={0}
        max={STEPS_IN_WEEK - 1}
        value={current}
        onChange={(event) => setStep(Number(event.target.value))}
        className="min-w-40 flex-1 accent-teal-400"
      />

      <div className="flex gap-1">
        {Array.from({ length: DAYS }, (_, index) => (
          <Button
            key={index}
            variant={index === day ? "default" : "outline"}
            size="sm"
            onClick={() => setStep(index * STEPS_PER_DAY + stepOfDay(current))}
          >
            {dayLabel(index)}
          </Button>
        ))}
      </div>

      <Tabs value={policy} onValueChange={(value) => setPolicy(value as typeof policy)}>
        <TabsList>
          <TabsTrigger value="baseline">Baseline</TabsTrigger>
          <TabsTrigger value="sentinel">Sentinel</TabsTrigger>
        </TabsList>
      </Tabs>
    </div>
  );
}
