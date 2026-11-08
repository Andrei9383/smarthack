"use client";

import { cn } from "@wattsentry/ui/lib/utils";
import { Volume2, VolumeX } from "lucide-react";
import { useEffect, useRef, useSyncExternalStore } from "react";

import type { AgentAction } from "@/components/shell/agent-actions";
import { useAgentActions } from "@/components/shell/agent-actions";
import { useReplayPlaying } from "@/demo/replay-bar";
import { clampStep, dayOfStep, HERO_DAY, STEPS_PER_DAY, stepToDate, timeLabel } from "@/demo/replay";
import { useReplayStep } from "@/demo/use-replay";
import type { FeederState } from "@/modes/overview/use-overview-data";
import { useOverviewData } from "@/modes/overview/use-overview-data";

import { loadLine, narrator, narratorLine } from "./narrator";
import { setNarrationOn, useNarrationOn, useVoiceStatus } from "./use-voice-status";

// "Play today's story", told by an ElevenLabs voice (card #207). The script is written from the
// same data the Overview shows: what the agent did overnight, the midday solar on the worst
// feeder, each action it takes during the day, the evening peak and the day against no WattSentry.

const DAY = HERO_DAY * STEPS_PER_DAY;
const at = (hour: number, minute = 0) => DAY + hour * 4 + Math.round(minute / 15);
export const STORY_START = at(6);
const STORY_END = at(20);

type Line = { step: number; text: string };

/** 13:15 → "1:15 p.m.", 06:00 → "6 a.m." */
function spokenTime(step: number) {
  const [h = 0, m = 0] = timeLabel(step).split(":").map(Number);
  const hour = h % 12 === 0 ? 12 : h % 12;
  return `${hour}${m ? `:${String(m).padStart(2, "0")}` : ""} ${h < 12 ? "a.m." : "p.m."}`;
}

const plural = (n: number, noun: string, many = `${noun}s`) => `${n} ${n === 1 ? noun : many}`;
const kw = (value: number) => `${Math.round(value)} kilowatts`;
const total = (feeders: FeederState[], pick: (f: FeederState) => number | undefined) =>
  feeders.some((f) => pick(f) !== undefined) ? feeders.reduce((sum, f) => sum + (pick(f) ?? 0), 0) : undefined;
const peak = (feeders: FeederState[], pick: (f: FeederState) => number | undefined) =>
  feeders.some((f) => pick(f) !== undefined) ? Math.max(...feeders.map((f) => pick(f) ?? 0)) : undefined;

/** "Scheduled 15 home batteries on …" → "WattSentry scheduled 15 home batteries on …" */
const asSentinel = (title: string) => `WattSentry ${title.charAt(0).toLowerCase()}${title.slice(1)}`;

function actionLine(group: AgentAction[]): string {
  const [lead] = group;
  if (!lead) return "";
  const others = group.length - 1;
  const also = others > 0 ? `, and on ${plural(others, "other feeder")}` : "";
  const why = lead.kind === "dispatch" ? "" : ` ${lead.detail}.`;
  return `At ${spokenTime(lead.step)}, ${asSentinel(lead.title)}${also}.${why}`;
}

export function buildStory(feeders: FeederState[], actions: AgentAction[], comfortBreaches?: number): Line[] {
  const lines: Line[] = [];
  const date = stepToDate(STORY_START).toLocaleDateString("en-GB", {
    weekday: "long",
    day: "numeric",
    month: "long",
    timeZone: "Europe/Bucharest",
  });

  // 06:00 — the night's decisions.
  const overnight = actions.filter((a) => dayOfStep(a.step) === HERO_DAY && a.step < STORY_START);
  const plans = overnight.filter((a) => a.kind === "dispatch");
  const homes = plans.reduce((sum, a) => sum + (a.homes ?? 0), 0);
  const crews = overnight.filter((a) => a.kind === "workorder");
  const remote = overnight.filter((a) => a.kind === "remote");
  let intro = `Good morning. It's 6 a.m. on ${date}, and this is today's story, with WattSentry running the grid.`;
  if (plans.length > 0) {
    intro += ` Overnight it forecast the sun and the load on every feeder, and planned ${plural(homes, "home battery", "home batteries")} across ${plural(plans.length, "feeder")} to soak up the midday solar.`;
  }
  if (crews.length > 0 || remote.length > 0) {
    const first = [...crews, ...remote].sort((a, b) => a.step - b.step)[0]!;
    const parts = [
      crews.length > 0 ? `sent ${plural(crews.length, "crew")} to real faults` : null,
      remote.length > 0 ? `booked ${plural(remote.length, "remote check")} that need no truck` : null,
    ].filter(Boolean);
    intro += ` At ${spokenTime(first.step)} it ranked the faults by lost energy and ${parts.join(", and ")}.`;
  }
  lines.push({ step: STORY_START, text: intro });

  // 10:00 — solar ramps up on the feeder that would suffer most without WattSentry.
  const hero = [...feeders]
    .filter((f) => f.baselinePeakKw !== undefined)
    .sort((a, b) => (b.baselinePeakKw ?? 0) / b.limitKw - (a.baselinePeakKw ?? 0) / a.limitKw)[0];
  if (hero && (hero.baselinePeakKw ?? 0) > hero.limitKw) {
    lines.push({
      step: at(10),
      text: `By 10, the solar is ramping up. Without WattSentry, ${hero.name} would push ${kw(hero.baselinePeakKw!)} into a line built for ${kw(hero.limitKw)}. Instead, its home batteries start charging and soak up the surplus.`,
    });
  } else {
    lines.push({
      step: at(10),
      text: "By 10, the solar is ramping up, and the home batteries in the plan start charging to soak up the surplus.",
    });
  }

  // 13:00 — the solar peak.
  const withPeak = peak(feeders, (f) => f.sentinelPeakKw);
  const withoutPeak = peak(feeders, (f) => f.baselinePeakKw);
  if (withPeak !== undefined && withoutPeak !== undefined) {
    lines.push({
      step: at(13),
      text: `1 p.m., the solar peak. With WattSentry, the highest feeder peak today is ${kw(withPeak)}, against ${kw(withoutPeak)} without it.`,
    });
  }

  // During the day — each action the agent takes, one line per moment.
  const byStep = new Map<number, AgentAction[]>();
  for (const action of actions) {
    if (action.step < STORY_START || action.step >= STORY_END) continue;
    byStep.set(action.step, [...(byStep.get(action.step) ?? []), action]);
  }
  for (const [step, group] of byStep) lines.push({ step, text: actionLine(group) });

  // 17:30 — the evening peak.
  lines.push({
    step: at(17, 30),
    text: "5:30 p.m., and the evening peak begins. The batteries now give back the solar they stored at midday, when the grid needs it most.",
  });

  // 20:00 — the day against no WattSentry.
  const withMinutes = total(feeders, (f) => f.sentinelMinutes);
  const withoutMinutes = total(feeders, (f) => f.baselineMinutes);
  let outro = "That's the day.";
  if (withMinutes !== undefined && withoutMinutes !== undefined) {
    outro += ` Across all feeders, ${Math.round(withMinutes)} minutes over the limit with WattSentry, against ${Math.round(withoutMinutes)} without it`;
    outro += comfortBreaches === 0 ? ", and not one home dipped below its comfort reserve." : ".";
  }
  outro += " Every decision is in the audit log, next to the model prediction behind it.";
  lines.push({ step: STORY_END, text: outro });

  return lines.sort((a, b) => a.step - b.step);
}

/**
 * Narrates the story while it plays. `runId` changes each time the story starts again from 06:00.
 * Pausing pauses the voice; scrubbing or jumping elsewhere ends the narration for that run.
 */
export function useStoryNarration(runId: number) {
  const status = useVoiceStatus();
  const wanted = useNarrationOn();
  const enabled = status.narration && wanted;
  const data = useOverviewData();
  const { actions } = useAgentActions();
  const [rawStep] = useReplayStep();
  const playing = useReplayPlaying();
  const step = clampStep(rawStep);

  const script = useRef<Line[] | null>(null);
  const last = useRef(step);
  const latest = useRef({ data, actions, enabled });
  latest.current = { data, actions, enabled };

  // A new run: write the script from today's data, fetch every line now, say the intro.
  // Turning narration on mid-story doesn't restart it; the next press of Play does.
  useEffect(() => {
    if (runId === 0 || !latest.current.enabled) return;
    narrator.stop();
    const { data: now, actions: acts } = latest.current;
    const lines = buildStory(now.feeders, acts, now.comfort.sentinel);
    for (const line of lines) void loadLine(line.text);
    script.current = lines;
    last.current = STORY_START;
    for (const line of lines) if (line.step <= STORY_START) narrator.say(line.text);
  }, [runId]);

  // Speak each line as the clock crosses its moment.
  useEffect(() => {
    const lines = script.current;
    const before = last.current;
    last.current = step;
    if (!lines || !enabled) return;
    if (step < before || step - before > 8) {
      narrator.stop();
      script.current = null;
      return;
    }
    for (const line of lines) if (line.step > before && line.step <= step) narrator.say(line.text);
    if (step >= STORY_END) script.current = null;
  }, [step, enabled]);

  useEffect(() => {
    if (playing) narrator.resume();
    else narrator.pause();
  }, [playing]);

  useEffect(() => {
    if (!enabled) {
      narrator.stop();
      script.current = null;
    }
  }, [enabled]);

  useEffect(() => () => narrator.stop(), []);
}

/** The line being spoken, for a caption under the briefing. */
export function useNarratorLine() {
  return useSyncExternalStore(narratorLine.subscribe, narratorLine.get, () => null);
}

/** Voice on / off for the story, beside "Play today's story". Hidden when the server has no key. */
export function NarrationToggle({ className }: { className?: string }) {
  const { narration } = useVoiceStatus();
  const on = useNarrationOn();
  if (!narration) return null;
  const label = on ? "Narration on: the story is read aloud" : "Narration off";
  return (
    <button
      type="button"
      onClick={() => setNarrationOn(!on)}
      aria-pressed={on}
      aria-label={label}
      title={label}
      className={cn(
        "grid size-10 shrink-0 cursor-pointer place-items-center rounded-lg bg-card text-muted-foreground shadow-raised transition-[background-color,color] duration-150 hover:bg-surface-raised hover:text-foreground focus-visible:outline-2 focus-visible:outline-offset-2 focus-visible:outline-ring",
        on && "text-foreground",
        className,
      )}
    >
      {on ? <Volume2 aria-hidden className="size-4" /> : <VolumeX aria-hidden className="size-4" />}
    </button>
  );
}
