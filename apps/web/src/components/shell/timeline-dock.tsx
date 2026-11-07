"use client";

import { cn } from "@sentinel-app/ui/lib/utils";
import { Pause, Play } from "lucide-react";
import { type KeyboardEvent, type PointerEvent, useEffect, useMemo, useRef, useState } from "react";

import { LiveDot, setPlaying, useReplayPlaying, useReplayTicker } from "@/demo/replay-bar";
import {
  clampStep,
  clockLabel,
  DAYS,
  dayLabel,
  dayOfStep,
  DEFAULT_STEP,
  STEPS_IN_WEEK,
  STEPS_PER_DAY,
} from "@/demo/replay";
import { type Policy, usePolicy, useReplayStep } from "@/demo/use-replay";

import { ACTION_LABEL, type ActionKind, groupMarkers, useAgentActions } from "./agent-actions";
import { showAgentToast } from "./agent-toast";
import { Term } from "@/components/help";

// The story's "now": the hero day at noon, where the demo starts and "Back to now" returns.
const NOW_STEP = DEFAULT_STEP;
const LAST = STEPS_IN_WEEK - 1;
const pct = (step: number) => `${(step / LAST) * 100}%`;

const isTyping = (target: EventTarget | null) =>
  target instanceof HTMLElement &&
  (target.isContentEditable || ["INPUT", "TEXTAREA", "SELECT", "BUTTON"].includes(target.tagName));

// Marker shape and colour per action type; the same pair is used in the lane key.
const MARKER: Record<ActionKind, string> = {
  dispatch: "rounded-full bg-primary",
  workorder: "rounded-[3px] bg-status-warn",
  review: "rotate-45 rounded-[2px] bg-status-review",
  remote: "rounded-full border-2 border-status-forecast bg-card",
};

// Short words for the lane key; markers and toasts use the full ACTION_LABEL.
const LANE_KEY: Record<ActionKind, string> = {
  dispatch: "Battery",
  workorder: "Repair",
  review: "Person",
  remote: "Remote",
};

function Marker({ kind, className }: { kind: ActionKind; className?: string }) {
  return <span aria-hidden className={cn("block size-2.5 shrink-0", MARKER[kind], className)} />;
}

function PolicySwitch({ policy, onChange }: { policy: Policy; onChange: (p: Policy) => void }) {
  const options: { value: Policy; label: string; dot: string }[] = [
    { value: "baseline", label: "Baseline", dot: "bg-foreground/55" },
    { value: "sentinel", label: "Sentinel", dot: "bg-primary" },
  ];
  return (
    <div role="radiogroup" aria-label="Policy shown" data-tour="policy" className="relative flex rounded-[9px] bg-muted p-[3px]">
      <span
        aria-hidden
        className={cn(
          "absolute top-[3px] bottom-[3px] w-[calc(50%-3px)] rounded-[7px] bg-card shadow-[0_1px_2px_rgb(13_13_13/0.1)] transition-[left] duration-250 ease-out motion-reduce:transition-none",
          policy === "baseline" ? "left-[3px]" : "left-1/2",
        )}
      />
      {options.map((option) => (
        <button
          key={option.value}
          type="button"
          role="radio"
          aria-checked={policy === option.value}
          onClick={() => onChange(option.value)}
          className={cn(
            "relative z-[1] flex h-8 flex-1 items-center justify-center gap-1.5 rounded-[7px] px-2.5 text-[13px] min-[1440px]:px-3 font-medium whitespace-nowrap transition-colors duration-200",
            policy === option.value ? "text-foreground" : "text-text-secondary hover:text-foreground",
          )}
        >
          <span aria-hidden className={cn("size-2 rounded-full", option.dot)} />
          {option.label}
        </button>
      ))}
    </div>
  );
}

/** The bottom dock: play, SCADA clock, the week scrubber with the agent-actions lane, policy. */
export function TimelineDock({ className }: { className?: string }) {
  useReplayTicker();
  const [rawStep, setStep] = useReplayStep();
  const [policy, setPolicy] = usePolicy();
  const playing = useReplayPlaying();
  const step = clampStep(rawStep);
  const { actions } = useAgentActions();
  const markers = useMemo(() => groupMarkers(actions), [actions]);
  const track = useRef<HTMLDivElement>(null);
  const [dragging, setDragging] = useState(false);
  const agentOn = policy === "sentinel";

  // Crossing an action while playing raises its toast (the last one crossed, if several).
  const previous = useRef(step);
  useEffect(() => {
    const before = previous.current;
    previous.current = step;
    if (!playing || !agentOn || step <= before || step - before > 8) return;
    const crossed = markers.filter((m) => m.step > before && m.step <= step);
    const last = crossed.at(-1);
    const lead = last?.items[0];
    if (lead) showAgentToast(lead, { others: last.items.length - 1, fast: true });
  }, [step, playing, agentOn, markers]);

  // Space plays/pauses anywhere outside a text field or button.
  useEffect(() => {
    const onKey = (event: globalThis.KeyboardEvent) => {
      if (event.code !== "Space" || isTyping(event.target)) return;
      event.preventDefault();
      setPlaying((value) => !value);
    };
    window.addEventListener("keydown", onKey);
    return () => window.removeEventListener("keydown", onKey);
  }, []);

  const stepAt = (clientX: number) => {
    const box = track.current?.getBoundingClientRect();
    if (!box) return step;
    return Math.round(Math.min(1, Math.max(0, (clientX - box.left) / box.width)) * LAST);
  };
  const onPointerDown = (event: PointerEvent<HTMLDivElement>) => {
    event.currentTarget.setPointerCapture(event.pointerId);
    setDragging(true);
    setStep(stepAt(event.clientX));
  };
  const onPointerMove = (event: PointerEvent<HTMLDivElement>) => {
    if (dragging) setStep(stepAt(event.clientX));
  };
  const onKeyDown = (event: KeyboardEvent<HTMLDivElement>) => {
    const moves: Record<string, number> = {
      ArrowLeft: -1,
      ArrowRight: 1,
      ArrowDown: -1,
      ArrowUp: 1,
      PageDown: -STEPS_PER_DAY,
      PageUp: STEPS_PER_DAY,
    };
    if (event.key in moves) {
      event.preventDefault();
      setStep(clampStep(step + (moves[event.key] ?? 0) * (event.shiftKey ? 4 : 1)));
    } else if (event.key === "Home") setStep(0);
    else if (event.key === "End") setStep(LAST);
  };

  const offset = step - NOW_STEP;
  const hours = Math.round(Math.abs(offset) / 4);
  const relative =
    hours >= 24 ? `${Math.floor(hours / 24)} d ${hours % 24} h` : hours > 0 ? `${hours} h` : "15 min";
  const status = playing
    ? "Replaying"
    : offset === 0
      ? "Story now"
      : `${offset < 0 ? "−" : "+"}${relative}`;

  return (
    <div
      data-tour="timeline"
      className={cn(
        "flex items-center gap-3 rounded-xl border border-border bg-card/95 px-3.5 pt-3 pb-2.5 shadow-card min-[1440px]:gap-4 min-[1440px]:px-4",
        className,
      )}
    >
      <button
        type="button"
        aria-label={playing ? "Pause the replay" : "Play the replay"}
        onClick={() => setPlaying((value) => !value)}
        className="grid size-10 shrink-0 place-items-center rounded-full bg-primary text-primary-foreground transition-[background-color,transform] duration-150 hover:bg-brand-700 active:scale-95"
      >
        {playing ? <Pause aria-hidden className="size-4 fill-current" /> : <Play aria-hidden className="size-4 translate-x-px fill-current" />}
      </button>

      <div className="w-[118px] shrink-0 min-[1440px]:w-[136px]">
        <b className="block font-mono text-[13px] leading-tight font-semibold tracking-tighter min-[1440px]:text-[15px] whitespace-nowrap text-foreground tabular-nums">
          {clockLabel(step)}
        </b>
        <span className="flex items-center gap-1.5 text-[11px] whitespace-nowrap text-muted-foreground">
          <LiveDot playing={playing} />
          {status}
          {!playing && offset !== 0 ? (
            <button
              type="button"
              onClick={() => setStep(NOW_STEP)}
              className="font-medium text-brand-700 hover:underline focus-visible:underline focus-visible:outline-none"
            >
              · back<span className="hidden min-[1440px]:inline"> to now</span>
            </button>
          ) : null}
        </span>
      </div>

      <div className="relative min-w-0 flex-1 select-none">
        {/* Agent actions lane */}
        <div className="mb-0.5 flex items-center justify-between gap-3 text-[10px] leading-none">
          <span className="font-medium tracking-[0.04em] whitespace-nowrap text-muted-foreground uppercase">
            Agent actions{agentOn ? "" : " · none in Baseline"}
          </span>
          <span className="hidden min-w-0 items-center gap-1.5 overflow-hidden text-text-secondary min-[1400px]:flex">
            {(Object.keys(ACTION_LABEL) as ActionKind[]).map((kind) => (
              <span key={kind} className="flex items-center gap-1 whitespace-nowrap">
                <Marker kind={kind} className="size-2" />
                {LANE_KEY[kind]}
              </span>
            ))}
          </span>
        </div>
        <div className="relative h-[18px]">
          <span aria-hidden className="absolute inset-x-0 top-[9px] h-px bg-border" />
          {markers.map((marker) => {
            const lead = marker.items[0];
            if (!lead) return null;
            const label = `${clockLabel(marker.step)} · ${lead.title}${marker.items.length > 1 ? ` (+${marker.items.length - 1} more)` : ""}`;
            return (
              <button
                key={`${marker.kind}-${marker.step}`}
                type="button"
                aria-label={label}
                title={label}
                onClick={() => {
                  setPlaying(false);
                  setStep(marker.step);
                  if (agentOn) showAgentToast(lead, { others: marker.kind === "dispatch" ? marker.items.length - 1 : 0 });
                }}
                style={{ left: pct(marker.step) }}
                className={cn(
                  "absolute top-0 -ml-[9px] grid size-[18px] place-items-center rounded-full transition-[opacity,transform] duration-150 hover:scale-125 focus-visible:outline-2 focus-visible:outline-ring",
                  !agentOn ? "opacity-15" : marker.step > step ? "opacity-45" : "opacity-100",
                )}
              >
                <Marker kind={marker.kind} className="ring-[1.5px] ring-card" />
              </button>
            );
          })}
        </div>

        {/* Track */}
        <div
          ref={track}
          role="slider"
          tabIndex={0}
          aria-label="Replay time"
          aria-valuemin={0}
          aria-valuemax={LAST}
          aria-valuenow={step}
          aria-valuetext={clockLabel(step)}
          onPointerDown={onPointerDown}
          onPointerMove={onPointerMove}
          onPointerUp={() => setDragging(false)}
          onPointerCancel={() => setDragging(false)}
          onKeyDown={onKeyDown}
          className="group relative h-[22px] cursor-pointer touch-none rounded-sm outline-none focus-visible:ring-2 focus-visible:ring-ring/40"
        >
          <div className="absolute inset-x-0 top-[9px] h-1 overflow-hidden rounded-full bg-muted">
            {/* nights, 20:00–06:00, so the days read at a glance */}
            {Array.from({ length: DAYS }, (_, day) => (
              <span key={day}>
                <span
                  className="absolute inset-y-0 bg-foreground/[0.06]"
                  style={{ left: pct(day * STEPS_PER_DAY), width: pct(24) }}
                />
                <span
                  className="absolute inset-y-0 bg-foreground/[0.06]"
                  style={{ left: pct(day * STEPS_PER_DAY + 80), width: pct(16) }}
                />
              </span>
            ))}
            <span className="absolute inset-y-0 left-0 rounded-full bg-primary" style={{ width: pct(step) }} />
          </div>
          <span
            aria-hidden
            title="The demo's present moment"
            className="absolute top-0 -ml-px h-[22px] w-0.5 rounded-full bg-foreground/55"
            style={{ left: pct(NOW_STEP) }}
          >
            <span className="absolute -top-px left-[5px] font-mono text-[9px] font-medium text-text-secondary">now</span>
          </span>
          <span
            aria-hidden
            className={cn(
              "absolute top-[3px] -ml-2 size-4 rounded-full border-2 border-primary bg-card shadow-[0_1px_3px_rgb(13_13_13/0.18)] transition-transform duration-150 group-hover:scale-110",
              dragging && "scale-110",
            )}
            style={{ left: pct(step) }}
          />
        </div>

        {/* Day ticks */}
        <div className="relative h-4">
          {Array.from({ length: DAYS }, (_, day) => (
            <span key={day}>
              <span aria-hidden className="absolute -top-6 h-1.5 w-px bg-border" style={{ left: pct(day * STEPS_PER_DAY) }} />
              <button
                type="button"
                tabIndex={-1}
                onClick={() => setStep(day * STEPS_PER_DAY + (step % STEPS_PER_DAY))}
                className={cn(
                  "absolute top-0 ml-[3px] text-[11px] whitespace-nowrap transition-colors duration-150 hover:text-foreground",
                  day === dayOfStep(step) ? "font-semibold text-foreground" : "text-muted-foreground",
                )}
                style={{ left: pct(day * STEPS_PER_DAY) }}
              >
                <span className="min-[1440px]:hidden">{dayLabel(day).split(" ")[0]}</span>
                <span className="hidden min-[1440px]:inline">{dayLabel(day)}</span>
              </button>
            </span>
          ))}
        </div>
      </div>

      <div className="flex shrink-0 flex-col items-end gap-1">
        <PolicySwitch policy={policy} onChange={setPolicy} />
        <span className="text-[11px] whitespace-nowrap text-muted-foreground">
          <Term id="baseline">Baseline</Term> = without Sentinel<span className="hidden min-[1440px]:inline"> · space plays</span>
        </span>
      </div>
    </div>
  );
}
