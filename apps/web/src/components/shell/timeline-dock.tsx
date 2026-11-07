"use client";

import { cn } from "@sentinel-app/ui/lib/utils";
import { Pause, Play } from "lucide-react";
import {
  type FocusEvent,
  type KeyboardEvent,
  type PointerEvent,
  useCallback,
  useEffect,
  useMemo,
  useRef,
  useState,
} from "react";

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
  timeLabel,
} from "@/demo/replay";
import { type Policy, usePolicy, useReplayStep } from "@/demo/use-replay";

import {
  ACTION_LABEL,
  type ActionBucket,
  type ActionKind,
  type ActionMarker,
  bucketMarkers,
  groupMarkers,
  useAgentActions,
} from "./agent-actions";
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

// The bucket chip takes its dominant kind's colour (remote's ring becomes a fill so the count stays legible).
const CHIP: Record<ActionKind, string> = {
  dispatch: "bg-primary",
  workorder: "bg-status-warn",
  review: "bg-status-review",
  remote: "bg-status-forecast",
};

// Half-day buckets once each would get this many pixels; below that, one bucket per day.
const HALF_DAY_MIN_PX = 34;

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
            "relative z-[1] flex h-8 flex-1 items-center justify-center gap-1.5 rounded-[7px] px-2 text-[13px] font-medium whitespace-nowrap transition-colors duration-200 min-[1440px]:px-3",
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

const plural = (n: number, noun: string) => `${n} ${noun}${n === 1 ? "" : "s"}`;

function bucketLabel(bucket: ActionBucket) {
  const day = dayLabel(dayOfStep(bucket.start));
  const span = bucket.end - bucket.start;
  if (span >= STEPS_PER_DAY) return day;
  return `${day} · ${timeLabel(bucket.start)}–${timeLabel(bucket.end % STEPS_PER_DAY === 0 ? bucket.end - 1 : bucket.end)}`;
}

/**
 * The agent-actions lane: one chip per day (or half-day) with a count in its dominant type's colour.
 * Hover or focus lists what the agent did; picking a row jumps there and plays the agent toast.
 */
function ActionLane({
  markers,
  step,
  agentOn,
  onPick,
}: {
  markers: ActionMarker[];
  step: number;
  agentOn: boolean;
  onPick: (marker: ActionMarker) => void;
}) {
  const lane = useRef<HTMLDivElement>(null);
  const [width, setWidth] = useState(0);
  const [open, setOpen] = useState<number | null>(null);
  const closeTimer = useRef<ReturnType<typeof setTimeout>>(undefined);

  useEffect(() => {
    const node = lane.current;
    if (!node) return;
    const observer = new ResizeObserver(([entry]) => setWidth(entry?.contentRect.width ?? 0));
    observer.observe(node);
    return () => observer.disconnect();
  }, []);
  useEffect(() => () => clearTimeout(closeTimer.current), []);

  const size = width / (DAYS * 2) >= HALF_DAY_MIN_PX ? STEPS_PER_DAY / 2 : STEPS_PER_DAY;
  const buckets = useMemo(() => bucketMarkers(markers, size), [markers, size]);

  const show = useCallback((start: number) => {
    clearTimeout(closeTimer.current);
    setOpen(start);
  }, []);
  const hide = useCallback((delay = 140) => {
    clearTimeout(closeTimer.current);
    closeTimer.current = setTimeout(() => setOpen(null), delay);
  }, []);

  return (
    <div ref={lane} className="relative h-[18px]">
      <span aria-hidden className="absolute inset-x-0 top-[9px] h-px bg-border" />
      {buckets.map((bucket) => {
        const centre = Math.min(LAST, (bucket.start + bucket.end) / 2);
        const fraction = centre / LAST;
        const lead = bucket.groups.find((g) => g.kind === bucket.kind) ?? bucket.groups[0];
        const count = bucket.groups.length;
        const isOpen = open === bucket.start;
        const label = `${bucketLabel(bucket)}: ${plural(count, "agent action")}. Mostly ${ACTION_LABEL[bucket.kind].toLowerCase()}.`;
        const onBlur = (event: FocusEvent<HTMLDivElement>) => {
          if (!event.currentTarget.contains(event.relatedTarget as Node | null)) hide(0);
        };
        return (
          <div
            key={bucket.start}
            className="absolute top-0"
            style={{ left: pct(centre) }}
            onMouseEnter={() => show(bucket.start)}
            onMouseLeave={() => hide()}
            onFocus={() => show(bucket.start)}
            onBlur={onBlur}
            onKeyDown={(event) => {
              if (event.key === "Escape" && isOpen) {
                event.stopPropagation();
                setOpen(null);
              }
            }}
          >
            <button
              type="button"
              aria-label={label}
              aria-expanded={isOpen}
              onClick={() => {
                if (lead) onPick(lead);
              }}
              className={cn(
                "absolute top-px flex h-4 min-w-[22px] -translate-x-1/2 items-center justify-center rounded-full px-1.5 font-mono text-[10px] leading-none font-semibold text-primary-foreground tabular-nums ring-2 ring-card transition-[opacity,transform] duration-150 hover:scale-110 focus-visible:outline-2 focus-visible:outline-offset-1 focus-visible:outline-ring motion-reduce:transition-none motion-reduce:hover:scale-100",
                CHIP[bucket.kind],
                !agentOn ? "opacity-20" : bucket.start > step ? "opacity-50" : "opacity-100",
                isOpen && "scale-110 motion-reduce:scale-100",
              )}
            >
              {count}
            </button>

            {isOpen ? (
              <div
                className="absolute bottom-0 z-30 w-[300px] pb-3"
                style={{ left: 0, transform: `translateX(-${fraction * 100}%)` }}
              >
                <div
                  role="group"
                  aria-label={`Agent actions, ${bucketLabel(bucket)}`}
                  className="overflow-hidden rounded-xl border border-border bg-card shadow-[0_1px_2px_rgb(13_13_13/0.05),0_16px_40px_rgb(13_13_13/0.10)] motion-safe:animate-in motion-safe:fade-in-0 motion-safe:slide-in-from-bottom-1 motion-safe:duration-150"
                >
                  <div className="flex items-baseline justify-between gap-3 border-b border-border px-3 py-2">
                    <span className="font-mono text-[11px] font-semibold whitespace-nowrap text-foreground tabular-nums">
                      {bucketLabel(bucket)}
                    </span>
                    <span className="text-[11px] whitespace-nowrap text-muted-foreground tabular-nums">
                      {agentOn ? plural(count, "action") : "Baseline: agent off"}
                    </span>
                  </div>
                  <ul
                    className={cn(
                      "max-h-[216px] overflow-y-auto overscroll-contain py-1 [scrollbar-width:thin]",
                      // more rows below: fade the cut-off one so it reads as "scroll", not as clipped
                      count > 4 && "pb-6 [mask-image:linear-gradient(to_bottom,black_calc(100%-32px),transparent)]",
                    )}
                  >
                    {bucket.groups.map((group) => {
                      const first = group.items[0];
                      if (!first) return null;
                      const more = group.items.length - 1;
                      return (
                        <li key={`${group.kind}-${group.step}`}>
                          <button
                            type="button"
                            onClick={() => {
                              setOpen(null);
                              onPick(group);
                            }}
                            className={cn(
                              "grid w-full grid-cols-[38px_10px_1fr] items-start gap-x-2 px-3 py-1.5 text-left text-xs leading-snug transition-colors duration-150 hover:bg-sidebar focus-visible:bg-sidebar focus-visible:outline-none",
                              group.step > step && "text-text-secondary",
                            )}
                          >
                            <span className="pt-px font-mono text-[11px] text-muted-foreground tabular-nums">
                              {timeLabel(group.step)}
                            </span>
                            <Marker kind={group.kind} className="mt-[3px] size-2" />
                            <span className="min-w-0">
                              <span className="line-clamp-2 text-foreground">{first.title}</span>
                              {more > 0 ? (
                                <span className="text-[11px] text-muted-foreground">
                                  +{more} more{group.kind === "dispatch" ? ` feeder${more === 1 ? "" : "s"}` : ""}
                                </span>
                              ) : null}
                            </span>
                          </button>
                        </li>
                      );
                    })}
                  </ul>
                </div>
              </div>
            ) : null}
          </div>
        );
      })}
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
        "flex items-center gap-2.5 rounded-xl border border-border bg-card/95 px-3 pt-2.5 pb-2 shadow-card min-[1440px]:gap-4 min-[1440px]:px-4",
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

      <div className="w-[100px] shrink-0 min-[1440px]:w-[136px]">
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

      <div className="@container relative min-w-0 flex-1 select-none">
        {/* Agent actions lane. The key drops to normal case, then hides, so it never clips. */}
        <div className="mb-0.5 flex h-3 items-center justify-between gap-2 text-[10px] leading-none">
          <span className="font-medium whitespace-nowrap text-muted-foreground @min-[294px]:tracking-[0.04em] @min-[294px]:uppercase">
            Agent actions{agentOn ? "" : " · none in Baseline"}
          </span>
          <span
            className={cn(
              "hidden shrink-0 items-center gap-1.5 text-text-secondary @min-[272px]:flex",
              !agentOn && "@min-[272px]:hidden @min-[400px]:flex",
            )}
          >
            {(Object.keys(ACTION_LABEL) as ActionKind[]).map((kind) => (
              <span key={kind} className="flex items-center gap-1 whitespace-nowrap">
                <Marker kind={kind} className="size-[7px]" />
                {LANE_KEY[kind]}
              </span>
            ))}
          </span>
        </div>
        <ActionLane
          markers={markers}
          step={step}
          agentOn={agentOn}
          onPick={(marker) => {
            const lead = marker.items[0];
            setPlaying(false);
            setStep(marker.step);
            if (lead && agentOn) showAgentToast(lead, { others: marker.kind === "dispatch" ? marker.items.length - 1 : 0 });
          }}
        />

        {/* "now" sits above the track, beside the top of its line, clear of the thumb and the markers. */}
        <div aria-hidden className="relative h-2.5">
          <span
            title="The demo's present moment"
            className="absolute top-0 font-mono text-[9px] leading-[10px] font-medium text-text-secondary"
            style={{ left: `calc(${pct(NOW_STEP)} + 4px)` }}
          >
            now
          </span>
          <span className="absolute top-0.5 bottom-0 -ml-px w-0.5 rounded-t-full bg-foreground/55" style={{ left: pct(NOW_STEP) }} />
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
          className="group relative h-5 cursor-pointer touch-none rounded-sm outline-none focus-visible:ring-2 focus-visible:ring-ring/40"
        >
          <div className="absolute inset-x-0 top-2 h-1 overflow-hidden rounded-full bg-muted">
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
          {/* day boundaries, just above the bar */}
          {Array.from({ length: DAYS - 1 }, (_, day) => (
            <span
              key={day}
              aria-hidden
              className="absolute top-0 h-1.5 w-px bg-border"
              style={{ left: pct((day + 1) * STEPS_PER_DAY) }}
            />
          ))}
          <span
            aria-hidden
            className="absolute top-0 bottom-1 -ml-px w-0.5 rounded-b-full bg-foreground/55"
            style={{ left: pct(NOW_STEP) }}
          />
          <span
            aria-hidden
            className={cn(
              "absolute top-0.5 -ml-2 size-4 rounded-full border-2 border-primary bg-card shadow-[0_1px_3px_rgb(13_13_13/0.18)] transition-transform duration-150 group-hover:scale-110 motion-reduce:transition-none",
              dragging && "scale-110",
            )}
            style={{ left: pct(step) }}
          />
        </div>

        {/* Day ticks */}
        <div className="relative h-3.5">
          {Array.from({ length: DAYS }, (_, day) => (
            <span key={day}>
              <button
                type="button"
                tabIndex={-1}
                onClick={() => setStep(day * STEPS_PER_DAY + (step % STEPS_PER_DAY))}
                className={cn(
                  "absolute top-0 ml-[3px] text-[11px] leading-[14px] whitespace-nowrap transition-colors duration-150 hover:text-foreground",
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
