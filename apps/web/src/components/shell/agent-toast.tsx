"use client";

import { cn } from "@sentinel-app/ui/lib/utils";
import { BatteryCharging, Radio, UserRound, Wrench, X } from "lucide-react";
import { useEffect, useRef, useState, useSyncExternalStore } from "react";

import { StatusMark } from "@/components/brand/status-mark";
import { clockLabel } from "@/demo/replay";

import { type ActionKind, type AgentAction, useExplainAction } from "./agent-actions";

// One toast for the agent's moment: Thinking → Decided → Acted, then the plain sentence and "why?".
type ToastState = { action: AgentAction; others: number; fast: boolean; key: number } | null;

let current: ToastState = null;
let counter = 0;
const listeners = new Set<() => void>();
const emit = () => {
  for (const listener of listeners) listener();
};

export function showAgentToast(action: AgentAction, options: { others?: number; fast?: boolean } = {}) {
  counter += 1;
  current = { action, others: options.others ?? 0, fast: options.fast ?? false, key: counter };
  emit();
}

export function hideAgentToast() {
  current = null;
  emit();
}

const subscribe = (listener: () => void) => {
  listeners.add(listener);
  return () => {
    listeners.delete(listener);
  };
};

export const KIND_STYLE: Record<ActionKind, { icon: typeof Wrench; tile: string }> = {
  dispatch: { icon: BatteryCharging, tile: "bg-sentinel" },
  workorder: { icon: Wrench, tile: "bg-status-warn" },
  review: { icon: UserRound, tile: "bg-status-review" },
  remote: { icon: Radio, tile: "bg-status-forecast" },
};

const STAGES = ["Thinking", "Decided", "Acted"] as const;

function reducedMotion() {
  return typeof window !== "undefined" && window.matchMedia("(prefers-reduced-motion: reduce)").matches;
}

export function AgentToast({ className }: { className?: string }) {
  const state = useSyncExternalStore(subscribe, () => current, () => null);
  const explain = useExplainAction();
  const [stage, setStage] = useState(0);
  const hovering = useRef(false);
  const hideTimer = useRef<ReturnType<typeof setTimeout>>(undefined);

  useEffect(() => {
    if (!state) return;
    const step = reducedMotion() ? 0 : state.fast ? 160 : 420;
    setStage(step === 0 ? 2 : 0);
    const timers = step === 0 ? [] : [setTimeout(() => setStage(1), step), setTimeout(() => setStage(2), step * 2)];
    clearTimeout(hideTimer.current);
    hideTimer.current = setTimeout(() => {
      if (!hovering.current) hideAgentToast();
    }, state.fast ? 3600 : 7000);
    return () => {
      for (const timer of timers) clearTimeout(timer);
    };
  }, [state]);

  const action = state?.action;
  const { icon: Icon, tile } = KIND_STYLE[action?.kind ?? "dispatch"];
  const sentence = !action ? "" : stage === 0 ? action.thinking : stage === 1 ? action.decided : action.title;

  return (
    <div
      role="status"
      aria-live="polite"
      onMouseEnter={() => {
        hovering.current = true;
        clearTimeout(hideTimer.current);
      }}
      onMouseLeave={() => {
        hovering.current = false;
        hideTimer.current = setTimeout(hideAgentToast, 2500);
      }}
      className={cn(
        "flex w-[420px] max-w-[calc(100%-2rem)] items-start gap-3 rounded-xl bg-card px-3.5 py-3 shadow-overlay transition-[opacity,transform] duration-250 ease-out motion-reduce:transition-none",
        state ? "translate-y-0 opacity-100" : "pointer-events-none translate-y-2.5 opacity-0",
        className,
      )}
    >
      <span className={cn("grid size-[30px] shrink-0 place-items-center rounded-lg text-primary-foreground shadow-inset-hi", tile)}>
        <Icon aria-hidden className="size-4" />
      </span>
      <div className="min-w-0 flex-1">
        <p className="text-[13px] leading-snug font-semibold text-foreground">
          {sentence}
          {action && stage === 2 ? (
            <>
              {" "}
              <button
                type="button"
                onClick={() => {
                  explain(action);
                  hideAgentToast();
                }}
                className="font-semibold text-foreground underline underline-offset-2 hover:underline focus-visible:underline focus-visible:outline-none"
              >
                why?
              </button>
            </>
          ) : null}
        </p>
        {action ? (
          <p className="mt-0.5 text-xs text-text-secondary">
            <span className="font-mono text-[11px] tabular-nums">{clockLabel(action.step)}</span> · {action.detail}
            {state && state.others > 0 ? ` · also ${state.others} other feeder${state.others === 1 ? "" : "s"}` : ""}
          </p>
        ) : null}
        <ol className="mt-2 flex items-center gap-1.5" aria-label="Agent steps">
          {STAGES.map((name, index) => {
            const done = index < stage || (index === 2 && stage === 2);
            const busy = index === stage && stage < 2;
            return (
              <li key={name} className="flex items-center gap-1.5">
                {index > 0 ? <span aria-hidden className="h-px w-3.5 bg-border" /> : null}
                <span
                  className={cn(
                    "flex items-center gap-1.5 text-[11px] font-medium whitespace-nowrap transition-colors duration-200",
                    done || busy ? "text-foreground" : "text-muted-foreground",
                  )}
                >
                  <StatusMark kind={done ? "ok" : busy ? "live" : "unknown"} size="xs" pulse={busy} />
                  {name}
                </span>
              </li>
            );
          })}
        </ol>
      </div>
      <button
        type="button"
        aria-label="Dismiss"
        onClick={hideAgentToast}
        className="grid size-[22px] shrink-0 place-items-center rounded-md text-muted-foreground transition-colors duration-150 hover:bg-surface-raised hover:text-foreground"
      >
        <X aria-hidden className="size-3.5" />
      </button>
    </div>
  );
}
