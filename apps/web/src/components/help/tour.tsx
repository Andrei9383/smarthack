"use client";

import {
  type CSSProperties,
  type KeyboardEvent,
  type ReactNode,
  useCallback,
  useEffect,
  useId,
  useLayoutEffect,
  useRef,
  useState,
} from "react";
import { createPortal } from "react-dom";

import { cn } from "@wattsentry/ui/lib/utils";

/**
 * Elements the tour points at. The shell marks them with `data-tour="…"` (see README.md):
 * rail = mode navigation · map = map canvas · inspector = right-hand details panel ·
 * timeline = bottom dock (play / scrubber) · policy = Baseline/WattSentry toggle.
 */
export type TourTarget = "rail" | "map" | "inspector" | "timeline" | "policy";

export type TourStep = {
  id: string;
  /** `data-tour` value to spotlight. Omit (or missing in the DOM) = a centred card over the full scrim. */
  target?: TourTarget;
  title: string;
  body: ReactNode;
  /** Preferred side for the card; falls back to whichever side fits. */
  placement?: "right" | "left" | "top" | "bottom" | "center";
};

export const defaultTourSteps: TourStep[] = [
  {
    id: "welcome",
    title: "Welcome to WattSentry",
    body: "An agent that forecasts rooftop solar, home batteries, wind turbines and demand across Romania, then acts on it: it schedules batteries, sends repair crews and flags risks before they happen.",
  },
  {
    id: "rail",
    target: "rail",
    placement: "right",
    title: "Five views, one map",
    body: "Start with Overview: everything that needs attention today. Feeders, O&M, Production and Demand each zoom into one problem the agent solves.",
  },
  {
    id: "map",
    target: "map",
    placement: "center",
    title: "Click anything on the map",
    body: "Feeders, solar homes, turbines and counties are all clickable. Colour shows status, and the legend in the corner says what each colour means.",
  },
  {
    id: "inspector",
    target: "inspector",
    placement: "left",
    title: "Details and the “why”",
    body: "Whatever you select opens here: what it is, what the agent is doing about it, and why. Dotted words have a plain-English definition when you hover them.",
  },
  {
    id: "timeline",
    target: "timeline",
    placement: "top",
    title: "Play the week",
    body: "Press play to replay 2–8 June 2025 in 15-minute steps. Each mark on the lane is an agent action, such as charging batteries before a feeder hits its limit.",
  },
  {
    id: "policy",
    target: "policy",
    placement: "top",
    title: "Baseline vs WattSentry",
    body: "Baseline is the same week without WattSentry: batteries only store their own solar. Flip the switch to see what the agent changes, with the same weather, homes and prices. You can reopen this tour from the ? button.",
  },
];

const PAD = 8; // spotlight padding around the target
const GAP = 14; // card ↔ spotlight
const EDGE = 16; // card ↔ viewport
const CARD_W = 344;
const EASE = "cubic-bezier(.2,.7,.2,1)";
const DUR = 260;

type Rect = { top: number; left: number; width: number; height: number };

function measure(target: TourTarget | undefined): Rect | null {
  if (!target || typeof document === "undefined") return null;
  const el = document.querySelector<HTMLElement>(`[data-tour="${target}"]`);
  if (!el) return null;
  const r = el.getBoundingClientRect();
  if (r.width === 0 && r.height === 0) return null;
  // Clip to the viewport so a target that runs off-screen (e.g. the full-bleed map) stays framed.
  const top = Math.max(0, r.top - PAD);
  const left = Math.max(0, r.left - PAD);
  const bottom = Math.min(window.innerHeight, r.bottom + PAD);
  const right = Math.min(window.innerWidth, r.right + PAD);
  return { top, left, width: right - left, height: bottom - top };
}

function placeCard(rect: Rect | null, placement: TourStep["placement"], cw: number, ch: number) {
  const vw = window.innerWidth;
  const vh = window.innerHeight;
  const clampX = (x: number) => Math.min(Math.max(EDGE, x), vw - cw - EDGE);
  const clampY = (y: number) => Math.min(Math.max(EDGE, y), vh - ch - EDGE);
  if (!rect) return { left: clampX((vw - cw) / 2), top: clampY((vh - ch) / 2) };

  const cx = rect.left + rect.width / 2;
  const cy = rect.top + rect.height / 2;
  const fits = {
    right: vw - (rect.left + rect.width) - GAP - EDGE >= cw,
    left: rect.left - GAP - EDGE >= cw,
    bottom: vh - (rect.top + rect.height) - GAP - EDGE >= ch,
    top: rect.top - GAP - EDGE >= ch,
  };
  const at = {
    right: { left: rect.left + rect.width + GAP, top: clampY(cy - ch / 2) },
    left: { left: rect.left - GAP - cw, top: clampY(cy - ch / 2) },
    bottom: { left: clampX(cx - cw / 2), top: rect.top + rect.height + GAP },
    top: { left: clampX(cx - cw / 2), top: rect.top - GAP - ch },
  };
  const order = (["right", "left", "bottom", "top"] as const).filter((s) => s !== placement);
  if (placement && placement !== "center" && fits[placement]) return at[placement];
  if (placement !== "center") for (const s of order) if (fits[s]) return at[s];
  // Large targets (the map) or nothing fits: centre the card inside the spotlight.
  return { left: clampX(cx - cw / 2), top: clampY(cy - ch / 2) };
}

const FOCUSABLE = 'button:not([disabled]), [href], [tabindex]:not([tabindex="-1"])';

/**
 * Spotlight tour: a dimmed scrim with a cut-out around `[data-tour=…]`, and a small card with
 * Back / Next / Skip and "3 of 7". Controlled: pair it with `useTour()`.
 *
 *   const tour = useTour();
 *   <button onClick={tour.start}>?</button>
 *   <Tour steps={defaultTourSteps} open={tour.open} onOpenChange={tour.onOpenChange} />
 *
 * Keyboard: ← / → move between steps, Esc skips, Tab stays inside the card.
 */
export function Tour({
  steps,
  open,
  onOpenChange,
  initialStep = 0,
}: {
  steps: TourStep[];
  open: boolean;
  onOpenChange: (open: boolean) => void;
  /** Step shown each time the tour opens (0-based). */
  initialStep?: number;
}) {
  const [index, setIndex] = useState(initialStep);
  const [rect, setRect] = useState<Rect | null>(null);
  const [card, setCard] = useState<{ left: number; top: number } | null>(null);
  const [mounted, setMounted] = useState(false);
  // The card glides between steps, but must not glide in from (0,0) on the first frame.
  const [settled, setSettled] = useState(false);
  const cardRef = useRef<HTMLDivElement>(null);
  const nextRef = useRef<HTMLButtonElement>(null);
  const restoreRef = useRef<HTMLElement | null>(null);
  const titleId = useId();
  const bodyId = useId();

  const step = steps[Math.min(index, steps.length - 1)];
  const last = index >= steps.length - 1;

  useEffect(() => setMounted(true), []);

  // Reset to the first step and remember focus each time the tour opens; restore focus on close.
  useEffect(() => {
    if (!open) return;
    setIndex(Math.min(initialStep, steps.length - 1));
    setSettled(false);
    restoreRef.current = document.activeElement as HTMLElement | null;
    return () => restoreRef.current?.focus?.();
  }, [open, initialStep, steps.length]);

  const layout = useCallback(() => {
    if (!step) return;
    const r = measure(step.target);
    setRect(r);
    const el = cardRef.current;
    const cw = el?.offsetWidth ?? CARD_W;
    const ch = el?.offsetHeight ?? 200;
    setCard(placeCard(r, step.target ? step.placement : undefined, cw, ch));
  }, [step]);

  useLayoutEffect(() => {
    if (!open || !mounted) return;
    const el = step?.target ? document.querySelector<HTMLElement>(`[data-tour="${step.target}"]`) : null;
    el?.scrollIntoView?.({ block: "nearest", inline: "nearest" });
    layout();
    // The card's height depends on its text; lay out again once it has rendered.
    const raf = requestAnimationFrame(() => {
      layout();
      setSettled(true);
    });
    const ro = new ResizeObserver(layout);
    if (el) ro.observe(el);
    if (cardRef.current) ro.observe(cardRef.current);
    window.addEventListener("resize", layout);
    window.addEventListener("scroll", layout, true);
    return () => {
      cancelAnimationFrame(raf);
      ro.disconnect();
      window.removeEventListener("resize", layout);
      window.removeEventListener("scroll", layout, true);
    };
  }, [open, mounted, step, layout]);

  useEffect(() => {
    if (open) nextRef.current?.focus({ preventScroll: true });
  }, [open, index]);

  const close = () => onOpenChange(false);
  const next = () => (last ? close() : setIndex((i) => i + 1));
  const back = () => setIndex((i) => Math.max(0, i - 1));

  const onKeyDown = (e: KeyboardEvent<HTMLDivElement>) => {
    if (e.key === "Escape") {
      e.preventDefault();
      e.stopPropagation();
      close();
    } else if (e.key === "ArrowRight") {
      e.preventDefault();
      next();
    } else if (e.key === "ArrowLeft") {
      e.preventDefault();
      back();
    } else if (e.key === "Tab" && cardRef.current) {
      const items = [...cardRef.current.querySelectorAll<HTMLElement>(FOCUSABLE)];
      const first = items[0];
      const end = items[items.length - 1];
      if (!first || !end) return;
      if (e.shiftKey && document.activeElement === first) {
        e.preventDefault();
        end.focus();
      } else if (!e.shiftKey && document.activeElement === end) {
        e.preventDefault();
        first.focus();
      }
    }
    // Keep the shell's own shortcuts (1–6, space) from firing behind the tour.
    e.stopPropagation();
  };

  if (!mounted || !open || !step) return null;

  const moving = `${DUR}ms ${EASE}`;
  const spot: CSSProperties = rect
    ? { top: rect.top, left: rect.left, width: rect.width, height: rect.height }
    : { top: window.innerHeight / 2, left: window.innerWidth / 2, width: 0, height: 0 };

  return createPortal(
    <div className="fixed inset-0 z-[110]" onKeyDown={onKeyDown}>
      {/* Click-catcher: the page behind is inert while the tour is open. */}
      <div className="absolute inset-0" aria-hidden />
      <div
        aria-hidden
        className="pointer-events-none absolute rounded-xl transition-[top,left,width,height,opacity] motion-reduce:transition-none"
        style={{
          ...spot,
          transitionDuration: `${DUR}ms`,
          transitionTimingFunction: EASE,
          boxShadow:
            "0 0 0 2px var(--ring), 0 0 0 9999px var(--scrim)",
        }}
      />

      <div
        ref={cardRef}
        role="dialog"
        aria-modal="true"
        aria-labelledby={titleId}
        aria-describedby={bodyId}
        className={cn(
          "absolute rounded-lg bg-card p-5 text-card-foreground shadow-overlay",
          "animate-in fade-in-0 duration-200 ease-out motion-reduce:animate-none motion-reduce:transition-none!",
          !card && "invisible",
        )}
        style={{
          width: CARD_W,
          maxWidth: `calc(100vw - ${EDGE * 2}px)`,
          left: card?.left ?? 0,
          top: card?.top ?? 0,
          transition: settled ? `left ${moving}, top ${moving}` : "none",
        }}
      >
        <div className="flex items-center gap-3">
          <span className="font-mono text-[11px] text-muted-foreground tabular-nums" aria-live="polite">
            {index + 1} of {steps.length}
          </span>
          <span aria-hidden className="flex flex-1 gap-1">
            {steps.map((s, i) => (
              <span
                key={s.id}
                className={cn(
                  "h-[3px] flex-1 rounded-full transition-colors duration-200 motion-reduce:transition-none",
                  i <= index ? "bg-primary" : "bg-muted",
                )}
              />
            ))}
          </span>
        </div>

        <h2 id={titleId} className="mt-3 text-base leading-snug font-semibold">
          {step.title}
        </h2>
        <div id={bodyId} className="mt-1.5 text-[13px] leading-relaxed text-text-secondary">
          {step.body}
        </div>

        <div className="mt-5 flex items-center gap-2">
          <button
            type="button"
            onClick={close}
            className="-ml-1 h-9 rounded-lg px-1.5 text-[13px] font-medium text-muted-foreground transition-colors duration-150 hover:text-foreground focus-visible:ring-2 focus-visible:ring-ring/50 focus-visible:outline-none"
          >
            Skip tour
          </button>
          <span className="flex-1" />
          {index > 0 && (
            <button
              type="button"
              onClick={back}
              className="h-9 rounded-lg border border-border bg-background px-3.5 text-[13px] font-medium transition-colors duration-150 hover:bg-surface-raised focus-visible:ring-2 focus-visible:ring-ring/50 focus-visible:outline-none"
            >
              Back
            </button>
          )}
          <button
            ref={nextRef}
            type="button"
            onClick={next}
            className="h-9 rounded-lg bg-primary px-4 text-[13px] font-medium text-primary-foreground transition-colors duration-150 hover:bg-primary/85 focus-visible:ring-[3px] focus-visible:ring-ring/25 focus-visible:outline-none"
          >
            {last ? "Start exploring" : index === 0 ? "Show me around" : "Next"}
          </button>
        </div>
      </div>
    </div>,
    document.body,
  );
}

const DEFAULT_KEY = "sentinel.tour.v1";

function seen(key: string) {
  try {
    return window.localStorage.getItem(key) === "done";
  } catch {
    return false; // storage blocked: treat as a first visit, but never throw
  }
}

function markSeen(key: string) {
  try {
    window.localStorage.setItem(key, "done");
  } catch {
    /* storage blocked: the tour may show again next visit */
  }
}

/**
 * Open/close state for <Tour>, with auto-open on the first visit (guarded by localStorage).
 * Auto-open is skipped for automated browsers (navigator.webdriver, e.g. screenshot scripts)
 * and when the URL has `?tour=0`, so demo links and screenshots stay clean.
 */
export function useTour({
  storageKey = DEFAULT_KEY,
  autoOpen = true,
  delayMs = 800,
}: { storageKey?: string; autoOpen?: boolean; delayMs?: number } = {}) {
  const [open, setOpen] = useState(false);

  useEffect(() => {
    if (!autoOpen || seen(storageKey)) return;
    if (navigator.webdriver) return;
    if (new URLSearchParams(window.location.search).get("tour") === "0") return;
    const t = setTimeout(() => setOpen(true), delayMs);
    return () => clearTimeout(t);
  }, [autoOpen, storageKey, delayMs]);

  const onOpenChange = useCallback(
    (next: boolean) => {
      setOpen(next);
      if (!next) markSeen(storageKey);
    },
    [storageKey],
  );

  const start = useCallback(() => setOpen(true), []);

  const reset = useCallback(() => {
    try {
      window.localStorage.removeItem(storageKey);
    } catch {
      /* ignore */
    }
  }, [storageKey]);

  return { open, onOpenChange, start, reset };
}
