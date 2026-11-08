"use client";

import {
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

import { type GlossaryId, getTerm } from "@/lib/glossary";
import { cn } from "@wattsentry/ui/lib/utils";

const OPEN_DELAY_MS = 120;
const CLOSE_DELAY_MS = 120;
const GAP = 8;
const EDGE = 12;
const MAX_WIDTH = 288;

type Pos = { left: number; top: number; side: "top" | "bottom" };

/**
 * A jargon word with a dotted underline and a plain-English definition on hover, keyboard focus or tap.
 *
 *   <Term id="p90">P90</Term>
 *
 * Accessible: the definition is the trigger's aria-describedby (a hidden node, so screen readers get it
 * without opening the popover); Esc closes; tap toggles on touch; taps on a Term don't bubble to a
 * clickable parent row. Ids are typed from lib/glossary.ts.
 *
 * Not built on packages/ui's TooltipContent on purpose: that one is a dark, square tooltip, and Base UI
 * tooltips do not open on tap. This is a white card, positioned with fixed coordinates.
 */
export function Term({
  id,
  children,
}: {
  id: GlossaryId;
  children: ReactNode;
}) {
  const entry = getTerm(id);
  const descId = useId();
  const triggerRef = useRef<HTMLSpanElement>(null);
  const popRef = useRef<HTMLDivElement>(null);
  const timer = useRef<ReturnType<typeof setTimeout> | null>(null);
  const pointer = useRef<string>("mouse");
  const [open, setOpen] = useState(false);
  const [pos, setPos] = useState<Pos | null>(null);

  const clear = () => {
    if (timer.current) clearTimeout(timer.current);
    timer.current = null;
  };
  const openSoon = () => {
    clear();
    timer.current = setTimeout(() => setOpen(true), OPEN_DELAY_MS);
  };
  const closeSoon = () => {
    clear();
    timer.current = setTimeout(() => setOpen(false), CLOSE_DELAY_MS);
  };
  useEffect(() => clear, []);

  const place = useCallback(() => {
    const t = triggerRef.current;
    const p = popRef.current;
    if (!t || !p) return;
    const r = t.getBoundingClientRect();
    const w = p.offsetWidth;
    const h = p.offsetHeight;
    const vw = window.innerWidth;
    const side: Pos["side"] = r.top - GAP - h >= EDGE ? "top" : "bottom";
    const left = Math.min(Math.max(EDGE, r.left + r.width / 2 - w / 2), vw - w - EDGE);
    const top = side === "top" ? r.top - GAP - h : r.bottom + GAP;
    setPos({ left, top, side });
  }, []);

  useLayoutEffect(() => {
    if (!open) return;
    place();
    const onMove = () => place();
    window.addEventListener("resize", onMove);
    window.addEventListener("scroll", onMove, true);
    const onDown = (e: PointerEvent) => {
      const target = e.target as Node;
      if (triggerRef.current?.contains(target) || popRef.current?.contains(target)) return;
      setOpen(false);
    };
    const onKey = (e: globalThis.KeyboardEvent) => {
      if (e.key === "Escape") setOpen(false);
    };
    document.addEventListener("pointerdown", onDown);
    document.addEventListener("keydown", onKey);
    return () => {
      window.removeEventListener("resize", onMove);
      window.removeEventListener("scroll", onMove, true);
      document.removeEventListener("pointerdown", onDown);
      document.removeEventListener("keydown", onKey);
    };
  }, [open, place]);

  const onKeyDown = (e: KeyboardEvent<HTMLSpanElement>) => {
    if (e.key === "Enter" || e.key === " ") {
      e.preventDefault();
      setOpen((v) => !v);
    } else if (e.key === "Escape" && open) {
      e.stopPropagation();
      setOpen(false);
    }
  };

  return (
    <>
      <span
        ref={triggerRef}
        tabIndex={0}
        aria-describedby={descId}
        aria-expanded={open}
        data-term={id}
        onPointerEnter={(e) => e.pointerType === "mouse" && openSoon()}
        onPointerLeave={(e) => e.pointerType === "mouse" && closeSoon()}
        onPointerDown={(e) => {
          pointer.current = e.pointerType;
        }}
        onFocus={(e) => {
          // Keyboard focus opens; focus that comes from a click or tap is handled by onClick.
          if (!e.currentTarget.matches(":focus-visible")) return;
          clear();
          setOpen(true);
        }}
        onBlur={() => closeSoon()}
        onClick={(e) => {
          // Touch and pen: a tap toggles. Mouse already opened on hover; a click keeps it open.
          e.stopPropagation();
          clear();
          setOpen((v) => (pointer.current === "mouse" ? true : !v));
        }}
        onKeyDown={onKeyDown}
        className={cn(
          "cursor-help rounded-[3px] underline decoration-muted-foreground decoration-dotted decoration-[1.5px] underline-offset-[3px] outline-none transition-colors duration-150",
          "hover:decoration-foreground focus-visible:ring-2 focus-visible:ring-ring/40 focus-visible:ring-offset-1",
          open && "decoration-foreground",
        )}
      >
        {children}
      </span>
      <span id={descId} hidden>
        {entry.short}
      </span>
      {open &&
        createPortal(
          <div
            ref={popRef}
            role="tooltip"
            aria-hidden
            onPointerEnter={(e) => e.pointerType === "mouse" && clear()}
            onPointerLeave={(e) => e.pointerType === "mouse" && closeSoon()}
            style={{ left: pos?.left ?? -9999, top: pos?.top ?? -9999, maxWidth: MAX_WIDTH }}
            className={cn(
              "fixed z-[120] w-max rounded-lg border border-border bg-popover px-3 py-2.5 text-left text-popover-foreground shadow-card",
              "animate-in fade-in-0 duration-150 ease-out motion-reduce:animate-none",
              pos?.side === "bottom" ? "slide-in-from-top-1" : "slide-in-from-bottom-1",
              !pos && "invisible",
            )}
          >
            <div className="text-[12px] leading-tight font-semibold">{entry.term}</div>
            <p className="mt-1 text-[12px] leading-[1.45] font-normal text-text-secondary normal-case tracking-normal">
              {entry.short}
            </p>
            {entry.more ? (
              <p className="mt-2 border-t border-border pt-1.5 text-[11px] leading-[1.45] font-normal text-muted-foreground normal-case tracking-normal">
                {entry.more}
              </p>
            ) : null}
          </div>,
          document.body,
        )}
    </>
  );
}
