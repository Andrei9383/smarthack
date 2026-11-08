"use client";

import {
  AnimatePresence,
  type AnimationPlaybackControls,
  animate,
  cancelFrame,
  domAnimation,
  frame,
  LazyMotion,
  m,
  type Process,
  useReducedMotion,
  useReducedMotionConfig,
  useSpring,
} from "motion/react";
import type { RefObject } from "react";
import { useLayoutEffect, useRef, useState } from "react";
import { createPortal } from "react-dom";

import { useChartPortalContainer } from "../chart-context";

interface TooltipBoxProps {
  /** Cursor x in container px. The panel opens right of it, or left near the right edge. */
  x: number;
  /** Panel top in container px. */
  top: number;
  visible: boolean;
  containerRef: RefObject<HTMLDivElement | null>;
  containerWidth: number;
  containerHeight: number;
  children: React.ReactNode;
}

/** Popover-style panel: a 1px --border-strong edge and the overlay shadow, theme tokens only. */
const PANEL_CLASS_NAME =
  "relative min-w-[160px] rounded-md border border-border-strong bg-clip-padding text-[12px] text-chart-tooltip-foreground shadow-overlay bg-chart-tooltip-background backdrop-blur-md";

/** Clips content to the panel's inner radius without clipping the edge highlight. */
const PANEL_INNER_CLASS_NAME = "overflow-hidden rounded-[calc(var(--radius-md)-1px)]";

/** Follow spring of the floating panel. */
const BOX_SPRING = { stiffness: 100, damping: 20 };
/** Gap between the cursor and the panel. */
const OFFSET = 16;

/** Panel scale/fade-in on show. */
const PANEL_ENTRANCE_SPRING = {
  type: "spring",
  stiffness: 300,
  damping: 25,
} as const;

/**
 * Panel size tween when its content grows or shrinks. Critically damped (no
 * overshoot, so a shrinking panel never dips under its content) and a little
 * quicker than the entrance, so it reads as the panel hugging the new content.
 */
const PANEL_SIZE_SPRING = {
  type: "spring",
  stiffness: 500,
  damping: 45,
} as const;

/** Layout sizes (unaffected by transforms) of the panel and its clip box. */
interface PanelSizes {
  panelWidth: number;
  panelHeight: number;
  clipWidth: number;
  clipHeight: number;
}

const mixNumber = (from: number, to: number, progress: number) =>
  from + (to - from) * progress;

function mixPanelSizes(
  from: PanelSizes,
  to: PanelSizes,
  progress: number
): PanelSizes {
  return {
    panelWidth: mixNumber(from.panelWidth, to.panelWidth, progress),
    panelHeight: mixNumber(from.panelHeight, to.panelHeight, progress),
    clipWidth: mixNumber(from.clipWidth, to.clipWidth, progress),
    clipHeight: mixNumber(from.clipHeight, to.clipHeight, progress),
  };
}

const samePanelSizes = (a: PanelSizes, b: PanelSizes) =>
  a.panelWidth === b.panelWidth &&
  a.panelHeight === b.panelHeight &&
  a.clipWidth === b.clipWidth &&
  a.clipHeight === b.clipHeight;

/** Pixel radius from the computed top-left radius, or `null` (e.g. `%`). */
function readPxRadius(element: HTMLElement): number | null {
  const value = getComputedStyle(element).borderTopLeftRadius;
  return value.endsWith("px") ? Number.parseFloat(value) : null;
}

/**
 * Radius `r` (px) as per-axis percentages of the drawn (scaled) size, so the
 * corners stay round while the box is scaled — Motion's own correction.
 */
function correctedRadius(r: number, width: number, height: number): string {
  const x = width === 0 ? 0 : (r / width) * 100;
  const y = height === 0 ? 0 : (r / height) * 100;
  return `${x}% / ${y}%`;
}

const ratio = (drawn: number, layout: number) =>
  layout === 0 ? 1 : drawn / layout;

interface PanelRadii {
  panel: number | null;
  clip: number | null;
  panelInline: string;
  clipInline: string;
}

/** Border-box layout size (fractional, unaffected by transforms). */
function readBorderBox(element: HTMLElement): [number, number] {
  const style = getComputedStyle(element);
  let width = Number.parseFloat(style.width);
  let height = Number.parseFloat(style.height);
  if (!(Number.isFinite(width) && Number.isFinite(height))) {
    return [element.offsetWidth, element.offsetHeight];
  }
  if (style.boxSizing !== "border-box") {
    width +=
      Number.parseFloat(style.paddingLeft) +
      Number.parseFloat(style.paddingRight) +
      Number.parseFloat(style.borderLeftWidth) +
      Number.parseFloat(style.borderRightWidth);
    height +=
      Number.parseFloat(style.paddingTop) +
      Number.parseFloat(style.paddingBottom) +
      Number.parseFloat(style.borderTopWidth) +
      Number.parseFloat(style.borderBottomWidth);
  }
  return [width, height];
}

function readPanelSizes(panel: HTMLElement, clip: HTMLElement): PanelSizes {
  const [panelWidth, panelHeight] = readBorderBox(panel);
  const [clipWidth, clipHeight] = readBorderBox(clip);
  return { panelWidth, panelHeight, clipWidth, clipHeight };
}

/**
 * One frame of the size tween at `progress` (0–1): the panel scaled to the
 * drawn size, the clip kept at its unscaled offset (the border) inside it and
 * scaled to its own drawn size, the content counter-scaled to 1.
 */
function drawPanelTween(
  {
    panel,
    clip,
    content,
  }: { panel: HTMLElement; clip: HTMLElement; content: HTMLElement },
  radii: PanelRadii,
  from: PanelSizes,
  target: PanelSizes,
  progress: number
) {
  const drawn = mixPanelSizes(from, target, progress);
  const scaleX = ratio(drawn.panelWidth, target.panelWidth);
  const scaleY = ratio(drawn.panelHeight, target.panelHeight);
  const clipScaleX = ratio(drawn.clipWidth, target.clipWidth);
  const clipScaleY = ratio(drawn.clipHeight, target.clipHeight);
  panel.style.transform = `scale(${scaleX}, ${scaleY})`;
  const offsetX = clip.offsetLeft / scaleX - clip.offsetLeft;
  const offsetY = clip.offsetTop / scaleY - clip.offsetTop;
  const translate =
    offsetX || offsetY ? `translate3d(${offsetX}px, ${offsetY}px, 0px) ` : "";
  clip.style.transform = `${translate}scale(${clipScaleX / scaleX}, ${
    clipScaleY / scaleY
  })`;
  content.style.transform = `scale(${1 / clipScaleX}, ${1 / clipScaleY})`;
  if (radii.panel !== null) {
    panel.style.borderRadius = correctedRadius(
      radii.panel,
      drawn.panelWidth,
      drawn.panelHeight
    );
  }
  if (radii.clip !== null) {
    clip.style.borderRadius = correctedRadius(
      radii.clip,
      drawn.clipWidth,
      drawn.clipHeight
    );
  }
}

/**
 * Panel size tween when the content grows or shrinks (FLIP, like Motion's `layout="size"` without the
 * layout bundle): the panel is drawn at its previous size and sprung to the new one, the clip box
 * follows and the content is counter-scaled, so text never stretches. Sizes are layout sizes, so
 * the flip / clamp measurements always see the final size. The first measurement only records it.
 */
function usePanelSizeTween(
  panelRef: RefObject<HTMLDivElement | null>,
  clipRef: RefObject<HTMLDivElement | null>,
  contentRef: RefObject<HTMLDivElement | null>,
  enabled: boolean
) {
  const checkRef = useRef<(() => void) | null>(null);

  useLayoutEffect(() => {
    const panel = panelRef.current;
    const clip = clipRef.current;
    const content = contentRef.current;
    if (!(enabled && panel && clip && content)) {
      return;
    }

    let target = readPanelSizes(panel, clip);
    let from: PanelSizes | null = null;
    let progress = 1;
    let controls: AnimationPlaybackControls | undefined;
    let pendingStart: Process | undefined;
    // Read when a tween starts; the inline values are restored when it ends.
    let radii: PanelRadii | null = null;

    const stop = () => {
      if (pendingStart) {
        cancelFrame(pendingStart);
        pendingStart = undefined;
      }
      controls?.stop();
      controls = undefined;
    };

    const reset = () => {
      panel.style.transform = "";
      clip.style.transform = "";
      content.style.transform = "";
      if (radii) {
        panel.style.borderRadius = radii.panelInline;
        clip.style.borderRadius = radii.clipInline;
        radii = null;
      }
    };

    const draw = (latest: number) => {
      progress = latest;
      if (from && radii) {
        drawPanelTween({ panel, clip, content }, radii, from, target, latest);
      }
    };

    const check = () => {
      const next = readPanelSizes(panel, clip);
      if (samePanelSizes(next, target)) {
        return;
      }
      from =
        from && progress < 1 ? mixPanelSizes(from, target, progress) : target;
      target = next;
      stop();
      radii ??= {
        panel: readPxRadius(panel),
        clip: readPxRadius(clip),
        panelInline: panel.style.borderRadius,
        clipInline: clip.style.borderRadius,
      };
      // This frame draws the old size; the spring starts on the next one.
      draw(0);
      pendingStart = frame.update(() => {
        pendingStart = undefined;
        controls = animate(0, 1000, {
          ...PANEL_SIZE_SPRING,
          velocity: 0,
          onUpdate: (latest) => draw(latest / 1000),
          onComplete: () => {
            progress = 1;
            from = null;
            controls = undefined;
            reset();
          },
        });
      });
    };

    checkRef.current = check;
    const observer =
      typeof ResizeObserver === "undefined" ? null : new ResizeObserver(check);
    observer?.observe(panel);
    observer?.observe(clip);
    return () => {
      checkRef.current = null;
      observer?.disconnect();
      stop();
      reset();
    };
  }, [panelRef, clipRef, contentRef, enabled]);

  // After every render (new content): measure before paint, as Motion did.
  useLayoutEffect(() => {
    checkRef.current?.();
  });
}

const ORIGIN_TOP_LEFT: React.CSSProperties = { transformOrigin: "0 0" };

/** Entrance wrapper (scale / slide around `transformOrigin`) around the panel, its clip box and content. */
function TooltipPanel({
  animateSize,
  entrance,
  transformOrigin,
  className,
  style,
  children,
}: {
  animateSize: boolean;
  /** Entrance scale/slide start, or `false` to only fade in. */
  entrance: { scale: number; x?: number } | false;
  transformOrigin: string;
  className: string;
  style: React.CSSProperties;
  children: React.ReactNode;
}) {
  const panelRef = useRef<HTMLDivElement>(null);
  const clipRef = useRef<HTMLDivElement>(null);
  const contentRef = useRef<HTMLDivElement>(null);
  // `MotionConfig reducedMotion` made Motion's layout tween instant.
  const reducedMotionConfig = useReducedMotionConfig() === true;
  usePanelSizeTween(
    panelRef,
    clipRef,
    contentRef,
    animateSize && !reducedMotionConfig
  );

  return (
    <m.div
      animate={{ scale: 1, x: 0 }}
      initial={entrance === false ? false : { x: 0, ...entrance }}
      style={{ transformOrigin }}
      transition={PANEL_ENTRANCE_SPRING}
    >
      <m.div
        animate={{ opacity: 1 }}
        className={className}
        initial={{ opacity: 0 }}
        ref={panelRef}
        style={{ ...style, ...ORIGIN_TOP_LEFT }}
        transition={PANEL_ENTRANCE_SPRING}
      >
        <div
          className={PANEL_INNER_CLASS_NAME}
          ref={clipRef}
          style={ORIGIN_TOP_LEFT}
        >
          <div ref={contentRef} style={ORIGIN_TOP_LEFT}>
            {children}
          </div>
        </div>
      </m.div>
    </m.div>
  );
}

const DEFAULT_TOOLTIP_WIDTH = 180;

/** Panel left beside the cursor: flips left near the right edge. */
function resolveLeft(x: number, width: number, containerWidth: number) {
  const flipped = x + width + OFFSET > containerWidth;
  return { flipped, left: flipped ? x - OFFSET - width : x + OFFSET };
}

// Inner only while visible, so the spring starts at the cursor instead of (0, 0) on first hover.
export function TooltipBox(props: TooltipBoxProps) {
  const container = useChartPortalContainer(props.containerRef);
  if (!container) return null;
  // Own feature scope: the box renders outside the chart's svg tree.
  return (
    <LazyMotion features={domAnimation}>
      <AnimatePresence>
        {props.visible ? <TooltipBoxInner {...props} container={container} key="tooltip-box" /> : null}
      </AnimatePresence>
    </LazyMotion>
  );
}

function TooltipBoxInner({
  x,
  top,
  containerWidth,
  children,
  container,
}: Omit<TooltipBoxProps, "visible" | "containerRef"> & { container: HTMLElement }) {
  // Reduced motion: position jumps instead of springing and the panel skips its scale / slide.
  const reducedMotion = useReducedMotion() === true;
  const tooltipRef = useRef<HTMLDivElement>(null);
  const widthRef = useRef(DEFAULT_TOOLTIP_WIDTH);
  const hasPositionedRef = useRef(false);

  // Flip side + replay key for the panel entrance, seeded from the default width and refined from
  // the measured one (a re-render only when the side changes).
  const [flip, setFlip] = useState(() => ({
    flipped: resolveLeft(x, DEFAULT_TOOLTIP_WIDTH, containerWidth).flipped,
    key: 0,
  }));
  const [initialLeft] = useState(() => resolveLeft(x, DEFAULT_TOOLTIP_WIDTH, containerWidth).left);
  const animatedLeft = useSpring(initialLeft, BOX_SPRING);

  useLayoutEffect(() => {
    const el = tooltipRef.current;
    if (!el) return;
    if (el.offsetWidth > 0) widthRef.current = el.offsetWidth;
    const target = resolveLeft(x, widthRef.current, containerWidth);
    setFlip((prev) => (prev.flipped === target.flipped ? prev : { flipped: target.flipped, key: prev.key + 1 }));
    // The first measured position jumps (no spring from the estimated width).
    if (reducedMotion || !hasPositionedRef.current) animatedLeft.jump(target.left);
    else animatedLeft.set(target.left);
    hasPositionedRef.current = true;
  }, [x, containerWidth, animatedLeft, reducedMotion]);

  return createPortal(
    <m.div
      animate={{ opacity: 1 }}
      className="pointer-events-none absolute z-50"
      exit={{ opacity: 0 }}
      initial={{ opacity: 0 }}
      ref={tooltipRef}
      style={{ left: animatedLeft, top }}
      transition={{ duration: 0.1 }}
    >
      <TooltipPanel
        animateSize={!reducedMotion}
        className={PANEL_CLASS_NAME}
        entrance={reducedMotion ? false : { scale: 0.85, x: flip.flipped ? 20 : -20 }}
        key={flip.key}
        style={{ backgroundColor: "var(--chart-tooltip-background)" }}
        transformOrigin={flip.flipped ? "right top" : "left top"}
      >
        {children}
      </TooltipPanel>
    </m.div>,
    container,
  );
}
