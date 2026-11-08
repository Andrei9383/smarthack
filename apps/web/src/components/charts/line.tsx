"use client";

import { type CurveFactory, curveNatural } from "d3-shape";
import { m, type MotionValue, useSpring } from "motion/react";
import { type RefObject, useCallback, useEffect, useId, useLayoutEffect, useRef, useState } from "react";

import {
  type ChartHoverContextValue,
  chartCssVars,
  DATA_TWEEN_MS,
  DISCRETE_INTERACTION_LABEL_THRESHOLD,
  selectIsChartHovering,
  useChartHoverSelector,
  useChartHoverStore,
  useChartStable,
  useSettledYScale,
  useYScale,
} from "./chart-context";
import { useAnimatedSeriesPath } from "./use-animated-series-path";

export interface LineProps {
  dataKey: string;
  /** Y-scale group id. Default: `"left"`. */
  yAxisId?: string | number;
  /** Default: var(--chart-line-primary) */
  stroke?: string;
  /** Default: 2.5 */
  strokeWidth?: number;
  /** Default: curveNatural */
  curve?: CurveFactory;
  /** Data index from which the stroke is dashed ("6,4"), e.g. a limit drawn as a dashed line. */
  dashFromIndex?: number;
}

const DASH_ARRAY = "6,4";
/** Opacity the rest of the line dims to while the chart is hovered. */
const HOVER_DIM_OPACITY = 0.3;
const HIGHLIGHT_SPRING = { stiffness: 180, damping: 28 };

/** `d` and length of the rendered stroke, re-measured after the deps change (dash tail only). */
function usePathStrokeMetrics(pathRef: RefObject<SVGPathElement | null>, deps: readonly unknown[], enabled: boolean) {
  const [metrics, setMetrics] = useState<{ pathD: string | null; pathLength: number }>({ pathD: null, pathLength: 0 });
  useEffect(() => {
    const path = pathRef.current;
    if (!(enabled && path)) return;
    const d = path.getAttribute("d");
    const len = d ? path.getTotalLength() : 0;
    setMetrics((prev) => (prev.pathD === d && prev.pathLength === len ? prev : { pathD: d, pathLength: len }));
  }, [enabled, ...deps]);
  return metrics;
}

/**
 * Solid head up to `dashFromIndex`, then the same curved path dashed and clipped to x ≥ the dash
 * start, so the dashes follow the curve.
 */
function DashTailStroke({
  pathD,
  pathLength,
  dashFromIndex,
  dataLength,
  dashStartX,
  stroke,
  strokeWidth,
}: {
  pathD: string | null;
  pathLength: number;
  dashFromIndex: number;
  dataLength: number;
  dashStartX: number;
  stroke: string;
  strokeWidth: number;
}) {
  const { innerWidth, innerHeight } = useChartStable();
  const clipPathId = useId().replace(/:/g, "");
  // Index-based approximation of the path length at the dash start: exact for evenly spaced data
  // on flat stretches, a pixel or two off on steep ones.
  const dashStartLength = (dashFromIndex / Math.max(1, dataLength - 1)) * pathLength;
  if (!pathD || pathLength <= 0 || dashStartLength >= pathLength) return null;
  const pad = strokeWidth * 2;
  return (
    <>
      <defs>
        <clipPath id={clipPathId}>
          <rect
            height={innerHeight + pad}
            width={Math.max(0, innerWidth - dashStartX + pad)}
            x={dashStartX - strokeWidth}
            y={-strokeWidth}
          />
        </clipPath>
      </defs>
      <path
        d={pathD}
        fill="none"
        stroke={stroke}
        strokeDasharray={`${dashStartLength} ${Math.max(1, pathLength - dashStartLength)}`}
        strokeLinecap="round"
        strokeWidth={strokeWidth}
      />
      <path
        clipPath={`url(#${clipPathId})`}
        d={pathD}
        fill="none"
        stroke={stroke}
        strokeDasharray={DASH_ARRAY}
        strokeLinecap="round"
        strokeWidth={strokeWidth}
      />
    </>
  );
}

/** Dims the series while the chart is hovered; re-renders on hover enter / leave only. */
function HoverDim({ children }: { children: React.ReactNode }) {
  const hovering = useChartHoverSelector(selectIsChartHovering);
  return (
    <m.g
      animate={{ opacity: hovering ? HOVER_DIM_OPACITY : 1 }}
      initial={{ opacity: 1 }}
      transition={{ duration: 0.4, ease: "easeInOut" }}
    >
      {children}
    </m.g>
  );
}

/**
 * Pixel band one data point either side of the hovered point (clamped to the ends), or the dragged
 * range of an active selection.
 */
function segmentBounds(
  data: Record<string, unknown>[],
  xScale: (value: Date) => number | undefined,
  xAccessor: (d: Record<string, unknown>) => Date,
  { tooltipData, selection }: ChartHoverContextValue,
): { x: number; width: number } | null {
  if (data.length === 0) return null;
  if (selection?.active) {
    return { x: Math.min(selection.startX, selection.endX), width: Math.abs(selection.endX - selection.startX) };
  }
  if (!tooltipData) return null;
  const start = data[Math.max(0, tooltipData.index - 1)];
  const end = data[Math.min(data.length - 1, tooltipData.index + 1)];
  if (!(start && end)) return null;
  const startX = xScale(xAccessor(start)) ?? 0;
  const endX = xScale(xAccessor(end)) ?? 0;
  return { x: startX, width: Math.max(0, endX - startX) };
}

/**
 * Hover highlight: re-strokes the base path at full opacity, clipped to a band whose x / width
 * spring to the hovered point. The springs are driven straight from the hover store, so a pointer
 * move retargets them without a React render.
 */
function HighlightSegment({
  pathRef,
  stroke,
  strokeWidth,
}: {
  pathRef: RefObject<SVGPathElement | null>;
  stroke: string;
  strokeWidth: number;
}) {
  const { data, xScale, xAccessor, dateLabels, innerHeight } = useChartStable();
  const snap = dateLabels.length > DISCRETE_INTERACTION_LABEL_THRESHOLD;
  const store = useChartHoverStore();
  const resolveBounds = useCallback(
    (hover: ChartHoverContextValue) => segmentBounds(data, xScale, xAccessor, hover),
    [data, xScale, xAccessor],
  );
  const isActive = useChartHoverSelector((hover) => resolveBounds(hover) !== null);
  const x = useSpring(0, HIGHLIGHT_SPRING);
  const width = useSpring(0, HIGHLIGHT_SPRING);
  const wasActive = useRef(false);

  useLayoutEffect(() => {
    const drive = () => {
      const bounds = resolveBounds(store.getSnapshot());
      if (!bounds) {
        wasActive.current = false;
        return;
      }
      if (!wasActive.current || snap) {
        x.jump(bounds.x);
        width.jump(bounds.width);
      }
      x.set(bounds.x);
      width.set(bounds.width);
      wasActive.current = true;
    };
    drive();
    return store.subscribe(drive);
  }, [resolveBounds, snap, store, x, width]);

  if (!isActive) return null;
  return <HighlightStroke height={innerHeight} pathRef={pathRef} stroke={stroke} strokeWidth={strokeWidth} width={width} x={x} />;
}

function HighlightStroke({
  pathRef,
  stroke,
  strokeWidth,
  height,
  x,
  width,
}: {
  pathRef: RefObject<SVGPathElement | null>;
  stroke: string;
  strokeWidth: number;
  height: number;
  x: MotionValue<number>;
  width: MotionValue<number>;
}) {
  const bandId = useId();
  const highlightRef = useRef<SVGPathElement>(null);
  const appliedRef = useRef<{ d: string; element: SVGPathElement | null } | null>(null);

  // Copy the base stroke's `d` after commit (before paint), only when it or the element changed:
  // re-setting a long `d` re-parses the path.
  useLayoutEffect(() => {
    const d = pathRef.current?.getAttribute("d") || "";
    const element = highlightRef.current;
    if (appliedRef.current?.d === d && appliedRef.current.element === element) return;
    element?.setAttribute("d", d);
    appliedRef.current = { d, element };
  });

  return (
    <>
      <defs>
        <clipPath id={bandId}>
          {/* Bleed by the stroke width so a stroke at the min / max value keeps its full width. */}
          <m.rect height={height + strokeWidth * 2} width={width} x={x} y={-strokeWidth} />
        </clipPath>
      </defs>
      <m.g
        animate={{ opacity: 1 }}
        clipPath={`url(#${bandId})`}
        exit={{ opacity: 0 }}
        initial={{ opacity: 0 }}
        transition={{ duration: 0.4, ease: "easeInOut" }}
      >
        <path fill="none" ref={highlightRef} stroke={stroke} strokeLinecap="round" strokeWidth={strokeWidth} />
      </m.g>
    </>
  );
}

export function Line({
  dataKey,
  yAxisId,
  stroke = chartCssVars.linePrimary,
  strokeWidth = 2.5,
  curve = curveNatural,
  dashFromIndex,
}: LineProps) {
  const { data, renderData, xScale, innerWidth, xAccessor } = useChartStable();
  const yScale = useYScale(yAxisId);
  const settledYScale = useSettledYScale(yAxisId);
  const pathD = useAnimatedSeriesPath({
    curve,
    dataKey,
    durationMs: DATA_TWEEN_MS,
    innerWidth,
    renderData,
    settledYScale,
    xAccessor,
    xScale,
    yScale,
  });

  const pathRef = useRef<SVGPathElement>(null);
  const hasDashTail = dashFromIndex != null && dashFromIndex >= 0 && dashFromIndex < data.length - 1;
  const metrics = usePathStrokeMetrics(pathRef, [renderData, innerWidth, dashFromIndex, pathD], hasDashTail);
  const dashPoint = hasDashTail ? data[dashFromIndex] : undefined;

  return (
    <>
      <HoverDim>
        {/* With a dash tail the base path stays (transparent) for measuring and the highlight. */}
        <path
          d={pathD}
          fill="none"
          ref={pathRef}
          stroke={hasDashTail ? "transparent" : stroke}
          strokeLinecap="round"
          strokeWidth={strokeWidth}
        />
        {hasDashTail ? (
          <DashTailStroke
            dashFromIndex={dashFromIndex}
            dashStartX={dashPoint ? (xScale(xAccessor(dashPoint)) ?? 0) : 0}
            dataLength={data.length}
            pathD={metrics.pathD}
            pathLength={metrics.pathLength}
            stroke={stroke}
            strokeWidth={strokeWidth}
          />
        ) : null}
      </HoverDim>
      <HighlightSegment pathRef={pathRef} stroke={stroke} strokeWidth={strokeWidth} />
    </>
  );
}

Line.displayName = "Line";
