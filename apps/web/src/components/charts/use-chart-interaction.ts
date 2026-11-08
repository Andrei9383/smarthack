"use client";

import type { ScaleTime } from "d3-scale";
import { useCallback, useEffect, useRef, useState } from "react";
import { flushSync } from "react-dom";

import {
  type BandConfig,
  type ChartHoverStore,
  createChartHoverStore,
  type LineConfig,
  type Margin,
  normalizeYAxisId,
  type TooltipData,
  type YScale,
} from "./chart-context";

interface UseChartInteractionParams {
  xScale: ScaleTime<number, number>;
  yScale: YScale;
  yScales: Record<string, YScale>;
  data: Record<string, unknown>[];
  lines: LineConfig[];
  bands: BandConfig[];
  margin: Margin;
  xAccessor: (d: Record<string, unknown>) => Date;
  bisectDate: (data: Record<string, unknown>[], date: Date, lo: number) => number;
}

type SvgEvent = React.MouseEvent<SVGGElement> | React.TouchEvent<SVGGElement>;

/** Pixel y per band edge and line key (lines last, so they win when a band edge shares a key). */
function resolveYPositions(
  d: Record<string, unknown>,
  lines: LineConfig[],
  bands: BandConfig[],
  yScales: Record<string, YScale>,
  yScale: YScale,
): Record<string, number> {
  const yPositions: Record<string, number> = {};
  const place = (key: string, yAxisId: string | number | undefined) => {
    const value = d[key];
    if (typeof value === "number") yPositions[key] = (yScales[normalizeYAxisId(yAxisId)] ?? yScale)(value) ?? 0;
  };
  for (const band of bands) {
    place(band.upperKey, band.yAxisId);
    place(band.lowerKey, band.yAxisId);
  }
  for (const line of lines) place(line.dataKey, line.yAxisId);
  return yPositions;
}

/** Pointer x in the svg's user space (handles transformed ancestors), minus the left margin. */
function chartX(event: SvgEvent, margin: Margin, touchIndex = 0): number | null {
  const svg = event.currentTarget.ownerSVGElement;
  const source = "touches" in event ? event.touches[touchIndex] : event;
  const ctm = svg?.getScreenCTM();
  if (!(svg && source && ctm)) return null;
  const point = svg.createSVGPoint();
  point.x = source.clientX;
  point.y = source.clientY;
  return point.matrixTransform(ctm.inverse()).x - margin.left;
}

/**
 * rAF-coalesced tooltip updates into the hover store, deduped on the hovered index and x. The
 * commit is flushed inside the frame, so the tooltip never paints one frame late.
 */
function useScheduledTooltip(store: ChartHoverStore) {
  const lastKeyRef = useRef<string | null>(null);
  const pendingRef = useRef<{ tooltip: TooltipData; key: string } | null>(null);
  const rafRef = useRef<number | null>(null);

  useEffect(
    () => () => {
      if (rafRef.current !== null) cancelAnimationFrame(rafRef.current);
    },
    [],
  );

  const scheduleTooltip = useCallback(
    (tooltip: TooltipData) => {
      const key = `${tooltip.index}:${Math.round(tooltip.x)}`;
      pendingRef.current = { tooltip, key };
      if (key === lastKeyRef.current || rafRef.current !== null) return;
      rafRef.current = requestAnimationFrame(() => {
        rafRef.current = null;
        const pending = pendingRef.current;
        if (!pending || pending.key === lastKeyRef.current) return;
        flushSync(() => {
          lastKeyRef.current = pending.key;
          store.setTooltipData(pending.tooltip);
        });
      });
    },
    [store],
  );

  const clearTooltip = useCallback(() => {
    if (rafRef.current !== null) {
      cancelAnimationFrame(rafRef.current);
      rafRef.current = null;
    }
    pendingRef.current = null;
    lastKeyRef.current = null;
    store.setTooltipData(null);
  }, [store]);

  return { scheduleTooltip, clearTooltip, resetTooltipDedupe: () => (lastKeyRef.current = null) };
}

/**
 * Hover (nearest data row to the pointer) and drag / two-finger selection over the plot. State lives
 * in a hover store, not React state, so a pointer move re-renders only hover consumers.
 */
export function useChartInteraction({
  xScale,
  yScale,
  yScales,
  data,
  lines,
  bands,
  margin,
  xAccessor,
  bisectDate,
}: UseChartInteractionParams) {
  const [hoverStore] = useState(createChartHoverStore);
  const { scheduleTooltip, clearTooltip, resetTooltipDedupe } = useScheduledTooltip(hoverStore);
  const dragStartXRef = useRef<number | null>(null);
  const lastHoveredXRef = useRef<number | null>(null);

  const nearestIndex = useCallback(
    (pixelX: number): number | null => {
      const x0 = xScale.invert(pixelX);
      const index = bisectDate(data, x0, 1);
      const d0 = data[index - 1];
      const d1 = data[index];
      if (!d0) return null;
      if (d1 && x0.getTime() - xAccessor(d0).getTime() > xAccessor(d1).getTime() - x0.getTime()) return index;
      return index - 1;
    },
    [xScale, data, xAccessor, bisectDate],
  );

  const hoverAt = useCallback(
    (pixelX: number) => {
      lastHoveredXRef.current = pixelX;
      const index = nearestIndex(pixelX);
      const point = index == null ? undefined : data[index];
      if (index == null || !point) return;
      scheduleTooltip({
        point,
        index,
        x: xScale(xAccessor(point)) ?? 0,
        yPositions: resolveYPositions(point, lines, bands, yScales, yScale),
      });
    },
    [nearestIndex, data, xScale, xAccessor, lines, bands, yScales, yScale, scheduleTooltip],
  );

  const select = (a: number, b: number) =>
    hoverStore.setSelection({ startX: Math.min(a, b), endX: Math.max(a, b), active: true });

  const onTouch = (event: React.TouchEvent<SVGGElement>, start: boolean) => {
    if (event.touches.length === 1) {
      event.preventDefault();
      const x = chartX(event, margin);
      if (x !== null) hoverAt(x);
    } else if (event.touches.length === 2) {
      event.preventDefault();
      if (start) {
        resetTooltipDedupe();
        clearTooltip();
      }
      const x0 = chartX(event, margin, 0);
      const x1 = chartX(event, margin, 1);
      if (x0 !== null && x1 !== null) select(x0, x1);
    }
  };

  // Re-anchor the tooltip when the scales or data change under a resting pointer.
  useEffect(() => {
    const x = lastHoveredXRef.current;
    if (x === null) return;
    if (nearestIndex(x) == null) clearTooltip();
    else hoverAt(x);
  }, [hoverAt, nearestIndex, clearTooltip]);

  const interactionHandlers = {
    onMouseMove: (event: React.MouseEvent<SVGGElement>) => {
      const x = chartX(event, margin);
      if (x === null) return;
      if (dragStartXRef.current !== null) select(dragStartXRef.current, x);
      else hoverAt(x);
    },
    onMouseLeave: () => {
      lastHoveredXRef.current = null;
      dragStartXRef.current = null;
      clearTooltip();
      hoverStore.setSelection(null);
    },
    onMouseDown: (event: React.MouseEvent<SVGGElement>) => {
      const x = chartX(event, margin);
      if (x === null) return;
      dragStartXRef.current = x;
      clearTooltip();
      hoverStore.setSelection(null);
    },
    onMouseUp: () => {
      dragStartXRef.current = null;
      hoverStore.setSelection(null);
    },
    onTouchStart: (event: React.TouchEvent<SVGGElement>) => onTouch(event, true),
    onTouchMove: (event: React.TouchEvent<SVGGElement>) => onTouch(event, false),
    onTouchEnd: () => {
      clearTooltip();
      hoverStore.setSelection(null);
    },
  };

  return { hoverStore, interactionHandlers };
}
