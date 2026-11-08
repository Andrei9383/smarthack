"use client";

import type { CurveFactory } from "d3-shape";
import { curveLinear } from "d3-shape";
import { animate, useReducedMotion } from "motion/react";
import { useEffect, useLayoutEffect, useMemo, useRef, useState } from "react";

import { DATA_TWEEN_EASE } from "./chart-context";
import {
  computeSeriesPathPoints,
  flattenSeriesPath,
  interpolateSeriesPathPoints,
  morphSeriesPolylines,
  type SeriesPathPoint,
  seriesDataSignature,
  seriesPathFromPoints,
} from "./series-path-utils";

interface UseAnimatedSeriesPathOptions {
  renderData: Record<string, unknown>[];
  xAccessor: (datum: Record<string, unknown>) => Date;
  xScale: (value: Date) => number | undefined;
  yScale: (value: number) => number | undefined;
  /** Y-scale at the end of the y-domain tween: data morphs animate toward it. */
  settledYScale: (value: number) => number | undefined;
  dataKey: string;
  curve: CurveFactory;
  durationMs: number;
  innerWidth: number;
}

/** Points on screen. `polyline` frames are dense samples of the curve, drawn with `curveLinear`. */
interface DisplayedSeriesPath {
  points: SeriesPathPoint[];
  polyline: boolean;
}

/**
 * Path `d` of a series that tweens when its data or the plot width / x-domain changes: new data
 * morphs vertically on a shared x grid, a moved x-domain slides points by key.
 */
export function useAnimatedSeriesPath({
  renderData,
  xAccessor,
  xScale,
  yScale,
  settledYScale,
  dataKey,
  curve,
  durationMs,
  innerWidth,
}: UseAnimatedSeriesPathOptions): string {
  const reducedMotion = useReducedMotion();
  const [animatedPath, setAnimatedPath] = useState<DisplayedSeriesPath | null>(null);
  const displayedRef = useRef<DisplayedSeriesPath | null>(null);
  const animatingRef = useRef(false);

  const xScaleDomain = useMemo(() => (xScale as unknown as { domain: () => [Date, Date] }).domain(), [xScale]);

  const targetPoints = useMemo(
    () => computeSeriesPathPoints(renderData, xAccessor, xScale, yScale, dataKey),
    [renderData, xAccessor, xScale, yScale, dataKey],
  );

  const dataSignature = useMemo(
    () => seriesDataSignature(renderData, xAccessor, dataKey),
    [renderData, xAccessor, dataKey],
  );
  const transitionSignature = `${innerWidth}|${xScaleDomain[0]?.getTime?.() ?? 0}|${xScaleDomain[1]?.getTime?.() ?? 0}|${dataSignature}`;
  const settledYDomainKey = (settledYScale as { domain?: () => number[] }).domain?.().join(",") ?? "";

  const prevTransitionSignatureRef = useRef(transitionSignature);
  const prevDataSignatureRef = useRef(dataSignature);
  const prevSettledYDomainKeyRef = useRef(settledYDomainKey);

  // Live inputs for the running tween. The y-domain tween rebuilds `yScale` every frame, so these
  // must not be effect deps, or the cleanup would stop the path tween after its first frame.
  const latestRef = useRef({ renderData, xAccessor, xScale, yScale, settledYScale, dataKey, curve });
  useLayoutEffect(() => {
    latestRef.current = { renderData, xAccessor, xScale, yScale, settledYScale, dataKey, curve };
  });
  const computeLatestTarget = (settled = false) => {
    const latest = latestRef.current;
    return computeSeriesPathPoints(
      latest.renderData,
      latest.xAccessor,
      latest.xScale,
      settled ? latest.settledYScale : latest.yScale,
      latest.dataKey,
    );
  };

  const shouldAnimate = !reducedMotion && durationMs > 0 && renderData.length > 0;

  // Drop a stale tween frame as soon as animation is no longer allowed.
  if (!shouldAnimate && animatedPath !== null) setAnimatedPath(null);

  // Layout effect so frame 0 (the previous shape) is committed before paint.
  useLayoutEffect(() => {
    const dataChanged = prevDataSignatureRef.current !== dataSignature;
    prevDataSignatureRef.current = dataSignature;
    const yDomainChanged = prevSettledYDomainKeyRef.current !== settledYDomainKey;
    prevSettledYDomainKeyRef.current = settledYDomainKey;

    if (!shouldAnimate) {
      animatingRef.current = false;
      displayedRef.current = { points: targetPoints, polyline: false };
      prevTransitionSignatureRef.current = transitionSignature;
      return;
    }
    if (prevTransitionSignatureRef.current === transitionSignature) return;
    prevTransitionSignatureRef.current = transitionSignature;

    const from = displayedRef.current ?? { points: targetPoints, polyline: false };
    if (from.points.length === 0) {
      displayedRef.current = { points: targetPoints, polyline: false };
      return;
    }

    animatingRef.current = true;
    const morph = dataChanged || from.polyline;
    const fromPolyline = from.polyline ? from.points : flattenSeriesPath(from.points, curve);
    // When new data also moves the y-domain, hold the line while the grid tweens to the new scale,
    // then morph to its settled pixels: morphing against the mid-tween scale sends the line far
    // outside the plot when the magnitude changes.
    const holdForYDomain = morph && dataChanged && yDomainChanged;

    // The morph target depends only on these inputs, so its flattened polyline is cached.
    let targetCache: { inputs: readonly unknown[]; polyline: SeriesPathPoint[] } | null = null;
    const settledTargetPolyline = () => {
      const latest = latestRef.current;
      const inputs = [latest.renderData, latest.xAccessor, latest.xScale, latest.settledYScale, latest.dataKey, latest.curve];
      const cached = targetCache;
      if (cached && inputs.every((input, i) => input === cached.inputs[i])) return cached.polyline;
      const polyline = flattenSeriesPath(computeLatestTarget(true), latest.curve);
      targetCache = { inputs, polyline };
      return polyline;
    };

    const renderFrame = (progress: number) => {
      const next: DisplayedSeriesPath = morph
        ? { points: morphSeriesPolylines(fromPolyline, settledTargetPolyline(), progress), polyline: true }
        : { points: interpolateSeriesPathPoints(from.points, computeLatestTarget(false), progress), polyline: false };
      displayedRef.current = next;
      setAnimatedPath(next);
    };
    renderFrame(0);

    const control = animate(0, 1, {
      delay: holdForYDomain ? durationMs / 1000 : 0,
      duration: durationMs / 1000,
      ease: [...DATA_TWEEN_EASE],
      onUpdate: renderFrame,
      onComplete: () => {
        animatingRef.current = false;
        displayedRef.current = { points: computeLatestTarget(), polyline: false };
        setAnimatedPath(null);
      },
    });
    return () => {
      control.stop();
      animatingRef.current = false;
    };
  }, [transitionSignature, dataSignature, shouldAnimate, durationMs, reducedMotion, renderData.length]);

  // After the tween effect: on a data change the tween needs the previously displayed points as
  // its `from`, not the new target.
  useEffect(() => {
    if (!animatingRef.current) displayedRef.current = { points: targetPoints, polyline: false };
  }, [targetPoints]);

  return useMemo(
    () =>
      animatedPath
        ? seriesPathFromPoints(animatedPath.points, animatedPath.polyline ? curveLinear : curve)
        : seriesPathFromPoints(targetPoints, curve),
    [animatedPath, targetPoints, curve],
  );
}
