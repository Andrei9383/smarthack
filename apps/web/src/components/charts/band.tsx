"use client";

import { type CurveFactory, curveLinear, curveNatural } from "d3-shape";
import { useMemo } from "react";

import { bandAreaPath, bandEdgePath } from "./band-geometry";
import { chartCssVars, DATA_TWEEN_MS, useChartStable, useYScale } from "./chart-context";
import { useAnimatedBand } from "./use-animated-band";

interface BandProps {
  /** Data key for the band's upper edge. */
  upperKey: string;
  /** Data key for the band's lower edge. */
  lowerKey: string;
  /** Y-scale group id. Default: `"left"`. */
  yAxisId?: string;
  /** Tooltip row label (the chart reads it from props). Default: `"lowerKey – upperKey"`. */
  label?: string;
  /** Default: curveNatural (same as Line). */
  curve?: CurveFactory;
  /** Band colour. Default: var(--chart-line-primary) */
  fill?: string;
  /** Default: 0.15 */
  fillOpacity?: number;
  /** Opacity of the 1px edge strokes. Default: 1 */
  strokeOpacity?: number;
}

/** A shaded range between two series (e.g. P10–P90), with thin edge strokes. */
export function Band({
  upperKey,
  lowerKey,
  yAxisId,
  curve = curveNatural,
  fill = chartCssVars.linePrimary,
  fillOpacity = 0.15,
  strokeOpacity = 1,
}: BandProps) {
  const { renderData, xScale, innerWidth, xAccessor } = useChartStable();
  const yScale = useYScale(yAxisId);

  // Same tween inputs as `<Line>`, so band edges morph in lockstep with lines on data changes.
  const { segments, polyline } = useAnimatedBand({
    curve,
    durationMs: DATA_TWEEN_MS,
    innerWidth,
    lowerKey,
    renderData,
    upperKey,
    xAccessor,
    xScale,
    yScale,
  });

  const paths = useMemo(() => {
    const pathCurve = polyline ? curveLinear : curve;
    return {
      area: bandAreaPath(segments, pathCurve),
      upper: bandEdgePath(segments, "upper", pathCurve),
      lower: bandEdgePath(segments, "lower", pathCurve),
    };
  }, [segments, polyline, curve]);

  if (paths.area === "") return null;

  const edge = { fill: "none", stroke: fill, strokeLinecap: "round", strokeOpacity, strokeWidth: 1 } as const;
  return (
    <g aria-hidden="true" className="chart-band" pointerEvents="none">
      <path d={paths.area} fill={fill} fillOpacity={fillOpacity} />
      <path d={paths.lower} {...edge} />
      <path d={paths.upper} {...edge} />
    </g>
  );
}

Band.displayName = "Band";
