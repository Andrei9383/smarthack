import { type CurveFactory, area as d3Area, line as d3Line } from "d3-shape";
import {
  flattenSeriesPath,
  interpolateSeriesPathPoints,
  type SeriesPathPoint,
  samplePolylineY,
} from "./series-path-utils";

/** One plotted row of a band, in pixels. */
export interface BandPoint {
  x: number;
  upper: number;
  lower: number;
  key: string;
}

/** Contiguous run of defined rows (gaps split the band into segments). */
export type BandSegment = BandPoint[];

function finiteNumber(value: unknown): value is number {
  return typeof value === "number" && Number.isFinite(value);
}

/**
 * Pixel band segments. Rows where either edge is null / undefined / NaN end
 * the current segment.
 */
export function computeBandSegments(
  data: Record<string, unknown>[],
  xAccessor: (datum: Record<string, unknown>) => Date,
  xScale: (value: Date) => number | undefined,
  yScale: (value: number) => number | undefined,
  upperKey: string,
  lowerKey: string
): BandSegment[] {
  const segments: BandSegment[] = [];
  let current: BandSegment = [];

  data.forEach((datum, index) => {
    const upper = datum[upperKey];
    const lower = datum[lowerKey];
    if (!(finiteNumber(upper) && finiteNumber(lower))) {
      if (current.length > 0) {
        segments.push(current);
        current = [];
      }
      return;
    }
    const xValue = xAccessor(datum);
    current.push({
      x: xScale(xValue) ?? 0,
      upper: yScale(upper) ?? 0,
      lower: yScale(lower) ?? 0,
      key: String(xValue.getTime?.() ?? index),
    });
  });

  if (current.length > 0) {
    segments.push(current);
  }
  return segments;
}

function edgePoints(
  segment: BandSegment,
  edge: "upper" | "lower"
): SeriesPathPoint[] {
  return segment.map((point) => ({
    x: point.x,
    y: point[edge],
    key: point.key,
  }));
}

/** Closed area path between the edges, one sub-path per segment. */
export function bandAreaPath(
  segments: BandSegment[],
  curve: CurveFactory
): string {
  const generator = d3Area<BandPoint>()
    .x((point) => point.x)
    .y0((point) => point.lower)
    .y1((point) => point.upper)
    .curve(curve);
  return segments
    .filter((segment) => segment.length > 1)
    .map((segment) => generator(segment) ?? "")
    .join("");
}

/** Open stroke path along one edge, one sub-path per segment. */
export function bandEdgePath(
  segments: BandSegment[],
  edge: "upper" | "lower",
  curve: CurveFactory
): string {
  const generator = d3Line<BandPoint>()
    .x((point) => point.x)
    .y((point) => point[edge])
    .curve(curve);
  return segments
    .filter((segment) => segment.length > 1)
    .map((segment) => generator(segment) ?? "")
    .join("");
}

/** Dense polylines of both edges, all segments concatenated and sorted by x. */
interface BandEdgePolylines {
  upper: SeriesPathPoint[];
  lower: SeriesPathPoint[];
}

export function flattenBandEdges(
  segments: BandSegment[],
  curve: CurveFactory,
  alreadyPolyline: boolean
): BandEdgePolylines {
  const flatten = (edge: "upper" | "lower") =>
    segments
      .flatMap((segment) => {
        const points = edgePoints(segment, edge);
        return alreadyPolyline ? points : flattenSeriesPath(points, curve);
      })
      .sort((a, b) => a.x - b.x);
  return { upper: flatten("upper"), lower: flatten("lower") };
}

/**
 * Vertical morph (same approach as `morphSeriesPolylines`): both edges are
 * sampled on a shared x grid per target segment and only y is interpolated,
 * so the band rises and falls into shape alongside morphing lines. Result is
 * a polyline — render with `curveLinear`.
 */
export function morphBandSegments(
  from: BandEdgePolylines,
  targets: BandSegment[],
  curve: CurveFactory,
  progress: number,
  sampleSpacingPx = 2
): BandSegment[] {
  if (from.upper.length === 0 || from.lower.length === 0) {
    return targets;
  }
  return targets.map((segment) => {
    const first = segment[0];
    const last = segment.at(-1);
    if (!(first && last) || segment.length < 2) {
      return segment;
    }
    const toUpper = flattenSeriesPath(edgePoints(segment, "upper"), curve);
    const toLower = flattenSeriesPath(edgePoints(segment, "lower"), curve);
    const span = last.x - first.x;
    const count = Math.max(2, Math.ceil(span / sampleSpacingPx) + 1);
    const result: BandSegment = new Array(count);
    for (let index = 0; index < count; index++) {
      const x = first.x + (span * index) / (count - 1);
      const fromUpper = samplePolylineY(from.upper, x);
      const fromLower = samplePolylineY(from.lower, x);
      result[index] = {
        x,
        upper: fromUpper + (samplePolylineY(toUpper, x) - fromUpper) * progress,
        lower: fromLower + (samplePolylineY(toLower, x) - fromLower) * progress,
        key: "",
      };
    }
    return result;
  });
}

/**
 * Keyed slide (same approach as `interpolateSeriesPathPoints`) for viewport
 * changes on unchanged data, e.g. brush pan / resize.
 */
export function interpolateBandSegments(
  from: BandSegment[],
  targets: BandSegment[],
  progress: number
): BandSegment[] {
  if (progress >= 1 || from.length === 0) {
    return targets;
  }
  const fromUpper = from.flatMap((segment) => edgePoints(segment, "upper"));
  const fromLower = from.flatMap((segment) => edgePoints(segment, "lower"));
  return targets.map((segment) => {
    const upper = interpolateSeriesPathPoints(
      fromUpper,
      edgePoints(segment, "upper"),
      progress
    );
    const lower = interpolateSeriesPathPoints(
      fromLower,
      edgePoints(segment, "lower"),
      progress
    );
    return upper.map((point, index) => ({
      x: point.x,
      upper: point.y,
      lower: lower[index]?.y ?? point.y,
      key: point.key,
    }));
  });
}

/** Signature of both edge values per row (ignores scales / width). */
export function bandDataSignature(
  renderData: Record<string, unknown>[],
  xAccessor: (datum: Record<string, unknown>) => Date,
  upperKey: string,
  lowerKey: string
): string {
  return renderData
    .map((datum) => {
      const upper = datum[upperKey];
      const lower = datum[lowerKey];
      return `${xAccessor(datum).getTime()}:${finiteNumber(upper) ? upper : ""}:${finiteNumber(lower) ? lower : ""}`;
    })
    .join(",");
}
