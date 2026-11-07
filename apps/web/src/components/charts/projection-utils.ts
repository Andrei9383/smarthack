export type ProjectionMode = "auto" | "target" | "manual";
export type ProjectionAutoMethod = "linearRegression" | "lastSegment";
/** How the projection segment is drawn between anchor and horizon. */
export type ProjectionCurveKind = "linear" | "bezier";
/** @deprecated Stepped density removed — projections always anchor → horizon. */
export type ProjectionPathDensity = "stepped" | "endpoints";

export interface ProjectionPoint {
  date: Date;
  value: number;
}

export interface BuildProjectionPathOptions {
  sourceData: Record<string, unknown>[];
  seriesKey: string;
  xDataKey?: string;
  mode: ProjectionMode;
  autoMethod?: ProjectionAutoMethod;
  /** Auto mode: stepped points per interval, or anchor + end only. Default: stepped */
  pathDensity?: ProjectionPathDensity;
  /** Index in sourceData where projection anchors (default: last point). */
  startIndex?: number;
  /** How many future points to generate (matches source cadence). */
  horizonPoints?: number;
  /** Target Y at the final projected date (target mode). */
  endValue?: number;
  /** Full manual path — anchor + future points (manual mode). */
  points?: ProjectionPoint[];
}

/** Cubic bezier with horizontal tangents at start and end (price-target S-curve). */
export function buildHorizontalTangentBezierPath(
  x0: number,
  y0: number,
  x1: number,
  y1: number,
  /** How far control points sit along the x span (0–0.5). Default: 0.45 */
  tension = 0.45
): string {
  const dx = x1 - x0;
  if (Math.abs(dx) < 1e-6) {
    return `M ${x0},${y0} L ${x1},${y1}`;
  }
  const t = Math.min(0.5, Math.max(0.05, tension));
  const c1x = x0 + dx * t;
  const c2x = x1 - dx * t;
  return `M ${x0},${y0} C ${c1x},${y0} ${c2x},${y1} ${x1},${y1}`;
}

/** Collect numeric Y extents from projection point arrays. */
export function projectionValueExtents(
  paths: ProjectionPoint[][]
): { minValue: number; maxValue: number } | null {
  let minValue = Number.POSITIVE_INFINITY;
  let maxValue = Number.NEGATIVE_INFINITY;

  for (const path of paths) {
    for (const point of path) {
      if (point.value < minValue) {
        minValue = point.value;
      }
      if (point.value > maxValue) {
        maxValue = point.value;
      }
    }
  }

  if (minValue === Number.POSITIVE_INFINITY) {
    return null;
  }

  return { minValue, maxValue };
}

/** Collect date extents from projection point arrays. */
export function projectionDateExtents(
  paths: ProjectionPoint[][]
): { minTime: number; maxTime: number } | null {
  let minTime = Number.POSITIVE_INFINITY;
  let maxTime = Number.NEGATIVE_INFINITY;

  for (const path of paths) {
    for (const point of path) {
      const time = point.date.getTime();
      if (time < minTime) {
        minTime = time;
      }
      if (time > maxTime) {
        maxTime = time;
      }
    }
  }

  if (minTime === Number.POSITIVE_INFINITY) {
    return null;
  }

  return { minTime, maxTime };
}
