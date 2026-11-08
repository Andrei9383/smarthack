"use client";

import { cn } from "@wattsentry/ui/lib/utils";
import { bisector, extent } from "d3-array";
import { scaleLinear, scaleTime } from "d3-scale";
import { animate, domAnimation, LazyMotion, useReducedMotion } from "motion/react";
import {
  Children,
  cloneElement,
  isValidElement,
  memo,
  type ReactElement,
  type ReactNode,
  useEffect,
  useMemo,
  useRef,
  useState,
} from "react";

import {
  type BandConfig,
  ChartHoverStoreProvider,
  type ChartStableContextValue,
  ChartStableContextProvider,
  DATA_TWEEN_EASE,
  DATA_TWEEN_MS,
  DEFAULT_Y_AXIS_ID,
  type LineConfig,
  type Margin,
  normalizeYAxisId,
  type YDomain,
  type YScale,
} from "./chart-context";
import { ParentSize } from "./chart-parent-size";
import { decimateTimeSeries } from "./decimate-time-series";
import { Line, type LineProps } from "./line";
import { useChartInteraction } from "./use-chart-interaction";

export { Line } from "./line";

interface LineChartProps {
  /** Rows with a `date` field and numeric series values. */
  data: Record<string, unknown>[];
  margin?: Partial<Margin>;
  /** Aspect ratio as "width / height". Default: "2 / 1". */
  aspectRatio?: string;
  className?: string;
  /**
   * Fixed y-domain max for single-axis charts: the domain becomes `[0, yScaleDomainMax × 1.1]`
   * instead of following the data max, to leave headroom above the series.
   */
  yScaleDomainMax?: number;
  /** Line, Band, Grid, ChartTooltip and custom overlays reading the chart context. */
  children: ReactNode;
}

const DEFAULT_MARGIN: Margin = { top: 40, right: 40, bottom: 40, left: 40 };
/** Grid and axes are drawn under the series wherever they appear among the children. */
const UNDERLAY_NAMES = new Set(["Grid", "XAxis", "YAxis"]);
/** Below a 2% move of either end the y-domain snaps instead of tweening. */
const Y_TWEEN_THRESHOLD = 0.02;
const shortDateFmt = new Intl.DateTimeFormat("en-US", { month: "short", day: "numeric" });

function componentName(child: ReactElement): string {
  const type = child.type as { displayName?: string; name?: string };
  return typeof child.type === "function" ? type.displayName || type.name || "" : "";
}

/** Walks the children (through fragments and wrappers) collecting `<Line>` and `<Band>` props. */
function extractSeries(children: ReactNode): { lines: LineConfig[]; bands: BandConfig[] } {
  const lines: LineConfig[] = [];
  const bands: BandConfig[] = [];
  const visit = (node: ReactNode) => {
    Children.forEach(node, (child) => {
      if (!isValidElement(child)) return;
      const name = componentName(child);
      if (child.type === Line || name === "Line") {
        const props = child.props as LineProps;
        if (props.dataKey) {
          lines.push({
            dataKey: props.dataKey,
            stroke: props.stroke || "var(--chart-line-primary)",
            strokeWidth: props.strokeWidth || 2.5,
            yAxisId: props.yAxisId,
          });
          return;
        }
      }
      if (name === "Band") {
        const { upperKey, lowerKey, yAxisId, fill, label } = child.props as Record<string, unknown>;
        if (typeof upperKey === "string" && upperKey && typeof lowerKey === "string" && lowerKey) {
          bands.push({
            yAxisId: normalizeYAxisId(yAxisId as string | undefined),
            upperKey,
            lowerKey,
            color: typeof fill === "string" && fill ? fill : "var(--chart-line-primary)",
            label: typeof label === "string" && label ? label : `${lowerKey} – ${upperKey}`,
          });
        }
        return;
      }
      const nested = (child.props as { children?: ReactNode }).children;
      if (nested) visit(nested);
    });
  };
  visit(children);
  return { lines, bands };
}

/** Grid / axes first, then everything else in order; unkeyed children get index keys. */
function layerChildren(children: ReactNode): ReactElement[] {
  const under: ReactElement[] = [];
  const over: ReactElement[] = [];
  Children.forEach(children, (child, index) => {
    if (!isValidElement(child)) return;
    const keyed = child.key == null ? cloneElement(child, { key: `chart-child-${index}` }) : child;
    (UNDERLAY_NAMES.has(componentName(child)) ? under : over).push(keyed);
  });
  return [...under, ...over];
}

/**
 * Line / area y-domain, nice'd: zero-anchored while every value is >= 0, padded min..max once any
 * value is negative, `[0, max × 1.1]` with a fixed `domainMax`.
 */
function resolveYDomain(data: Record<string, unknown>[], keys: string[], domainMax: number | undefined): YDomain {
  let raw: YDomain;
  if (domainMax != null && domainMax > 0) {
    raw = [0, domainMax * 1.1];
  } else {
    let min = Number.POSITIVE_INFINITY;
    let max = Number.NEGATIVE_INFINITY;
    for (const d of data) {
      for (const key of keys) {
        const value = d[key];
        if (typeof value === "number") {
          min = Math.min(min, value);
          max = Math.max(max, value);
        }
      }
    }
    if (min === Number.POSITIVE_INFINITY) [min, max] = [0, 100];
    const padding = (max - min) * 0.05 || 1;
    raw = min >= 0 ? [0, max <= 0 ? 100 : max * 1.1] : [min - padding, max + padding];
  }
  const nice = scaleLinear().domain(raw).nice().domain();
  return [nice[0] ?? raw[0], nice[1] ?? raw[1]];
}

function domainsEqual(a: Record<string, YDomain>, b: Record<string, YDomain>): boolean {
  const keys = Object.keys(a);
  return (
    keys.length === Object.keys(b).length &&
    keys.every((axisId) => a[axisId]?.[0] === b[axisId]?.[0] && a[axisId]?.[1] === b[axisId]?.[1])
  );
}

function shouldTween(from: YDomain, to: YDomain): boolean {
  const span = Math.max(Math.abs(to[1] - to[0]), Math.abs(from[1] - from[0]), 1);
  return (
    Math.abs(to[0] - from[0]) / span >= Y_TWEEN_THRESHOLD || Math.abs(to[1] - from[1]) / span >= Y_TWEEN_THRESHOLD
  );
}

/** Y-domains that tween to `target` when the data changes (snap under reduced motion). */
function useAnimatedYDomains(target: Record<string, YDomain>): Record<string, YDomain> {
  const reducedMotion = useReducedMotion();
  const [animated, setAnimated] = useState(target);
  const animatedRef = useRef(animated);
  const targetRef = useRef(target);
  const signature = JSON.stringify(target);
  const prevSignatureRef = useRef(signature);

  useEffect(() => {
    targetRef.current = target;
  });
  useEffect(() => {
    animatedRef.current = animated;
  }, [animated]);

  useEffect(() => {
    if (prevSignatureRef.current === signature) return;
    prevSignatureRef.current = signature;
    const destination = targetRef.current;
    const snap = () => {
      if (domainsEqual(animatedRef.current, destination)) return;
      animatedRef.current = destination;
      setAnimated(destination);
    };
    const axisIds = Object.keys(destination);
    const from: Record<string, YDomain> = {};
    for (const axisId of axisIds) from[axisId] = animatedRef.current[axisId] ?? destination[axisId] ?? [0, 100];
    if (domainsEqual(animatedRef.current, destination)) return;
    if (reducedMotion || !axisIds.some((axisId) => shouldTween(from[axisId]!, destination[axisId]!))) {
      snap();
      return;
    }
    const control = animate(0, 1, {
      duration: DATA_TWEEN_MS / 1000,
      ease: [...DATA_TWEEN_EASE],
      onUpdate: (progress) => {
        const next: Record<string, YDomain> = {};
        for (const axisId of axisIds) {
          const a = from[axisId]!;
          const b = destination[axisId]!;
          next[axisId] = shouldTween(a, b) ? [a[0] + (b[0] - a[0]) * progress, a[1] + (b[1] - a[1]) * progress] : b;
        }
        animatedRef.current = next;
        setAnimated(next);
      },
      onComplete: snap,
    });
    return () => control.stop();
  }, [signature, reducedMotion]);

  return reducedMotion ? target : animated;
}

interface ChartCoreProps {
  width: number;
  height: number;
  data: Record<string, unknown>[];
  margin: Margin;
  containerRef: React.RefObject<HTMLDivElement | null>;
  yScaleDomainMax?: number;
  children: ReactNode;
}

const xAccessor = (d: Record<string, unknown>): Date => {
  const value = d.date;
  return value instanceof Date ? value : new Date(value as string | number);
};
const bisectDate = bisector<Record<string, unknown>, Date>(xAccessor).left;

const ChartCore = memo(function ChartCore({
  width,
  height,
  data,
  margin,
  containerRef,
  yScaleDomainMax,
  children,
}: ChartCoreProps) {
  const innerWidth = width - margin.left - margin.right;
  const innerHeight = height - margin.top - margin.bottom;

  // Keyed by a string signature so new `children` identities on hover re-renders skip decimation.
  const seriesSignature = useMemo(() => JSON.stringify(extractSeries(children)), [children]);
  const { lines, bands } = useMemo(
    () => JSON.parse(seriesSignature) as { lines: LineConfig[]; bands: BandConfig[] },
    [seriesSignature],
  );

  const xScale = useMemo(() => {
    const [min = 0, max = min] = extent(data, (d) => xAccessor(d).getTime());
    return scaleTime().domain([min, max]).range([0, innerWidth]);
  }, [innerWidth, data]);

  const renderData = useMemo(() => {
    const keys = lines.map((line) => line.dataKey);
    for (const band of bands) {
      for (const key of [band.upperKey, band.lowerKey]) if (!keys.includes(key)) keys.push(key);
    }
    // ~1.5 points per pixel: crisp curves without over-drawing.
    return decimateTimeSeries(data, Math.max(64, Math.ceil(innerWidth * 1.5)), keys);
  }, [data, innerWidth, lines, bands]);

  const yDomainTargetByAxis = useMemo(() => {
    const keysByAxis = new Map<string, string[]>();
    const add = (axisId: string | number | undefined, keys: string[]) => {
      const id = normalizeYAxisId(axisId);
      const list = keysByAxis.get(id) ?? [];
      for (const key of keys) if (!list.includes(key)) list.push(key);
      keysByAxis.set(id, list);
    };
    for (const line of lines) add(line.yAxisId, [line.dataKey]);
    for (const band of bands) add(band.yAxisId, [band.upperKey, band.lowerKey]);
    const lineAxes = new Set(lines.map((line) => normalizeYAxisId(line.yAxisId)));
    const singleAxis = lineAxes.size === 1 && lineAxes.has(DEFAULT_Y_AXIS_ID);
    const domains: Record<string, YDomain> = {};
    for (const [axisId, keys] of keysByAxis) {
      domains[axisId] = resolveYDomain(data, keys, singleAxis ? yScaleDomainMax : undefined);
    }
    domains[DEFAULT_Y_AXIS_ID] ??= [0, 100];
    return domains;
  }, [data, lines, bands, yScaleDomainMax]);

  const yDomains = useAnimatedYDomains(yDomainTargetByAxis);

  const yScales = useMemo(() => {
    const scales: Record<string, YScale> = {};
    const axisIds = new Set([...lines.map((line) => normalizeYAxisId(line.yAxisId)), DEFAULT_Y_AXIS_ID]);
    for (const axisId of axisIds) {
      const domain = yDomains[axisId] ?? yDomains[DEFAULT_Y_AXIS_ID] ?? [0, 100];
      scales[axisId] = scaleLinear().domain(domain).range([innerHeight, 0]);
    }
    return scales;
  }, [yDomains, innerHeight, lines]);
  const yScale = yScales[DEFAULT_Y_AXIS_ID]!;

  const dateLabels = useMemo(() => data.map((d) => shortDateFmt.format(xAccessor(d))), [data]);

  const { hoverStore, interactionHandlers } = useChartInteraction({
    bisectDate,
    bands,
    data,
    lines,
    margin,
    xAccessor,
    xScale,
    yScale,
    yScales,
  });

  const layers = useMemo(() => layerChildren(children), [children]);

  const contextValue = useMemo<ChartStableContextValue>(
    () => ({
      data,
      renderData,
      xScale,
      yScale,
      yScales,
      width,
      height,
      innerWidth,
      innerHeight,
      margin,
      containerRef,
      lines,
      bands,
      yDomainTargetByAxis,
      xAccessor,
      dateLabels,
    }),
    [
      data,
      renderData,
      xScale,
      yScale,
      yScales,
      width,
      height,
      innerWidth,
      innerHeight,
      margin,
      containerRef,
      lines,
      bands,
      yDomainTargetByAxis,
      dateLabels,
    ],
  );

  return (
    <ChartStableContextProvider value={contextValue}>
      <ChartHoverStoreProvider store={hoverStore}>
        <svg aria-hidden="true" className="overflow-visible" height={height} width={width}>
          <rect fill="transparent" height={height} width={width} x={0} y={0} />
          <g
            {...interactionHandlers}
            style={{ cursor: "crosshair", touchAction: "none" }}
            transform={`translate(${margin.left},${margin.top})`}
          >
            <rect fill="transparent" height={innerHeight} width={innerWidth} x={0} y={0} />
            {layers}
          </g>
        </svg>
      </ChartHoverStoreProvider>
    </ChartStableContextProvider>
  );
});

export function LineChart({
  data,
  margin: marginProp,
  aspectRatio = "2 / 1",
  className = "",
  yScaleDomainMax,
  children,
}: LineChartProps) {
  const containerRef = useRef<HTMLDivElement>(null);
  const { top, right, bottom, left } = { ...DEFAULT_MARGIN, ...marginProp };
  // Same object while the numbers are unchanged, so an inline `margin={{…}}` keeps ChartCore memoised.
  const margin = useMemo(() => ({ top, right, bottom, left }), [top, right, bottom, left]);

  return (
    <LazyMotion features={domAnimation}>
      <div
        className={cn("relative w-full", className)}
        ref={containerRef}
        style={{ aspectRatio, touchAction: "none" }}
      >
        <ParentSize debounceTime={10}>
          {({ width, height }) =>
            width >= 10 && height >= 10 ? (
              <ChartCore
                containerRef={containerRef}
                data={data}
                height={height}
                margin={margin}
                width={width}
                yScaleDomainMax={yScaleDomainMax}
              >
                {children}
              </ChartCore>
            ) : null
          }
        </ParentSize>
      </div>
    </LazyMotion>
  );
}
