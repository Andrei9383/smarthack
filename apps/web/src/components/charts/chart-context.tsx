"use client";

import { type ScaleLinear, type ScaleTime, scaleLinear } from "d3-scale";
import {
  createContext,
  type Dispatch,
  type ReactNode,
  type RefObject,
  type SetStateAction,
  useContext,
  useEffect,
  useLayoutEffect,
  useMemo,
  useState,
  useSyncExternalStore,
} from "react";

export type YScale = ScaleLinear<number, number>;
export type YDomain = [number, number];

export const chartCssVars = {
  background: "var(--chart-background)",
  linePrimary: "var(--chart-line-primary)",
  crosshair: "var(--chart-crosshair)",
  grid: "var(--chart-grid)",
  tooltipBackground: "var(--chart-tooltip-background)",
};

/** Duration of the line / band / y-domain tweens when the data changes (ms). */
export const DATA_TWEEN_MS = 500;
export const DATA_TWEEN_EASE = [0.85, 0, 0.15, 1] as const;

/**
 * Above this many points the crosshair, tooltip dots and date pill snap between points instead of
 * springing (springs would lag behind a dense series). The hover highlight band follows the same rule.
 */
export const DISCRETE_INTERACTION_LABEL_THRESHOLD = 60;

/** Default axis id when `yAxisId` is omitted. */
export const DEFAULT_Y_AXIS_ID = "left";

export function normalizeYAxisId(id?: string | number): string {
  return id == null || id === "" ? DEFAULT_Y_AXIS_ID : String(id);
}

export interface Margin {
  top: number;
  right: number;
  bottom: number;
  left: number;
}

export interface TooltipData {
  point: Record<string, unknown>;
  index: number;
  /** X of the hovered point, relative to the plot. */
  x: number;
  /** Y per series / band edge key, relative to the plot. */
  yPositions: Record<string, number>;
}

export interface ChartSelection {
  startX: number;
  endX: number;
  active: boolean;
}

export interface LineConfig {
  dataKey: string;
  stroke: string;
  strokeWidth: number;
  yAxisId?: string | number;
}

/** Y-domain and tooltip inputs collected from a `<Band>` child. */
export interface BandConfig {
  yAxisId: string;
  upperKey: string;
  lowerKey: string;
  /** Tooltip swatch / edge-dot colour (the band's `fill`). */
  color: string;
  label: string;
}

/** Everything that doesn't change on hover. */
export interface ChartStableContextValue {
  data: Record<string, unknown>[];
  /** Decimated subset for SVG paths; equals `data` when no decimation is needed. */
  renderData: Record<string, unknown>[];
  xScale: ScaleTime<number, number>;
  /** Primary (left) y-scale. */
  yScale: YScale;
  yScales: Record<string, YScale>;
  width: number;
  height: number;
  innerWidth: number;
  innerHeight: number;
  margin: Margin;
  containerRef: RefObject<HTMLDivElement | null>;
  lines: LineConfig[];
  bands: BandConfig[];
  /** Nice'd y-domains per axis for the current data (where the y-domain tween ends). */
  yDomainTargetByAxis: Record<string, YDomain>;
  xAccessor: (d: Record<string, unknown>) => Date;
  /** One label per data row, for the date pill. */
  dateLabels: string[];
}

/** Hover state: every field changes on mouse movement. */
export interface ChartHoverContextValue {
  tooltipData: TooltipData | null;
  selection: ChartSelection | null;
}

const ChartStableContext = createContext<ChartStableContextValue | null>(null);
export const ChartStableContextProvider = ChartStableContext.Provider;

/**
 * Subscribable hover state. Hover lives here instead of React state, so a pointer move re-renders
 * only the components that read hover, not the chart shell and its layers.
 */
export interface ChartHoverStore {
  getSnapshot: () => ChartHoverContextValue;
  subscribe: (listener: () => void) => () => void;
  setTooltipData: Dispatch<SetStateAction<TooltipData | null>>;
  setSelection: (selection: ChartSelection | null) => void;
}

const ChartHoverStoreContext = createContext<ChartHoverStore | null>(null);
const ChartHoverContext = createContext<ChartHoverContextValue | null>(null);

export function createChartHoverStore(): ChartHoverStore {
  const listeners = new Set<() => void>();
  let snapshot: ChartHoverContextValue = { tooltipData: null, selection: null };
  const update = (next: Partial<ChartHoverContextValue>) => {
    snapshot = { ...snapshot, ...next };
    for (const listener of listeners) listener();
  };
  return {
    getSnapshot: () => snapshot,
    subscribe: (listener) => {
      listeners.add(listener);
      return () => listeners.delete(listener);
    },
    setTooltipData: (action) => {
      const next = typeof action === "function" ? action(snapshot.tooltipData) : action;
      if (next !== snapshot.tooltipData) update({ tooltipData: next });
    },
    setSelection: (selection) => {
      if (selection !== snapshot.selection) update({ selection });
    },
  };
}

/** Publishes the hover store. Only this provider re-renders on a hover change. */
export function ChartHoverStoreProvider({ children, store }: { children: ReactNode; store: ChartHoverStore }) {
  const hover = useSyncExternalStore(store.subscribe, store.getSnapshot, store.getSnapshot);
  return (
    <ChartHoverStoreContext.Provider value={store}>
      <ChartHoverContext.Provider value={hover}>{children}</ChartHoverContext.Provider>
    </ChartHoverStoreContext.Provider>
  );
}

function required<T>(value: T | null, hook: string): T {
  if (!value) throw new Error(`${hook} must be used within a <LineChart>.`);
  return value;
}

export function useChartStable(): ChartStableContextValue {
  return required(useContext(ChartStableContext), "useChartStable");
}

/** The hover slice; re-renders on every mouse move. */
export function useChartHover(): ChartHoverContextValue {
  return required(useContext(ChartHoverContext), "useChartHover");
}

export function useChartHoverStore(): ChartHoverStore {
  return required(useContext(ChartHoverStoreContext), "useChartHoverStore");
}

/** One value derived from hover; re-renders only when it changes (`Object.is`). */
export function useChartHoverSelector<T>(selector: (hover: ChartHoverContextValue) => T): T {
  const store = useChartHoverStore();
  const getSelected = () => selector(store.getSnapshot());
  return useSyncExternalStore(store.subscribe, getSelected, getSelected);
}

export function selectIsChartHovering(hover: ChartHoverContextValue): boolean {
  return hover.tooltipData !== null || hover.selection?.active === true;
}

const useIsomorphicLayoutEffect = typeof window === "undefined" ? useEffect : useLayoutEffect;

/**
 * The chart container element, as a `createPortal` target. Seeded from the ref (already attached
 * when the chart renders after measuring), re-captured after mount; `null` during SSR.
 */
export function useChartPortalContainer<T extends HTMLElement>(containerRef: RefObject<T | null> | undefined): T | null {
  const [container, setContainer] = useState<T | null>(() => containerRef?.current ?? null);
  useIsomorphicLayoutEffect(() => {
    setContainer(containerRef?.current ?? null);
  }, [containerRef]);
  useEffect(() => {
    const element = containerRef?.current ?? null;
    setContainer((previous) => (previous === element ? previous : element));
  }, [containerRef]);
  return container;
}

/** Y-scale for a series axis. */
export function useYScale(yAxisId?: string | number): YScale {
  const { yScales, yScale } = useChartStable();
  return yScales[normalizeYAxisId(yAxisId)] ?? yScale;
}

/**
 * Y-scale at the end of the y-domain tween, for series that morph toward their final pixels rather
 * than the mid-tween scale.
 */
export function useSettledYScale(yAxisId?: string | number): YScale {
  const { yDomainTargetByAxis, innerHeight } = useChartStable();
  const liveScale = useYScale(yAxisId);
  const domain = yDomainTargetByAxis[normalizeYAxisId(yAxisId)] ?? yDomainTargetByAxis[DEFAULT_Y_AXIS_ID];
  const min = domain?.[0];
  const max = domain?.[1];
  const settled = useMemo(
    () => (min == null || max == null ? null : scaleLinear().domain([min, max]).range([innerHeight, 0])),
    [min, max, innerHeight],
  );
  return settled ?? liveScale;
}
