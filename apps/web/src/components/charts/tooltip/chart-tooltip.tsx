"use client";

import { m, useSpring } from "motion/react";
import { memo, useEffect, useId, useMemo } from "react";
import { createPortal } from "react-dom";

import {
  chartCssVars,
  DISCRETE_INTERACTION_LABEL_THRESHOLD,
  useChartHover,
  useChartPortalContainer,
  useChartStable,
} from "../chart-context";
import { DateTicker } from "./date-ticker";
import { TooltipBox } from "./tooltip-box";

/** Crosshair, tooltip dots and date pill. */
const TOOLTIP_SPRING = { stiffness: 300, damping: 30 };
/** Entrance spring for the dot core and halo radius. */
const HALO_SPRING = { type: "spring", stiffness: 420, damping: 28 } as const;
const DOT_SIZE = 5;
const HALO_SCALE = 2.6;

const weekdayDateFmt = new Intl.DateTimeFormat("en-US", { weekday: "short", month: "short", day: "numeric" });
const intFmt = new Intl.NumberFormat("en-US").format;

interface TooltipRow {
  color: string;
  label: string;
  value: string | number;
}

/** `"low – high"` for a band row, or `null` when either edge is missing. */
function formatBandRange(point: Record<string, unknown>, upperKey: string, lowerKey: string): string | null {
  const upper = point[upperKey];
  const lower = point[lowerKey];
  if (!(typeof upper === "number" && Number.isFinite(upper) && typeof lower === "number" && Number.isFinite(lower))) {
    return null;
  }
  return `${intFmt(Math.min(lower, upper))} – ${intFmt(Math.max(lower, upper))}`;
}

/** Vertical 1px crosshair, faded at both ends. Mounted only while hovering. */
function Crosshair({ x, height, animate }: { x: number; height: number; animate: boolean }) {
  const gradientId = `tooltip-indicator-gradient-${useId()}`;
  const rectX = x - 0.5;
  const animatedX = useSpring(rectX, TOOLTIP_SPRING);
  const animatedHeight = useSpring(height, TOOLTIP_SPRING);
  if (animate) {
    animatedX.set(rectX);
    animatedHeight.set(height);
  }
  useEffect(() => {
    animatedX.set(rectX);
    animatedHeight.set(height);
  }, [animatedX, animatedHeight, rectX, height]);

  const color = chartCssVars.crosshair;
  const stops: [string, number][] = [
    ["0%", 0],
    ["10%", 1],
    ["50%", 1],
    ["90%", 1],
    ["100%", 0],
  ];
  return (
    <g>
      <defs>
        <linearGradient id={gradientId} x1="0%" x2="0%" y1="0%" y2="100%">
          {stops.map(([offset, opacity]) => (
            <stop key={offset} offset={offset} style={{ stopColor: color, stopOpacity: opacity }} />
          ))}
        </linearGradient>
      </defs>
      {animate ? (
        <m.rect fill={`url(#${gradientId})`} height={animatedHeight} width={1} x={animatedX} y={0} />
      ) : (
        <rect fill={`url(#${gradientId})`} height={height} width={1} x={rectX} y={0} />
      )}
    </g>
  );
}

/** Filled dot with a soft same-colour halo that springs to the hovered point. */
function TooltipDot({ x, y, color, animate }: { x: number; y: number; color: string; animate: boolean }) {
  const animatedX = useSpring(x, TOOLTIP_SPRING);
  const animatedY = useSpring(y, TOOLTIP_SPRING);
  if (animate) {
    animatedX.set(x);
    animatedY.set(y);
  }
  const halo = DOT_SIZE * HALO_SCALE;
  if (!animate) {
    return (
      <g>
        <circle cx={x} cy={y} fill={color} fillOpacity={0.3} r={halo} />
        <circle cx={x} cy={y} fill={color} r={DOT_SIZE} />
      </g>
    );
  }
  return (
    <g>
      <m.circle
        animate={{ r: halo }}
        cx={animatedX}
        cy={animatedY}
        fill={color}
        fillOpacity={0.3}
        initial={{ r: DOT_SIZE }}
        transition={HALO_SPRING}
      />
      <m.circle
        animate={{ r: DOT_SIZE }}
        cx={animatedX}
        cy={animatedY}
        fill={color}
        initial={{ r: DOT_SIZE * 0.5 }}
        transition={HALO_SPRING}
      />
    </g>
  );
}

function TooltipContent({ title, rows }: { title: string; rows: TooltipRow[] }) {
  return (
    <div>
      <div className="border-b px-3.5 py-2.5 text-left font-medium text-chart-tooltip-foreground text-sm">{title}</div>
      {rows.length > 0 && (
        <div className="px-3.5 py-3">
          <div className="space-y-2">
            {rows.map((row) => (
              <div className="flex items-center justify-between gap-6" key={`${row.label}-${row.color}`}>
                <div className="flex min-w-0 items-center gap-2">
                  <span className="size-2 shrink-0 rounded-full" style={{ backgroundColor: row.color }} />
                  <span className="truncate text-chart-tooltip-muted text-sm">{row.label}</span>
                </div>
                <span className="text-right font-medium text-chart-tooltip-foreground text-sm tabular-nums">
                  {typeof row.value === "number" ? intFmt(row.value) : row.value}
                </span>
              </div>
            ))}
          </div>
        </div>
      )}
    </div>
  );
}

const ChartTooltipInner = memo(function ChartTooltipInner({ container }: { container: HTMLElement }) {
  const { width, height, innerHeight, margin, lines, bands, xAccessor, dateLabels, containerRef } = useChartStable();
  const { tooltipData } = useChartHover();
  const discrete = dateLabels.length > DISCRETE_INTERACTION_LABEL_THRESHOLD;
  const visible = tooltipData !== null;
  const x = tooltipData?.x ?? 0;
  const xWithMargin = x + margin.left;

  const rows = useMemo(() => {
    if (!tooltipData) return [];
    const result: TooltipRow[] = lines.map((line) => ({
      color: line.stroke,
      label: line.dataKey,
      value: (tooltipData.point[line.dataKey] as number) ?? 0,
    }));
    for (const band of bands) {
      const value = formatBandRange(tooltipData.point, band.upperKey, band.lowerKey);
      if (value != null) result.push({ color: band.color, label: band.label, value });
    }
    return result;
  }, [tooltipData, lines, bands]);

  // One dot per band edge, deduped; edges shared with a `<Line>` use its dot.
  const bandDots = useMemo(() => {
    const seen = new Set(lines.map((line) => line.dataKey));
    const dots: { key: string; color: string }[] = [];
    for (const band of bands) {
      for (const key of [band.upperKey, band.lowerKey]) {
        if (seen.has(key)) continue;
        seen.add(key);
        dots.push({ key, color: band.color });
      }
    }
    return dots;
  }, [lines, bands]);

  const overlay = "pointer-events-none absolute inset-0 overflow-visible";
  return createPortal(
    <>
      {visible && (
        <>
          <svg aria-hidden="true" className={overlay} height="100%" width="100%">
            <g transform={`translate(${margin.left},${margin.top})`}>
              <Crosshair animate={!discrete} height={innerHeight} x={x} />
            </g>
          </svg>
          <svg aria-hidden="true" className={overlay} height="100%" width="100%">
            <g transform={`translate(${margin.left},${margin.top})`}>
              {lines.map((line) => (
                <TooltipDot
                  animate={!discrete}
                  color={line.stroke}
                  key={line.dataKey}
                  x={x}
                  y={tooltipData.yPositions[line.dataKey] ?? 0}
                />
              ))}
              {bandDots.map((dot) => {
                const y = tooltipData.yPositions[dot.key];
                return y == null ? null : (
                  <TooltipDot animate={!discrete} color={dot.color} key={`band-${dot.key}`} x={x} y={y} />
                );
              })}
            </g>
          </svg>
        </>
      )}

      <TooltipBox
        containerHeight={height}
        containerRef={containerRef}
        containerWidth={width}
        top={margin.top}
        visible={visible}
        x={xWithMargin}
      >
        {tooltipData && <TooltipContent rows={rows} title={weekdayDateFmt.format(xAccessor(tooltipData.point))} />}
      </TooltipBox>

      {visible && dateLabels.length > 0 && (
        <DatePill
          currentIndex={tooltipData.index}
          discrete={discrete}
          labels={dateLabels}
          xWithMargin={xWithMargin}
        />
      )}
    </>,
    container,
  );
});

/** Crosshair, dots on every series and band edge, a floating panel and a date pill on hover. */
export function ChartTooltip() {
  const { containerRef } = useChartStable();
  const container = useChartPortalContainer(containerRef);
  return container ? <ChartTooltipInner container={container} /> : null;
}

ChartTooltip.displayName = "ChartTooltip";

// Mounted on show, so the spring starts at the cursor instead of the left margin.
function DatePill({
  labels,
  currentIndex,
  xWithMargin,
  discrete,
}: {
  labels: string[];
  currentIndex: number;
  xWithMargin: number;
  discrete: boolean;
}) {
  const animatedX = useSpring(xWithMargin, TOOLTIP_SPRING);
  if (!discrete) animatedX.set(xWithMargin);
  return (
    <m.div
      className="pointer-events-none absolute z-50 w-max"
      style={{ left: discrete ? xWithMargin : animatedX, transform: "translateX(-50%)", bottom: 4 }}
    >
      <DateTicker currentIndex={currentIndex} labels={labels} />
    </m.div>
  );
}
