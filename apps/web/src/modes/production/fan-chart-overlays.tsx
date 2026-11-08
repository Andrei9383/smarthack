"use client";

import { createPortal } from "react-dom";

import { useChartPortalContainer, useChartStable } from "@/components/charts/chart-context";

// HTML overlays for the park fan chart, portalled into the chart container like the shared XAxis.
// The shared axis picks ticks by spacing, which lands on odd hours; these follow the clock instead.

const LABEL = "pointer-events-none absolute text-[11px] leading-none text-muted-foreground tabular-nums";

function useOverlayContainer() {
  const { containerRef } = useChartStable();
  return useChartPortalContainer(containerRef);
}

/** MW values on the left, at the same `yScale.ticks(count)` the Grid draws its rows at. */
export function FanYAxis({ numTicks, unit }: { numTicks: number; unit: string }) {
  const container = useOverlayContainer();
  const { yScale, margin } = useChartStable();
  if (!container) return null;
  const ticks = yScale.ticks(numTicks);
  // One decimal when the ticks are closer than 1 apart (e.g. MWh per 15 min), else whole numbers.
  const step = ticks.length > 1 ? Math.abs(ticks[1]! - ticks[0]!) : 1;
  const digits = step >= 1 ? 0 : step >= 0.1 ? 1 : 2;
  return createPortal(
    <div className="pointer-events-none absolute inset-0">
      <span className={LABEL} style={{ left: 0, top: 2 }}>
        {unit}
      </span>
      {ticks.map((tick) => (
        <span
          key={tick}
          className={`${LABEL} -translate-y-1/2 text-right`}
          style={{ left: 0, width: margin.left - 8, top: margin.top + yScale(tick) }}
        >
          {tick.toFixed(digits)}
        </span>
      ))}
    </div>,
    container,
  );
}

/** Hour labels on whole multiples of `everyHours` (wall-clock dates, see wallClock). */
export function FanHourTicks({ everyHours }: { everyHours: number }) {
  const container = useOverlayContainer();
  const { data, xAccessor, xScale, margin, innerHeight } = useChartStable();
  if (!container) return null;
  // One label per whole hour: data at 15-min steps would otherwise stack four labels on each hour.
  const ticks = data
    .map(xAccessor)
    .filter((date) => date.getMinutes() === 0 && date.getHours() % everyHours === 0);
  return createPortal(
    <div className="pointer-events-none absolute inset-0">
      {ticks.map((date) => (
        <span
          key={date.getTime()}
          className={`${LABEL} -translate-x-1/2`}
          style={{ left: margin.left + xScale(date), top: margin.top + innerHeight + 8 }}
        >
          {`${String(date.getHours()).padStart(2, "0")}:00`}
        </span>
      ))}
    </div>,
    container,
  );
}

/** Vertical hairline with a small label at the replay clock. */
export function FanNowMarker({ date, label }: { date: Date; label: string }) {
  const container = useOverlayContainer();
  const { xScale, margin, innerHeight, innerWidth } = useChartStable();
  if (!container) return null;
  const x = xScale(date);
  if (!Number.isFinite(x) || x < 0 || x > innerWidth) return null;
  return createPortal(
    <div className="pointer-events-none absolute inset-0">
      <span
        className="absolute w-px bg-foreground/40"
        style={{ left: margin.left + x, top: margin.top - 8, height: innerHeight + 8 }}
      />
      <span
        className="absolute rounded-full bg-foreground px-1.5 py-0.5 text-[10px] leading-none font-medium text-background"
        style={{ left: margin.left + x + 4, top: margin.top - 20 }}
      >
        {label}
      </span>
    </div>,
    container,
  );
}
