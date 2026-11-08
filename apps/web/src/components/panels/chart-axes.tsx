"use client";

import { createPortal } from "react-dom";

import {
  useChartPortalContainer,
  useChartStable,
  useYScale,
} from "@/components/charts/chart-context";

// Lightweight axes for the panel charts. The shared XAxis picks "evenly spaced" data rows
// (00:00, 04:45, 09:30, …) and there is no YAxis component, so these render readable
// fixed-hour ticks and a 3–4 value y-axis as HTML overlays, the same way XAxis does.

/** Same count the Grid should use (`numTicksRows`) so lines and labels line up. */
export const Y_TICKS = 3;

const tickClass =
  "absolute text-[11px] leading-none whitespace-nowrap text-muted-foreground tabular-nums";

const HOUR_TICKS = [0, 6, 12, 18, 24];

/** Ticks every 6 hours across one wall-clock day, labelled 00:00 … 24:00. */
export function HourTicks({ day }: { day: Date }) {
  const { xScale, margin, containerRef, innerHeight } = useChartStable();
  const container = useChartPortalContainer(containerRef);
  if (!container) return null;
  const midnight = new Date(day.getFullYear(), day.getMonth(), day.getDate());
  return createPortal(
    <div className="pointer-events-none absolute inset-0">
      {HOUR_TICKS.map((h) => {
        const x =
          (xScale(new Date(midnight.getTime() + h * 3_600_000)) ?? 0) +
          margin.left;
        return (
          <span
            key={h}
            className={tickClass}
            style={{
              left: x,
              top: margin.top + innerHeight + 8,
              transform: "translateX(-50%)",
            }}
          >
            {`${String(h).padStart(2, "0")}:00`}
          </span>
        );
      })}
    </div>,
    container,
  );
}

const fmt = (v: number) =>
  (Math.abs(v) >= 100 || Number.isInteger(v) ? v.toFixed(0) : v.toFixed(1))
    // typographic minus
    .replace("-", "\u2212");

/** Y-axis labels (3–4 values) with the unit on the top tick. */
export function YTicks({
  unit,
  side = "left",
  yAxisId,
}: {
  unit: string;
  side?: "left" | "right";
  yAxisId?: string;
}) {
  const { margin, containerRef, width } = useChartStable();
  const yScale = useYScale(yAxisId);
  const container = useChartPortalContainer(containerRef);
  if (!container) return null;
  // Only label values inside the plotted range (a "nice" tick can fall just outside it).
  const [lo, hi] = yScale.domain() as [number, number];
  const ticks = yScale
    .ticks(Y_TICKS)
    .filter((t) => t >= Math.min(lo, hi) && t <= Math.max(lo, hi));
  const top = ticks.at(-1);
  return createPortal(
    <div className="pointer-events-none absolute inset-0">
      {ticks.map((t) => (
        <span
          key={t}
          className={tickClass}
          style={{
            top: (yScale(t) ?? 0) + margin.top,
            transform: "translateY(-50%)",
            ...(side === "left"
              ? { right: width - margin.left + 6 }
              : { left: width - margin.right + 6 }),
          }}
        >
          {fmt(t)}
          {t === top ? (
            <span className={unit === "%" ? "" : "ml-0.5"}>{unit}</span>
          ) : null}
        </span>
      ))}
    </div>,
    container,
  );
}
