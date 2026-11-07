"use client";

import { scaleLinear } from "d3-scale";
import type { LucideIcon } from "lucide-react";
import type { ReactNode } from "react";

// Small presentational pieces shared by the Feeders and Demand panels (team/contracts/ui-style.md).

export const SECTION_LABEL =
  "text-[11px] font-medium tracking-[0.04em] text-text-secondary uppercase";

/** Plot margins: room on the left for y labels, below for time ticks. */
export const AXIS_MARGIN = { left: 36, right: 10, top: 22, bottom: 24 };

const DAY_TICKS_H = [0, 6, 12, 18, 24];

/**
 * The y-domain LineChart will settle on when given `yScaleDomainMax`: it pads by 10% and then
 * applies d3 `nice()`. Doing the same here lets the overlay labels line up with the grid.
 */
export function yAxisFor(values: number[]) {
  const finite = values.filter((value) => Number.isFinite(value));
  const max = finite.length > 0 ? Math.max(0, ...finite) : 0;
  const domainMax = max > 0 ? max : 1;
  const scale = scaleLinear()
    .domain([0, domainMax * 1.1])
    .nice();
  const top = scale.domain()[1] ?? domainMax * 1.1;
  return { domainMax, top, ticks: scale.ticks(3) };
}

const formatTick = (value: number) =>
  Math.abs(value) >= 1000 ? `${(value / 1000).toFixed(1)}k` : `${Number(value.toFixed(1))}`;

type AxisMark = { value: number; label: string; className?: string };

/**
 * Time and value labels drawn over a LineChart (which has no y-axis and crowds its own time
 * ticks). Pass the same dates the chart plots; ticks land on 00:00/06:00/12:00/18:00/24:00.
 */
export function ChartAxes({
  dates,
  top,
  ticks,
  unit,
  marks = [],
  now,
}: {
  dates: Date[];
  top: number;
  ticks: number[];
  unit: string;
  marks?: AxisMark[];
  /** The replay clock, drawn as a thin "now" rule when it falls inside the plotted day. */
  now?: Date;
}) {
  const first = dates[0];
  const last = dates.at(-1);
  if (!first || !last || top <= 0) return null;
  const start = first.getTime();
  const span = Math.max(1, last.getTime() - start);
  const midnight = new Date(first.getFullYear(), first.getMonth(), first.getDate()).getTime();
  const step = dates.length > 1 ? span / (dates.length - 1) : 0;
  const timeTicks = DAY_TICKS_H.flatMap((hour) => {
    const fraction = (midnight + hour * 3_600_000 - start) / span;
    // Allow one sample past the end so 24:00 sits on the last 23:45 point's right edge.
    if (fraction < -0.001 || fraction > 1 + step / span + 0.001) return [];
    return [{ hour, fraction: Math.min(1, fraction) }];
  });
  const yOf = (value: number) => `${(1 - value / top) * 100}%`;
  const nowFraction = now ? (now.getTime() - start) / span : null;
  const showNow = nowFraction != null && nowFraction >= 0 && nowFraction <= 1;

  return (
    <div
      aria-hidden
      className="pointer-events-none absolute"
      style={{
        top: AXIS_MARGIN.top,
        bottom: AXIS_MARGIN.bottom,
        left: AXIS_MARGIN.left,
        right: AXIS_MARGIN.right,
      }}
    >
      <span className="absolute right-full bottom-full mb-2.5 pr-2 text-[10px] leading-none font-medium text-muted-foreground">
        {unit}
      </span>
      {ticks.map((value) => (
        <span
          key={value}
          className="absolute right-full -translate-y-1/2 pr-2 text-[11px] leading-none text-muted-foreground tabular-nums"
          style={{ top: yOf(value) }}
        >
          {formatTick(value)}
        </span>
      ))}
      {timeTicks.map(({ hour, fraction }, index) => (
        <span
          key={hour}
          className={`absolute top-full mt-2 text-[11px] leading-none text-muted-foreground tabular-nums ${
            index === 0 ? "" : index === timeTicks.length - 1 ? "-translate-x-full" : "-translate-x-1/2"
          }`}
          style={{ left: `${fraction * 100}%` }}
        >
          {`${String(hour).padStart(2, "0")}:00`}
        </span>
      ))}
      {showNow ? (
        <span className="absolute inset-y-0 border-l border-dashed border-foreground/35" style={{ left: `${(nowFraction ?? 0) * 100}%` }}>
          <span className="absolute bottom-full left-0 mb-1 -translate-x-1/2 rounded-sm bg-foreground px-1 py-px font-mono text-[9px] leading-none font-medium text-background">
            now
          </span>
        </span>
      ) : null}
      {marks.map(({ value, label, className }) => (
        <span
          key={label}
          className={`absolute right-0 -translate-y-full rounded-sm bg-card/90 px-1 pb-0.5 text-[11px] leading-none font-semibold whitespace-nowrap tabular-nums ${className ?? ""}`}
          style={{ top: yOf(value) }}
        >
          {label}
        </span>
      ))}
    </div>
  );
}

export function ChartLegend({ items }: { items: { label: string; color: string; dashed?: boolean }[] }) {
  return (
    <div className="flex flex-wrap items-center justify-end gap-x-3 gap-y-1 text-[11px] text-text-secondary">
      {items.map(({ label, color, dashed }) => (
        <span key={label} className="flex items-center gap-1.5 whitespace-nowrap">
          <span
            className="inline-block w-3.5 border-t-[2.5px]"
            style={{ borderColor: color, borderStyle: dashed ? "dashed" : "solid" }}
          />
          {label}
        </span>
      ))}
    </div>
  );
}

/** Section title on the left, optional legend on the right, one line. */
export function SectionHeader({ title, children }: { title: ReactNode; children?: ReactNode }) {
  return (
    <div className="flex items-center justify-between gap-3">
      <h3 className={SECTION_LABEL}>{title}</h3>
      {children}
    </div>
  );
}

function provenanceDot(label: string) {
  if (/non-ro/i.test(label)) return "bg-status-forecast";
  if (/real/i.test(label)) return "bg-primary";
  return "bg-status-unknown";
}

/** The neutral provenance pill shared by every mode (synthetic RO / real RO / non-RO). */
export function ProvenancePill({ label }: { label: string }) {
  return (
    <span className="inline-flex items-center gap-1.5 rounded-full bg-muted px-2 py-0.5 text-[11px] font-medium whitespace-nowrap text-text-secondary">
      <span className={`size-1.5 rounded-full ${provenanceDot(label)}`} />
      {label}
    </span>
  );
}

export function EmptyNote({ icon: Icon, children }: { icon: LucideIcon; children: ReactNode }) {
  return (
    <div className="flex flex-col items-center gap-2 rounded-lg bg-sidebar px-4 py-6 text-center text-xs text-muted-foreground">
      <Icon className="size-4" aria-hidden />
      <p>{children}</p>
    </div>
  );
}
