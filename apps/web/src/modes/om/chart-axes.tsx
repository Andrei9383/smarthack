"use client";

import { useChartStable, useYScale } from "@/components/charts/chart-context";

import { OM_COLORS } from "./colors";

// Axes for the shared <LineChart>, drawn from its scales so they line up exactly. The display names
// match the chart shell's axis list, so they stay outside the series clip.

/** Left y-axis with its grid lines: 3–4 values, the unit on the top tick. */
export function OmYAxis({ unit }: { unit: string }) {
  const yScale = useYScale();
  const { innerWidth } = useChartStable();
  let values = yScale.ticks(4);
  if (values.length > 4) values = values.filter((_, i) => i % 2 === 0);
  const top = values.at(-1);
  return (
    <g fontSize={11}>
      {values.map((v) => (
        <g key={v}>
          <line
            x1={0}
            x2={innerWidth}
            y1={yScale(v)}
            y2={yScale(v)}
            stroke={OM_COLORS.grid}
          />
          <text
            x={-8}
            y={yScale(v)}
            dy="0.32em"
            textAnchor="end"
            className="fill-muted-foreground tabular-nums"
          >
            {v}
            {v === top ? ` ${unit}` : ""}
          </text>
        </g>
      ))}
    </g>
  );
}
OmYAxis.displayName = "YAxis";

const DAY = new Intl.DateTimeFormat("en-GB", {
  weekday: "short",
  day: "numeric",
});

/**
 * Time axis: a faint guide every 6 h, labelled at midnight with the day ("Mon 2"). For a span of a
 * day or less every 6 h tick is labelled with its hour (00:00, 06:00, …).
 */
export function OmTimeAxis() {
  const { xScale, innerHeight, innerWidth } = useChartStable();
  const [start, end] = xScale.domain() as [Date, Date];
  if (!start || !end) return null;
  const short = end.getTime() - start.getTime() <= 36 * 3600 * 1000;
  const first = new Date(start);
  first.setMinutes(0, 0, 0);
  first.setHours(Math.ceil(first.getHours() / 6) * 6);
  const ticks: Date[] = [];
  for (let t = first; t <= end; t = new Date(t.getTime() + 6 * 3600 * 1000))
    ticks.push(t);
  return (
    <g fontSize={11}>
      {ticks.map((t) => {
        const x = xScale(t);
        const midnight = t.getHours() === 0;
        const label = short
          ? `${String(t.getHours()).padStart(2, "0")}:00`
          : midnight
            ? DAY.format(t)
            : null;
        return (
          <g key={t.getTime()}>
            <line
              x1={x}
              x2={x}
              y1={0}
              y2={innerHeight}
              stroke={OM_COLORS.grid}
              strokeWidth={midnight ? 1 : 0.75}
              strokeDasharray={midnight ? undefined : "2,3"}
            />
            {label && x >= 12 && x <= innerWidth - 12 ? (
              <text
                x={x}
                y={innerHeight + 16}
                textAnchor="middle"
                className="fill-muted-foreground tabular-nums"
              >
                {label}
              </text>
            ) : null}
          </g>
        );
      })}
    </g>
  );
}
OmTimeAxis.displayName = "XAxis";
