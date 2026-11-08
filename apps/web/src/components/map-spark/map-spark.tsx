"use client";

import { cn } from "@wattsentry/ui/lib/utils";
import { curveMonotoneX } from "d3-shape";
import { domAnimation, LazyMotion, useReducedMotion } from "motion/react";
import { useId, useMemo } from "react";

import { bandAreaPath, bandEdgePath, computeBandSegments } from "@/components/charts/band-geometry";
import { ChartRevealClip } from "@/components/charts/chart-reveal-clip";
import { linePathD } from "@/components/charts/line-path";

// A map-sized member of the Bklit chart family (components/charts): the same band geometry, line
// paths, dashed projection and left-to-right reveal clip, without the shell's axes and tooltip,
// which can't shrink to 120 × 36. Grammar, everywhere on the map:
//   measured (past)  solid line          forecast (ahead)  dashed line from the last measurement
//   likely range     soft P10–P90 band   now               hairline tick at the replay clock
//   peak             dot + "17 MW · 11:00"                 limit  dashed line + "62 kW"

export type SparkPoint = {
  /** Where the point sits on the time axis, epoch ms. */
  t: number;
  /** Shown for the peak instead of formatTime(t), e.g. the hour a bucket starts. */
  label?: string;
  actual?: number | null;
  p10?: number | null;
  p50?: number | null;
  p90?: number | null;
};

export type MapSparkProps = {
  /** Sorted by t. Actual values after `now` are ignored; forecast values start at the last actual. */
  points: SparkPoint[];
  /** The replay clock, epoch ms. */
  now: number;
  /** Visible time window, epoch ms. */
  from: number;
  to: number;
  unit: string;
  expanded?: boolean;
  /** Measured line colour: near-black for telemetry, green for WattSentry's own series. */
  tone?: "actual" | "brand";
  /** Dashed horizontal limit with its value at the end, e.g. a feeder's 62 kW. */
  limit?: { value: number; label: string };
  /** Hour labels for the peak, in the app's wall-clock zone. */
  formatTime: (ms: number) => string;
  /** Ticks under the expanded chart, as hours after now. */
  tickHours?: number[];
  /** Explains the grammar under the expanded chart; the legend says it once per mode. */
  showKey?: boolean;
  /** Peak value text in both sizes, or only when opened (where a limit already crowds the compact chart). */
  peakLabel?: "always" | "expanded";
  /** Words for that line, e.g. "today so far" / "rest of the day" on a feeder. */
  keyLabels?: Partial<SparkKeyLabels>;
  className?: string;
};

const SIZE = {
  compact: { width: 120, height: 36, pad: { top: 5, right: 3, bottom: 3, left: 3 } },
  expanded: { width: 220, height: 90, pad: { top: 16, right: 6, bottom: 16, left: 6 } },
} as const;

const HOUR_MS = 3_600_000;
const REVEAL = { type: "tween", duration: 0.3, ease: [0.22, 1, 0.36, 1] } as const;
const PEAK_LABEL_GAP = 5;
// Rough width of a label character, to keep the peak label inside the chart.
const CHAR_PX = { compact: 5, expanded: 5.6 } as const;

const isNumber = (value: number | null | undefined): value is number =>
  typeof value === "number" && Number.isFinite(value);

export function formatAmount(value: number) {
  return value >= 10 ? Math.round(value).toString() : value.toFixed(1);
}

function sanitizeId(id: string) {
  return id.replace(/[^a-zA-Z0-9_-]/g, "");
}

export function MapSpark({
  points,
  now,
  from,
  to,
  unit,
  expanded = false,
  tone = "actual",
  limit,
  formatTime,
  tickHours = [0, 6, 12],
  showKey = false,
  keyLabels,
  peakLabel = "always",
  className,
}: MapSparkProps) {
  const reduceMotion = useReducedMotion();
  const clipId = `map-spark-${sanitizeId(useId())}`;
  const { width, height, pad } = expanded ? SIZE.expanded : SIZE.compact;
  const innerWidth = width - pad.left - pad.right;
  const innerHeight = height - pad.top - pad.bottom;

  const model = useMemo(() => {
    const visible = points.filter((point) => point.t >= from - HOUR_MS && point.t <= to + HOUR_MS);
    const past = visible.filter((point) => point.t <= now && isNumber(point.actual));
    const anchor = past.at(-1);
    const measured = past.length > 0;
    // With no measurements the whole window is forecast; otherwise it picks up where they stop.
    const ahead = visible.filter((point) => (anchor ? point.t > anchor.t : true) && isNumber(point.p50));
    const projection = anchor
      ? [{ t: anchor.t, value: anchor.actual as number }, ...ahead.map((p) => ({ t: p.t, value: p.p50 as number }))]
      : ahead.map((p) => ({ t: p.t, value: p.p50 as number }));
    // The band opens from the last measurement, so it reads as a fan of where things may go.
    const bandRows = [
      ...(anchor ? [{ t: anchor.t, upper: anchor.actual as number, lower: anchor.actual as number }] : []),
      ...ahead
        .filter((p) => (anchor ? true : p.t >= now - HOUR_MS) && isNumber(p.p10) && isNumber(p.p90))
        .map((p) => ({ t: p.t, upper: p.p90 as number, lower: p.p10 as number })),
    ];
    const peak = ahead
      .filter((p) => p.t > now && p.t <= to)
      .reduce<SparkPoint | null>((best, p) => (best && (best.p50 as number) >= (p.p50 as number) ? best : p), null);
    const values = [
      ...past.map((p) => p.actual as number),
      ...ahead.flatMap((p) => [p.p50 as number, ...(isNumber(p.p90) ? [p.p90] : [])]),
      ...(limit ? [limit.value] : []),
    ];
    const max = Math.max(...values, 0);
    return { past, projection, bandRows, peak, measured, yMax: max > 0 ? max * 1.08 : 1 };
  }, [points, now, from, to, limit]);

  const x = (t: number) => pad.left + ((t - from) / (to - from)) * innerWidth;
  const y = (value: number) => pad.top + innerHeight - (Math.max(0, value) / model.yMax) * innerHeight;

  const pastD = linePathD(model.past, {
    x: (p) => x(p.t),
    y: (p) => y(p.actual as number),
    curve: curveMonotoneX,
  });
  const projectionD = linePathD(model.projection, {
    x: (p) => x(p.t),
    y: (p) => y(p.value),
    curve: curveMonotoneX,
  });
  const bandData = model.bandRows.map((row) => ({ date: new Date(row.t), upper: row.upper, lower: row.lower }));
  const segments = computeBandSegments(
    bandData,
    (datum) => datum.date as Date,
    (date) => x(date.getTime()),
    (value) => y(value),
    "upper",
    "lower",
  );
  const bandD = bandAreaPath(segments, curveMonotoneX);
  const bandEdgeD = expanded ? bandEdgePath(segments, "upper", curveMonotoneX) : "";

  const nowX = x(now);
  const baseY = pad.top + innerHeight;
  const peak = model.peak;
  const peakX = peak ? x(peak.t) : 0;
  const peakY = peak ? y(peak.p50 as number) : 0;
  const peakText = peak
    ? `${formatAmount(peak.p50 as number)} ${unit} · ${peak.label ?? formatTime(peak.t)}`
    : "";
  const peakTextWidth = peakText.length * (expanded ? CHAR_PX.expanded : CHAR_PX.compact);
  const fontPx = expanded ? 10 : 9;
  // The drawn curves in chart pixels, to keep the label off them.
  const curve = [
    ...model.past.map((p) => ({ x: x(p.t), y: y(p.actual as number) })),
    ...model.projection.map((p) => ({ x: x(p.t), y: y(p.value) })),
  ].sort((a, b) => a.x - b.x);
  const curveYAt = (at: number) => {
    const i = curve.findIndex((p) => p.x >= at);
    if (i <= 0) return curve[Math.max(i, 0)]?.y ?? null;
    const a = curve[i - 1]!;
    const b = curve[i]!;
    return a.y + ((b.y - a.y) * (at - a.x)) / Math.max(b.x - a.x, 1e-6);
  };
  // Candidate spots: beside the dot (above or below it) or in the top strip, sliding sideways away
  // from the dot. Keep the spot that stays inside the chart and clear of the line, nearest the dot.
  const peakPlacement = (() => {
    const baselines = [peakY - 3, fontPx, peakY + fontPx + 2];
    const shifts = [0, 8, 16, 28, 40, 56];
    const options = baselines.flatMap((baseline) =>
      shifts.flatMap((shift) =>
        [false, true].map((left) => {
          const anchorX = left ? peakX - PEAK_LABEL_GAP - shift : peakX + PEAK_LABEL_GAP + shift;
          const x0 = left ? anchorX - peakTextWidth : anchorX;
          const x1 = x0 + peakTextWidth;
          const top = baseline - fontPx + 1;
          const fits = x0 >= 1 && x1 <= width - 1 && top >= 0 && baseline <= height - 1;
          let hits = 0;
          for (let at = x0; at <= x1; at += 2) {
            const cy = curveYAt(at);
            if (cy != null && cy >= top - 2 && cy <= baseline + 2) hits += 1;
          }
          // The label shouldn't cover its own dot either.
          if (peakX >= x0 - 3 && peakX <= x1 + 3 && peakY >= top - 3 && peakY <= baseline + 3) hits += 3;
          const distance = Math.abs((x0 + x1) / 2 - peakX) + Math.abs(baseline - fontPx / 2 - peakY);
          return { left, baseline, anchorX, fits, hits, distance };
        }),
      ),
    );
    const fitting = options.filter((o) => o.fits);
    return (
      fitting.filter((o) => o.hits === 0).sort((a, b) => a.distance - b.distance)[0] ??
      fitting.sort((a, b) => a.hits - b.hits || a.distance - b.distance)[0] ??
      options[0]!
    );
  })();
  const peakLabelLeft = peakPlacement.left;
  const peakAnchorX = peakPlacement.anchorX;
  const peakLabelY = peakPlacement.baseline;

  const lineWidth = expanded ? 1.75 : 1.5;
  const dash = expanded ? "4 3" : "3 2.5";
  const measuredStroke = tone === "brand" ? "var(--sentinel)" : "var(--status-actual)";
  const ticks = tickHours
    .map((hours) => ({ hours, at: x(now + hours * HOUR_MS) }))
    .filter((tick) => tick.at >= pad.left - 1 && tick.at <= width - pad.right + 1);

  const summary = [
    model.measured ? `measured ${formatAmount(model.past.at(-1)?.actual ?? 0)} ${unit} now` : "forecast only",
    peak ? `forecast peak ${formatAmount(peak.p50 as number)} ${unit} at ${peak.label ?? formatTime(peak.t)}` : null,
    limit ? `limit ${limit.label}` : null,
  ]
    .filter(Boolean)
    .join(", ");

  return (
    <span className={cn("block", className)}>
      <LazyMotion features={domAnimation}>
        <svg
          width={width}
          height={height}
          viewBox={`0 0 ${width} ${height}`}
          role="img"
          aria-label={summary}
          className="block overflow-visible"
        >
          <defs>
            <ChartRevealClip
              clipPathId={clipId}
              height={height}
              targetWidth={width}
              revealEpoch={0}
              enterTransition={REVEAL}
              animating={!reduceMotion}
              paddingY={4}
            />
          </defs>

          {/* Zero line: gives the series a floor without a full axis. */}
          <line x1={pad.left} x2={width - pad.right} y1={baseY} y2={baseY} stroke="var(--chart-grid)" strokeWidth={1} />

          {expanded && (
            <text x={pad.left} y={9} className="fill-muted-foreground text-[9.5px] font-medium">
              {unit}
            </text>
          )}

          <g clipPath={`url(#${clipId})`}>
            {bandD && <path d={bandD} fill="var(--status-forecast)" fillOpacity={expanded ? 0.13 : 0.15} />}
            {bandEdgeD && (
              <path d={bandEdgeD} fill="none" stroke="var(--status-forecast)" strokeOpacity={0.3} strokeWidth={1} />
            )}
            {limit && (
              <line
                x1={pad.left}
                x2={width - pad.right}
                y1={y(limit.value)}
                y2={y(limit.value)}
                stroke="var(--status-critical)"
                strokeOpacity={0.75}
                strokeWidth={1}
                strokeDasharray="2 2"
              />
            )}
            {pastD && (
              <path
                d={pastD}
                fill="none"
                stroke={measuredStroke}
                strokeWidth={lineWidth}
                strokeLinecap="round"
                strokeLinejoin="round"
              />
            )}
            {projectionD && (
              <path
                d={projectionD}
                fill="none"
                stroke="var(--status-forecast)"
                strokeWidth={lineWidth}
                strokeDasharray={dash}
                strokeLinecap="round"
              />
            )}
            {peak && (
              <>
                <circle cx={peakX} cy={peakY} r={expanded ? 3 : 2.5} fill="var(--status-forecast)" stroke="var(--card)" strokeWidth={1.5} />
                {(expanded || peakLabel === "always") && <text
                  x={peakAnchorX}
                  y={peakLabelY}
                  textAnchor={peakLabelLeft ? "end" : "start"}
                  paintOrder="stroke"
                  stroke="var(--card)"
                  strokeWidth={3}
                  strokeLinejoin="round"
                  className={cn("fill-foreground font-semibold tabular-nums", expanded ? "text-[10px]" : "text-[9px]")}
                >
                  {peakText}
                </text>}
              </>
            )}
          </g>

          {/* Now: a hairline at the replay clock, drawn outside the reveal so it never moves. */}
          {nowX >= pad.left && nowX <= width - pad.right && (
            <line
              x1={nowX}
              x2={nowX}
              y1={pad.top - (expanded ? 4 : 3)}
              y2={baseY + (expanded ? 3 : 2)}
              stroke="var(--foreground)"
              strokeOpacity={0.45}
              strokeWidth={1}
            />
          )}

          {limit && expanded && (
            <text
              x={pad.left + 14}
              y={y(limit.value) - 3}
              textAnchor="start"
              paintOrder="stroke"
              stroke="var(--card)"
              strokeWidth={3}
              className="fill-status-critical text-[9.5px] font-semibold tabular-nums"
            >
              {limit.label}
            </text>
          )}

          {expanded &&
            ticks.map((tick) => (
              <text
                key={tick.hours}
                x={tick.at}
                y={height - 3}
                textAnchor="middle"
                className={cn(
                  "text-[9.5px] tabular-nums",
                  tick.hours === 0 ? "fill-foreground font-semibold" : "fill-muted-foreground",
                )}
              >
                {tick.hours === 0 ? "now" : `+${tick.hours} h`}
              </text>
            ))}
        </svg>
      </LazyMotion>
      {expanded && showKey && (
        <SparkKey
          measured={model.measured}
          range={model.bandRows.length > 1}
          limit={Boolean(limit)}
          tone={tone}
          labels={keyLabels}
          className="mt-1"
        />
      )}
    </span>
  );
}

export type SparkKeyLabels = { measured: string; forecast: string; range: string; limit: string };

const DEFAULT_KEY: SparkKeyLabels = {
  measured: "measured",
  forecast: "forecast",
  range: "likely range",
  limit: "limit",
};

/** One line explaining the sparkline grammar, for legends and the expanded chip. */
export function SparkKey({
  measured = true,
  range = true,
  limit = false,
  tone = "actual",
  labels,
  className,
}: {
  measured?: boolean;
  range?: boolean;
  limit?: boolean;
  tone?: "actual" | "brand";
  labels?: Partial<SparkKeyLabels>;
  className?: string;
}) {
  const text = { ...DEFAULT_KEY, ...labels };
  return (
    <span className={cn("flex items-center gap-2 text-[10px] leading-none whitespace-nowrap text-muted-foreground", className)}>
      {measured && (
        <span className="flex items-center gap-1">
          <span
            aria-hidden
            className={cn("block h-[1.5px] w-3 rounded-full", tone === "brand" ? "bg-sentinel" : "bg-status-actual")}
          />
          {text.measured}
        </span>
      )}
      <span className="flex items-center gap-1">
        <span aria-hidden className="block w-3 border-t-[1.5px] border-dashed border-status-forecast" />
        {text.forecast}
      </span>
      {range && (
        <span className="flex items-center gap-1">
          <span aria-hidden className="block h-2 w-3 rounded-[2px] bg-status-forecast/15" />
          {text.range}
        </span>
      )}
      {limit && (
        <span className="flex items-center gap-1">
          <span aria-hidden className="block w-3 border-t border-dashed border-status-critical" />
          {text.limit}
        </span>
      )}
    </span>
  );
}
