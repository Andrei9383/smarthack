"use client";

// Live chart: measured export up to "now", the P10–P90 band and P50 ahead of it, and the
// past P50 predictions behind it so each tick's miss is visible. Plain SVG: the shared LineChart
// draws one gap-free series per key, and here every series stops or starts at "now".
import type { LivePoint, LiveView } from "@wattsentry/api/contract/live";
import { scaleLinear, scaleTime } from "d3-scale";
import { area, line } from "d3-shape";
import { useMemo } from "react";

import { hm } from "@/lib/utils";
import { ACTUAL_SERIES as ACTUAL, FORECAST_SERIES as FORECAST, LIMIT_SERIES } from "./colors";

const W = 400;
const H = 150;
const M = { left: 34, right: 8, top: 14, bottom: 20 };
const PAST_STEPS = 24; // six hours of 15-min ticks behind "now"

type XY = { t: Date; v: number };
const series = (points: LivePoint[], key: keyof LivePoint): XY[] =>
  points.flatMap((p) => {
    const v = p[key];
    return typeof v === "number" ? [{ t: new Date(p.ts), v }] : [];
  });

export function LiveChart({ view, limitKw }: { view: LiveView; limitKw: number }) {
  const past = view.past.slice(-PAST_STEPS);
  const geom = useMemo(() => {
    const all = [...past, ...view.ahead];
    if (all.length < 2) return null;
    const t0 = new Date(all[0]!.ts);
    const t1 = new Date(all.at(-1)!.ts);
    const values = all.flatMap((p) => [p.actualKw, p.q50Kw, p.q90Kw]).filter((v): v is number => v != null);
    const x = scaleTime().domain([t0, t1]).range([M.left, W - M.right]);
    const y = scaleLinear()
      .domain([Math.min(0, ...values), Math.max(limitKw, ...values) * 1.08])
      .nice()
      .range([H - M.bottom, M.top]);
    const toLine = line<XY>()
      .x((d) => x(d.t))
      .y((d) => y(d.v));
    const actual = series(past, "actualKw");
    const predictedPast = series(past, "q50Kw");
    // The forecast starts from the last reading, so the P50 reads as a continuation.
    const lastActual = actual.at(-1);
    const ahead50 = [...(lastActual ? [lastActual] : []), ...series(view.ahead, "q50Kw")];
    const bandRows = view.ahead.filter((p) => p.q10Kw != null && p.q90Kw != null);
    const band = area<LivePoint>()
      .x((p) => x(new Date(p.ts)))
      .y0((p) => y(p.q10Kw ?? 0))
      .y1((p) => y(p.q90Kw ?? 0));
    const nowX = x(new Date(view.nowTs));
    const next = view.next?.q50Kw != null ? { x: x(new Date(view.next.ts)), y: y(view.next.q50Kw) } : null;
    const xTicks = x.ticks(4);
    return {
      x,
      y,
      nowX,
      next,
      xTicks,
      yTicks: y.ticks(3),
      limitY: y(limitKw),
      actualPath: toLine(actual) ?? "",
      predictedPastPath: toLine(predictedPast) ?? "",
      aheadPath: toLine(ahead50) ?? "",
      bandPath: bandRows.length > 1 ? (band(bandRows) ?? "") : "",
    };
  }, [past, view, limitKw]);

  if (!geom) return null;
  const { y, x } = geom;
  return (
    <svg
      viewBox={`0 0 ${W} ${H}`}
      className="h-auto w-full overflow-visible"
      role="img"
      aria-label="Feeder export: measured up to now, forecast band ahead"
    >
      {geom.yTicks.map((v) => (
        <g key={v}>
          <line x1={M.left} x2={W - M.right} y1={y(v)} y2={y(v)} stroke="var(--chart-grid)" strokeWidth={1} />
          <text x={M.left - 6} y={y(v)} dy="0.32em" textAnchor="end" className="fill-muted-foreground text-caption tabular-nums">
            {v}
          </text>
        </g>
      ))}
      {geom.xTicks.map((t) => (
        <text
          key={t.getTime()}
          x={x(t)}
          y={H - M.bottom + 14}
          textAnchor="middle"
          className="fill-muted-foreground text-caption tabular-nums"
        >
          {hm(t)}
        </text>
      ))}
      <text x={M.left - 6} y={M.top - 6} textAnchor="end" className="fill-muted-foreground text-caption font-medium">
        kW
      </text>

      {/* ahead of now: shaded forecast area */}
      <rect
        x={geom.nowX}
        y={M.top}
        width={Math.max(0, W - M.right - geom.nowX)}
        height={H - M.top - M.bottom}
        fill="var(--muted)"
        opacity={0.35}
      />

      {geom.bandPath ? <path d={geom.bandPath} fill="var(--chart-band)" stroke={FORECAST} strokeOpacity={0.35} strokeWidth={1} /> : null}
      <path d={geom.predictedPastPath} fill="none" stroke={FORECAST} strokeWidth={1.25} strokeDasharray="3 3" opacity={0.7} />
      <path d={geom.actualPath} fill="none" stroke={ACTUAL} strokeWidth={2} strokeLinejoin="round" />
      <path d={geom.aheadPath} fill="none" stroke={FORECAST} strokeWidth={2} strokeDasharray="5 3" />

      <line x1={M.left} x2={W - M.right} y1={geom.limitY} y2={geom.limitY} stroke={LIMIT_SERIES} strokeWidth={1.5} strokeDasharray="6 4" />
      <text x={W - M.right} y={geom.limitY - 4} textAnchor="end" className="fill-status-critical text-caption font-semibold tabular-nums">
        {Math.round(limitKw)} kW limit
      </text>

      <line x1={geom.nowX} x2={geom.nowX} y1={M.top} y2={H - M.bottom} stroke="var(--foreground)" strokeOpacity={0.45} strokeDasharray="2 2" />
      <text x={geom.nowX} y={M.top - 4} textAnchor="middle" className="fill-foreground font-mono text-caption font-medium">
        now
      </text>
      {geom.next ? <circle cx={geom.next.x} cy={geom.next.y} r={3.5} fill={FORECAST} stroke="var(--card)" strokeWidth={1.5} /> : null}
    </svg>
  );
}
