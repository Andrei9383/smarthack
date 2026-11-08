"use client";

import { chartCssVars, useChartStable, useYScale } from "./chart-context";

interface GridProps {
  /** Number of horizontal grid lines (`yScale.ticks(n)`). Default: 5 */
  numTicksRows?: number;
  /** Explicit tick values for the rows. Overrides numTicksRows. */
  rowTickValues?: number[];
  /** Default: "4,4" */
  strokeDasharray?: string;
  /** Y-scale for the rows. Default: primary (`"left"`) axis. */
  yAxisId?: string | number;
}

/** Horizontal grid lines at the y-scale ticks. */
export function Grid({ numTicksRows = 5, rowTickValues, strokeDasharray = "4,4", yAxisId }: GridProps) {
  const { innerWidth } = useChartStable();
  const yScale = useYScale(yAxisId);
  const ticks = rowTickValues?.length ? rowTickValues : yScale.ticks(numTicksRows);
  return (
    <g className="chart-grid">
      <g>
        {ticks.map((tick, index) => {
          const y = yScale(tick) ?? 0;
          return (
            <line
              fill="transparent"
              key={`row-line-${index}`}
              shapeRendering="crispEdges"
              stroke={chartCssVars.grid}
              strokeDasharray={strokeDasharray}
              strokeOpacity={1}
              strokeWidth={1}
              x1={0}
              x2={innerWidth}
              y1={y}
              y2={y}
            />
          );
        })}
      </g>
    </g>
  );
}

Grid.displayName = "Grid";
