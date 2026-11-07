"use client";

import type { ReactNode } from "react";
import { Circles as PatternCircles } from "./patterns/circles";
import { Lines as PatternLines } from "./patterns/lines";
import type { PatternOrientationType } from "./patterns/constants";

export const PATTERN_PRESET_IDS = [
  "none",
  "diagonal",
  "horizontal",
  "vertical",
  "cross",
  "dots",
  "circles",
  "accent",
] as const;

export type PatternPresetId = (typeof PATTERN_PRESET_IDS)[number];

export interface PatternPresetOptions {
  color?: string;
  scale?: number;
  strokeWidth?: number;
  radius?: number;
  complement?: boolean;
  fill?: string;
  /** Dot grid only — when false, render hollow dots (stroke only). Default: true */
  dotFill?: boolean;
  tileBackground?: string;
}

export function patternPresetTileSize(
  preset: PatternPresetId,
  scale = 1
): { width: number; height: number; strokeWidth: number } {
  let base = { width: 6, height: 6, strokeWidth: 1 };
  if (preset === "dots") {
    base = { width: 10, height: 10, strokeWidth: 0 };
  } else if (preset === "cross") {
    base = { width: 8, height: 8, strokeWidth: 1 };
  } else if (preset === "circles") {
    base = { width: 6, height: 6, strokeWidth: 1 };
  }

  return {
    width: base.width * scale,
    height: base.height * scale,
    strokeWidth: base.strokeWidth * scale,
  };
}

function renderPatternCircles(
  preset: "dots" | "circles",
  _id: string,
  color: string,
  common: {
    id: string;
    height: number;
    width: number;
    strokeWidth: number;
    background?: string;
  },
  options: PatternPresetOptions,
  scale: number
) {
  const isDotGrid = preset === "dots";
  const radius =
    options.radius ?? (isDotGrid ? Math.max(0.5, 1.5 * scale) : 2 * scale);
  const dotFillEnabled = options.dotFill !== false;

  if (isDotGrid) {
    const dotFill = dotFillEnabled ? options.fill || color : undefined;
    return (
      <PatternCircles
        {...common}
        complement={options.complement}
        fill={dotFill}
        radius={radius}
        stroke={dotFillEnabled && options.fill ? undefined : color}
        strokeWidth={
          dotFillEnabled && !options.fill
            ? (options.strokeWidth ?? 0)
            : (options.strokeWidth ?? 1)
        }
      />
    );
  }

  return (
    <PatternCircles
      {...common}
      complement={options.complement}
      fill={options.fill || undefined}
      radius={radius}
      stroke={color}
      strokeWidth={options.strokeWidth ?? common.strokeWidth}
    />
  );
}

const LINE_ORIENTATIONS: Record<
  Exclude<PatternPresetId, "none" | "dots" | "circles">,
  PatternOrientationType[]
> = {
  diagonal: ["diagonal"],
  horizontal: ["horizontal"],
  vertical: ["vertical"],
  cross: ["diagonal", "diagonalRightToLeft"],
  accent: ["diagonal"],
};

/** Renders a pattern definition node for the given preset. */
export function renderPatternPreset(
  preset: PatternPresetId,
  id: string,
  options: PatternPresetOptions = {}
): ReactNode {
  if (preset === "none") {
    return null;
  }

  const color = options.color ?? "var(--chart-1)";
  const scale = options.scale ?? 1;
  const tile = patternPresetTileSize(preset, scale);
  const common = {
    id,
    height: tile.height,
    width: tile.width,
    strokeWidth: tile.strokeWidth,
    ...(options.tileBackground ? { background: options.tileBackground } : {}),
  };

  if (preset === "dots" || preset === "circles") {
    return renderPatternCircles(preset, id, color, common, options, scale);
  }

  const strokeWidth = options.strokeWidth ?? tile.strokeWidth;

  const orientation = LINE_ORIENTATIONS[preset];
  return (
    <PatternLines
      {...common}
      orientation={orientation}
      stroke={preset === "accent" ? "var(--chart-line-secondary)" : color}
      strokeWidth={strokeWidth}
    />
  );
}
