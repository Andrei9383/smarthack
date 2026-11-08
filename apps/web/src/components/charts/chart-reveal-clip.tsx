"use client";

import { m, type Transition } from "motion/react";

/**
 * Left-to-right clip reveal: the clip rect width grows 0 → full (scaleX would reveal from the
 * centre). Render inside `<defs>` and clip the series with `url(#clipPathId)`.
 */
export function ChartRevealClip({
  clipPathId,
  height,
  targetWidth,
  enterTransition,
  revealEpoch,
  padding = 0,
  paddingY = padding,
  animating = true,
}: {
  clipPathId: string;
  height: number;
  targetWidth: number;
  /** A tween: spring does not reliably animate SVG width. */
  enterTransition: Transition & { type: "tween" };
  /** Bump to replay the reveal. */
  revealEpoch: number;
  /** Inset around the clip rect so edge glyphs are not cut off. */
  padding?: number;
  /** Vertical inset. Default: `padding`. */
  paddingY?: number;
  /** When false the clip stays at full width. */
  animating?: boolean;
}) {
  const width = Math.max(0, targetWidth + padding * 2);
  const rect = { height: height + paddingY * 2, x: -padding, y: -paddingY };
  return (
    <clipPath id={clipPathId}>
      {animating ? (
        <m.rect
          animate={{ width }}
          initial={{ width: 0 }}
          key={revealEpoch}
          transition={enterTransition}
          width={width}
          {...rect}
        />
      ) : (
        <rect width={width} {...rect} />
      )}
    </clipPath>
  );
}
