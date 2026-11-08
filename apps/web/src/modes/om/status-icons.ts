import type { Map as MapLibreMap } from "maplibre-gl";

import { OM_COLORS } from "./colors";

// The O&M map draws the same shape-coded status marks as the legend and the panels
// (components/status-mark.tsx): ok = LED, derated = triangle, stopped = diamond,
// waiting for a person = hexagon with a hollow core, left out / no data = hollow ring.
// MapLibre symbol layers need bitmaps, so each mark is painted once on a canvas and added as an
// image; the geometry mirrors StatusMark's 16-unit SVG viewBox.

const STATUS_ICON = {
  ok: "om-mark-ok",
  warn: "om-mark-warn",
  critical: "om-mark-critical",
  review: "om-mark-review",
  unknown: "om-mark-unknown",
} as const;

type Kind = keyof typeof STATUS_ICON;

const UNIT = 16; // StatusMark viewBox
const PAD = 4; // room for the contact shadow
const SCALE = 4; // device pixels per unit; addImage's pixelRatio keeps it crisp
const BOX = (UNIT + PAD * 2) * SCALE;

const COLOR: Record<Kind, string> = {
  ok: OM_COLORS.ok,
  warn: OM_COLORS.warn,
  critical: OM_COLORS.critical,
  review: OM_COLORS.review,
  unknown: OM_COLORS.unknown,
};

const TRIANGLE = "M8 1.6c.5 0 .9.3 1.2.7l5.6 9.7c.5.9-.1 2-1.2 2H2.4c-1 0-1.7-1.1-1.2-2l5.6-9.7c.3-.4.7-.7 1.2-.7Z";
const HEXAGON = "M8 1.3 13.8 4.7v6.6L8 14.7 2.2 11.3V4.7Z";

function paint(kind: Kind): ImageData | null {
  if (typeof document === "undefined") return null;
  const canvas = document.createElement("canvas");
  canvas.width = BOX;
  canvas.height = BOX;
  const ctx = canvas.getContext("2d");
  if (!ctx) return null;
  const color = COLOR[kind];
  ctx.scale(SCALE, SCALE);
  ctx.translate(PAD, PAD);

  const body = new Path2D();
  if (kind === "ok") body.arc(8, 8, 6.5, 0, Math.PI * 2);
  else if (kind === "warn") body.addPath(new Path2D(TRIANGLE));
  else if (kind === "review") body.addPath(new Path2D(HEXAGON));
  else if (kind === "unknown") body.arc(8, 8, 5.6, 0, Math.PI * 2);
  else {
    // critical: a 9.6-unit rounded square turned 45°
    const m = new DOMMatrix().translate(8, 8).rotate(45).translate(-8, -8);
    const square = new Path2D();
    square.roundRect(3.2, 3.2, 9.6, 9.6, 2);
    body.addPath(square, m);
  }

  // contact shadow lifts the mark off the light basemap
  ctx.save();
  ctx.shadowColor = OM_COLORS.markShadow;
  ctx.shadowBlur = 1.6 * SCALE;
  ctx.shadowOffsetY = 0.6 * SCALE;
  ctx.fillStyle = kind === "unknown" ? OM_COLORS.ring : color;
  ctx.fill(body);
  ctx.restore();

  ctx.lineJoin = "round";
  if (kind === "unknown") {
    ctx.strokeStyle = color;
    ctx.lineWidth = 2;
    ctx.stroke(body);
  } else {
    // domed fill: a soft light from the top left
    const dome = ctx.createRadialGradient(6, 4.8, 0, 6, 4.8, 9);
    dome.addColorStop(0, OM_COLORS.markLight);
    dome.addColorStop(0.5, OM_COLORS.markLightEnd);
    ctx.fillStyle = dome;
    ctx.fill(body);
    ctx.strokeStyle = OM_COLORS.ring;
    ctx.lineWidth = 1.3;
    ctx.stroke(body);
  }
  if (kind === "review") {
    ctx.fillStyle = OM_COLORS.ring;
    ctx.beginPath();
    ctx.arc(8, 8, 2.3, 0, Math.PI * 2);
    ctx.fill();
  }
  return ctx.getImageData(0, 0, BOX, BOX);
}

/** Add the five marks to the map's current style (no-op for any already there). */
export function addStatusIcons(map: MapLibreMap) {
  for (const [kind, name] of Object.entries(STATUS_ICON) as [Kind, string][]) {
    if (map.hasImage(name)) continue;
    const image = paint(kind);
    if (image) map.addImage(name, image, { pixelRatio: SCALE });
  }
}

/** Status property → icon name (stopped turbines carry status "stopped"). */
export const statusIcon: unknown[] = [
  "match",
  ["get", "status"],
  "eligible",
  STATUS_ICON.ok,
  "derated",
  STATUS_ICON.warn,
  "stopped",
  STATUS_ICON.critical,
  "human_review",
  STATUS_ICON.review,
  STATUS_ICON.unknown,
];
