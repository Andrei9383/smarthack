import type { Line } from "d3-shape";

/**
 * Formats like d3-path's `appendRound(3)`: `String(Math.round(v * 1000) / 1000)`.
 * Builds the decimal from the rounded integer instead of formatting a
 * fractional double, which is the bulk of the cost of a long path `d`.
 * For |v| below 1e9 the result is identical: a decimal with at most 12
 * significant digits is the shortest round-trip form of its double.
 */
function formatCoordinate(value: number): string {
  const scaled = Math.round(value * 1000);
  if (!(scaled > -1e12 && scaled < 1e12)) {
    // NaN, ±Infinity and huge values: keep d3's exact formatting.
    return String(scaled / 1000);
  }
  if (scaled === 0) {
    return "0";
  }
  const negative = scaled < 0;
  const abs = negative ? -scaled : scaled;
  const whole = Math.floor(abs / 1000);
  const fraction = abs - whole * 1000;
  const head = negative ? `-${whole}` : `${whole}`;
  if (fraction === 0) {
    return head;
  }
  if (fraction % 100 === 0) {
    return `${head}.${fraction / 100}`;
  }
  if (fraction % 10 === 0) {
    return `${head}.${fraction < 100 ? "0" : ""}${fraction / 10}`;
  }
  let pad = "";
  if (fraction < 10) {
    pad = "00";
  } else if (fraction < 100) {
    pad = "0";
  }
  return `${head}.${pad}${fraction}`;
}

/**
 * Path context for d3 line generators that emits the same `d` string as d3-path with 3 digits
 * (d3-shape's default), but faster. Line curves only call moveTo / lineTo / bezierCurveTo /
 * quadraticCurveTo / closePath.
 */
class FastPath {
  d = "";
  private open = false;

  moveTo(x: number, y: number) {
    this.d += `M${formatCoordinate(+x)},${formatCoordinate(+y)}`;
    this.open = true;
  }

  lineTo(x: number, y: number) {
    this.d += `L${formatCoordinate(+x)},${formatCoordinate(+y)}`;
    this.open = true;
  }

  closePath() {
    if (this.open) {
      this.d += "Z";
    }
  }

  quadraticCurveTo(x1: number, y1: number, x: number, y: number) {
    this.d += `Q${formatCoordinate(+x1)},${formatCoordinate(+y1)},${formatCoordinate(+x)},${formatCoordinate(+y)}`;
    this.open = true;
  }

  bezierCurveTo(
    x1: number,
    y1: number,
    x2: number,
    y2: number,
    x: number,
    y: number
  ) {
    this.d += `C${formatCoordinate(+x1)},${formatCoordinate(+y1)},${formatCoordinate(+x2)},${formatCoordinate(+y2)},${formatCoordinate(+x)},${formatCoordinate(+y)}`;
    this.open = true;
  }
}

/**
 * `generator(data) || ""` for a d3 `line()` that uses the default 3-digit rounding and no context
 * of its own: same string, built faster.
 */
export function fastShapePath<Datum>(generator: Line<Datum>, data: Datum[]): string {
  const context = new FastPath();
  // FastPath implements the subset of CanvasRenderingContext2D that d3 curves call.
  generator.context(context as unknown as CanvasRenderingContext2D);
  generator(data);
  generator.context(null);
  return context.d;
}
