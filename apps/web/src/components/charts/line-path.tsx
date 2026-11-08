import { type CurveFactory, line } from "d3-shape";

import { fastShapePath } from "./fast-path";

/** `d` of a d3 line through `data` (an empty result is `""`). */
export function linePathD<Datum>(
  data: Datum[],
  { x, y, curve }: { x: (datum: Datum) => number; y: (datum: Datum) => number; curve: CurveFactory },
): string {
  return fastShapePath(line<Datum>().x(x).y(y).curve(curve), data);
}
