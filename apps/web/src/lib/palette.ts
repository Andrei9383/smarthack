// RGB mirrors of the --status-* and brand tokens in packages/ui/src/styles/globals.css, for
// deck.gl and canvas code that can't read CSS variables. Keep the two in step.
export type Rgb = [number, number, number];

export const PALETTE = {
  brand50: [236, 251, 242],
  brand500: [18, 183, 106],
  primary: [14, 159, 90],
  brand700: [11, 127, 72],
  statusOk: [14, 159, 90],
  statusWarn: [217, 119, 6],
  statusCritical: [229, 72, 77],
  statusForecast: [2, 132, 199],
  statusSoc: [99, 102, 241],
  statusReview: [139, 92, 246],
  statusUnknown: [148, 163, 184],
  statusActual: [13, 13, 13],
} as const satisfies Record<string, Rgb>;
