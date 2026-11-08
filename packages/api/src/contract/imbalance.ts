import { oc } from "@orpc/contract";
import { z } from "zod";

// Imbalance replay tables written by wattsentry/eval/imbalance_replay.py; MWh per 15-min step, lei.
const num = z.number().nullable();

const imbalanceStepSchema = z.object({
  ts: z.string(),
  policy: z.string(),
  p10: num, // the policy's day-ahead band (NULL for point policies)
  p50: num,
  p90: num,
  bought: num, // bought_da + traded_id
  actual: num,
  devMwh: num,
  avoidableLei: num,
});

const policyTotalsSchema = z.object({
  policy: z.string(),
  avoidableLei: z.number(),
  intradayLei: z.number(),
  imbalanceLei: z.number(),
  absDevMwh: z.number(),
  exportMwh: z.number(),
  leiPerMwhInjected: num,
  doubleCover1821Mwh: z.number(), // bought beyond need, 18:00-21:00 Bucharest
  days: z.number(),
});
export type PolicyTotals = z.infer<typeof policyTotalsSchema>;

const strategyRowSchema = z.object({ strategy: z.string(), homes: z.number(), confidence: num });

const windowInput = z.object({ from: z.string().optional(), to: z.string().optional() });

export const imbalanceContract = {
  // from inclusive, to exclusive (ISO UTC). If that day was not replayed, the nearest replayed day at the same
  // clock offset is returned instead (from/to in the output say which).
  getImbalanceDay: oc
    .input(windowInput.extend({ policy: z.string().optional() }))
    .output(
      z.object({
        from: z.string().nullable(),
        to: z.string().nullable(),
        steps: z.array(imbalanceStepSchema),
        totals: z.array(policyTotalsSchema),
      }),
    ),
  // Totals per policy over the window (whole replay when omitted): the waterfall.
  getPolicyComparison: oc.input(windowInput.optional()).output(z.array(policyTotalsSchema)),
  // Homes per battery strategy (site_strategy); falls back to assets.has_battery when not classified.
  getStrategyBreakdown: oc
    .input(z.object({}).optional())
    .output(z.object({ source: z.enum(["site_strategy", "assets", "none"]), rows: z.array(strategyRowSchema) })),
};
