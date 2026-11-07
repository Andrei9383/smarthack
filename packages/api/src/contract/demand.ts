import { oc } from "@orpc/contract";
import { z } from "zod";

export const zoneDemandItemSchema = z.object({
  zoneId: z.string(),
  level: z.string(),
  zoneName: z.string(),
  ts: z.string(),
  q10Mw: z.number().nullable().optional(),
  q50Mw: z.number(),
  q90Mw: z.number().nullable().optional(),
  peakStress: z.number(),
  isPeakHour: z.number(),
  label: z.string(),
});
export type ZoneDemandItem = z.infer<typeof zoneDemandItemSchema>;

export const stressedFeederItemSchema = z.object({
  feederId: z.string(),
  name: z.string(),
  county: z.string(),
  peakStress: z.number(),
  slackKw: z.number(),
});
export type StressedFeederItem = z.infer<typeof stressedFeederItemSchema>;

export const demandContract = {
  getZones: oc
    .input(
      z
        .object({
          level: z.string().optional(),
          runId: z.string().optional(),
          ts: z.string().optional(),
        })
        .optional(),
    )
    .output(z.array(zoneDemandItemSchema)),
  getTopStressed: oc
    .input(
      z
        .object({
          runId: z.string().optional(),
          limit: z.number().optional(),
        })
        .optional(),
    )
    .output(z.array(stressedFeederItemSchema)),
};
