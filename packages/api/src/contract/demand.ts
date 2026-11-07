import { oc } from "@orpc/contract";
import { z } from "zod";

export const zoneDemandItemSchema = z.object({
  zoneId: z.string(),
  zoneType: z.string(),
  zoneName: z.string(),
  p50Mw: z.number(),
  p10Mw: z.number().nullable().optional(),
  p90Mw: z.number().nullable().optional(),
  peakStress: z.number(),
  isPeakHour: z.number(),
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
          zoneType: z.string().optional(),
          runId: z.string().optional(),
          t: z.number().optional(),
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
