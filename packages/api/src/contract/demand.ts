import { oc } from "@orpc/contract";
import { z } from "zod";

export const zoneDemandItemSchema = z.object({
  zoneId: z.string(),
  level: z.string(),
  zoneName: z.string(),
  parentZoneId: z.string().nullable().optional(),
  ts: z.string(),
  q10Mw: z.number().nullable().optional(),
  q50Mw: z.number(),
  q90Mw: z.number().nullable().optional(),
  peakStress: z.number(),
  isPeakHour: z.number(),
  label: z.string(),
});
export type ZoneDemandItem = z.infer<typeof zoneDemandItemSchema>;

// Fixtures store peak_stress per step, replay per zone per day. Compare days, or derive per-step
// stress as max(peak_stress) * q90 / max(q90) over the day (modes/demand/data.ts).
export const demandContract = {
  // from inclusive, to exclusive; ISO-8601 UTC. Omit both for the whole run.
  getZones: oc
    .input(
      z
        .object({
          level: z.string().optional(),
          runId: z.string().optional(),
          from: z.string().optional(),
          to: z.string().optional(),
        })
        .optional(),
    )
    .output(z.array(zoneDemandItemSchema)),
};
