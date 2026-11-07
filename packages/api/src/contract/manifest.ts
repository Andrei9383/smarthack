import { oc } from "@orpc/contract";
import { z } from "zod";

export const runItemSchema = z.object({
  id: z.string(),
  kind: z.string(),
  policy: z.string().nullable().optional(),
  label: z.string(),
  weekStart: z.string().nullable().optional(),
  gitSha: z.string().nullable().optional(),
  modelVersionsJson: z.string().nullable().optional(),
  createdAt: z.string(),
  finishedAt: z.string().nullable().optional(),
  notes: z.string().nullable().optional(),
});
export type RunItem = z.infer<typeof runItemSchema>;

export const kpiItemSchema = z.object({
  runId: z.string(),
  policy: z.string(),
  name: z.string(),
  value: z.number(),
  unit: z.string().nullable().optional(),
  label: z.string(),
});
export type KpiItem = z.infer<typeof kpiItemSchema>;

export const manifestContract = {
  getRuns: oc
    .input(z.object({}).optional())
    .output(z.array(runItemSchema)),
  getKpis: oc
    .input(z.object({ runId: z.string().optional() }).optional())
    .output(z.array(kpiItemSchema)),
};
