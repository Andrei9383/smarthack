import { oc } from "@orpc/contract";
import { z } from "zod";

const runItemSchema = z.object({
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

const kpiItemSchema = z.object({
  runId: z.string(),
  policy: z.string(),
  name: z.string(),
  value: z.number(),
  unit: z.string().nullable().optional(),
  label: z.string(),
});

const overviewSchema = z.object({
  homes: z.number(),
  feeders: z.number(),
  turbines: z.number(),
  parks: z.number(),
  productionMwh: z.number().nullable(),
  avgWindKmh: z.number().nullable(),
  availabilityPct: z.number().nullable(),
  activeIssues: z.number().nullable(),
});

export const manifestContract = {
  // KPI strip for one replay window; from/to are ISO-8601 UTC, `now` is the replay time.
  getOverview: oc
    .input(z.object({ from: z.string(), to: z.string(), now: z.string() }))
    .output(overviewSchema),
  getRuns: oc
    .input(z.object({}).optional())
    .output(z.array(runItemSchema)),
  getKpis: oc
    .input(z.object({ runId: z.string().optional() }).optional())
    .output(z.array(kpiItemSchema)),
};
