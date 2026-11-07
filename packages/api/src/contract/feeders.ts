import { oc } from "@orpc/contract";
import { z } from "zod";

export const feederSummarySchema = z.object({
  id: z.string(),
  name: z.string(),
  county: z.string(),
  locality: z.string(),
  limitKw: z.number(),
  gMaxKw: z.number().nullable().optional(),
  gConservativeKw: z.number().nullable().optional(),
  tau: z.number().nullable().optional(),
  gisTrust: z.enum(["ok", "low"]).nullable().optional(),
  geometryJson: z.string().nullable().optional(),
});
export type FeederSummary = z.infer<typeof feederSummarySchema>;

export const feederCurvePointSchema = z.object({
  ts: z.string(),
  exportKw: z.number(),
  importKw: z.number().default(0),
  limitKw: z.number(),
  slackKw: z.number(),
  policy: z.string(),
  isViolation: z.number().default(0),
});
export type FeederCurvePoint = z.infer<typeof feederCurvePointSchema>;

export const commitmentPointSchema = z.object({
  ts: z.string(),
  nominalKw: z.number(),
  availableKw: z.number(),
  committedKw: z.number(),
  realisedKw: z.number().nullable().optional(),
  hitP10: z.number().nullable().optional(),
});
export type CommitmentPoint = z.infer<typeof commitmentPointSchema>;

export const feedersContract = {
  list: oc
    .input(z.object({ runId: z.string().optional() }).optional())
    .output(z.array(feederSummarySchema)),
  getCurves: oc
    .input(z.object({ feederId: z.string(), runId: z.string().optional() }))
    .output(z.array(feederCurvePointSchema)),
  getCommitments: oc
    .input(z.object({ feederId: z.string(), runId: z.string().optional() }))
    .output(z.array(commitmentPointSchema)),
};
