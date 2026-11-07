import { oc } from "@orpc/contract";
import { z } from "zod";

export const feederSummarySchema = z.object({
  id: z.string(),
  name: z.string(),
  county: z.string(),
  locality: z.string(),
  limitKw: z.number(),
  installedPvKw: z.number(),
  installedBattKwh: z.number(),
  gisTrust: z.string().default("high"),
});
export type FeederSummary = z.infer<typeof feederSummarySchema>;

export const feederCurvePointSchema = z.object({
  t: z.number(),
  step: z.number(),
  exportKw: z.number(),
  importKw: z.number().default(0),
  limitKw: z.number(),
  slackKw: z.number(),
  policy: z.string(),
});
export type FeederCurvePoint = z.infer<typeof feederCurvePointSchema>;

export const commitmentPointSchema = z.object({
  t: z.number(),
  step: z.number(),
  nominalKw: z.number(),
  availableKw: z.number(),
  committedKw: z.number(),
  realisedKw: z.number().nullable().optional(),
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
