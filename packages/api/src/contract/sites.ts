import { oc } from "@orpc/contract";
import { z } from "zod";

export const siteDetailSchema = z.object({
  id: z.string(),
  name: z.string(),
  assetKind: z.string(),
  county: z.string(),
  locality: z.string().nullable().optional(),
  ratedKw: z.number(),
  hasBattery: z.number(),
  batteryKwh: z.number().nullable().optional(),
  status: z.string(),
  reason: z.string().optional(),
  gisTrust: z.enum(["ok", "low"]).nullable().optional(),
});
export type SiteDetail = z.infer<typeof siteDetailSchema>;

export const siteTelemetryPointSchema = z.object({
  ts: z.string(),
  pvAcKw: z.number().nullable().optional(),
  loadKw: z.number().nullable().optional(),
  battKw: z.number().nullable().optional(),
  soc: z.number().nullable().optional(),
});
export type SiteTelemetryPoint = z.infer<typeof siteTelemetryPointSchema>;

export const assetStepPointSchema = z.object({
  ts: z.string(),
  actualKw: z.number().nullable().optional(),
  q10Kw: z.number().nullable().optional(),
  q50Kw: z.number().nullable().optional(),
  q90Kw: z.number().nullable().optional(),
});
export type AssetStepPoint = z.infer<typeof assetStepPointSchema>;

export const sitePredictionPointSchema = z.object({
  ts: z.string(),
  t0: z.string(),
  horizonIdx: z.number(),
  model: z.string(),
  modelVersion: z.string(),
  q10: z.number().nullable().optional(),
  q50: z.number().nullable().optional(),
  q90: z.number().nullable().optional(),
  faultProbsJson: z.string().nullable().optional(),
  anomalyClass: z.string().nullable().optional(),
  oodScore: z.number().nullable().optional(),
  memberStd: z.number().nullable().optional(),
  isFallback: z.number(),
});
export type SitePredictionPoint = z.infer<typeof sitePredictionPointSchema>;

export const sitesContract = {
  getDetail: oc
    .input(z.object({ assetId: z.string(), runId: z.string().optional() }))
    .output(siteDetailSchema),
  getTelemetry: oc
    .input(z.object({ assetId: z.string(), runId: z.string().optional() }))
    .output(z.array(siteTelemetryPointSchema)),
  getSteps: oc
    .input(z.object({ assetId: z.string(), runId: z.string().optional() }))
    .output(z.array(assetStepPointSchema)),
  getPredictions: oc
    .input(z.object({ assetId: z.string(), runId: z.string().optional(), model: z.string().optional() }))
    .output(z.array(sitePredictionPointSchema)),
};
