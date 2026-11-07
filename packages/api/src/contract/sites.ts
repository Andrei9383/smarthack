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
});
export type SiteDetail = z.infer<typeof siteDetailSchema>;

export const siteTelemetryPointSchema = z.object({
  ts: z.number(),
  step: z.number(),
  pvAcKw: z.number().nullable().optional(),
  loadKw: z.number().nullable().optional(),
  battKw: z.number().nullable().optional(),
  soc: z.number().nullable().optional(),
});
export type SiteTelemetryPoint = z.infer<typeof siteTelemetryPointSchema>;

export const sitePredictionPointSchema = z.object({
  ts: z.number(),
  step: z.number(),
  q10Kw: z.number(),
  q50Kw: z.number(),
  q90Kw: z.number(),
  anomalyClass: z.string().nullable().optional(),
});
export type SitePredictionPoint = z.infer<typeof sitePredictionPointSchema>;

export const sitesContract = {
  getDetail: oc
    .input(z.object({ assetId: z.string(), runId: z.string().optional() }))
    .output(siteDetailSchema),
  getTelemetry: oc
    .input(z.object({ assetId: z.string(), runId: z.string().optional() }))
    .output(z.array(siteTelemetryPointSchema)),
  getPredictions: oc
    .input(z.object({ assetId: z.string(), runId: z.string().optional() }))
    .output(z.array(sitePredictionPointSchema)),
};
