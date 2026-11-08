import { oc } from "@orpc/contract";
import { z } from "zod";

import { workOrderItemSchema } from "./om";

const siteDetailSchema = z.object({
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

const assetStepPointSchema = z.object({
  ts: z.string(),
  actualKw: z.number().nullable().optional(),
  q10Kw: z.number().nullable().optional(),
  q50Kw: z.number().nullable().optional(),
  q90Kw: z.number().nullable().optional(),
});

const sitePredictionPointSchema = z.object({
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

// Contiguous rows with the same status and reason, so the timeline is a handful of segments, not 672 rows.
const statusSegmentSchema = z.object({
  from: z.string(),
  to: z.string(),
  status: z.string(),
  reason: z.string(),
  headroomFactor: z.number(),
});
export type StatusSegment = z.infer<typeof statusSegmentSchema>;

const schedulePointSchema = z.object({
  ts: z.string(),
  battKw: z.number(),
  soc: z.number(),
  comfortReserveKwh: z.number().nullable(),
  comfortBreached: z.number(),
  clipped: z.number(),
});

const siteInput = z.object({ assetId: z.string(), runId: z.string().optional() });

export const sitesContract = {
  getDetail: oc.input(siteInput).output(siteDetailSchema),
  getSteps: oc.input(siteInput).output(z.array(assetStepPointSchema)),
  getPredictions: oc
    .input(siteInput.extend({ model: z.string().optional() }))
    .output(z.array(sitePredictionPointSchema)),
  getStatuses: oc.input(siteInput).output(z.array(statusSegmentSchema)),
  getSchedule: oc.input(siteInput).output(z.array(schedulePointSchema)),
  getWorkOrders: oc.input(siteInput).output(z.array(workOrderItemSchema)),
};
