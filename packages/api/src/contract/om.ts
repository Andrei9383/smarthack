import { oc } from "@orpc/contract";
import { z } from "zod";

export const assetListItemSchema = z.object({
  id: z.string(),
  name: z.string(),
  assetKind: z.string(),
  status: z.string(),
  headroomFactor: z.number(),
  reason: z.string(),
  county: z.string(),
  locality: z.string().nullable().optional(),
  gisTrust: z.enum(["ok", "low"]).nullable().optional(),
  // #32: the O&M map needs a position and an owner for every marker
  lat: z.number().optional(),
  lon: z.number().optional(),
  feederId: z.string().nullable().optional(),
  parkId: z.string().nullable().optional(),
  ratedKw: z.number().optional(),
});
export type AssetListItem = z.infer<typeof assetListItemSchema>;

export const alertItemSchema = z.object({
  actionId: z.string(),
  assetId: z.string().nullable().optional(),
  parkId: z.string().nullable().optional(),
  feederId: z.string().nullable().optional(),
  severity: z.string(),
  code: z.string(),
  message: z.string(),
  ts: z.string(),
  createsWorkOrder: z.number(),
  curtailedMwh: z.number().nullable().optional(),
  evidenceJson: z.string().nullable().optional(),
});
export type AlertItem = z.infer<typeof alertItemSchema>;

export const workOrderItemSchema = z.object({
  actionId: z.string(),
  assetId: z.string(),
  assetKind: z.string(),
  crewType: z.string(),
  action: z.string(),
  status: z.string(),
  score: z.number(),
  day: z.string(),
  rank: z.number().nullable().optional(),
  ts: z.string(),
  fault: z.string().nullable().optional(),
  confidence: z.number(),
  lostKwhDay: z.number().nullable().optional(),
  lostFlexMwhDay: z.number().nullable().optional(),
  lostLeiDay: z.number().nullable().optional(),
  mwAtRisk: z.number().nullable().optional(),
  daysToVisit: z.number().nullable().optional(),
  evidenceJson: z.string().nullable().optional(),
});
export type WorkOrderItem = z.infer<typeof workOrderItemSchema>;

export const turbineDrawerSchema = z.object({
  assetId: z.string(),
  name: z.string(),
  parkId: z.string().nullable().optional(),
  ratedKw: z.number().nullable().optional(),
  turbineModel: z.string().nullable().optional(),
  powerCurve: z.array(
    z.object({
      wdSpdKmh: z.number(),
      actPwtKw: z.number(),
      expectedPwtKw: z.number().nullable().optional(),
    }),
  ),
  tempTrends: z.array(
    z.object({
      ts: z.string(),
      ambientTempC: z.number().nullable().optional(), // #32: residual = component - ambient
      convTempC: z.number().nullable().optional(),
      gearboxTempC: z.number().nullable().optional(),
      genTempC: z.number().nullable().optional(),
      transformerTempC: z.number().nullable().optional(),
    }),
  ),
});
export type TurbineDrawerData = z.infer<typeof turbineDrawerSchema>;

export const parkReportItemSchema = z.object({
  actionId: z.string(),
  parkId: z.string().nullable().optional(),
  date: z.string(),
  energyMwh: z.number().nullable().optional(),
  expectedMwh: z.number().nullable().optional(),
  availabilityPct: z.number().nullable().optional(),
  curtailedMwh: z.number().nullable().optional(),
  dataQualityPct: z.number().nullable().optional(),
  issuesJson: z.string().nullable().optional(),
});
export type ParkReportItem = z.infer<typeof parkReportItemSchema>;

export const reviewItemSchema = z.object({
  actionId: z.string(),
  assetId: z.string(),
  ts: z.string(),
  reason: z.string(),
  oodScore: z.number().nullable().optional(),
  intervalWidth: z.number().nullable().optional(),
  memberStd: z.number().nullable().optional(),
  resolved: z.number(),
});
export type ReviewItem = z.infer<typeof reviewItemSchema>;

export const omContract = {
  listAssets: oc
    .input(
      z
        .object({
          status: z.string().optional(),
          assetKind: z.string().optional(),
          runId: z.string().optional(),
          at: z.string().optional(), // status as of this ts (default: newest step)
        })
        .optional(),
    )
    .output(z.array(assetListItemSchema)),
  listAlerts: oc
    .input(
      z
        .object({
          runId: z.string().optional(),
          severity: z.string().optional(),
          parkId: z.string().optional(),
          until: z.string().optional(), // only rows with ts <= until
        })
        .optional(),
    )
    .output(z.array(alertItemSchema)),
  listWorkOrders: oc
    .input(
      z
        .object({
          runId: z.string().optional(),
          status: z.string().optional(),
          crewType: z.string().optional(),
          assetId: z.string().optional(),
          until: z.string().optional(),
        })
        .optional(),
    )
    .output(z.array(workOrderItemSchema)),
  listReviewQueue: oc
    .input(
      z
        .object({
          runId: z.string().optional(),
          resolved: z.boolean().optional(),
          until: z.string().optional(),
        })
        .optional(),
    )
    .output(z.array(reviewItemSchema)),
  getTurbineDrawer: oc
    .input(z.object({ assetId: z.string(), runId: z.string().optional() }))
    .output(turbineDrawerSchema),
  listReports: oc
    .input(
      z
        .object({
          parkId: z.string().optional(),
          runId: z.string().optional(),
        })
        .optional(),
    )
    .output(z.array(parkReportItemSchema)),
};
