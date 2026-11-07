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
  curtailedMwh: z.number().nullable().optional(),
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
  lostLeiDay: z.number().nullable().optional(),
});
export type WorkOrderItem = z.infer<typeof workOrderItemSchema>;

export const turbineDrawerSchema = z.object({
  assetId: z.string(),
  name: z.string(),
  powerCurve: z.array(
    z.object({
      wdSpd: z.number(),
      actPwt: z.number(),
      expectedPwt: z.number().optional(),
    }),
  ),
  tempTrends: z.array(
    z.object({
      ts: z.string(),
      convTemp: z.number().optional(),
      gearboxTemp: z.number().optional(),
      genTemp: z.number().optional(),
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
});
export type ParkReportItem = z.infer<typeof parkReportItemSchema>;

export const omContract = {
  listAssets: oc
    .input(
      z
        .object({
          status: z.string().optional(),
          assetKind: z.string().optional(),
          runId: z.string().optional(),
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
        })
        .optional(),
    )
    .output(z.array(workOrderItemSchema)),
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
