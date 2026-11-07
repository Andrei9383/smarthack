import { oc } from "@orpc/contract";
import { z } from "zod";

export const flexOverviewSchema = z.object({
  totalBatteries: z.number(),
  totalKwh: z.number(),
  totalKw: z.number(),
  dispatchedKw: z.number(),
  availableKw: z.number(),
});
export type FlexOverview = z.infer<typeof flexOverviewSchema>;

export const feederFlexItemSchema = z.object({
  feederId: z.string(),
  flexibleHeadroomKw: z.number(),
  eligibleSites: z.number(),
  deratedSites: z.number(),
  excludedSites: z.number(),
});
export type FeederFlexItem = z.infer<typeof feederFlexItemSchema>;

export const flexContract = {
  getOverview: oc
    .input(z.object({ runId: z.string().optional() }).optional())
    .output(flexOverviewSchema),
  getFeedersFlex: oc
    .input(z.object({ runId: z.string().optional() }).optional())
    .output(z.array(feederFlexItemSchema)),
};
