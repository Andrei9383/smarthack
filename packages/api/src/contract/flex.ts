import { oc } from "@orpc/contract";
import { z } from "zod";

const flexOverviewSchema = z.object({
  totalBatteries: z.number(),
  totalKwh: z.number(),
  totalKw: z.number(),
  dispatchedKw: z.number(),
  availableKw: z.number(),
});

const feederFlexItemSchema = z.object({
  feederId: z.string(),
  flexibleHeadroomKw: z.number(),
  eligibleSites: z.number(),
  deratedSites: z.number(),
  excludedSites: z.number(),
});
export type FeederFlexItem = z.infer<typeof feederFlexItemSchema>;

// Hourly, because a DSO commitment is bought per hour; each value is the mean of the four 15-min steps.
const hourlyCommitmentSchema = z.object({
  hour: z.string(), // ISO-8601 UTC, start of the hour
  nominalKw: z.number(),
  availableKw: z.number(),
  committedKw: z.number(),
  realisedKw: z.number().nullable(), // null outside the hours the DSO buys
  hitRate: z.number().nullable(), // share of the hour's steps where realised >= committed
});

const runInput = z.object({ runId: z.string().optional() }).optional();

export const flexContract = {
  getOverview: oc.input(runInput).output(flexOverviewSchema),
  getFeedersFlex: oc.input(runInput).output(z.array(feederFlexItemSchema)),
  // One feeder, or the whole fleet when feederId is omitted.
  getCommitments: oc
    .input(z.object({ feederId: z.string().optional(), runId: z.string().optional() }).optional())
    .output(z.array(hourlyCommitmentSchema)),
};
