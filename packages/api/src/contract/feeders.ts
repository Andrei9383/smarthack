import { oc } from "@orpc/contract";
import { z } from "zod";

const feederSummarySchema = z.object({
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

const feederCurvePointSchema = z.object({
  ts: z.string(),
  exportKw: z.number(),
  importKw: z.number().default(0),
  limitKw: z.number(),
  slackKw: z.number(),
  policy: z.string(),
  isViolation: z.number().default(0),
});

const gisFlagItemSchema = z.object({
  actionId: z.string(),
  sectionId: z.string(),
  recordedFeeder: z.string().nullable().optional(),
  recordedPhase: z.string().nullable().optional(),
  suggestedFeeder: z.string().nullable().optional(),
  suggestedPhase: z.string().nullable().optional(),
  tau: z.number().nullable().optional(),
  geometryJson: z.string().nullable().optional(),
});

const windowInput = {
  from: z.string().optional(), // ISO-8601 UTC, inclusive
  to: z.string().optional(), // ISO-8601 UTC, exclusive
  runId: z.string().optional(),
};

const feederOverLimitSchema = z.object({
  feederId: z.string(),
  policy: z.string(),
  minutesOver: z.number(),
  peakExportKw: z.number(),
});

const socFanPointSchema = z.object({
  ts: z.string(),
  socQ10: z.number(),
  socQ50: z.number(),
  socQ90: z.number(),
});

const absorbingSiteSchema = z.object({
  assetId: z.string(),
  absorbedKwh: z.number(),
});

// Grid results, read from the hosting-capacity, voltage and DSO-value JSON documents (no DB table).
const nn = z.number().nullable();
const voltagePolicySchema = z.object({
  violations: z.number(),
  minutes: z.number(),
  tripKwh: z.number(),
  tripLei: z.number(),
  vMax: z.number(),
});
const gridResultsSchema = z.object({
  hosting: z
    .object({
      label: z.string(),
      runId: z.string(),
      tolMin: z.number(),
      variant: z.string(),
      totalWithout: z.number(),
      totalThreshold: nn,
      totalWith: z.number(),
      feeders: z.array(
        z.object({
          feederId: z.string(),
          nHomes: z.number(),
          without: z.number(),
          threshold: nn,
          with: z.number(),
          censored: z.boolean(),
        }),
      ),
    })
    .nullable(),
  voltage: z
    .object({
      label: z.string(),
      runId: z.string(),
      vLimit: z.number(),
      nHomes: z.number(),
      baseline: voltagePolicySchema,
      sentinel: voltagePolicySchema,
    })
    .nullable(),
  value: z
    .object({
      label: z.string(),
      labelDeo: z.string(),
      runId: z.string(),
      deferredCapexLei: z.number(),
      deoBasisLei: z.array(z.number()),
      gain: z.number(),
      feeders: z.array(z.object({ feederId: z.string(), gain: z.number(), yearsDeferred: z.number(), lei: z.number() })),
      assumptions: z.array(z.object({ label: z.string(), value: z.string() })),
    })
    .nullable(),
  twin: z
    .object({
      recoveryRate: nn,
      nCorrupted: nn,
      moved: z
        .object({
          siteId: z.string(),
          recordedFeeder: z.string(),
          suggestedFeeder: z.string(),
          peakGisKw: nn,
          peakM8Kw: nn,
          sitesChanged: z.number(),
        })
        .nullable(),
    })
    .nullable(),
});
export type GridResults = z.infer<typeof gridResultsSchema>;

export const feedersContract = {
  list: oc
    .input(z.object({ runId: z.string().optional() }).optional())
    .output(z.array(feederSummarySchema)),
  getCurves: oc
    .input(z.object({ feederId: z.string(), ...windowInput }))
    .output(z.array(feederCurvePointSchema)),
  getOverLimit: oc
    .input(z.object(windowInput).optional())
    .output(z.array(feederOverLimitSchema)),
  // The 15-min steps where at least one feeder exported over its limit, with how many (the timeline's red marks).
  getOverLimitSteps: oc
    .input(z.object({ policy: z.string(), ...windowInput }))
    .output(z.array(z.object({ ts: z.string(), feeders: z.number() }))),
  getSocFan: oc
    .input(z.object({ feederId: z.string(), policy: z.string(), ...windowInput }))
    .output(z.array(socFanPointSchema)),
  // WattSentry-policy charging during the steps where the baseline is over the limit.
  getAbsorption: oc
    .input(z.object({ feederId: z.string(), ...windowInput }))
    .output(z.array(absorbingSiteSchema)),
  listGisFlags: oc
    .input(z.object({ runId: z.string().optional() }).optional())
    .output(z.array(gisFlagItemSchema)),
  getGrid: oc.input(z.object({}).optional()).output(gridResultsSchema),
};
