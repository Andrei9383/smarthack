import { oc } from "@orpc/contract";
import { z } from "zod";

// Congestion notices: day-ahead congestion forecast per feeder and household emails.

const congestionLevelSchema = z.enum(["none", "watch", "congested", "severe"]);
export type CongestionLevelValue = z.infer<typeof congestionLevelSchema>;

const noticeStepSchema = z.object({
  ts: z.string(),
  q10Kw: z.number(),
  q50Kw: z.number(),
  q90Kw: z.number(),
  limitKw: z.number(),
  actualKw: z.number().nullable(), // baseline export, what happened without WattSentry
});

const householdSchema = z.object({
  assetId: z.string(),
  locality: z.string(),
  county: z.string(),
  feederId: z.string(),
  kwp: z.number(),
  hasBattery: z.boolean(),
  batteryKwh: z.number(),
  comfortReserveKwh: z.number(),
  hasEv: z.boolean(),
  hasAc: z.boolean(),
  hasBoiler: z.boolean(),
  worksFromHome: z.boolean(),
  pvWindowKwh: z.number(),
  exportWindowKwh: z.number(),
  batteryPlanKwh: z.number(),
});
export type HouseholdItem = z.infer<typeof householdSchema>;

const feederNoticeSchema = z.object({
  feederId: z.string(),
  feederName: z.string(),
  limitKw: z.number(),
  level: congestionLevelSchema,
  reason: z.string(),
  windowFrom: z.string().nullable(),
  windowTo: z.string().nullable(),
  stepsOverP10: z.number(),
  stepsOverP50: z.number(),
  stepsOverP90: z.number(),
  peakP50Kw: z.number(),
  peakP90Kw: z.number(),
  overLimitP50Kwh: z.number(),
  loadSource: z.enum(["yesterday", "weekly_profile", "none"]),
  modelVersion: z.string(),
  /** Hindsight check against the baseline curve: did congestion happen, and inside the window? */
  actualStepsOver: z.number(),
  actualStepsInWindow: z.number(),
  actualPeakKw: z.number(),
  steps: z.array(noticeStepSchema),
  households: z.array(householdSchema),
});
export type FeederNotice = z.infer<typeof feederNoticeSchema>;

const sentLogSchema = z.object({
  at: z.string(),
  feederId: z.string(),
  day: z.string(),
  level: congestionLevelSchema,
  count: z.number(),
  ids: z.array(z.string()),
  to: z.array(z.string()), // masked
  ok: z.boolean(),
  error: z.string().nullable(),
});
export type SentLogItem = z.infer<typeof sentLogSchema>;

const dayInput = {
  from: z.string(), // ISO-8601 UTC, start of the Bucharest day
  to: z.string(), // exclusive
  runId: z.string().optional(),
};

export const notifyContract = {
  getConfig: oc.input(z.object({}).optional()).output(
    z.object({
      enabled: z.boolean(), // RESEND_API_KEY set
      from: z.string(),
      demoTo: z.array(z.string()), // masked
      hourlyCap: z.number(),
    }),
  ),
  getForecast: oc.input(z.object({ feederId: z.string(), ...dayInput })).output(feederNoticeSchema.nullable()),
  preview: oc
    .input(z.object({ feederId: z.string(), assetId: z.string(), ...dayInput }))
    .output(z.object({ subject: z.string(), html: z.string(), text: z.string() }).nullable()),
  send: oc
    .input(
      z.object({
        feederId: z.string(),
        assetIds: z.array(z.string()).optional(), // default: every home on the feeder
        minLevel: congestionLevelSchema.default("congested"),
        ...dayInput,
      }),
    )
    .output(sentLogSchema),
  listSent: oc.input(z.object({}).optional()).output(z.array(sentLogSchema)),
};
