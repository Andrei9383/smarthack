import { oc } from "@orpc/contract";
import { z } from "zod";

const parkItemSchema = z.object({
  id: z.string(),
  name: z.string(),
  county: z.string(),
  lat: z.number(),
  lon: z.number(),
  ratedMw: z.number(),
  turbineCount: z.number(),
});
export type ParkItem = z.infer<typeof parkItemSchema>;

const parkForecastItemSchema = z.object({
  parkId: z.string(),
  ts: z.string(),
  p10Mw: z.number().nullable().optional(),
  p50Mw: z.number().nullable().optional(),
  p90Mw: z.number().nullable().optional(),
  expectedMwh: z.number().nullable().optional(),
  availabilityPct: z.number().nullable().optional(),
  curtailedMwh: z.number().nullable().optional(),
});
export type ParkForecastItem = z.infer<typeof parkForecastItemSchema>;

const countyTotalItemSchema = z.object({
  county: z.string(),
  lat: z.number(),
  lon: z.number(),
  expectedMwh: z.number(),
  solarMwh: z.number(),
  windMwh: z.number(),
});

// One hour of one county: mean MW over the hour. Solar sums the county's rooftop homes
// (asset_steps); wind sums the county's turbines' measured output (measurements.act_pwt_kw).
const countySeriesPointSchema = z.object({
  ts: z.string(), // hour start, ISO-8601 UTC
  solarActualMw: z.number().nullable(),
  solarP10Mw: z.number().nullable(),
  solarP50Mw: z.number().nullable(),
  solarP90Mw: z.number().nullable(),
  windActualMw: z.number().nullable(),
});
export type CountySeriesPoint = z.infer<typeof countySeriesPointSchema>;

const countySeriesItemSchema = z.object({
  county: z.string(),
  points: z.array(countySeriesPointSchema),
});
export type CountySeriesItem = z.infer<typeof countySeriesItemSchema>;

// What the park sold day-ahead vs what it made, with the forecast band. MWh per 15 min.
const windPositionItemSchema = z.object({
  parkId: z.string(),
  policy: z.string(),
  ts: z.string(),
  actualMwh: z.number().nullable(),
  soldDaMwh: z.number(),
  fcP10Mwh: z.number().nullable(),
  fcP50Mwh: z.number().nullable(),
  fcP90Mwh: z.number().nullable(),
  avoidableLei: z.number().nullable(),
  removedTurbines: z.number().nullable(),
});

// Park battery per variant (none | rule | lp). MWh per 15 min.
const parkBatteryItemSchema = z.object({
  parkId: z.string(),
  variant: z.string(),
  ts: z.string(),
  chargeMwh: z.number().nullable(),
  dischargeMwh: z.number().nullable(),
  socMwh: z.number().nullable(),
  devNoneMwh: z.number().nullable(),
  devMwh: z.number().nullable(),
  avoidableLei: z.number().nullable(),
  avoidableNoneLei: z.number().nullable(),
});

const windowInput = z.object({
  parkId: z.string().optional(),
  runId: z.string().optional(),
  from: z.string().optional(),
  to: z.string().optional(),
});

// The wind field is static, not a procedure: /windfield/meta.json + NN.png + NN-speed.png.
export const productionContract = {
  listParks: oc.input(z.object({}).optional()).output(z.array(parkItemSchema)),
  getParkForecasts: oc
    .input(
      z
        .object({
          parkId: z.string().optional(),
          runId: z.string().optional(),
        })
        .optional(),
    )
    .output(z.array(parkForecastItemSchema)),
  // from inclusive, to exclusive; ISO-8601 UTC. Omit both for the whole run.
  getCountyTotals: oc
    .input(
      z
        .object({
          runId: z.string().optional(),
          from: z.string().optional(),
          to: z.string().optional(),
        })
        .optional(),
    )
    .output(z.array(countyTotalItemSchema)),
  // Hourly buckets per county for the map sparklines; same run selection as getCountyTotals.
  getCountySeries: oc
    .input(
      z
        .object({
          runId: z.string().optional(),
          from: z.string().optional(),
          to: z.string().optional(),
        })
        .optional(),
    )
    .output(z.array(countySeriesItemSchema)),
  // from inclusive, to exclusive; sentinel policy unless `policy` is given.
  getWindPositions: oc
    .input(windowInput.extend({ policy: z.string().optional() }).optional())
    .output(z.array(windPositionItemSchema)),
  getParkBattery: oc.input(windowInput.optional()).output(z.array(parkBatteryItemSchema)),
};
