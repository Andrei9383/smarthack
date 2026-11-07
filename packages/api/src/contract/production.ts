import { oc } from "@orpc/contract";
import { z } from "zod";

export const productionRunSchema = z.object({
  runId: z.string(),
  label: z.string(), // 'synthetic RO' | 'real RO' | 'non-RO': shown on screen
});
export type ProductionRun = z.infer<typeof productionRunSchema>;

export const parkItemSchema = z.object({
  id: z.string(),
  name: z.string(),
  county: z.string(),
  lat: z.number(),
  lon: z.number(),
  ratedMw: z.number(),
  turbineCount: z.number(),
});
export type ParkItem = z.infer<typeof parkItemSchema>;

export const parkForecastItemSchema = z.object({
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

export const countyTotalItemSchema = z.object({
  county: z.string(),
  lat: z.number(),
  lon: z.number(),
  expectedMwh: z.number(),
  solarMwh: z.number(),
  windMwh: z.number(),
});
export type CountyTotalItem = z.infer<typeof countyTotalItemSchema>;

// One hour of one county: mean MW over the hour. Solar sums the county's rooftop homes
// (asset_steps); wind sums the county's turbines' measured output (measurements.act_pwt_kw).
export const countySeriesPointSchema = z.object({
  ts: z.string(), // hour start, ISO-8601 UTC
  solarActualMw: z.number().nullable(),
  solarP10Mw: z.number().nullable(),
  solarP50Mw: z.number().nullable(),
  solarP90Mw: z.number().nullable(),
  windActualMw: z.number().nullable(),
});
export type CountySeriesPoint = z.infer<typeof countySeriesPointSchema>;

export const countySeriesItemSchema = z.object({
  county: z.string(),
  points: z.array(countySeriesPointSchema),
});
export type CountySeriesItem = z.infer<typeof countySeriesItemSchema>;

// The wind field is static, not a procedure: /windfield/meta.json + NN.png + NN-speed.png.
export const productionContract = {
  getRun: oc.input(z.object({}).optional()).output(productionRunSchema.nullable()),
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
  // Hourly buckets per county for the map sparklines (#137); same run selection as getCountyTotals.
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
};
