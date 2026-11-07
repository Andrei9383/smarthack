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
};
