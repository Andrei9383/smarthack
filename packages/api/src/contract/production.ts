import { oc } from "@orpc/contract";
import { z } from "zod";

export const windfieldMetaSchema = z.object({
  source: z.string(), // 'mock' until real ERA5/ICON hours replace it (#55)
  bounds: z.tuple([z.number(), z.number(), z.number(), z.number()]),
  imageUnscale: z.tuple([z.number(), z.number()]),
  speedMax: z.number(),
  hours: z.array(z.object({ ts: z.string() })), // index = NN in /windfield/NN.png and NN-speed.png
});
export type WindfieldMeta = z.infer<typeof windfieldMetaSchema>;

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
  expectedMwh: z.number(),
  solarMwh: z.number(),
  windMwh: z.number(),
});
export type CountyTotalItem = z.infer<typeof countyTotalItemSchema>;

export const productionContract = {
  getWindfield: oc
    .input(
      z
        .object({
          ts: z.string().optional(),
          runId: z.string().optional(),
        })
        .optional(),
    )
    .output(windfieldMetaSchema),
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
  getCountyTotals: oc
    .input(z.object({ runId: z.string().optional() }).optional())
    .output(z.array(countyTotalItemSchema)),
};
