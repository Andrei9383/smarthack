import { oc } from "@orpc/contract";
import { z } from "zod";

export const windfieldMetaSchema = z.object({
  bounds: z.tuple([z.number(), z.number(), z.number(), z.number()]),
  particlesUrl: z.string(),
  uMax: z.number(),
  vMax: z.number(),
});
export type WindfieldMeta = z.infer<typeof windfieldMetaSchema>;

export const parkForecastItemSchema = z.object({
  parkId: z.string(),
  t: z.number(),
  step: z.number(),
  p10Mw: z.number().nullable().optional(),
  p50Mw: z.number().nullable().optional(),
  p90Mw: z.number().nullable().optional(),
  expectedMwh: z.number().nullable().optional(),
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
          t: z.number().optional(),
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
