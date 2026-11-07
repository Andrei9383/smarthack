import { oc } from "@orpc/contract";
import { z } from "zod";

export const runItemSchema = z.object({
  id: z.string(),
  kind: z.string(),
  policy: z.string().nullable().optional(),
  label: z.string(),
  startedAt: z.number(),
  notes: z.string().nullable().optional(),
});
export type RunItem = z.infer<typeof runItemSchema>;

export const manifestContract = {
  getRuns: oc
    .input(z.object({}).optional())
    .output(z.array(runItemSchema)),
};
