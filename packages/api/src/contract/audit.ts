import { oc } from "@orpc/contract";
import { z } from "zod";

export const auditLogItemSchema = z.object({
  id: z.string(),
  actionId: z.string().nullable().optional(),
  actionType: z.string(),
  ts: z.string(),
  policy: z.string().nullable().optional(),
  actor: z.string(),
  assetId: z.string().nullable().optional(),
  feederId: z.string().nullable().optional(),
  parkId: z.string().nullable().optional(),
  status: z.string().nullable().optional(),
  note: z.string().nullable().optional(),
  model: z.string().nullable().optional(),
  modelVersion: z.string().nullable().optional(),
  committedKw: z.number().nullable().optional(),
  realisedKw: z.number().nullable().optional(),
});
export type AuditLogItem = z.infer<typeof auditLogItemSchema>;

export const auditContract = {
  listLogs: oc
    .input(
      z
        .object({
          runId: z.string().optional(),
          assetId: z.string().optional(),
          actionType: z.string().optional(),
          actionId: z.string().optional(),
          limit: z.number().optional(),
        })
        .optional(),
    )
    .output(z.array(auditLogItemSchema)),
};
