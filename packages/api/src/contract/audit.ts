import { oc } from "@orpc/contract";
import { z } from "zod";

export const auditLogItemSchema = z.object({
  id: z.string(),
  ts: z.number(),
  actor: z.string(),
  actionType: z.string(),
  assetId: z.string().nullable().optional(),
  status: z.string().nullable().optional(),
  note: z.string().nullable().optional(),
  model: z.string().nullable().optional(),
  modelVersion: z.string().nullable().optional(),
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
          limit: z.number().optional(),
        })
        .optional(),
    )
    .output(z.array(auditLogItemSchema)),
};
