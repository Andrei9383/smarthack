// Audit trail (#32): every action the agent took, with the model version, input hash, prediction
// and threshold it was based on, and committed vs realised where it was a commitment.
import { implement } from "@orpc/server";
import { auditLog } from "@sentinel-app/db/schema/sentinel";
import { and, desc, eq, lte, type SQL } from "drizzle-orm";

import type { Context } from "../context";
import { auditContract } from "../contract/audit";
import { emptyIfNoTables } from "./runs";
import { resolveRunId } from "./sites";

const os = implement(auditContract).$context<Context>();

const DEFAULT_LIMIT = 200;
const MAX_LIMIT = 2000;

export const auditRouter = os.router({
  listLogs: os.listLogs.handler(({ context: { db }, input }) =>
    emptyIfNoTables(async () => {
      const runId = await resolveRunId(db, input?.runId);
      if (!runId) return [];
      const where = [
        eq(auditLog.runId, runId),
        input?.assetId ? eq(auditLog.assetId, input.assetId) : undefined,
        input?.actionType
          ? eq(auditLog.actionType, input.actionType)
          : undefined,
        input?.actionId ? eq(auditLog.actionId, input.actionId) : undefined,
        input?.until ? lte(auditLog.ts, input.until) : undefined,
      ].filter((c): c is SQL => c !== undefined);
      return db
        .select({
          id: auditLog.id,
          actionId: auditLog.actionId,
          actionType: auditLog.actionType,
          ts: auditLog.ts,
          policy: auditLog.policy,
          actor: auditLog.actor,
          assetId: auditLog.assetId,
          feederId: auditLog.feederId,
          parkId: auditLog.parkId,
          status: auditLog.status,
          note: auditLog.note,
          model: auditLog.model,
          modelVersion: auditLog.modelVersion,
          committedKw: auditLog.committedKw,
          realisedKw: auditLog.realisedKw,
          inputHash: auditLog.inputHash,
          predictionJson: auditLog.predictionJson,
          threshold: auditLog.threshold,
        })
        .from(auditLog)
        .where(and(...where))
        .orderBy(desc(auditLog.ts))
        .limit(Math.min(input?.limit ?? DEFAULT_LIMIT, MAX_LIMIT));
    }),
  ),
});
