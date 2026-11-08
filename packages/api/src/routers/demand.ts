import { implement } from "@orpc/server";
import { zoneDemand } from "@wattsentry/db/schema/replay";
import { and, asc, eq } from "drizzle-orm";

import type { Context } from "../context";
import { demandContract } from "../contract/demand";
import { inWindow, latestRunWith } from "./runs";

const os = implement(demandContract).$context<Context>();

export const demandRouter = os.router({
  getZones: os.getZones.handler(async ({ context, input }) => {
    const runId = input?.runId ?? (await latestRunWith(context.db, zoneDemand))?.runId;
    if (!runId) return [];
    return context.db
      .select({
        zoneId: zoneDemand.zoneId,
        level: zoneDemand.level,
        zoneName: zoneDemand.zoneName,
        parentZoneId: zoneDemand.parentZoneId,
        ts: zoneDemand.ts,
        q10Mw: zoneDemand.q10Mw,
        q50Mw: zoneDemand.q50Mw,
        q90Mw: zoneDemand.q90Mw,
        peakStress: zoneDemand.peakStress,
        isPeakHour: zoneDemand.isPeakHour,
        label: zoneDemand.label,
      })
      .from(zoneDemand)
      .where(
        and(
          eq(zoneDemand.runId, runId),
          input?.level ? eq(zoneDemand.level, input.level) : undefined,
          inWindow(zoneDemand.ts, input?.from, input?.to),
        ),
      )
      .orderBy(asc(zoneDemand.ts));
  }),
});
