import { implement } from "@orpc/server";
import { kpis, runs } from "@sentinel-app/db/schema/sentinel";
import { desc, inArray } from "drizzle-orm";

import type { Context } from "../context";
import { manifestContract } from "../contract/manifest";
import { currentRunIds, emptyIfNoTables } from "./runs";

const os = implement(manifestContract).$context<Context>();

export const manifestRouter = os.router({
  getRuns: os.getRuns.handler(({ context }) =>
    emptyIfNoTables(() => context.db.select().from(runs).orderBy(desc(runs.createdAt))),
  ),
  getKpis: os.getKpis.handler(({ context, input }) =>
    emptyIfNoTables(async () =>
      context.db
        .select()
        .from(kpis)
        .where(inArray(kpis.runId, await currentRunIds(context.db, input?.runId))),
    ),
  ),
});
