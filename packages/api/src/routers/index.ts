import type { RouterClient } from "@orpc/server";

import { auditRouter } from "./audit";
import { demandRouter } from "./demand";
import { feedersRouter } from "./feeders";
import { flexRouter } from "./flex";
import { imbalanceRouter } from "./imbalance";
import { judgeRouter } from "./judge";
import { liveRouter } from "./live";
import { manifestRouter } from "./manifest";
import { notifyRouter } from "./notify";
import { omRouter } from "./om";
import { productionRouter } from "./production";
import { sitesRouter } from "./sites";

export const appRouter = {
  demand: demandRouter,
  feeders: feedersRouter,
  manifest: manifestRouter,
  production: productionRouter,
  flex: flexRouter,
  imbalance: imbalanceRouter,
  sites: sitesRouter,
  om: omRouter,
  audit: auditRouter,
  notify: notifyRouter,
  judge: judgeRouter,
  live: liveRouter,
};
export type AppRouterClient = RouterClient<typeof appRouter>;
