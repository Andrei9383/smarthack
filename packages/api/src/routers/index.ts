import type { RouterClient } from "@orpc/server";

import { protectedProcedure, publicProcedure } from "../index";
import { auditRouter } from "./audit";
import { demandRouter } from "./demand";
import { feedersRouter } from "./feeders";
import { flexRouter } from "./flex";
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
  sites: sitesRouter,
  om: omRouter,
  audit: auditRouter,
  notify: notifyRouter,
  healthCheck: publicProcedure.handler(() => {
    return "OK";
  }),
  privateData: protectedProcedure.handler(({ context }) => {
    return {
      message: "This is private",
      user: context.session?.user,
    };
  }),
};
export type AppRouter = typeof appRouter;
export type AppRouterClient = RouterClient<typeof appRouter>;
