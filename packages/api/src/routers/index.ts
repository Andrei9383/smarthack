import type { RouterClient } from "@orpc/server";

import { protectedProcedure, publicProcedure } from "../index";
import { feedersRouter } from "./feeders";
import { flexRouter } from "./flex";
import { manifestRouter } from "./manifest";
import { productionRouter } from "./production";
import { sitesRouter } from "./sites";

export const appRouter = {
  feeders: feedersRouter,
  manifest: manifestRouter,
  production: productionRouter,
  flex: flexRouter,
  sites: sitesRouter,
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
