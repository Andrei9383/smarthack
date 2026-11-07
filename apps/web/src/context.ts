import type { Context as ApiContext } from "@sentinel-app/api/context";
import type { NextRequest } from "next/server";

import { db } from "./services";
import { auth } from "./services";

export async function createContext(req: NextRequest): Promise<ApiContext> {
  const session = await auth.api.getSession({
    headers: req.headers,
  });
  return {
    db,
    session,
  };
}

export type Context = Awaited<ReturnType<typeof createContext>>;
