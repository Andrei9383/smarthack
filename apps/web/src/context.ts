import type { Context as ApiContext } from "@sentinel-app/api/context";
import type { NextRequest } from "next/server";

import { auth, db } from "./services";

export async function createContext(req: NextRequest): Promise<ApiContext> {
  return { db, session: await auth.api.getSession({ headers: req.headers }) };
}
