import { createAuth } from "@sentinel-app/auth";
import { createDb } from "@sentinel-app/db";

import { ENV } from "./env.server";

export const db = createDb(ENV);
export const auth = createAuth(ENV, db);
