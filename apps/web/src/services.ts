import { createAuth } from "@wattsentry/auth";
import { createDb } from "@wattsentry/db";

import { ENV } from "./env.server";

export const db = createDb(ENV);
export const auth = createAuth(ENV, db);
