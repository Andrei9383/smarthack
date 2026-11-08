import { createClient } from "@libsql/client";
import { defineRelations } from "drizzle-orm";
import { drizzle } from "drizzle-orm/libsql";

import * as schema from "./schema";

const relations = { ...defineRelations(schema), ...schema.authRelations };

export function createDb(env: { DATABASE_URL: string }) {
  return drizzle({ client: createClient({ url: env.DATABASE_URL }), relations });
}

export type Database = ReturnType<typeof createDb>;
