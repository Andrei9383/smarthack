import type { Session } from "@sentinel-app/auth";
import type { Database } from "@sentinel-app/db";

export type Context = {
  session: Session | null;
  db: Database;
};
