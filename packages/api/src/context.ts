import type { Session } from "@wattsentry/auth";
import type { Database } from "@wattsentry/db";

export type Context = {
  session: Session | null;
  db: Database;
};
