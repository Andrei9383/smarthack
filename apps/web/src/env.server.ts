import { randomBytes } from "node:crypto";
import { existsSync } from "node:fs";
import path from "node:path";

// Demo store lookup (plan R18): promoted replay, then #30 fixtures, then an empty
// in-memory DB so the page still renders on a clean checkout.
function resolveDatabaseUrl(): string {
  if (process.env.DATABASE_URL) return process.env.DATABASE_URL;
  const roots = [process.cwd(), path.resolve(process.cwd(), "../..")];
  for (const file of ["demo/wattsentry.sqlite", "demo/fixtures.sqlite"]) {
    for (const root of roots) {
      const candidate = path.join(/*turbopackIgnore: true*/ root, file);
      if (existsSync(candidate)) return `file:${candidate}`;
    }
  }
  return ":memory:";
}

export const ENV = {
  DATABASE_URL: resolveDatabaseUrl(),
  // No login wall in the demo, so a per-process secret is enough when none is set.
  BETTER_AUTH_SECRET: process.env.BETTER_AUTH_SECRET ?? randomBytes(32).toString("hex"),
  BETTER_AUTH_URL:
    process.env.BETTER_AUTH_URL ?? `http://localhost:${process.env.PORT ?? "3000"}`,
};
