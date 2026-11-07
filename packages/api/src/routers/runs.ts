import type { Database } from "@sentinel-app/db";
import { runs } from "@sentinel-app/db/schema/sentinel";
import { desc } from "drizzle-orm";

// An empty DB (clean checkout, no fixtures yet) has no tables: the app shows no data, not an error.
export async function emptyIfNoTables<T>(query: () => Promise<T[]>): Promise<T[]> {
  try {
    return await query();
  } catch (error) {
    const cause = (error as { cause?: { message?: string } }).cause?.message ?? "";
    if (`${(error as Error).message} ${cause}`.includes("no such table")) return [];
    throw error;
  }
}

// Fixtures hold both policies in one run; replay.py writes one run per policy. Either way, take
// the newest run for each policy.
export async function currentRunIds(db: Database, runId?: string): Promise<string[]> {
  if (runId) return [runId];
  const rows = await db
    .select({ id: runs.id, policy: runs.policy })
    .from(runs)
    .orderBy(desc(runs.createdAt));
  const newest = new Map<string, string>();
  for (const row of rows) {
    const key = row.policy ?? "all";
    if (!newest.has(key)) newest.set(key, row.id);
  }
  return [...newest.values()];
}
