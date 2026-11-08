import type { Database } from "@wattsentry/db";
import { assetSteps, parkForecasts, runs, zoneDemand } from "@wattsentry/db/schema/sentinel";
import { desc, sql } from "drizzle-orm";

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

// Replay writes shared inputs into only one of its per-policy runs, so ask per table.
export async function latestRunWith(
  db: Database,
  table: typeof parkForecasts | typeof assetSteps | typeof zoneDemand,
) {
  const [run] = await emptyIfNoTables(() =>
    db
      .select({ runId: runs.id, label: runs.label })
      .from(runs)
      .where(sql`EXISTS (SELECT 1 FROM ${table} WHERE ${table.runId} = ${runs.id})`)
      .orderBy(desc(runs.createdAt))
      .limit(1),
  );
  return run ?? null;
}
