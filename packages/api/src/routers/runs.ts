// Run selection and query helpers shared by the routers.
import type { Database } from "@wattsentry/db";
import {
  assetSteps,
  parkBattery,
  parkForecasts,
  runs,
  windPositions,
  zoneDemand,
} from "@wattsentry/db/schema/replay";
import { and, type AnyColumn, desc, gte, lt, sql } from "drizzle-orm";

const errorMentions = (error: unknown, text: string) => {
  const { message, cause } = error as Error & { cause?: { message?: string } };
  return `${message} ${cause?.message ?? ""}`.includes(text);
};

// An empty DB (clean checkout, no fixtures yet) has no tables: the app shows no data, not an error.
export async function emptyIfNoTables<T>(query: () => Promise<T[]>): Promise<T[]> {
  try {
    return await query();
  } catch (error) {
    if (errorMentions(error, "no such table")) return [];
    throw error;
  }
}

// Older DBs lack some newer columns: run `full`, and fall back to `base` when SQLite reports a missing column.
export async function withoutMissingColumns<T>(full: () => Promise<T[]>, base: () => Promise<T[]>) {
  try {
    return await full();
  } catch (error) {
    if (errorMentions(error, "no such column")) return base();
    throw error;
  }
}

// from inclusive, to exclusive; either may be omitted. `and()` skips the undefined bounds.
export const inWindow = (ts: AnyColumn, from?: string, to?: string) =>
  and(from ? gte(ts, from) : undefined, to ? lt(ts, to) : undefined);

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

// Without an explicit run, read the newest run, preferring the sentinel policy over baseline: the panels
// show what WattSentry did. Fixtures have policy NULL, which sorts with sentinel.
export async function resolveRunId(db: Database, runId?: string): Promise<string | null> {
  if (runId) return runId;
  const [row] = await emptyIfNoTables(() =>
    db
      .select({ id: runs.id })
      .from(runs)
      .orderBy(sql`case when ${runs.policy} = 'baseline' then 1 else 0 end`, desc(runs.createdAt))
      .limit(1),
  );
  return row?.id ?? null;
}

// Replay writes shared inputs into only one of its per-policy runs, so ask per table.
export async function latestRunWith(
  db: Database,
  table: typeof parkForecasts | typeof assetSteps | typeof zoneDemand | typeof windPositions | typeof parkBattery,
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
