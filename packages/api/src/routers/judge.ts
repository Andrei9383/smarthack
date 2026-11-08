// Judge: verdicts per asset and the trust ledger per witness, read from the judge_verdicts / trust / outcomes
// tables with raw SQL. A DB without them returns no data, never an error.
import { os } from "@orpc/server";
import type { Database } from "@wattsentry/db";
import { sql, type SQL } from "drizzle-orm";
import { z } from "zod";

import type { Context } from "../context";
import { emptyIfNoTables } from "./runs";

const publicProcedure = os.$context<Context>();

type Row = Record<string, unknown>;
const num = (v: unknown) => (v == null || v === "" ? null : Number(v));
const str = (v: unknown) => (v == null ? null : String(v));
const flag = (v: unknown) => !(v == null || v === 0 || v === "0" || v === "" || v === false);
const where = (parts: (SQL | false | undefined)[]) =>
  sql.join(parts.filter((p): p is SQL => !!p), sql` and `);
function groupBy(rows: Row[], key: (r: Row) => string) {
  const out = new Map<string, Row[]>();
  for (const r of rows) out.set(key(r), [...(out.get(key(r)) ?? []), r]);
  return out;
}

// Newest run with verdicts (the Judge may write under its own run id).
async function judgeRun(db: Database, runId?: string) {
  if (runId) return runId;
  const [row] = await emptyIfNoTables(() =>
    db.all<Row>(
      sql`select t.run_id from judge_verdicts t left join runs r on r.id = t.run_id
          group by t.run_id order by max(r.created_at) desc, t.run_id desc limit 1`,
    ),
  );
  return row ? String(row.run_id) : null;
}

function reasonsOf(json: unknown): string[] {
  if (typeof json !== "string" || !json) return [];
  try {
    const parsed = JSON.parse(json) as unknown;
    return Array.isArray(parsed)
      ? parsed.slice(0, 3).map((r) => (typeof r === "string" ? r : JSON.stringify(r)))
      : [];
  } catch {
    return [json];
  }
}

const input = z
  .object({ runId: z.string().optional(), until: z.string().optional(), assetId: z.string().optional() })
  .optional();


export const judgeRouter = {
  // Latest verdict per (asset, witness) on or before `until` (local day YYYY-MM-DD), with its graded outcomes once known.
  verdicts: publicProcedure.input(input).handler(({ context: { db }, input }) =>
    emptyIfNoTables(async () => {
      const runId = await judgeRun(db, input?.runId);
      if (!runId) return [];
      const until = input?.until;
      const rows = await db.all<Row>(sql`
        select v.* from judge_verdicts v
        where ${where([
          sql`v.run_id = ${runId}`,
          input?.assetId !== undefined && sql`v.asset_id = ${input.assetId}`,
          sql`v.day = (select max(v2.day) from judge_verdicts v2 where v2.run_id = v.run_id
                and v2.asset_id = v.asset_id and v2.model = v.model ${until ? sql`and v2.day <= ${until}` : sql``})`,
        ])}
        order by v.asset_id, v.model`);
      const outcomes = await emptyIfNoTables(() =>
        db.all<Row>(sql`
          select asset_id, model, day, score_name, score, source, gated from outcomes
          where ${where([
            sql`run_id = ${runId}`,
            input?.assetId !== undefined && sql`asset_id = ${input.assetId}`,
            until !== undefined && sql`day <= ${until}`,
          ])}`),
      );
      const key = (r: Row) => `${r.asset_id}|${r.model}|${r.day}`;
      const byKey = groupBy(outcomes, key);
      return rows.map((r) => ({
        runId,
        day: String(r.day),
        assetId: String(r.asset_id),
        model: String(r.model),
        verdict: String(r.verdict),
        pValid: num(r.p_valid),
        trust: num(r.trust),
        nEff: num(r.n_eff),
        bandScale: num(r.band_scale),
        substitute: str(r.substitute),
        reasons: reasonsOf(r.reasons_json),
        rule: str(r.rule),
        judgeVersion: str(r.judge_version),
        outcomes: (byKey.get(key(r)) ?? []).map((o) => ({
          scoreName: String(o.score_name),
          score: num(o.score),
          source: str(o.source),
          gated: flag(o.gated),
        })),
      }));
    }),
  ),
};
