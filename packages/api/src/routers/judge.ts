// Sentinel Judge (#151, team/judge-design.md §8): verdicts per asset and the trust ledger per witness.
// The judge_verdicts / trust / outcomes tables are written by the Judge (J4) and are not in the
// Drizzle schema, so this reads them with raw SQL. A DB without them returns no data, never an error.
import type { Database } from "@sentinel-app/db";
import { sql, type SQL } from "drizzle-orm";
import { z } from "zod";

import { publicProcedure } from "../index";
import { emptyIfNoTables } from "./runs";

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
const mean = (rows: Row[], field: string) => {
  const v = rows.map((r) => num(r[field])).filter((x): x is number => x !== null && Number.isFinite(x));
  return v.length ? v.reduce((a, b) => a + b, 0) / v.length : null;
};

// Newest run with rows in `table` (the Judge may write under its own run id).
async function judgeRun(db: Database, table: "judge_verdicts" | "trust", runId?: string) {
  if (runId) return runId;
  const [row] = await emptyIfNoTables(() =>
    db.all<Row>(
      sql`select t.run_id from ${sql.identifier(table)} t left join runs r on r.id = t.run_id
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

export type TrustWitness = {
  model: string;
  series: { day: string; trust: number | null; prior: number | null; nEff: number }[];
  alerts: { bucket: string; trust: number | null; prior: number | null; message: string | null; since: string }[];
};

export const judgeRouter = {
  // Latest verdict per (asset, witness) on or before `until` (local day YYYY-MM-DD), with its graded outcomes once known.
  verdicts: publicProcedure.input(input).handler(({ context: { db }, input }) =>
    emptyIfNoTables(async () => {
      const runId = await judgeRun(db, "judge_verdicts", input?.runId);
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

  // Trust over the week per witness (mean over its context buckets per day) and its open health alerts (§6.5).
  trustBoard: publicProcedure.input(input).handler(async ({ context: { db }, input }) => {
    const runId = await judgeRun(db, "trust", input?.runId);
    if (!runId) return { runId: null, witnesses: [] as TrustWitness[] };
    const rows = await emptyIfNoTables(() =>
      db.all<Row>(sql`
        select day, model, bucket, prior, n_eff, trust, health_alert from trust
        where ${where([sql`run_id = ${runId}`, input?.until !== undefined && sql`day <= ${input.until}`])}
        order by model, day, bucket`),
    );
    const witnesses = [...groupBy(rows, (r) => String(r.model))].map(([model, list]): TrustWitness => {
      const days = [...groupBy(list, (r) => String(r.day))];
      const series = days.map(([day, xs]) => ({
        day,
        trust: mean(xs, "trust"),
        prior: mean(xs, "prior"),
        nEff: xs.reduce((a, x) => a + (num(x.n_eff) ?? 0), 0),
      }));
      const latest = days.at(-1)?.[1] ?? [];
      const alerts = latest.filter((x) => flag(x.health_alert)).map((x) => {
        // first day of the current unbroken alert streak for this bucket
        const history = list.filter((y) => y.bucket === x.bucket);
        let i = history.length - 1;
        while (i > 0 && flag(history[i - 1]?.health_alert)) i--;
        return {
          bucket: String(x.bucket),
          trust: num(x.trust),
          prior: num(x.prior),
          message: typeof x.health_alert === "string" && Number.isNaN(Number(x.health_alert)) ? x.health_alert : null,
          since: String(history[i]?.day ?? x.day),
        };
      });
      return { model, series, alerts };
    });
    return { runId, witnesses };
  }),
};
