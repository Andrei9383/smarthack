# Reading #38's generated schema — notes from #30

Written while building `make_fixtures.py` against `card/38 @ afc5ef4`. The detail behind
notes [42] and [62] on #38, kept out of the card thread because it is long and only matters
to whoever touches the schema or the seeder. Delete when #38 merges and the questions close.

## 1. The time column is not pinned (blocker, note [62])

Three sources on the same branch disagree:

| Source | Column | Type |
| --- | --- | --- |
| `packages/db/src/schema/sentinel.ts`, generated `migration.sql` | `ts` | `integer`, no unit documented |
| `team/contracts/app.md` §2.4–2.6 | `t` | `TEXT` ISO-8601 UTC |
| `app.md` §1 | — | "ISO-8601 strings **or** Unix epoch ms" |
| Tudor's note [48], `replay.md` per [55] | `ts` | TEXT ISO-8601 |

The schema uses a plain `integer("ts")` — not `integer({ mode: "timestamp" })` — so Drizzle
treats it as a bare number and nothing in the types says seconds or milliseconds.

This one fails *silently*, which is why it is worth a blocker rather than a guess. SQLite is
dynamically typed: inserting `'2025-06-02T11:15:00Z'` into an `integer` column succeeds, and
then `ORDER BY ts` and `WHERE ts BETWEEN ? AND ?` compare wrong, because SQLite orders all
integers before all text regardless of content. #40's 24 h slider and every windowed query
return plausible-looking wrong rows with no error anywhere.

Until it is pinned, every timestamp this seeder writes goes through `Clock` in
`make_fixtures.py`, so the answer is `--ts-unit s|ms|iso` and not a rewrite. Default is unix
seconds, which is what the migration already generates.

## 2. Tables the generated schema has that neither list had

`kpis`, `measurements`, `gis_flags`, `feeders`, `parks` all landed — these came out of #30's
draft and #17's `replay.md`. Good: the demo KPI strip and #32's power-curve scatter now have
somewhere to read from.

## 3. `asset_steps` is gone, and that is fine

`replay.md` proposed `asset_steps(actual_kw, q10/q50/q90)`. The generated schema instead keeps
actuals in `measurements` and quantiles in `predictions`. #31's expected-vs-actual chart is
therefore a join:

```sql
SELECT m.ts, m.pv_ac_kw AS actual, p.q10_kw, p.q50_kw, p.q90_kw
  FROM measurements m
  JOIN predictions p
    ON p.run_id = m.run_id AND p.asset_id = m.asset_id AND p.ts = m.ts AND p.model = 'M1'
 WHERE m.asset_id = ? ORDER BY m.ts;
```

Both sides are indexed on `(run_id, asset_id, ts)`, so this is cheap. No change requested.

## 4. `work_orders` has no `crew_type`

§3.5 and #32 both specify top-K per day **per crew type** (residential installers vs wind
technicians). The table has `asset_kind` and `action` instead. The seeder derives crew from
`asset_kind` 1:1 (`prosumer` → residential, `turbine` → wind) and uses `action` for
`truck | remote_check | inspection`, which covers "sensor faults never get a truck". That works,
but it means the crew split is a convention rather than a column — if a residential crew ever
services a turbine, or a wind crew a site, the model cannot express it.

## 5. `predictions` has no forecast vintage

Columns are `(asset_id, model, ts, step, created_at)` with an autoincrement `id`. There is no
`t0` or `horizon_idx`, so two forecasts made at different times *for the same* `ts` are
distinguishable only by `created_at`. Fine for fixtures (one vintage) and for a replay that
writes each tick once. It matters if #17 re-forecasts and the UI then has to pick "the forecast
as known at time X" — a day-ahead plan and an hour-ahead re-solve would collide.

## 6. `assets` collapsed recorded vs true PV metadata

Plan §3.4 asks for recorded *and* true tilt / azimuth / kWp side by side, because the gap
between them is what #13 / M8's GIS-error story is about. The schema has one `pv_kwp`,
`tilt_deg`, `azimuth_deg` plus a free-form `metadata_json`. Not #30's problem — the seeder
writes the recorded values — but whoever builds M8 will need somewhere structured to put the
corrected ones, and `gis_flags` only covers phase, not kWp or orientation.

## 7. `feeders` has no geometry

`feeders` carries `limit_kw`, `tau`, `gis_trust` but no shape, so Feeders mode (#19) cannot draw
a feeder as a line from this table alone — only infer one from its sites' positions.
`gis_flags.geometry_json` exists for flagged sections. Tudor's call, flagged on the card.

## 8. Not a schema issue: where `demo/fixtures.sqlite` comes from

Full fidelity (300 homes × 672 steps, every table) is ~128 MB, over GitHub's 100 MB per-file
hard limit and well over R18's 50 MB for a committed SQLite. Indexes are ~55 MB of it.
So the generator is committed and the database is not: `make_fixtures.py` rebuilds it
deterministically in ~5 s from stdlib only.

That needs one line wherever the preview is set up (`lab.toml [runs.preview]`, Tudor's file):

```
python3 demo/fixtures/make_fixtures.py
```

and `demo/fixtures.sqlite` added to the root `.gitignore` (also Tudor's, via #21's paths), so
nobody commits the blob by accident. Asked on the card rather than edited here, per §12.
