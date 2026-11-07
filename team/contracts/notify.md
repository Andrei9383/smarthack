# Congestion notices contract (card #120)

Code: `packages/api/src/notify/` (forecast, email, Resend client), router `packages/api/src/routers/notify.ts`,
contract `packages/api/src/contract/notify.ts`, UI `apps/web/src/modes/feeders/notice-section.tsx`.
Reads the shared SQLite only; no schema change, nothing written to the DB (the sent log is in memory).

## Decision: congestion level per feeder and Bucharest day

Export forecast per 15-min step = Σ over the feeder's homes of M1 PVNet PV q10/q50/q90 (`asset_steps`) minus the
feeder's load **yesterday at the same time** (`measurements.load_kw`, persistence). All inputs exist the evening before,
so the notice is day-ahead. First replay day has no "yesterday": falls back to the week's mean load profile and says so
(`loadSource = weekly_profile`).

| level | rule | email |
|---|---|---|
| `severe` | P10 export over the limit ≥ 1 h, or P50 peak ≥ 1.3 × limit (with ≥ 30 min over) | yes, "Action needed" |
| `congested` | P50 export over the limit ≥ 30 min | yes, "use your solar at home" |
| `watch` | only P90 crosses the limit, or P50 peak ≥ 0.9 × limit | no (shown in the panel) |
| `none` | otherwise | no |

Why: P10 over the limit means it congests even on a dull day, so ask firmly; P50 means more likely than not; P90 only is
a sunny-side risk, not worth an email. Window = first to last step over the limit at the level's quantile.
Demo week check (hindsight vs the baseline curve): 97.5% of over-limit steps are forecast (step precision 78%); every
feeder-day is graded right except F6 on Jun 2 (fallback load day, false `congested`).

## Email (one per home on the feeder)

Personalised from the DB: the home's P50 solar and surplus in the window, its Sentinel battery charge planned in the window
(`schedules`, comfort reserve untouched), and tips from `assets.metadata_json` (`has_ev`, `has_electric_boiler`, `has_ac`,
`works_from_home`). Footer "Why you got this" carries the level rule, P50/P90 peak vs limit, PVNet version and load source.

## Delivery

Resend batch API, server-side only. Environment (never committed; `apps/web/.env.local` locally, lab secrets on the server):

| var | meaning |
|---|---|
| `RESEND_API_KEY` | Resend key; unset = sending off, UI says so |
| `NOTIFY_DEMO_TO` | comma-separated inbox(es). Synthetic homes have no address, so every notice goes here, subject tagged `[home id]` |
| `RESEND_FROM` | default `Sentinel grid alerts <alerts@devsolution.ro>` (verified domain) |
| `NOTIFY_HOURLY_CAP` | max emails per process per hour, default 300 (the demo endpoint is public) |

Recipients are never taken from the request, so the public page cannot send mail anywhere else. The UI shows the inbox masked.

## API (`notify.*`)

`getConfig`, `getForecast({feederId, from, to})`, `getFleet({from, to})`, `preview({feederId, assetId, from, to})`,
`send({feederId, from, to, assetIds?, minLevel = congested})` → log entry with Resend ids, `listSent`.
