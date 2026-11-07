# Demo readiness checklist (card #62)

Demo data: `demo/sentinel.sqlite` = replay#506 (runs `506:baseline`, `506:sentinel`, week Mon 2 – Sun 8 June 2025).
Write what you saw on card #62 (`lab note 62 "..."`), one line per section.

## 0. Automated smoke check (1 min)

```sh
node apps/web/src/demo/offline-check.mjs                       # local build, default http://localhost:3000
node apps/web/src/demo/offline-check.mjs https://app.<domain>  # the shared app, from the hotspot
```

Every line must say `ok`. It loads all 6 modes, every script/style they reference, and the summary data.
`basemaps.cartocdn.com` is the only expected external host (offline it falls back to the plain sea + Romania outline).

## 1. Summary shows the real replay

Press `6` (or open `?summary=1`). Expect, with no yellow "Fixture data … Not a result" banner:

| Metric | Baseline | Sentinel |
|---|---|---|
| Peak reduction | — | 19.1 % |
| Minutes over limit | 6645 min | 1905 min (−71 %) |
| Energy over limit | 1397 kWh | 94.46 kWh (−93 %) |
| Peak over limit | 29.14 kW | 10.57 kW (−64 %) |
| P10 hit rate | — | 92.0 % |
| Comfort breaches | 1165 | 0 (−100 %) |
| Lei per home | — | 23.73 lei/home/month |
| Alerts per turbine-day | 0.70 | 0.11 (−84 %) |

Footer: `Run ID 506:sentinel · 506:baseline`. Keys 1–5 land on: Production WP01, Demand 19:00, F7 Wed 4 noon
(270 → 0 min), 1042 on Thu 5, O&M WP01.

## 2. Laptop offline (Wi-Fi really off, not just blocked requests)

1. While online: `npm ci && npm run build` at the repo root (the build needs the network, the app does not).
2. Turn Wi-Fi **off** (and unplug Ethernet). `npm start` → http://localhost:3000.
3. Run the smoke check (section 0): all `ok`.
4. In Chromium click through Overview, Feeders, O&M, Production, Demand, Imbalance. Each one: renders, no
   endless spinner, map shows the offline basemap (sea + Romania) with the offline badge, DevTools console has no
   red errors other than the failed CARTO fetch.
5. Keys 1–6 and the replay bar play/scrub still work.

## 3. Phone hotspot on the shared app

1. Laptop on the phone hotspot only. Open the shared app once **before** presenting (first load was ~5 s; later
   loads are cached).
2. Run the smoke check against the shared URL; note the `ms` of the first mode.
3. Click through all modes and keys 1–6 as in section 2.

## 4. Lab public on (with Andrei)

`lab public on`, then open `demo.<domain>` from a phone on mobile data and repeat section 3, step 3.

## 5. Rehearsal

Two timed run-throughs of the plan §7 script on the demo laptop; note the time and whether the map stays smooth
(no visible stutter while the replay plays).
