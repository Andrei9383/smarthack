# demo/sentinel.sqlite — the demo replay (card #29)

**Label: synthetic RO** (ERA5-driven simulator: #11 fleet + #12 faults, #22/#23 wind) with **real OPCOM prices**.
**Weights: fleet-trained** — M1/M2/M4 are #14's train#383-392 on the shared #11 fleet + #12 faults (#10 windows,
train 2022-23 / val 2024 / test 2025; the demo week is in the test year), M6 WindNBM on the pre-#22 wind stand-in
(#41 run 135). models/README.md has the runs and their test-split numbers.

| | |
|---|---|
| Lab run | **replay#506** (card #29, worker rtx5070, commit `4ad365d074`, `--gpu-mem 12G`) |
| Command | `python -m sentinel.data.windows && python -m sentinel.sim.wind --no-plots && python -m sentinel.gridtwin.run && python replay.py --world sim --policy baseline,sentinel --out $LAB_OUTPUT_DIR/sentinel.sqlite` |
| File | the exact worker output: `scp rtx5070:/var/lib/lab-gpu/work/outputs/506/sentinel.sqlite`, sha256 `25b178dfe6539b4ad2639a6cd5d87c070254befb76da35d4b6418ba4d241d196` (= the `sha256sum` line at the end of `lab logs 506`); 245,591 rows, `PRAGMA integrity_check` ok |
| Verify | `shasum -a 256 demo/sentinel.sqlite` against `lab logs 506`. Rebuilding locally (`LAB_RUN_ID=506 python replay.py --world sim --policy baseline,sentinel --out /tmp/s.sqlite` after the cache commands) gives the same stories, row count and LAB_METRICs but not the same bytes (wall-clock `runs.created_at`) |
| Size | 45.7 MB (R18 < 50 MB), strict schema mode |
| Week | `runs.week_start = 2025-06-02`: Mon 2 – Sun 8 June 2025, 00:00 Europe/Bucharest (= 2025-06-01T21:00Z, the app's `WEEK_START_UTC`); `ts` in UTC |
| Runs | `506:baseline`, `506:sentinel` |
| World | 5 of the fleet's 15 feeders (118 sites, the hero + 3 other congested + 1 uncongested), wind park WP01 (70 turbines, Fântânele-Cogealac area, Constanța) |

## Components (runs.notes)
- Real: M1 PVNet forecast mode (`pvnet-392-s0`, day-ahead PV), M2 LoadNet 4-seed ensemble (`loadnet-387-s0x4`,
  on forecast weather), M4 SiteHealth (`sitehealth-391-s0`, on the 24 h ending at local midnight plus re-checks at
  10/12/14/16 h, expected power from the M1 nowcast ensemble `pvnet-383-s0x4`), M6 WindNBM (`windnbm-r135`,
  per-step classes), planner LP (#18) with aggregate backup, qualify + triage (#26), committer (#27), demand (#44),
  grid-twin (#13: tau + effective G per feeder, 11 GIS flags), baseline wind alerts from WindMon's four rules (#24,
  `sentinel.eval.baselines.wind.windmon_rules`, on all PPC sensors of the #22 cache). 0 prediction fallbacks
  (`predictions.is_fallback`); no fallback component. The LP (#18, with #80's negative-price fix) is optimal on all
  35 feeder-days (`runs.notes.plan_status`).
- P10 commitment: the committer samples PV/load bands recalibrated online by split conformal
  (`sentinel.eval.uncertainty.conformal`, hourly buckets) on the replay's past days only (observed telemetry;
  site-days off the forecast by > 3x in energy are dropped, offsets clipped to ±2 band widths); day 0 is
  uncalibrated. `runs.notes.commit_calibration` has n and raw coverage per day.
- Telemetry loss (#78 drills): every telemetry read goes through `World.observed_site_day` (physics stays
  `site_day`); a pool site dropped only for missing telemetry keeps running the schedule on its device, rejoins the
  pool when fresh, and flex expected to be blind is not committed (`commitments.haircut_h` per step; loop.py
  docstring).
- M6 park-consensus rule (replay only, `sentinel/agent/predictors.py`): #41's `decide()` vetoes curtailment unless
  15 % of the park sits at one power ceiling; a cap to 0 % leaves noisy near-zero power, so the 70-turbine cap
  below came out "normal" although the classifier gave p(non-normal) ≈ 0.999. The replay marks steps where ≥ 50 %
  of the park has p(curtailment) ≥ its threshold (`runs.notes.wind_m6_park_consensus_steps`).

## Topology geometry and grid-twin
- `feeders.geometry_json`: a GeoJSON LineString per feeder, `[lon, lat]`: the depth-first walk of a minimum spanning
  tree from the substation head (the feeder's locality centre in #11's feeders table) through the feeder's sites
  (`replay.feeder_line`, <= 200 points). A plausible radial layout, not a surveyed one.
- Feeder `tau`, `limit_kw` (= G_f = tau·G_max + (1-tau)·G_conservative, only ever tightens) and `gis_trust`
  (`low` if tau < 0.8) come from the grid-twin (#13: `python -m sentinel.gridtwin.run`, cache
  ~/.cache/sentinel/data-v4/gridtwin; in-process fallback, else contract G with tau = 1). `gis_flags` rows are the
  twin's phase / feeder flags on demo sites (Point geometry), audited as `gridtwin:sentinel.gridtwin.twin@#13`.

## Mock world (`replay.py --world mock`)
Same ids and week as this file (F7, 1042, WP01.TIDxx, F5/F6/F8, week from 2025-06-01T21:00Z), so the UI can be
tested without the caches: 1042's string outage starts day 3 05:00 and is derated + gets a truck work order at the
10:00 re-check; ~49 MB. Label synthetic RO, all predictors mock (`is_fallback = 1`).

## Ids (match apps/web: demo-jumps.tsx, replay.ts)
Feeders `F<n>` (fleet `F0n` without the zero), sites `1000 + n` (fleet `S0nnn`), park `WP01` (fleet `WP1`), turbines
`WP01.TIDxx`. Two labels are moved, numbers are not: the hero feeder (fleet **F06 Oșorhei, Bihor**, picked from the
data as the congested feeder with the largest midday baseline overload on day 2) is labelled **F7** (fleet F07 Măgura
becomes F6), and the story site (fleet **S0284**, 6.1 kWp + 8.5 kWh battery on that feeder) is labelled **1042**.
`runs.notes.world.feeder_ids` and `story_site` keep the mapping.

## Stories (verify with sqlite3)
| story | when (local) | what the DB shows |
|---|---|---|
| F7 congested at midday | day 2 (Wed 4 Jun) | limit 61.8 kW (grid-twin tau 0.685: G = tau·65.5 + (1-tau)·53.9); baseline peak export 90.3 kW, 18 steps (270 min, 75.9 kWh) over; Sentinel peak 61.7 kW, 0 steps over |
| F10 (alternative jump) | day 2 (Wed 4 Jun) | limit 63.6 kW; baseline peak 77.8 kW, 270 min / 44.0 kWh over; Sentinel 57.1 kW, 0 min. F10 is clean all week under Sentinel (1605 -> 0 min) |
| String outage at 1042 | from day 3 05:00 (Thu 5 Jun) | one of two strings open (#12's string_outage physics). M4 re-check at 11:00: `derated` (headroom 0.50) + truck work order `string_outage`, residential installer; stays `derated` (headroom 0.70) on the following days |
| Human review (OOD) | from day 0 | 7 sites, 11 `review_queue` rows (M4 OOD robust-z ≥ 3: 1187, 1200, 1280, 1288, 1304, 1316, 1424); no work order |
| Turbine stopped while neighbours produce | day 2 13:45 – day 3 15:30 | WP01.TID24 (#23 fault_stop) at standby (≈ −9.5 kW) while the other 69 average 177-551 kW on day 3 09:00-15:00; M6 `fault_stop` -> `excluded` from day 2 13:45, wind-technician work order |
| Park-wide curtailment | day 5 18:30 – day 6 00:00 | #23's cap to 0 % on all 70 WP01 turbines: one `curtailment` alert, 105.5 MWh, `creates_work_order = 0`, statuses `eligible / curtailed` |

The app's jumps: HERO_DAY = 2 and FAULT_DAY = 3 match. The O&M jump (`FAULT_STEP + 1 day` = day 4 noon) shows the
open turbine work orders; the stop itself is best seen on day 3 noon, the curtailment on day 5 ~19:00 (Tudor: #62).

## KPIs (run 506, synthetic RO, fleet-trained weights)
| KPI (kpis table name) | baseline | Sentinel |
|---|---|---|
| minutes over limit (`minutes_over_limit`, 5 feeders x 7 days) | 6645 | 1905 |
| energy over the limit (`energy_over_limit`, kWh) | 1397 | 94.5 |
| worst overshoot above the limit (`peak_overshoot`, kW, any feeder-day) | 29.1 | 10.6 |
| peak reduction (mean over feeder-days / congested feeder-days) | — | 19.1 % / 24.1 % |
| P10 commitment hit rate (`p10_hit_rate`; 1845 kWh committed, 16 kWh short) | — | 92.0 % |
| comfort breaches (battery below reserve, steps) | 1165 | 0 |
| lei per home per month (vs baseline) | — | 23.73 saved |
| alerts per turbine-day | 0.698 (WindMon rules, #24) | 0.114 |

Hero feeder F7 day by day (`minutes_over_limit_F7_d<d>`, `energy_over_limit_F7_d<d>`, `peak_overshoot_F7_d<d>`):

| day (local) | baseline min / kWh / overshoot kW | Sentinel min / kWh / overshoot kW |
|---|---|---|
| d0 Mon 2 Jun | 285 / 79.1 / 27.4 | 375 / 27.3 / 8.6 |
| d1 Tue 3 Jun | 300 / 97.7 / 28.7 | 165 / 4.2 / 3.6 |
| d2 Wed 4 Jun | 270 / 75.9 / 28.4 | 0 / 0 / 0 |
| d3 Thu 5 Jun | 285 / 80.9 / 23.5 | 150 / 3.0 / 2.7 |
| d4 Fri 6 Jun | 300 / 89.8 / 24.4 | 180 / 2.8 / 2.9 |
| d5 Sat 7 Jun | 225 / 34.2 / 16.7 | 0 / 0 / 0 |
| d6 Sun 8 Jun | 120 / 20.2 / 16.3 | 0 / 0 / 0 |

Minutes alone mislead: on Mon the LP trades a few more steps just over the limit for a third of the energy over it
(the LP penalises energy above G, not the count of steps). F7 is still over the limit on Mon, Tue, Thu and Fri.

Earlier runs: 322 (interim weights, LP fallback on 10 days) 2880 min / P10 83.4 %; 380 (interim weights, LP fixed)
2565 min / P10 82.2 % / 15.0 %; 506 = fleet-trained weights + conformal commitment + #24 WindMon baseline.

Reproduce: build the caches (commands above, ~6 min), then the replay (~20 s). `make replay REPLAY_WORLD=sim` still
uses the old `SENTINEL_DATA` fleet path; use the command above until #35 updates the Makefile.
