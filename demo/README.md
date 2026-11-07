# demo/sentinel.sqlite — the demo replay (card #29)

**Label: synthetic RO** (ERA5-driven simulator: #11 fleet + #12 faults, #22/#23 wind) with **real OPCOM prices**.
**Weights: INTERIM** — M1/M2/M4 trained on #14's own synthetic data (models/README.md, runs 201-210), M6 WindNBM
on the pre-#22 wind stand-in (#41 run 135). Not fleet-trained; numbers below are a pipeline demo of the agent on
the real simulator week, not model results. The fleet-retrained weights replace them in stage 2.

| | |
|---|---|
| Lab run | **replay#322** (card #29, worker server-gpu, commit `cc846941ea`) |
| Command | `python -m sentinel.data.windows && python -m sentinel.sim.wind --no-plots && python -m sentinel.gridtwin.run && python replay.py --world sim --policy baseline,sentinel --out $LAB_OUTPUT_DIR/sentinel.sqlite` |
| File | regenerated locally from the same commit and the same deterministic caches with `LAB_RUN_ID=322` (run 322 landed on server-gpu, whose outputs are not reachable; rtx5070 was busy with #14's retrain). Not byte-identical: the worker file's sha256 is `4323c70a…3d1f9` (printed in `lab logs 322`), this file's is `14f13d4c…01baaa`, because `runs.created_at/finished_at` are wall-clock. Verified equal instead: all 13 LAB_METRICs of run 322 and the row count (245,583) |
| Verify | `LAB_RUN_ID=322 python replay.py --world sim --policy baseline,sentinel --out /tmp/s.sqlite` after the cache commands, then compare its LAB_METRIC lines with `lab logs 322` |
| Size | 45.7 MB (R18 < 50 MB), strict schema mode |
| Week | `runs.week_start = 2025-06-02`: Mon 2 – Sun 8 June 2025, 00:00 Europe/Bucharest (= 2025-06-01T21:00Z, the app's `WEEK_START_UTC`); `ts` in UTC |
| Runs | `322:baseline`, `322:sentinel` |
| World | 5 of the fleet's 15 feeders (118 sites, the hero + 3 other congested + 1 uncongested), wind park WP01 (70 turbines, Fântânele-Cogealac area, Constanța) |

## Components (runs.notes)
- Real: M1 PVNet forecast mode (`pvnet-210-s0`, day-ahead PV), M2 LoadNet 4-seed ensemble (`loadnet-205-s0x4`,
  on forecast weather), M4 SiteHealth (`sitehealth-209-s0`, on the 24 h ending at local midnight plus re-checks at
  10/12/14/16 h, expected power from the M1 nowcast ensemble `pvnet-201-s0x4`), M6 WindNBM (`windnbm-r135`,
  per-step classes), planner LP (#18) with aggregate backup, qualify + triage (#26), committer (#27), demand (#44),
  grid-twin (#13: tau + effective G per feeder, 11 GIS flags). 0 prediction fallbacks (`predictions.is_fallback`).
- Fallback: WindMon baseline (#24 stand-in rules) for the baseline policy's wind alerts; the LP returned
  "unbounded" on 10 of 35 feeder-days (days 4-5), which then used the built-in greedy plan (counted in
  `runs.notes.component_errors`; follow-up for #18).
- M6 park-consensus rule (replay only, `sentinel/agent/predictors.py`): #41's `decide()` vetoes curtailment unless
  15 % of the park sits at one power ceiling; a cap to 0 % leaves noisy near-zero power, so the 70-turbine cap
  below came out "normal" although the classifier gave p(non-normal) ≈ 0.999. The replay marks steps where ≥ 50 %
  of the park has p(curtailment) ≥ its threshold (`runs.notes.wind_m6_park_consensus_steps`).

## Ids (match apps/web: demo-jumps.tsx, replay.ts)
Feeders `F<n>` (fleet `F0n` without the zero), sites `1000 + n` (fleet `S0nnn`), park `WP01` (fleet `WP1`), turbines
`WP01.TIDxx`. Two labels are moved, numbers are not: the hero feeder (fleet **F06 Oșorhei, Bihor**, picked from the
data as the congested feeder with the largest midday baseline overload on day 2) is labelled **F7** (fleet F07 Măgura
becomes F6), and the story site (fleet **S0284**, 6.1 kWp + 8.5 kWh battery on that feeder) is labelled **1042**.
`runs.notes.world.feeder_ids` and `story_site` keep the mapping.

## Stories (verify with sqlite3)
| story | when (local) | what the DB shows |
|---|---|---|
| F7 congested at midday | day 2 (Wed 4 Jun) | limit 61.8 kW (grid-twin tau 0.685: G = tau·65.5 + (1-tau)·53.9); baseline peak export 90.3 kW, 18 steps over; Sentinel peak 63.9 kW, 5 steps over |
| String outage at 1042 | from day 3 05:00 (Thu 5 Jun) | one of two strings open (#12's string_outage physics). M4 re-check at 11:00: `derated` (headroom 0.52) + truck work order `string_outage`, residential installer. Day 4 11:00 M4 flips to sensor_fault -> `excluded` (model error, see insights) |
| Human review (OOD) | from day 0 | 5 sites, 15 `review_queue` rows (M4 OOD robust-z ≥ 3), e.g. site 1403 (ood 6.0) at day 0; no work order |
| Turbine stopped while neighbours produce | day 2 13:45 – day 3 15:30 | WP01.TID24 (#23 fault_stop) at standby (≈ −9.5 kW) while the other 69 average 177-551 kW on day 3 09:00-15:00; M6 `fault_stop` -> `excluded` from day 2 13:45, wind-technician work order |
| Park-wide curtailment | day 5 18:30 – day 6 00:00 | #23's cap to 0 % on all 70 WP01 turbines: one `curtailment` alert, 105.5 MWh, `creates_work_order = 0`, statuses `eligible / curtailed` |

The app's jumps: HERO_DAY = 2 and FAULT_DAY = 3 match. The O&M jump (`FAULT_STEP + 1 day` = day 4 noon) shows the
open turbine work orders; the stop itself is best seen on day 3 noon, the curtailment on day 5 ~19:00 (Tudor: #62).

## KPIs (run 322, synthetic RO, interim weights)
| KPI | baseline | Sentinel |
|---|---|---|
| minutes over limit (5 feeders x 7 days) | 6645 | 2880 |
| peak reduction (mean over feeder-days / congested feeder-days) | — | 13.6 % / 17.8 % |
| P10 commitment hit rate | — | 83.4 % |
| comfort breaches (battery below reserve, steps) | 1165 | 0 |
| lei per home per month (vs baseline) | — | 15.7 saved |
| alerts per turbine-day | 0.706 (WindMon rules) | 0.114 |

Reproduce: build the caches (commands above, ~6 min), then the replay (~20 s). `make replay REPLAY_WORLD=sim` still
uses the old `SENTINEL_DATA` fleet path; use the command above until #35 updates the Makefile.
