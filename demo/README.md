# demo/sentinel.sqlite — the demo replay (card #29)

**Label: synthetic RO** (ERA5-driven simulator: #11 fleet + #12 faults, #22/#23 wind) with **real OPCOM prices**.
**Weights: INTERIM** — M1/M2/M4 trained on #14's own synthetic data (models/README.md, runs 201-210), M6 WindNBM
on the pre-#22 wind stand-in (#41 run 135). Not fleet-trained; numbers below are a pipeline demo of the agent on
the real simulator week, not model results. The fleet-retrained weights replace them in stage 2.

| | |
|---|---|
| Lab run | **replay#380** (card #29, worker rtx5070, commit `28f94fbf08`, `--gpu-mem 12G`) |
| Command | `python -m sentinel.data.windows && python -m sentinel.sim.wind --no-plots && python -m sentinel.gridtwin.run && python replay.py --world sim --policy baseline,sentinel --out $LAB_OUTPUT_DIR/sentinel.sqlite` |
| File | the exact worker output: `scp rtx5070:/var/lib/lab-gpu/work/outputs/380/sentinel.sqlite`, sha256 `c992a97c75abceeaf7fbb7566901eb52b7d920829d6fa30792d41f0e1d6bd8f6` (= the `sha256sum` line at the end of `lab logs 380`); 245,583 rows, `PRAGMA integrity_check` ok |
| Verify | `shasum -a 256 demo/sentinel.sqlite` against `lab logs 380`. Rebuilding locally (`LAB_RUN_ID=380 python replay.py --world sim --policy baseline,sentinel --out /tmp/s.sqlite` after the cache commands) gives the same stories and row count but not the same bytes (wall-clock `runs.created_at`; macOS vs Linux floats move peak reduction 15.04 -> 15.02 %, lei 24.66 -> 24.65) |
| Size | 45.7 MB (R18 < 50 MB), strict schema mode |
| Week | `runs.week_start = 2025-06-02`: Mon 2 – Sun 8 June 2025, 00:00 Europe/Bucharest (= 2025-06-01T21:00Z, the app's `WEEK_START_UTC`); `ts` in UTC |
| Runs | `380:baseline`, `380:sentinel` |
| World | 5 of the fleet's 15 feeders (118 sites, the hero + 3 other congested + 1 uncongested), wind park WP01 (70 turbines, Fântânele-Cogealac area, Constanța) |

## Components (runs.notes)
- Real: M1 PVNet forecast mode (`pvnet-210-s0`, day-ahead PV), M2 LoadNet 4-seed ensemble (`loadnet-205-s0x4`,
  on forecast weather), M4 SiteHealth (`sitehealth-209-s0`, on the 24 h ending at local midnight plus re-checks at
  10/12/14/16 h, expected power from the M1 nowcast ensemble `pvnet-201-s0x4`), M6 WindNBM (`windnbm-r135`,
  per-step classes), planner LP (#18) with aggregate backup, qualify + triage (#26), committer (#27), demand (#44),
  grid-twin (#13: tau + effective G per feeder, 11 GIS flags). 0 prediction fallbacks (`predictions.is_fallback`).
- Fallback: WindMon baseline (#24 stand-in rules) for the baseline policy's wind alerts only. The LP (#18, with
  #80's negative-price fix) is optimal on all 35 feeder-days (`runs.notes.plan_status`); run 322 had 10 "unbounded"
  days on the greedy fallback.
- Telemetry loss (#78 drills): every telemetry read goes through `World.observed_site_day` (physics stays
  `site_day`); a pool site dropped only for missing telemetry keeps running the schedule on its device, rejoins the
  pool when fresh, and flex expected to be blind is not committed (`commitments.haircut_h` per step; loop.py
  docstring). On this clean sim week it changes only the P10 hit rate (82.5 -> 82.2 %, re-admission).
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
| F7 congested at midday | day 2 (Wed 4 Jun) | limit 61.8 kW (grid-twin tau 0.685: G = tau·65.5 + (1-tau)·53.9); baseline peak export 90.3 kW, 18 steps over; Sentinel peak 63.9 kW, 5 steps over |
| String outage at 1042 | from day 3 05:00 (Thu 5 Jun) | one of two strings open (#12's string_outage physics). M4 re-check at 11:00: `derated` (headroom 0.52) + truck work order `string_outage`, residential installer. Day 4 11:00 M4 flips to sensor_fault -> `excluded` (model error, see insights) |
| Human review (OOD) | from day 0 | 5 sites, 15 `review_queue` rows (M4 OOD robust-z ≥ 3), e.g. site 1403 (ood 6.0) at day 0; no work order |
| Turbine stopped while neighbours produce | day 2 13:45 – day 3 15:30 | WP01.TID24 (#23 fault_stop) at standby (≈ −9.5 kW) while the other 69 average 177-551 kW on day 3 09:00-15:00; M6 `fault_stop` -> `excluded` from day 2 13:45, wind-technician work order |
| Park-wide curtailment | day 5 18:30 – day 6 00:00 | #23's cap to 0 % on all 70 WP01 turbines: one `curtailment` alert, 105.5 MWh, `creates_work_order = 0`, statuses `eligible / curtailed` |

The app's jumps: HERO_DAY = 2 and FAULT_DAY = 3 match. The O&M jump (`FAULT_STEP + 1 day` = day 4 noon) shows the
open turbine work orders; the stop itself is best seen on day 3 noon, the curtailment on day 5 ~19:00 (Tudor: #62).

## KPIs (run 380, synthetic RO, interim weights)
| KPI | baseline | Sentinel |
|---|---|---|
| minutes over limit (5 feeders x 7 days) | 6645 | 2565 |
| peak reduction (mean over feeder-days / congested feeder-days) | — | 15.0 % / 19.5 % |
| P10 commitment hit rate | — | 82.2 % |
| comfort breaches (battery below reserve, steps) | 1165 | 0 |
| lei per home per month (vs baseline) | — | 24.66 saved |
| alerts per turbine-day | 0.706 (WindMon rules) | 0.114 |

Run 322 (before #80's LP fix and the telemetry-loss fix): 2880 minutes over, 13.6 % / 17.8 %, 83.4 %, 15.7 lei.

Reproduce: build the caches (commands above, ~6 min), then the replay (~20 s). `make replay REPLAY_WORLD=sim` still
uses the old `SENTINEL_DATA` fleet path; use the command above until #35 updates the Makefile.
