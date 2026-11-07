# Plan: Prosumer Sentinel v2

**A fleet flexibility agent that coordinates home batteries on each LV feeder, so midday solar export stays under
the feeder's limit and is released into the evening peak, and that commits only the flexibility healthy,
predictable installations can actually deliver.**

One model (SentinelNet), one optimiser (a per-feeder LP), no hardware at the home. The O&M challenge decides which
installations the fleet may count on (the work order is how excluded sites get back into the pool). The GIS
challenge supplies feeder limits and how far to trust them. If PPC leads with O&M or GIS, the same build is
re-weighted in the pitch, not rebuilt.

Source: team doc "Prosumer Sentinel v2 — Smarthack XI PPC" (Nov 7), which supersedes v1. This file is the binding,
sorted version; where they differ, this file wins (see §8 Review notes).

---

## 1. Users and value

| User | Gets | Value |
|---|---|---|
| PPC distribution / network ops (primary payer) | Day-ahead plan per feeder; export peaks held under the limit | Deferred reinforcement, fewer curtailments and overvoltage trips, more connections approved |
| PPC supply / aggregator desk | Hourly committed flexibility (MW) with a P10 guarantee and a delivery record | Lower imbalance cost; balancing revenue later (not assumed) |
| PPC O&M | Daily top-K inspection list ranked by lost energy and lost flexibility | Fewer blind truck rolls; repairs return MW to the pool |
| PPC GIS / planning | Feeder/phase mismatch flags + trust score per LV section (GeoJSON) | Review requests where the GIS is likely wrong |
| Prosumers (opt-in, free) | Battery schedule that never breaks their comfort reserve; tips; early fault alerts | Lower bills where the tariff allows; share of flexibility value |

---

## 2. Success criteria

Targets are aims; results are reported as measured, misses included. Every slide number cites a lab run ID.
Simulated and real-data results are always labelled separately.

| # | Claim | Baselines | Target | Run |
|---|---|---|---|---|
| 1 | **Feeder peak shaving (headline)**: export peak reduction, minutes above limit per feeder-day | self-consumption rule; tuned per-home export-threshold rule | zero violations where battery headroom suffices; residual slack reported otherwise | `replay` |
| 2 | Evening import peak reduction per feeder | same | reported | `replay` |
| 3 | Flexibility reliability: realised ≥ committed (P10) | n/a | ≥ 90% of feeder-steps; available/nominal and committed/nominal reported | `replay` |
| 4 | Nowcast nMAE on real PVDAQ | pvlib (recorded metadata), persistence, LightGBM | ≥ 20% lower than pvlib; 10–90% coverage 78–82% | `train` / `eval` |
| 5 | Load forecast on held-out London households | persistence, LightGBM | beat persistence | `train` / `eval` |
| 6 | Fault detection: PR-AUC + hours-to-detect (synthetic, PVDAQ + injected) | threshold rule (< 80% of pvlib for 3 days); CUSUM on pvlib residual | beat both | `train` / `eval` |
| 7 | Customer: comfort-reserve breaches; lei per home per month (OPCOM, net of degradation) | default rule | zero breaches; lei reported | `replay` |
| 8 | Grid-twin: % of corrupted GIS feeder/phase records recovered | n/a | reported | `gridtwin` |

`train` prints `LAB_METRIC val_nmae, val_pinball, val_load_pinball, val_prauc, hours_to_detect`.
`replay` prints `LAB_METRIC peak_reduction, minutes_over_limit, p10_hit_rate, comfort_breaches, lei_per_home`.

---

## 3. Architecture

### 3.1 The loop (15-min tick)

```
 perceive ──► estimate ─────► qualify ──────► decide ───────────► act ────────────► log
 telemetry    SentinelNet     site status     per-feeder LP       battery schedules  SQLite audit:
 weather      PV   P10/50/90  eligible        + P10 commitment    feeder commitment  model version,
 prices       load P10/50/90  derated         (limit G from       work orders        input hash,
 metadata     fault class     excluded        grid-twin, comfort  alerts             prediction,
 (Pandera     embedding/OOD   human review    reserve R)          GeoJSON flags      threshold, status,
  checks)                        │                                  ▲                schedule; realised
                                 └──── O&M top-K triage ────────────┘                vs committed
                                       (repaired sites rejoin the pool)
```

In the demo the agent replays a test week from precomputed predictions and a precomputed replay; the preview
CPU never runs torch or the LP.

### 3.2 One model output, two uses

| SentinelNet output | O&M uses it as | Fleet optimiser uses it as |
|---|---|---|
| PV P10/P50/P90 | reference a fault is measured against | charge headroom (surplus to absorb) |
| Load P10/P50/P90 | context for battery-underuse faults | discharge headroom, comfort reserve |
| Fault probability + class | work order and its priority | exclusion or derating of the site |
| Ensemble spread, interval width, OOD | human review instead of a truck roll | haircut on committed MW, or exclusion |

### 3.3 Components

| Component | Path | In → Out | Card |
|---|---|---|---|
| Contracts + mock predictor | `sentinel/contracts/`, `team/contracts/` | types, schemas, `MockPredictor` | #7 |
| Raw data | `sentinel/data/download.py`, `data/raw/` | public sources → committed Parquet | #9 |
| Fleet simulator | `sentinel/sim/fleet/` | weather + load → sites on feeders, telemetry under the baseline policy | #11 |
| PV fault injector | `sentinel/sim/faults/` | telemetry → telemetry + labels with onset | #12 |
| Windows, splits, Pandera | `sentinel/data/{windows,splits,schemas}.py` | Parquet → memmapped float16 windows | #10 |
| SentinelNet + training | `sentinel/model/`, `train.py` | windows → ONNX + calibration JSON | #14 |
| Baselines (PV, load, fault) | `sentinel/eval/baselines/pv/` | windows → baseline predictions + metrics | #15 |
| Evaluation | `eval.py`, `sentinel/eval/{metrics,report}.py` | predictions → LAB_METRIC + JSON report | #16 |
| Uncertainty + P10 rollout | `sentinel/eval/uncertainty/`, `sentinel/opt/commit.py` | quantiles → calibrated quantiles, thresholds, committed MW | #27 |
| Qualify + triage | `sentinel/agent/{qualify,triage}.py` | predictions → site status, top-K work orders | #26 |
| Per-feeder LP + baseline policies | `sentinel/opt/` | forecasts + status + limits → schedules | #18 |
| Agent loop + replay + audit | `sentinel/agent/{loop,audit,replay}.py`, `replay.py` | all of the above → actions, audit log, replay metrics | #17 |
| Grid-twin | `sentinel/gridtwin/` | IEEE feeder → flags, trust τ, effective limit G | #13 |
| API + UI (Windmon frontend) | `sentinel/api/`, `web/`, `demo/fixtures/` | `demo/replay/` or fixtures → dashboard | #21, #30, #19, #31, #32, #33 |
| Robustness drills | `sentinel/data/drills.py` | test set → drill variants | #28 |
| Promote | `models/`, `demo/` | chosen runs → committed weights + replay outputs | #29 |

### 3.4 Key interfaces (frozen by #7; change only with `lab announce`)

- `predict(window) -> {pv_q[3×96], load_q[3×96], fault_probs[8], embedding[d], ood_score, member_std}`.
  PV window = 24 h weather + solar geometry (no target power) + reference window of the site's own past power
  (7–14 days) + 14-day daily residual summary + battery channels. Forecast mode uses noisy weather, never ERA5 actuals.
- `qualify(site, prediction, telemetry_age) -> {status: eligible|derated|excluded|human_review, pv_headroom_factor, reason}`.
- `plan_feeder(feeder, eligible_sites, rest_of_feeder, forecasts, G, prices) -> {schedule[site][96], export_curve[96], slack[96]}`.
- `commit(feeder, schedule, scenarios) -> {nominal[96], available[96], committed[96]}`; `realised` is logged next tick.
- Actions (JSON): `battery_schedule`, `feeder_commitment`, `work_order`, `alert`, `gis_flag`, `human_review`,
  each with an `audit_record` (model_version, input_hash, prediction, threshold, site_status, schedule).

### 3.5 Decision maths (written out here; the formulas in the v2 doc were images)

**Qualify** (per site, per tick): stale telemetry or sensor fault → excluded (delivery can't be verified);
ensemble spread / conformal width / OOD above validation-tuned thresholds → human review (no work order, out of
the pool); confident PV fault → derated (PV headroom cut by the estimated loss; battery stays in if its own
telemetry is healthy); battery fault → excluded. Feeder available/committed MW are recomputed next tick.

**O&M triage score** (top-K per day; sensor faults → remote check, never a truck):

```
score_i = P_fault,i × ( lost_kWh_day,i × π  +  lost_flex_MWh_day,i × v_flex ) × D_i
lost_kWh_day,i = Σ_daylight (P50_i,t − actual_i,t) · Δt          D_i = days until the next possible visit
```

**Per-feeder LP** (feeder f, t = 1..96, Δt = 0.25 h, eligible sites i; all variables ≥ 0):

```
min  Σ_t Σ_i [ π_t·m_it·Δt − β·π_t·x_it·Δt + c_deg·(c_it + d_it)·Δt ]  +  M·Σ_t s_t·Δt  +  λ·z

site balance (P50):   P̂50_it − L̂50_it − c_it + d_it = x_it − m_it
battery:              e_i,t+1 = e_it + η_c·c_it·Δt − d_it·Δt/η_d ;   R_i ≤ e_it ≤ E_i ;   c_it, d_it ≤ P_i ;   e_i,96 ≥ e_i,0
feeder export limit:  Σ_i (P̂90_it − L̂10_it − c_it + d_it) + X_t^rest ≤ G_f + s_t        (worst plausible export)
evening peak:         z ≥ Σ_i (m_it − x_it) + M_t^rest      for t in 17:00–22:00
```

- `X_t^rest`, `M_t^rest`: forecast net export / import of the feeder's **non-eligible** sites, which keep running
  their own self-consumption rule. They still sit on the feeder (review note R3).
- `M ≫ λ ≫ π`. LP, not MILP: with η_c·η_d < 1 and c_deg > 0, charging and discharging at once only costs money.
  Check solutions; add a binary only if it appears (negative prices).
- `G_f = τ·G^max + (1 − τ)·G^conservative` from the grid-twin; low trust only ever tightens the limit.
- `c_deg = C_pack / (2 · N_cycles · E_usable)` ≈ €0.02–0.03/kWh per direction (assumed €250–400/kWh LFP, 6,000
  cycles); round-trip ~90%. Inputs, not results. `β` = export remuneration ratio, tariff-dependent (to verify).
- 10–20 independent LPs of ~25–50k variables; HiGHS should solve each in seconds (confirm on our machines).
- Fallback: one aggregate virtual battery per feeder, same LP, split to sites by headroom, then a per-site
  feasibility clip.

**P10 commitment**: conformal-calibrate each ensemble member's quantiles → ~200 scenarios per feeder-day (one
shared PV quantile level per feeder, independent load levels per site, one ensemble member per scenario) → replay
the schedule through the battery model with clipping → deliverable = export reduction vs the baseline rule →

```
committed_t = (1 − h) · Q10(deliverable_t)        h = telemetry-loss haircut
```

Four numbers, never conflated: **nominal** (Σ rated battery kW) ≥ **available** (what eligible batteries can move
at t given SoC, reserve, surplus, load) ≥ **committed** (P10 of available, minus haircut); **realised** is measured
afterwards (simulator ground truth in the hackathon). Hit rate and size of misses are published.

### 3.6 Runs and artifacts

| Run | Where | Command | Output |
|---|---|---|---|
| `check` | server sandbox | `pytest tests` (contract tests included) | pass/fail; gates auto-merge |
| `preview` | server sandbox | FastAPI serving `web/dist` + `/api` from `demo/replay/` (fixtures until promoted) | the shared app; `lab public on` for the demo |
| `train` | GPU workers | `python train.py` (`--grid` for sweeps and `seed=0,1,2,3`) | LAB_METRIC; ONNX + calibration JSON in `$LAB_OUTPUT_DIR`, uploaded as a release asset |
| `baseline` | CPU | `python eval.py --baselines` | baseline LAB_METRIC |
| `replay` | CPU | `python replay.py --policy baseline,sentinel` | replay LAB_METRIC; per-feeder curves, statuses, schedules, commitments, work orders, audit DB |
| `eval` | GPU/CPU | `python eval.py` | test-set LAB_METRIC + JSON report |

Run outputs stay on the worker, so the promote card (#29, review) commits the chosen weights to `models/` and the
chosen replay outputs to `demo/replay/`. The preview and `make reproduce` read only committed files.

### 3.7 Repo layout

```
sentinel/
  contracts/  data/  sim/fleet/  sim/faults/  model/  eval/baselines/pv/  eval/uncertainty/
  opt/  agent/  gridtwin/  api/
web/                      Windmon React frontend (files only, credited)
train.py  eval.py  replay.py  Makefile (make reproduce)
data/raw/                 committed Parquet (10–30 MB)
models/  demo/replay/     promoted artifacts (cite run IDs)
tests/                    test_contracts.py + per-card tests
team/                     brief, plan, contracts, decisions, insights
```

---

## 4. Data

| Source | We commit / use | Role |
|---|---|---|
| Open-Meteo ERA5 | hourly irradiance, temp, cloud, wind, snow; 20–40 RO locations, 2022–2025 | model input; drives the simulator |
| pvlib | solar position, clear-sky, PV model | simulator core + physical baseline |
| NREL PVDAQ | a handful of real multi-year PV systems | real-data nowcast test; real background for injected faults |
| OPCOM PZU | hourly day-ahead prices 2023–2025 (fallback ENTSO-E) | LP prices; kWh → lei |
| London Smart Meter | ~200 households, half-hourly | load profiles; load-head train/test |
| pandapower IEEE European LV | 55-customer 3-phase feeder | grid-twin only |

- **Synthetic fleet**: 500–1,000 residential prosumers × 2–3 years at 15 min on 10–20 feeders with clustered PV
  penetration (some congest, some don't). kWp 3–10, random tilt/azimuth, battery 0–15 kWh / 3–5 kW, comfort
  reserve, an export limit per feeder. Voltage = feeder base + PV-driven rise + noise. History is generated under
  the self-consumption rule (the baseline policy).
- **Fair pvlib baseline**: wrong recorded tilt/azimuth on ~20% of sites (±10–30°), inverter curves, 0.5–1%/yr
  degradation, snow, horizon shading; pvlib sees only recorded metadata. A suspiciously large synthetic win is a bug
  until explained; the headline nowcast claim is on PVDAQ.
- **Faults**: soiling (reset by rain), partial shading, string outage (1/N step), clipping/derating, overvoltage
  trips, battery underuse/fade, sensor faults; random onset and severity; ~15% of site-days; the same injector runs
  on PVDAQ.
- **Splits**: train 2022–2023 × 70% of sites; val 2024; test 2025 + the 30% unseen sites; PVDAQ clean; PVDAQ +
  injected; a separate calibration slice. Thresholds frozen on validation. Planning never sees ERA5 actuals.
- **Generator**: deterministic, < 5 min, cached at `~/.cache/sentinel/data-v{N}` (bump N on change).
- **Known limitation**: London load is not Romanian load; stated on the slide, with a scaled/shifted sensitivity test.

---

## 5. Scope tiers

| Tier | Items |
|---|---|
| **MUST** | simulator + injector · SentinelNet (PV, load, fault) or LightGBM fallback · baselines · site qualification · top-K O&M · per-feeder LP · P10 commitment · baseline-vs-Sentinel replay · audit log · dashboard (Windmon frontend) |
| **SHOULD** | grid-twin trust + conservative limits · per-customer comfort reserve · degradation cost · similar past cases · human-review queue UI · robustness drills · hourly LP re-solve |
| **OPTIONAL** | wind turbines from Windmon (#22–#24, O&M list only) · Integrated Gradients · LLM text · routing · GIS correction beyond flags · real vendor API integration |

**Cut order if behind**: LLM text → routing → similar cases → wind → grid-twin to one slide → full LP to the
aggregate per-feeder LP → PV forecast mode (use the pvlib forecast × the nowcast's per-site correction).
If SentinelNet isn't ready by hour 12, LightGBM quantile models feed the same LP; nothing downstream changes.
**Never cut**: simulator, PV + load forecasts, qualification, the LP, the P10 commitment, the baseline comparison,
the work order.

---

## 6. Steps (phases, gates, cards)

| Phase | Hours | Cards | Gate |
|---|---|---|---|
| 1. Setup & contracts | 0–2 | **#21** preview fix (Windmon UI + FastAPI) · **#6** skeleton + run types · **#7** contracts (review) · **#8** cache + CPU-slot probe · **#30** UI fixtures | preview passes; contracts merged; cache result on all 3 workers |
| 2. Data, baselines, LP on mock | 2–7 | #9 raw data · #11 simulator on feeders · #12 PV injector · #10 windows/splits · #15 baselines · #18 LP + baseline policies · #17 agent loop + replay on mock | data committed; data-v1 cached; baselines in `lab top`; LP solves one feeder from mock forecasts; baseline violations visible |
| 3. Model v1 + full loop | 7–12 | #14 SentinelNet · #16 eval.py · #26 qualify + triage · #27 conformal + P10 rollout · #19 #31 #32 UI pages | SentinelNet beats LightGBM on val (or fallback declared); qualify + LP on all feeders; P10 rollout runs |
| 4. Sweeps + integration (night) | 12–17 | #14 sweeps (`--grid`), 4-seed ensemble · #29 promote · #13 grid-twin | 2–3 grid rounds; ensemble; promoted ONNX; full replay with the real model logged |
| 5. Evaluation & robustness | 17–21 | #16 PVDAQ + faults on real data · #28 drills · tuned-rule comparison · grid-twin keep/drop | error analysis in `team/insights.md` |
| 6. Demo freeze | 21–24 | #35 README · #34 slides + rehearsal · #33 demo mode | `lab freeze on` at h22; `make reproduce` on a clean checkout; slides from `lab report --out`; rehearse ×2 |

Optional, only when MUST is on track: #22 wind data, #23 wind faults, #24 wind baselines.

---

## 7. Demo script (3 minutes)

Both runs replay the same test week (identical weather, PV, load, fleet, limits, prices); only the policy differs.
[Brackets] are filled from logged runs; every panel says "simulated" or "real data (PVDAQ)". Every panel is also
precomputed, so it can fall back to replay mode if a live step fails.

1. 0:00–0:20 Fleet map: [N] homes on [n] feeders by PV penetration. "Same roofs, same sun, same hour."
2. 0:20–0:50 Baseline, feeder F7: export crosses the dashed limit at midday; batteries full by late morning.
   "Every battery did its job locally, and was full before the problem started."
3. 0:50–1:30 Sentinel, same day: charging held back, concentrated 11:00–14:00; F7 under the limit; discharge
   18:00–21:00 lowers the evening import. "Same batteries, same energy, different timing."
4. 1:30–1:55 Flexibility panel: nominal vs available vs committed (P10) vs realised. "We never sell the nominal number."
5. 1:55–2:30 String failure at site 1042 on F7: band diverges → derated → F7's committed MW drops → work order in
   tomorrow's top-K with kWh, lei and MW at risk. "O&M is how the pool stays honest."
6. 2:30–2:50 An unusual new site (OOD) → human review: not in the pool, no work order.
7. 2:50–3:00 Summary: peak reduction, minutes over limit, P10 hit rate, comfort breaches, lei per home, each with its run ID.

The demo week is picked from the 2025 test year (a sunny late-spring week with congestion). Headline metrics are
over the whole test year, so the week illustrates; it doesn't prove.

---

## 8. Review notes (v2 reviewed Nov 7; these override the v2 doc)

- **R1 UI**: Windmon React frontend + FastAPI (decided earlier), not Streamlit. The preview reads precomputed
  `demo/replay/`, so it stays CPU-light.
- **R2 Wind**: Windmon wind work is OPTIONAL (#22–#24); wind turbines only join the O&M list, never the LP.
- **R3 Excluded sites still load the feeder**: the LP's feeder constraint includes the forecast net export of
  non-eligible sites (running their own rule) as a constant. Otherwise excluding a site falsely frees capacity.
- **R4 Plan vs reality**: the day-ahead plan fixes the commitment. MUST: day-ahead plan + a per-tick feasibility
  clip in the replay. SHOULD: re-solve every hour with updated SoC and nowcast. Count solves before the night run
  (15 feeders × 24 re-solves × 7 days ≈ 2.5k solves).
- **R5 Replay outputs need a path to main**: runs write to the worker's disk, so #29 promotes them with the weights.
- **R6 CPU runs**: `baseline` and `replay` need a CPU slot and no worker lists one yet. #8 tests `--needs cpu`; if
  none, ask Andrei to start a worker with `--cpu-slots`, or run them as GPU runs.
- **R7 Fleet realism**: residential kWp 3–10 (v1's 3–200 was industrial); history generated under the
  self-consumption rule so SoC telemetry matches the baseline.
- **R8 Coverage**: the 10–90% band target is 78–82% (v1's 88–92% was inconsistent). Adopted from v2.
- **Kept from the v1 review**: nowcast reference window of the site's own past power; fault head gets 14-day
  context; "beats pvlib" is the gate and 20% the stretch; grid-twin power flow may be slow (time 10 steps first; one
  feeder = phase ID only); ERA5 hourly → 15-min via clear-sky index; OPCOM fallback ENTSO-E; contracts allow
  dropping in PPC's own data.

### To verify (never claimed as fact)

- Market figures (150k+ prosumers / 1.7+ GW in 2024–25; 400k / 4 GW by 2030).
- β (export remuneration) and the Romanian compensation scheme; whether residential bills can save at all.
- Aggregator rules for small residential assets (aFRR/mFRR); unbundling and the legal basis for sharing data
  between distribution and supply; GDPR consent; the battery control contract.
- €80–120/home/year in the reference brief has no derivation. We report Σ committed kW(t) × value(t) × share kept,
  minus degradation and losses, from simulation.

---

## 9. Pitch material (from v2, for slides)

- **Why one agent, not two products**: without the health layer the optimiser silently counts broken sites and
  over-commits; without the optimiser, fault detection is a dashboard. Every repair returns MW to the pool.
- **Why a feeder plan, not a smarter home rule**: homes differ; only a feeder plan puts absorption where the
  headroom is. We benchmark a tuned per-home threshold rule and say so if it comes close.
- **Physical limit**: when surplus exceeds total battery headroom, we show the residual slack; we don't hide it
  with curtailment.
- **No hardware**: a pilot uses vendor cloud APIs or the customer's EMS where write access exists; others are
  monitor-only with zero committed MW. Hackathon control is simulated, stated openly. A pilot needs 50–100
  consenting homes on 2–3 congested feeders, vendor permissions, feeder limits from distribution, and a shadow
  phase (plan, don't dispatch).
- **Customer**: opt-in, customer-chosen comfort reserve as a hard constraint, override any time, every action logged.
- **Commercial**: PPC pays; prosumers opt in free. Feeder congestion relief (distribution: avoided capex/opex) is
  the first pilot value; imbalance cost (supply) is plausible; balancing revenue is not assumed; bill savings are
  reported, not promised.
- **Risk answers**: circular pvlib → hidden simulator effects + PVDAQ headline; future weather → noisy forecast
  inputs; easy synthetic faults → also injected into PVDAQ; London load → sensitivity test; wrong GIS → low trust
  tightens the limit, flags are review requests; missing telemetry → excluded after a timeout and removed from the
  commitment; infra failure → LightGBM fallback, contract tests, mock from hour 2, data + weights in the repo,
  phone hotspot.

---

## 10. Windmon integration

Andrei's Windmon (github.com/Andrei9383/Windmon, PPC HackITall, April 2026): React + TS frontend, Spring Boot API,
Postgres, Redis, Python ingestion with quality and anomaly rules.

- **Reuse (MUST)**: the React frontend (Leaflet map, detail charts, anomalies and reports pages) as our UI, served
  by `sentinel/api/` (FastAPI) as the lab preview (#21); pages adapted to feeders, flexibility, O&M queue, review
  queue and trust layer (#19).
- **Reuse (OPTIONAL)**: wind turbines as a second O&M asset type: the Windmon tag schema
  (`<park>.<turbine>.<signal>`), its quality rules and anomaly rules as the wind baseline, and public Kelmarsh /
  Penmanshiel SCADA (CC-BY-4.0) instead of PPC's private S3 data (#22–#24).
- **Drop**: Java API, Postgres, Redis, Prometheus/Grafana, docker-compose, DeepSeek.
- **Copy files only, never Windmon's git history**: it still contains AWS and DeepSeek keys (Andrei notified).
- **Prior work** is credited in the README and slides; the model, data pipeline, LP and agent are built in these 24 h.

---

## 11. Not doing

- GPVS-Faults; DDP; Optuna; W&B; any external ML API (an LLM is optional and only rewrites text).
- Curtailing PV to hide violations; balancing-market claims; real battery control.
- Replacing inverters, EMS or GIS; raw personal data (IDs pseudonymised; public or synthetic data only).

## 12. Lab rules for the 24 h

- Every card has `--paths`; agents never touch each other's files. Contracts and promote cards are `merge=review`;
  code cards `auto`; training runs `never`.
- Contract change = `lab announce` + a line in `team/decisions.md`.
- After every training or replay run: `lab note` + one line in `team/insights.md` (the error-analysis slide).
- Runs ≤ 20–30 min; memmapped float16 windows; mixed precision; `--compile` only on server-gpu; cheapest grid
  configs to the 4 GB laptop.
- Until #21 lands, a failed preview blocks auto-merge: submit with `lab submit <id> --run check`.
- Keep a phone hotspot ready: every submit and worker fetch goes through GitHub.

## 13. Who does what

Each person filters their cards with `lab cards --tag <name>`. Do your cards in this order; the first one is for
right now. Critical path: #21 (unblocks merges) → #7 (everyone builds on its mock) → #9 → #11 (data for everything).

| Person | Area | Cards, in order | Start now |
|---|---|---|---|
| **Andrei** | Data + simulation + grid-twin; lab admin | #8 cache/CPU probe → #11 fleet simulator → #12 PV fault injector → #10 windows/splits → #13 grid-twin (phase 4) → #28 drills (phase 5) → OPTIONAL #22 #23 #24 wind | #8, then #11 (stub weather until #9 lands) |
| **Stefan** | ML + training infra + baselines + eval | #6 skeleton → #9 raw data → #15 baselines + LightGBM fallback → #14 SentinelNet (on the #7 mock, then real windows) → #16 eval.py → sweeps + ensemble → #29 promote | #6, then #9 |
| **yolooo332** | Contracts, optimiser, agent, uncertainty, README | #7 contracts → #18 per-feeder LP + baseline policies → #17 agent loop + replay → #26 qualify + triage → #27 conformal + P10 → #35 README + make reproduce | #7 |
| **Tudor** | Front-end: shell, map, feeder view, demo | #21 Windmon port + FastAPI preview → #19 fleet map + feeder view → #33 demo mode → #34 slides + rehearsal | #21 |
| **Alexandra** | Front-end: data fixtures, flexibility, site, queues | #30 UI fixtures → #31 flexibility panel + site view → #32 O&M queue, review queue, audit → help on #34 slides | #30 |

Andrei also handles lab admin: start a worker with `--cpu-slots` if #8 finds none, `lab public on` before the
demo, `lab freeze on` at h22, and revoking the Windmon keys.

Hand-offs to agree early:
- yolooo332 ↔ Alexandra: the `demo/replay` format (`team/contracts/replay.md` in #7 = the shape of #30's fixtures).
- Andrei ↔ yolooo332: the battery model and self-consumption rule are shared by #11 (simulator) and #18 (LP baseline).
- Stefan ↔ yolooo332: the `predict()` contract (#7), implemented by both SentinelNet (#14) and the LightGBM fallback (#15).
- Tudor ↔ Alexandra: Tudor owns layout, navigation and `web/src/components/**`; each page card adds its own
  folder under `web/src/pages/` and its own route file under `sentinel/api/routes/`.

| Area | Paths | Owner |
|---|---|---|
| Data, simulation, grid-twin | `sentinel/data/**` (except `download.py`), `sentinel/sim/**`, `sentinel/gridtwin/**` | Andrei |
| ML, training, baselines, eval, promote | `sentinel/model/**`, `train.py`, `sentinel/eval/{baselines,metrics,report}*`, `eval.py`, `sentinel/data/download.py`, `data/raw/**`, `lab.toml`, `requirements*.txt`, `models/**`, `demo/replay/**` | Stefan |
| Contracts, optimiser, agent, uncertainty | `team/contracts/**`, `sentinel/contracts/**`, `sentinel/opt/**`, `sentinel/agent/**`, `sentinel/eval/uncertainty/**`, `replay.py`, `README.md`, `Makefile` | yolooo332 |
| UI shell, map, feeder view, demo, slides | `web/` (shell, `components/`, `pages/fleet`, `pages/feeder`, `src/demo`), `sentinel/api/main.py`, `sentinel/api/routes/{fleet,feeders}.py`, `slides/**` | Tudor |
| UI fixtures, flexibility, site, queues | `demo/fixtures/**`, `web/src/pages/{flex,site,queue,audit}/**`, `sentinel/api/routes/{flex,sites,queue,audit}.py` | Alexandra |
| Plan / docs | `team/plan.md`, `team/decisions.md` | yolooo332 |
