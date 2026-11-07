# Plan: Prosumer Sentinel v3

**One energy agent for Romania. It coordinates home batteries on each LV feeder so midday solar export stays under
the feeder's limit and is released into the evening peak. It watches wind turbine clusters to catch what is going
wrong, and it commits only what healthy, predictable assets can deliver.**

Feeder peak shaving stays the hero (MUST). Wind O&M is the second pillar (SHOULD), and the map UI changes with the
problem (Feeders, O&M, Production, Demand). Every problem has its own model behind one prediction envelope, and one
qualification layer decides which assets each plan may count on. Everything is located in, generated for and
validated against Romania; non-Romanian data only validates methods and is always labelled. If PPC leads with wind
O&M, prosumers or GIS, the same build is re-weighted in the pitch, not rebuilt.

Source: team doc "Prosumer Sentinel v3 — Smarthack XI PPC" (Nov 7), which extends v2. This file is the binding,
sorted version; where they differ, this file wins (see §8 Review notes). Model and card numbers below are current.

---

## 1. Users and value

| User | Gets | Value |
|---|---|---|
| PPC distribution / network ops (primary payer) | Day-ahead plan per feeder; export peaks held under the limit; demand map of where load will peak | Deferred reinforcement, fewer curtailments and overvoltage trips, more connections approved |
| PPC Renewables wind O&M | Per-turbine expected-vs-actual; anomalies classified (fault vs curtailment vs icing vs sensor); ranked inspection list | Fewer false alarms, less downtime, earlier fault detection |
| PPC supply / trading desk | Hourly committed prosumer flexibility (P10) with a delivery record; wind park P10/P50/P90 (OPTIONAL) | Lower imbalance cost; balancing revenue later (not assumed) |
| PPC O&M (residential) | Daily top-K inspection list ranked by lost energy and lost flexibility | Fewer blind truck rolls; repairs return MW to the pool |
| PPC GIS / planning | Feeder/phase mismatch flags + trust score per LV section (GeoJSON) | Review requests where the GIS is likely wrong |
| Prosumers (opt-in, free) | Battery schedule that never breaks their comfort reserve; tips; early fault alerts | Lower bills where the tariff allows |

---

## 2. Success criteria

Targets are aims; results are reported as measured, misses included. Every slide number cites a lab run ID and a
label: **synthetic RO**, **real RO** or **non-RO validation**.

| # | Tier | Claim | Baselines | Target | Label | Run |
|---|---|---|---|---|---|---|
| 1 | MUST | **Feeder peak shaving (headline)**: export peak reduction, minutes above limit per feeder-day | self-consumption rule; tuned per-home threshold rule | zero violations where headroom suffices; residual slack reported | synthetic RO | `replay` |
| 2 | MUST | Evening import peak reduction per feeder | same | reported | synthetic RO | `replay` |
| 3 | MUST | Flexibility reliability: realised ≥ committed (P10) | n/a | ≥ 90% of feeder-steps | synthetic RO | `replay` |
| 4 | MUST | PV nowcast nMAE + 10–90% coverage | pvlib (recorded metadata), persistence, LightGBM | ≥ 20% lower than pvlib; coverage 78–82% | synthetic RO **and** PVDAQ (non-RO) | `train` / `eval` |
| 5 | MUST | Household load forecast | persistence (day/week), LightGBM | beat persistence | synthetic RO | `train` / `eval` |
| 6 | MUST | Site faults: PR-AUC + hours-to-detect | threshold rule; CUSUM on pvlib residual | beat both | synthetic RO; PVDAQ + injected (non-RO) | `train` / `eval` |
| 7 | MUST | Customer: comfort-reserve breaches; lei/home/month on OPCOM, net of degradation | default rule | zero breaches; lei reported | synthetic RO | `replay` |
| 8 | SHOULD | Wind anomalies: precision, recall, alerts per turbine-day, curtailment-vs-fault confusion matrix | WindMon rules (HIGH_WIND_ZERO_POWER, TEMPERATURE_SPIKE, DATA_GAP, SUSPECT_DATA) | fewer alerts at equal or better recall | synthetic RO, held-out park | `train` / `baseline` |
| 9 | SHOULD | Area demand: zone P50/P90, peak hour | persistence, national-shape scaling | beat persistence | real RO (ENTSO-E load) + synthetic RO | `eval` |
| 10 | SHOULD | Grid-twin: % of corrupted GIS feeder/phase records recovered | GIS as-is | reported | IEEE feeder (non-RO topology) | `gridtwin` |
| 11 | OPTIONAL | Wind park forecast 24 h: nMAE, pinball | persistence; class power curve × NWP | beat both | synthetic RO; real RO (ENTSO-E wind) | `train` |
| 12 | OPTIONAL | Early failure warning: lead time; CARE score | fixed residual threshold | real lead time on held-out events | synthetic RO; CARE-to-Compare (non-RO) | `train` / `eval` |
| 13 | MUST | UI: switching mode changes map, panels and data in < 1 s on the demo laptop | n/a | < 1 s | n/a | manual |

`train` prints per model: `val_nmae, val_pinball` (forecasters), `val_prauc, hours_to_detect` (detectors),
`wind_precision, wind_recall, alerts_per_turbine_day, curtail_vs_fault_f1` (M6), `lead_time_h, care_score` (M7).
`replay` prints `peak_reduction, minutes_over_limit, p10_hit_rate, comfort_breaches, lei_per_home, alerts_per_turbine_day`.

---

## 3. Architecture

### 3.1 The loop (15-min tick)

```
 perceive ─────► estimate ─────────► qualify ─────────► decide ──────────────► act ──────────────► log
 prosumer +      M1 PV  P10/50/90     one status per     per-feeder LP          battery schedules   SQLite audit:
 wind SCADA      M2 load P10/50/90    asset:             + P10 commitment       feeder commitment   model version,
 RO weather      M4 site health       eligible           (limit G from          work orders         input hash,
 OPCOM prices    M6 wind NBM          derated            grid-twin, comfort     alerts              prediction,
 GIS metadata    M3 zone demand       excluded           reserve R)             park reports        threshold, status;
 (Pandera +      (M5, M7 optional)    human review       wind: no control,      GeoJSON flags       realised vs
  WindMon rules)                         │               monitor + forecast     zone demand         committed
                                         └──── O&M top-K triage, per crew type ──┘
                                               (repaired assets rejoin the pool)
```

In the demo the agent replays a test week from precomputed predictions written to SQLite; the preview never runs
torch, the LP or Python.

### 3.2 Model registry: one model per problem, one envelope

All models are trained from scratch in our repo (≤ 30 min on a 4–8 GB GPU or CPU) and write the same envelope
(§3.4). A slipped model runs on its fallback, so it degrades a panel, never the demo.

**Compute:** lab workers desktop (6 GB), laptop (4 GB), server-gpu (8 GB), plus a **12 GB RTX 5070** we can also
train on (not yet a lab worker; register it with `gpu_mem` 12G). Send the largest grid configs and the WindNBM sweeps
there; models must still fit 4–8 GB so any worker can retrain them.

| # | Model | Question | Architecture | Tier | Card | Fallback |
|---|---|---|---|---|---|---|
| M1 | PVNet | What should this PV system produce? | PatchTST-style encoder, ~0.3 M params, weather + solar geometry + site's own reference window | MUST | #14 | LightGBM quantile (#15) |
| M2 | LoadNet | What will this household consume? | Same encoder code as M1, separate weights, ~0.2 M params, temp + calendar | MUST | #14 | persistence |
| M4 | SiteHealth | Is this prosumer site healthy? | 1D-CNN on M1/M2 residuals + telemetry, 14-day summary, focal loss, 8 classes | MUST | #14 | threshold + CUSUM (#15) |
| M6 | WindNBM | Is this turbine behaving normally? | Small GRU: expected power + component temps; classifier on residuals + park context | SHOULD | #41 | WindMon rules (#24) |
| M3 | DemandNet | Where and when will area demand peak? | v1: aggregate M2 scenarios per zone; v2: LightGBM quantile per county | SHOULD | #44 | persistence |
| M8 | GridTwin | Is the GIS feeder/phase right? | Voltage correlation + spectral clustering | SHOULD | #13 | GIS as-is |
| M5 | WindPower | What will each turbine / park produce? | Probabilistic power-curve MLP + turbine embedding + park aggregation | OPTIONAL | #42 | binned empirical curve (#24) |
| M7 | WindFailure | Will a component fail soon? | LightGBM on daily M6 residual features | OPTIONAL | #43 | residual-trend chart, no score |

Not ML: the per-feeder LP (cvxpy + HiGHS), the P10 rollout and the qualification rules.

**One prediction, several uses:**

| Envelope field | O&M uses it as | Planner / forecast uses it as |
|---|---|---|
| q10/q50/q90 (PV, load, turbine power) | reference a fault is measured against | charge / discharge headroom; park availability |
| fault_probs / anomaly_class | work order and its priority | exclusion or derating of the asset |
| member_std, interval width, ood_score | human review instead of a truck roll | haircut on committed MW, or exclusion |

### 3.3 Components

| Component | Path | In → Out | Card |
|---|---|---|---|
| Contracts + mock predictor | `sentinel/contracts/`, `team/contracts/` | types, schemas, `MockPredictor` | #7 |
| App contract (shared SQLite) | `packages/db/src/schema/`, `packages/api/src/contract/` | Drizzle tables + generated SQL, oRPC signatures, ModeConfig | #38 |
| Raw RO data | `sentinel/data/download.py`, `data/raw/` | public sources → committed Parquet | #9 |
| Prosumer simulator | `sentinel/sim/fleet/` | RO weather + archetypes → sites on feeders | #11 |
| PV fault injector | `sentinel/sim/faults/` | telemetry → telemetry + labels with onset | #12 |
| Wind simulator (PPC schema) | `sentinel/sim/wind/` | ERA5 100 m wind → 3 RO parks, ~120 turbines, PPC tags | #22 |
| Wind fault injector | `sentinel/sim/wind_faults/` | wind telemetry → + labels with onset | #23 |
| Windows, splits, Pandera | `sentinel/data/{windows,splits,schemas}.py` | Parquet → memmapped float16 windows | #10 |
| M1 + M2 + M4, `train.py` | `sentinel/model/`, `train.py --model …` | windows → ONNX + calibration JSON | #14 |
| PV / load / fault baselines | `sentinel/eval/baselines/pv/` | baseline predictions + metrics | #15 |
| Wind baselines (WindMon rules, binned curve) | `sentinel/eval/baselines/wind/` | baseline alerts + metrics | #24 |
| M6 WindNBM | `sentinel/model/wind/` | wind windows → anomaly class + score | #41 |
| M3 DemandNet | `sentinel/demand/` | load scenarios + ENTSO-E → zone_demand | #44 |
| Evaluation | `eval.py`, `sentinel/eval/{metrics,report}.py` | predictions → LAB_METRIC + JSON report, by label | #16 |
| Uncertainty + P10 rollout | `sentinel/eval/uncertainty/`, `sentinel/opt/commit.py` | calibrated quantiles, thresholds, committed MW | #27 |
| Qualify + triage (sites + turbines) | `sentinel/agent/{qualify,triage}.py` | predictions → status, top-K per crew type | #26 |
| Per-feeder LP + baseline policies | `sentinel/opt/` | forecasts + status + limits → schedules | #18 |
| Agent loop + replay + audit | `sentinel/agent/`, `replay.py` | all of the above → rows in SQLite | #17 |
| Grid-twin | `sentinel/gridtwin/` | IEEE feeder → flags, trust τ, effective limit G | #13 |
| App: shell + map + modes | `apps/web/`, `packages/api/` | SQLite → Feeders / O&M / Production / Demand | #21, #19, #31, #32, #39, #40, #33 |
| UI fixtures | `demo/fixtures/` | stdlib script → `demo/fixtures.sqlite` | #30 |
| Robustness drills | `sentinel/data/drills.py` | test set → drill variants | #28 |
| Promote | `models/`, `demo/` | chosen runs → weights + `demo/sentinel.sqlite` | #29 |

### 3.4 Key interfaces (frozen by #7 and #38; change only with `lab announce`)

- **Prediction envelope** (every model): `predict(asset, window) -> {asset_id, model, model_version, t0, horizon,
  q10[H], q50[H], q90[H], fault_probs?, anomaly_class?, ood_score, member_std}`. PV window = 24 h weather + solar
  geometry (no target power) + reference window of the site's own past power (7–14 days) + 14-day residual summary
  + battery channels. Forecast mode uses noisy weather, never ERA5 actuals.
- **Prosumer telemetry** (15-min): `pv_ac_w, inv_voltage_v, inv_status, soc, batt_w, load_w`; metadata incl.
  recorded vs true tilt/azimuth/kWp, feeder/phase as the GIS records it, comfort reserve.
- **Wind telemetry** = PPC HackITall schema as documented by WindMon: tags `WP<park>.TID<turbine>.<sensor>`,
  `act_pwt` kW, `wd_spd` km/h, converter / gearbox / generator-bearing / transformer / ambient temps °C, a 5-min
  meteo wind series, turbine metadata (id, model, park, lat/lon); integer values are error codes. Real PPC data
  must drop in unchanged.
- `qualify(asset, prediction, telemetry_age) -> {status: eligible|derated|excluded|human_review, headroom_factor, reason}`
  for both asset types.
- `plan_feeder(feeder, eligible_sites, rest_of_feeder, forecasts, G, prices) -> {schedule[site][96], export_curve[96], slack[96]}`.
- `commit(feeder, schedule, scenarios) -> {nominal[96], available[96], committed[96]}`; `realised` is logged next tick.
- Actions (JSON + a row in SQLite): `battery_schedule`, `feeder_commitment`, `work_order`, `alert`, `gis_flag`,
  `human_review`, `park_report`, `zone_demand`, each with an `audit_record` (model_version, input_hash,
  prediction, threshold, status, schedule).
- **Storage**: the Drizzle schema in `packages/db` is the source of truth for the shared SQLite. `drizzle-kit
  generate` commits SQL files; Python applies them with `sqlite3` and only inserts rows. Times are UTC in storage,
  Europe/Bucharest in the UI.

### 3.5 Decision maths

**Qualify** (per asset, per tick): stale telemetry or sensor fault / data gap → excluded; ensemble spread,
conformal width or OOD above validation-tuned thresholds → human review (no work order, out of the pool); confident
PV fault → derated (PV headroom cut by the estimated loss; battery stays in if its own telemetry is healthy);
battery fault → excluded. Turbines: fault stop → excluded; underperformance / yaw → derated; curtailment → eligible
(logged as curtailed energy, no work order); icing → excluded + safety alert. Feeder and park availability are
recomputed next tick.

**O&M triage score** (top-K per day **per crew type**: residential installers, wind technicians; sensor faults →
remote check, never a truck):

```
score_i = P_fault,i × ( lost_kWh_day,i × π  +  lost_flex_MWh_day,i × v_flex ) × D_i
lost_kWh_day,i = Σ (P50_i,t − actual_i,t) · Δt          D_i = days until the next possible visit
v_flex = configurable value per committed MWh (0 for turbines)
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
  their own self-consumption rule (R3).
- `M ≫ λ ≫ π`. LP, not MILP: with η_c·η_d < 1 and c_deg > 0, charging and discharging at once only costs money.
  Check solutions; add a binary only if it appears (negative prices).
- `G_f = τ·G^max + (1 − τ)·G^conservative` from the grid-twin; low trust only ever tightens the limit.
- `c_deg = C_pack / (2 · N_cycles · E_usable)` ≈ €0.02–0.03/kWh per direction (assumed €250–400/kWh LFP, 6,000
  cycles); round-trip ~90%. Inputs, not results. `β` = export remuneration ratio (to verify).
- 10–20 independent LPs of ~25–50k variables; HiGHS should solve each in seconds (confirm on our machines).
- Fallback: one aggregate virtual battery per feeder, split to sites by headroom, then a per-site feasibility clip.

**P10 commitment**: conformal-calibrate each ensemble member's quantiles → ~200 scenarios per feeder-day (one
shared PV quantile level per feeder, independent load levels per site, one ensemble member per scenario) → replay
the schedule through the battery model with clipping → deliverable = export reduction vs the baseline rule →

```
committed_t = (1 − h) · Q10(deliverable_t)        h = telemetry-loss haircut
```

Four numbers, never conflated: **nominal** ≥ **available** ≥ **committed**; **realised** is measured afterwards.
Hit rate and size of misses are published. The same scenarios feed M3 v1 (zone demand = scenario sums, not sums of
quantiles).

**Wind normalisation** (M6 inputs, simulator, M5):

```
v_hub  = v_100 · (h_hub / 100)^α              α ≈ 0.14, fitted per park and season on train years
ρ      = p / (R_d · T)                         R_d = 287.05 J/(kg·K), p in Pa, T in K
v_norm = v_hub · (ρ / 1.225)^(1/3)             IEC 61400-12-1 density correction
```

**Wind anomaly classes** (M6): fault stop (one turbine at zero/negative power above cut-in while neighbours
produce) → work order with lost MWh and lei; curtailment (many turbines capped at one level at the same time) → no
work order; icing (power < P10 near or below 0 °C with high humidity / precipitation) → safety alert, no truck in a
storm; underperformance / yaw (sustained < P10, sector-dependent) → inspection; overheating trend (component
residual rising vs the turbine's own baseline) → feeds M7; anemometer fault (nacelle wind disagrees with the meteo
wind while power matches the meteo wind) → remote check; data gap / suspect data → one offline window per turbine.

### 3.6 Runs and artifacts

| Run | Where | Command | Output |
|---|---|---|---|
| `check` | server sandbox (Node 22, npm, Python, uv) | `npm test` (root) + `pytest tests` | pass/fail; gates auto-merge |
| `preview` | server sandbox | `npm ci && npm run build`, then `npm start` (Next on `$HOST:$PORT`) reading `demo/sentinel.sqlite`, else `demo/fixtures.sqlite` | the shared app; `lab public on` for the demo |
| `train` | GPU workers (no Node) | `python train.py --model pvnet\|loadnet\|sitehealth\|windnbm` (`--grid seed=0,1,2,3`) | LAB_METRIC; ONNX + calibration JSON as a release asset |
| `train_cpu` | CPU | `python train.py --model demandnet\|windpower\|windfailure\|gridtwin` | LAB_METRIC |
| `baseline` | CPU | `python eval.py --baselines` (PV, load, fault, WindMon rules) | baseline LAB_METRIC |
| `replay` | CPU | `python replay.py --policy baseline,sentinel --out $LAB_OUTPUT_DIR/sentinel.sqlite` | replay LAB_METRIC; all rows in SQLite |
| `eval` | GPU/CPU | `python eval.py` | test-set LAB_METRIC + JSON report, grouped by label |

Run outputs stay on the worker; #29 (review) commits the chosen weights to `models/` and the chosen replay to
`demo/sentinel.sqlite`. The preview and `make reproduce` read only committed files.

### 3.7 Repo layout

```
apps/web/                 Next.js app (better-t-stack, npm): shell, map, modes, panels
packages/api/             oRPC routers (one file per mode)          packages/db/  Drizzle schema + generated SQL
packages/auth/            Better Auth (role → default mode)
sentinel/                 Python: contracts/ data/ sim/{fleet,faults,wind,wind_faults}/ model/ demand/
                          eval/ opt/ agent/ gridtwin/
train.py  eval.py  replay.py  Makefile (make reproduce)
data/raw/                 committed Parquet (RO weather incl. 100 m wind, OPCOM, ENTSO-E RO, OSM, PVDAQ)
models/  demo/            promoted weights; demo/sentinel.sqlite; demo/fixtures/ (cite run IDs)
tests/  team/
```

### 3.8 App and map modes

One map of Romania, four modes. A mode is a config `{model, layers, panels, queries, legend}`, so a new problem is
a new config, not a new page. The mode lives in the URL (`?mode=feeders`), so a judge can be sent straight to it.

| Mode | Tier | Reads | Map | Panels | Card |
|---|---|---|---|---|---|
| Feeders (rețea) | MUST | LP, commitments, M8 | feeders coloured by forecast export vs limit; baseline/Sentinel toggle; low-trust LV sections hatched | export vs limit curve; nominal / available / committed / realised; site view | #19, #31 |
| O&M (mentenanță) | MUST | M4, M6, (M7) | sites and turbines by status, clustered when zoomed out; risk halo | alert queue, work orders, human-review queue, turbine drawer (power-curve scatter, temperature-residual trend), audit, daily park report | #32 |
| Production (producție) | SHOULD | M1, binned curve (M5) | wind particles from the 100 m wind field (fallback: wind-speed raster); parks and solar-dense counties by expected MWh | park fan charts, Dobrogea vs Moldova totals, best hours | #39 |
| Demand (cerere) | SHOULD | M3, M2 | red zones county → locality → feeder by peak-stress score; 24 h slider (fallback: static county choropleth) | peak hour and MW per zone, top stressed feeders, P50 with P10–P90 band | #40 |

- Scaffold: `bun create better-t-stack@latest` with the v3 options **but `--package-manager npm`** (R11):
  `--frontend next --backend self --runtime none --api orpc --auth better-auth --payments none --database sqlite
  --orm drizzle --db-setup none --package-manager npm --no-git --web-deploy docker --server-deploy none --install
  --addons turborepo --examples none`. Run it with `npx create-better-t-stack@latest …` if bun isn't installed.
- Libraries: shadcn/ui (sidebar shell, KPI cards, data table, sheet, tabs) · Bklit charts via the `@bklit`
  registry (line/area bands, scatter for power curves, gauges); shadcn charts (Recharts) as fallback · mapcn
  (`@mapcn`, MapLibre GL) · maplibre-gl-wind + deck.gl for particles · TanStack Table · nuqs (`?mode=`, `?t=`,
  selected asset) · Turf.js · sonner, motion, lucide. next-intl RO/EN is OPTIONAL.
- Look: WindMon's dashboard is the visual reference (dark theme, KPI strip on top: production MWh, avg wind,
  availability, active issues; map in the centre; detail drawer; reports). Open its frontend before inventing a
  layout; its code is not ported (Vite, not Next).
- Roles: a role switcher sets the default mode (network operator → Feeders, wind O&M → O&M, trader → Production,
  prosumer → single-site view). No login wall for the jury.

---

## 4. Data: Romania only

Real Romanian inputs drive everything; synthetic data is generated where labels don't exist; non-Romanian data
only validates methods and is always labelled. Raw inputs are committed as zstd Parquet (target ≤ 50 MB);
generated data is not committed (deterministic, cached per worker).

**Real Romanian sources**

| Source | We commit / use | Role |
|---|---|---|
| Open-Meteo Historical (ERA5 0.25°, ERA5-Land 0.1°) | hourly irradiance, cloud, temp, snow, 10 m + 100 m wind, direction, gusts, pressure, humidity, precipitation; 2022–2025; prosumer localities + every wind park + a coarse grid over RO for the wind field | simulator driver; model covariates; Production-mode particles |
| OPCOM PZU | hourly day-ahead prices 2023–2025 (fallback ENTSO-E RO day-ahead) | LP prices; MWh → lei |
| ENTSO-E Transparency, Romania | actual + day-ahead total load; generation by type (wind onshore, solar) | real-RO checks for M3 and M5; national load shape for synthetic households |
| Public park data | Fântânele-Cogealac-Grădina (Constanța, 240 × 2.5 MW), Deleni (Vaslui, 23 × 6.1 MW), other Dobrogea parks (to verify) | park locations, turbine counts, rated power |
| OpenStreetMap | turbine positions where mapped; county and locality boundaries (simplified GeoJSON) | real layouts; Demand-mode zones |

**Non-Romanian, validation only (always labelled)**

| Source | Why |
|---|---|
| NREL PVDAQ | real PV systems: the check that M1's synthetic win isn't circular (R13) |
| CARE-to-Compare (Zenodo) | real wind SCADA with labelled pre-fault events: scores M6 / M7 (OPTIONAL); downloaded in the eval run, never committed |
| pandapower IEEE European LV feeder | grid-twin topology only |
| London Smart Meter (OPTIONAL) | real-household check for M2 only, never in training (R14) |

- **Synthetic prosumers** (#11): 500–1,000 sites in real Romanian localities (county weights configurable; calibrate
  against ANRE prosumer statistics), 10–20 feeders with clustered PV penetration (some congest, some don't). kWp
  3–10, battery 0–15 kWh / 3–5 kW, comfort reserve, feeder limit (G_max, G_conservative). Household archetypes:
  electric heating / heat pump in winter, AC peaks in summer, Romanian public holidays (incl. Orthodox Easter),
  shaped by the ENTSO-E national load curve. History generated under the self-consumption rule.
- **Fair pvlib baseline**: wrong recorded tilt/azimuth on ~20% of sites (±10–30°), inverter curves, 0.5–1%/yr
  degradation, snow, horizon shading; pvlib sees only recorded metadata.
- **Prosumer faults** (#12): soiling, shading, string outage, clipping, overvoltage trips, battery fade, sensor
  faults; random onset and severity; ~15% of site-days; the same injector runs on PVDAQ.
- **Synthetic wind** (#22): ~120 turbines in 3 parks at real coordinates (Dobrogea × 2, Vaslui × 1), 2 years at 15
  min. ERA5 100 m wind → hub height (shear), turbulence noise, direction-dependent wakes, air density from T and
  p. Generic 2.5 MW and 6.1 MW class power curves (not vendor curves). Component temps from a thermal model
  (ambient + load-dependent rise + lag). Output in the PPC schema (§3.4).
- **Wind faults** (#23), exact onset labels, rates configurable: single-turbine stops, park-wide curtailment caps,
  icing (T < ~1 °C + humidity / precipitation), yaw misalignment (sector-dependent loss), pitch / power-curve
  degradation, gearbox and generator-bearing degradation (temperature residual drifting for 3–21 days before
  failure), anemometer faults, communication gaps and integer error codes.
- **Splits**: train 2022–2023, validate 2024, test 2025; 30% of prosumer sites and one whole wind park never seen
  in training; test failure events never in training; separate calibration slice; thresholds frozen on validation.
  Planning never sees ERA5 actuals; normalisation statistics from training data only.
- **Generator**: deterministic (fixed seed), < 10 min with wind, cached at `~/.cache/sentinel/data-v{N}` (bump N on
  change).

---

## 5. Scope tiers

| Tier | Items |
|---|---|
| **MUST** | RO prosumer simulator + PV injector · M1 PVNet · M2 LoadNet · M4 SiteHealth (or fallbacks) · PV/load/fault baselines · qualification · top-K O&M · per-feeder LP · P10 commitment · baseline-vs-Sentinel replay into SQLite · audit log · app with Feeders + O&M modes |
| **SHOULD** | wind simulator + injector in PPC schema · M6 WindNBM vs WindMon rules · turbines in O&M mode · M3 demand + Demand mode · Production mode (static wind layer first, particles second) · grid-twin trust · robustness drills · daily park report · roles · hourly LP re-solve |
| **OPTIONAL** | M5 WindPower · M7 WindFailure + CARE · park-battery scheduling (slide only) · similar cases · Integrated Gradients · LLM text · RO/EN i18n · routing · docker compose · Rust kernels |

**Cut order if behind**: LLM text → routing → similar cases → Rust / docker compose → M7 → M5 → wind particles
(keep the static wind layer) → Demand mode (keep a static county choropleth) → grid-twin to one slide → M6 (keep
WindMon rules in O&M mode) → full LP to the aggregate per-feeder LP → PV forecast mode (pvlib forecast × nowcast
correction). If M1/M2 aren't ready by hour 12, LightGBM feeds the same LP.
**Never cut**: RO simulator, M1, M2, qualification, the LP, the P10 commitment, the baseline comparisons, work
orders, Feeders + O&M modes.

---

## 6. Steps (phases, gates, cards)

| Phase | Hours | Cards | Gate |
|---|---|---|---|
| 1. Setup & contracts | 0–2 | **#21** Next scaffold + preview fix · **#38** app contract (review) · **#6** skeleton + run types · **#7** contracts (review) · **#8** cache + CPU-slot probe · **#30** fixtures SQLite · ENTSO-E token requested | preview passes; both contracts merged; cache result on all workers |
| 2. Data, baselines, LP on mock | 2–7 | #9 RO data · #11 prosumer sim · #12 PV injector · #10 windows/splits · #15 baselines · #18 LP · #17 replay on mock | data committed; data-v1 cached; baselines in `lab top`; LP solves one feeder; map renders Romania from fixtures |
| 3. Models v1 + full loop | 7–12 | #14 M1/M2/M4 · #16 eval · #26 qualify + triage · #27 conformal + P10 · #22 #23 wind sim + faults · #24 wind baselines · #19 #31 #32 Feeders + O&M modes | M1/M2 beat baselines on val (or fallback declared); qualify + LP on all feeders; Feeders and O&M modes read replay rows |
| 4. Sweeps + integration (night) | 12–17 | #14 sweeps + 4-seed ensemble · #41 WindNBM · #44 demand · #13 grid-twin · #39 Production · #40 Demand · #29 promote | full replay in SQLite with real models; wind anomalies in O&M mode; Production + Demand on real rows or fallbacks |
| 5. Evaluation & robustness | 17–21 | #16 real-RO (ENTSO-E) + non-RO (PVDAQ, CARE if #43) · #28 drills · tuned-rule comparison · grid-twin keep/drop | error analysis in `team/insights.md`; every number labelled |
| 6. Demo freeze | 21–24 | #35 README · #34 slides + rehearsal · #33 demo mode | `lab freeze on` at h22; `make reproduce` on a clean checkout; rehearse ×2 |

OPTIONAL, only when MUST and SHOULD are on track: #42 WindPower, #43 WindFailure.

---

## 7. Demo script (3 minutes)

Baseline and Sentinel replay the same 2025 test week (identical weather, assets, limits, prices). [Brackets] come
from logged runs; every panel says synthetic RO, real RO or non-RO. Every panel is precomputed, so it can fall back
to replay mode if a live step fails. SHOULD steps have a fallback panel.

1. 0:00–0:15 Map of Romania, KPI strip: [N] homes on [n] feeders, [T] turbines in 3 parks. "One map, four problems."
2. 0:15–0:40 *(SHOULD)* Production mode: wind sweeps across Dobrogea; parks light up with expected MWh for
   tomorrow. "This is where energy will be made."
3. 0:40–0:55 *(SHOULD)* Demand mode: red zones where consumption peaks this evening. "And where it will be needed."
4. 0:55–1:40 Feeders mode, F7: baseline export crosses the limit at noon; Sentinel holds charging back, absorbs
   11:00–14:00, stays under the limit, discharges 18:00–21:00. Flexibility panel: nominal vs available vs
   committed (P10) vs realised. "We never sell the nominal number."
5. 1:40–2:05 String failure at site 1042: band diverges → derated → F7's committed MW drops → work order in the
   top-K with kWh, lei and MW at risk. An OOD site goes to human review.
6. 2:05–2:45 O&M mode, a Constanța park *(SHOULD; fallback: WindMon rules on the same week)*: one turbine high wind
   / zero power while neighbours produce → fault + work order; minutes later the whole park is capped →
   curtailment, no work order (the WindMon rules fired on both); a gearbox residual drifting for days (risk score
   only if #43 shipped).
7. 2:45–3:00 Summary: peak reduction, P10 hit rate, comfort breaches, lei per home, alerts per turbine-day vs
   WindMon rules, each with its run ID.

---

## 8. Review notes

### v3 (reviewed Nov 7; these override the v3 doc)

- **R9 Hero unchanged**: v3 adds wind and map modes around v2's core (qualify → per-feeder LP → P10 commitment). The
  pitch still opens with feeder congestion; wind is the second pillar.
- **R10 Models**: registry and one envelope adopted. Eight models in 24 h is too many to make MUST: M1, M2, M4 are
  MUST; M6 and M3 SHOULD; M5 and M7 OPTIONAL (v3 had M5 and M6 as MUST). M1/M2 share encoder code; M4 is a separate
  small CNN. All three live in #14 behind `train.py --model`. M3 starts as aggregation of M2 scenarios, so Demand
  mode needs no new model.
- **R11 Stack**: v3's Next.js stack (better-t-stack, oRPC, Drizzle + SQLite, mapcn, Bklit) replaces Windmon React +
  FastAPI. Tweaks: `--package-manager npm`, because the lab sandbox has Node 22 + npm, bun isn't guaranteed and GPU
  workers have no Node. The root `package.json` needs a `test` script (check runs `npm test`) and a `start` script
  that serves Next on `$HOST:$PORT`; commit `package-lock.json` (check runs `npm ci`). The Python package stays
  `sentinel/` at the root (not `ml/`), so card paths and lab.toml commands don't move. No login wall. FastAPI is dropped.
- **R12 Wind claims**: curtailment vs fault is reported with its confusion matrix, mistakes included. M7 ships only
  with real lead time on held-out events; otherwise the drawer shows the residual trend without a score.
- **R13 Circularity**: the simulator is built on pvlib physics, so a synthetic-RO nowcast win is partly circular;
  PVDAQ (non-RO) stays on the slide next to it. For wind, synthetic faults are backed by CARE (if #43) and by the
  WindMon rules baseline on the same data.
- **R14 Load**: London leaves training (Romania only). Synthetic RO households come from archetypes shaped by
  ENTSO-E national load. A model trained on our own synthetic load mainly shows it learns our generator, so this is
  said on the slide. ENTSO-E national load (real RO) checks M3, and London is an OPTIONAL non-RO check for M2.
- **R15 ENTSO-E access**: the API needs a free security token by email request; ask at hour 0. Fallbacks: manual
  CSV export from the Transparency Platform, or Transelectrica's public data. Commit the files; no runtime calls.
- **R16 Data size**: commit raw inputs only (≤ 50 MB); generated prosumer and wind data is rebuilt deterministically
  and cached; CARE is downloaded in the eval run.
- **R17 Model code reaches main**: model cards are `merge=never` while experimenting; once a config is chosen, flip
  the card with `lab edit <id> --merge review` and approve, so `train.py` and the model code are on main for `make
  reproduce`. Weights still go through #29.
- **R18 Demo store**: replay writes one SQLite file; #29 promotes it to `demo/sentinel.sqlite` (< 50 MB). The
  preview opens it read-only; until then it reads `demo/fixtures.sqlite` from #30. WAL is not needed for a
  read-only demo.
- **R19 Time**: storage in UTC, UI in Europe/Bucharest. Avoid DST-change days in the demo week; tests cover one DST
  day (92/100 local steps).
- **R20 Wind never enters the LP**: no turbine control; park-battery scheduling is a slide at most.
- **R21 Rust**: v3's Rust core (window builder, rollout kernel, inference service) is OPTIONAL with no card. The
  pure-Python reference is the plan; Rust only replaces a loop that a logged run shows is the bottleneck.
- **R22 Map tiles**: CARTO basemaps are fine for the hackathon (non-commercial); a pilot switches to OpenFreeMap or
  self-hosted tiles. No MapTiler key.

### v2 (still in force)

- **R3** The LP feeder constraint includes non-eligible sites' forecast net export. **R4** Day-ahead plan + per-tick
  feasibility clip is MUST; hourly re-solve SHOULD (~2.5k solves for 15 feeders × 24 × 7). **R5** Replay outputs
  reach main through #29. **R6** `baseline`, `replay`, `train_cpu` need a CPU slot; #8 tests `--needs cpu`.
  **R7** Residential kWp 3–10. **R8** Coverage target 78–82%.
- Kept from v1: nowcast reference window of the site's own past power; fault model gets 14-day context; "beats
  pvlib" is the gate and 20% the stretch; grid-twin power flow may be slow (time 10 steps first); ERA5 hourly →
  15-min via clear-sky index; contracts allow PPC's own data to drop in.

### To verify (never claimed as fact)

- Market figures (150k+ prosumers / 1.7+ GW in 2024–25; 400k / 4 GW by 2030).
- PPC wind assets: Fântânele-Cogealac-Grădina 600 MW = 240 × GE 2.5 MW; 84 MW Dobrogea park bought from Lukoil;
  Deleni 140 MW = 23 × GE Vernova 6.1 MW; 45.72 MW / 91.44 MWh battery at a Dobrogea wind park.
- WindMon's 16-day run: ~15,000 rule alerts, ~70% data gaps from four offline turbines (check its anomaly docs).
- CARE-to-Compare: 95 datasets, 36 turbines, 3 farms, 44 labelled pre-fault events; download size.
- β (export remuneration) and the Romanian compensation scheme; aggregator rules for small residential assets;
  unbundling and data sharing between distribution and supply; GDPR consent; the battery control contract.
- €80–120/home/year in the reference brief has no derivation; we report Σ committed kW(t) × value(t) from simulation.

---

## 9. Pitch material

- **Why one product, not three**: the same qualification contract links every model to every decision. A broken PV
  string lowers a feeder's committed MW; a turbine with a drifting gearbox leaves the park's availability before it
  trips; every repair returns capacity to a plan.
- **Shared root cause**: neither side has a model of what each asset should do given its weather. With that model,
  anomalies, forecasts and commitments all become measurable.
- **Why a feeder plan, not a smarter home rule**: only a feeder plan puts absorption where the headroom is. We
  benchmark a tuned per-home rule and say so if it comes close.
- **Alert fatigue**: rules like "wind > 15 km/h and power < 10 kW" fire on every stop, including curtailment and
  maintenance; fleet-wide temperature thresholds miss a turbine that runs hot by design. We compare against
  WindMon's own rules on PPC's schema.
- **Physical limit**: when surplus exceeds battery headroom, we show the residual slack; we don't hide it.
- **No hardware**: vendor APIs or EMS only where write access exists, otherwise monitor-only with zero committed MW;
  wind is monitoring and forecasting only. Hackathon control is simulated, stated openly.
- **Customer**: opt-in, customer-chosen comfort reserve as a hard constraint, override any time, every action logged.
- **Commercial**: PPC pays; prosumers opt in free; wind is PPC's own fleet. Feeder congestion relief is the first
  pilot value; fewer wind false alarms and less downtime are plausible (need PPC downtime data); imbalance savings
  plausible; balancing revenue not assumed; bill savings reported, not promised. For wind we quote lost MWh and lei
  per detected event from simulation, not a fleet-wide saving.
- **Risk answers**: eight models → tiers + a fallback each; easy synthetic faults → physics-based injection with
  random onset, PVDAQ / CARE checks, WindMon baseline; curtailment vs fault → park context + confusion matrix;
  overclaimed failure prediction → shipped only with real lead time; Romanian load → archetypes on ENTSO-E shape,
  stated; wrong GIS → low trust tightens the limit; missing telemetry → excluded and removed from the commitment;
  infra → fallbacks, contract tests, fixtures from hour 2, data + weights in the repo, phone hotspot.

---

## 10. WindMon

Andrei's WindMon (github.com/Andrei9383/Windmon, PPC HackITall, April 2026) ingests 15-min turbine data, runs
rule-based anomaly detection and serves a dark shadcn dashboard. Its documented gaps (static thresholds, no
per-turbine baselines, no power-curve analysis, no trend detection, no suppression) are what M6 / M5 / M7 fill.

- **Reuse**: the PPC SCADA schema from its data dictionary (§3.4); its six quality rules and OK / DEGRADED /
  SUSPECT flags (#22); its four anomaly rules as the wind baseline (#24); deduplicated anomalies table and the
  TSO-style daily park report idea; its dashboard layout as the visual reference (§3.8).
- **Not reused**: its code (Vite React, Spring Boot, Postgres, Redis, Grafana, DeepSeek). Python rules are re-written
  on polars; if any file is copied, copy files only, never its git history (it holds AWS and DeepSeek keys).
- **Credit** in the README and slides as prior work; models, data pipeline, LP, agent and app are built in these 24 h.

---

## 11. Not doing

- External ML APIs (an LLM is optional and only rewrites text); DDP; Optuna; W&B; GPVS-Faults.
- Turbine control; real battery control; curtailing PV to hide violations; balancing-market claims.
- Replacing inverters, EMS, wind SCADA or GIS; raw personal data (pseudonymised IDs; public or synthetic data only).
- Streamlit, FastAPI, MapTiler; a Rust service unless a logged run proves a bottleneck (R21).

## 12. Lab rules for the 24 h

- Every card has `--paths`; agents never touch each other's files. Contract and promote cards are `merge=review`;
  code cards `auto`; training cards `never` until a config is chosen (R17).
- Contract change = `lab announce` + a line in `team/decisions.md`.
- After every training or replay run: `lab note` + one line in `team/insights.md`, with the label (synthetic RO /
  real RO / non-RO).
- Runs ≤ 20–30 min; memmapped float16 windows; mixed precision; cheapest grid configs to the 4 GB laptop, largest
  to the 12 GB RTX 5070.
- `requirements.txt` stays light (no torch); `requirements-train.txt` adds torch, so the sandbox never pulls CUDA.
- Until #21 lands, a failed preview blocks auto-merge: submit with `lab submit <id> --run check`.
- Keep a phone hotspot ready.

## 13. Who does what

Each person filters their cards with `lab cards --tag <name>`. Do cards in this order; the first is for right now.
Critical path: #21 (unblocks merges) → #38 + #7 (everyone builds on mocks) → #9 → #11 (data for everything).

| Person | Area | Cards, in order | Start now |
|---|---|---|---|
| **Andrei** | Data + simulation + grid-twin; lab admin | #8 → #11 RO prosumer sim → #12 PV faults → #10 windows/splits → #22 wind sim (PPC schema) → #23 wind faults → #13 grid-twin → #28 drills | #8, then #11 |
| **Stefan** | ML + baselines + eval | #6 → #9 RO data (ask for the ENTSO-E token first) → #15 baselines → #14 M1/M2/M4 → #16 eval → #24 wind baselines → sweeps + #41 WindNBM (night) → #29 promote · OPTIONAL #42 | #6, then #9 |
| **yolooo332** | Contracts, optimiser, agent, uncertainty, demand, README | #7 → #18 LP → #17 agent + replay to SQLite → #26 qualify + triage (sites + turbines) → #27 conformal + P10 → #44 demand → #35 README · OPTIONAL #43 | #7 |
| **Tudor** | App shell, contract, map, Feeders + Production, demo | #21 scaffold + preview → #38 app contract → #19 map shell + Feeders mode → #39 Production → #33 demo mode → #34 slides | #21 |
| **Alexandra** | Fixtures, panels, O&M + Demand | #30 fixtures SQLite → #31 flex panel + site view → #32 O&M mode → #40 Demand → help on #34 | #30 (draft rows from #38's draft schema) |

Andrei also handles lab admin: `--cpu-slots` worker if #8 finds none, `lab public on` before the demo, `lab freeze
on` at h22.

Hand-offs to agree early:
- Tudor ↔ yolooo332 ↔ Alexandra: the Drizzle schema (#38) is what #17 writes and #30 seeds.
- Andrei ↔ yolooo332: the battery model and self-consumption rule are shared by #11 and #18.
- Stefan ↔ yolooo332: the prediction envelope (#7), implemented by #14, #15, #41.
- Andrei ↔ Stefan: the wind telemetry schema (#7 §wind) between #22/#23 and #24/#41.

| Area | Paths | Owner |
|---|---|---|
| Data, simulation, grid-twin | `sentinel/data/**` (except `download.py`), `sentinel/sim/**`, `sentinel/gridtwin/**` | Andrei |
| ML, training, baselines, eval, promote | `sentinel/model/**`, `train.py`, `sentinel/eval/{baselines,metrics,report}*`, `sentinel/eval/care/**`, `eval.py`, `sentinel/data/download.py`, `data/raw/**`, `lab.toml` (except `[runs.preview]`), `requirements*.txt`, `models/**`, `demo/sentinel.sqlite` | Stefan |
| Contracts, optimiser, agent, uncertainty, demand | `team/contracts/**` (except `app.md`), `sentinel/contracts/**`, `sentinel/opt/**`, `sentinel/agent/**`, `sentinel/eval/uncertainty/**`, `sentinel/demand/**`, `replay.py`, `README.md`, `Makefile` | yolooo332 |
| App scaffold, contract, shell, Feeders, Production, demo, slides | root `package.json` / `package-lock.json` / `turbo.json`, `lab.toml [runs.preview]`, `packages/{db,auth}/**`, `packages/api/src/{contract,index}*`, `packages/api/src/routers/{feeders,production,manifest}.ts`, `apps/web/src/{app,components}/**`, `apps/web/src/modes/{types,registry}.ts`, `apps/web/src/modes/{feeders,production}/**`, `apps/web/src/demo/**`, `scripts/windfield.py`, `team/contracts/app.md`, `slides/**` | Tudor |
| Fixtures, panels, O&M, Demand | `demo/fixtures/**`, `apps/web/src/panels/**`, `apps/web/src/modes/{om,demand}/**`, `packages/api/src/routers/{om,demand,flex,sites,audit}.ts` | Alexandra |
| Plan / docs | `team/plan.md`, `team/decisions.md` | yolooo332 |

Folder names under `apps/web` and `packages/` follow what the scaffold creates; #21 adjusts the globs on the cards
if they differ.
