# Prosumer Sentinel

**An energy agent for PPC (Romania).** On each low-voltage feeder it schedules home batteries so midday solar export
stays under the feeder's limit and is released into the evening peak. It watches wind-turbine fleets to separate
real faults from curtailment and sensor noise. It commits only the flexibility that healthy, predictable assets
can actually deliver (P10). Every decision is written to an audit log that names the model prediction behind it.

Built in 24 h at Smarthack XI (PPC theme "24h to Power PPC"). Challenge summary: [team/brief.md](team/brief.md);
binding plan: [team/plan.md](team/plan.md).

```bash
make reproduce                                 # offline after the first install, ~9 min on a laptop CPU
npm ci && npm run build && npm start           # the app on http://localhost:3000
```

| Who at PPC | Gets |
|---|---|
| Distribution / network ops | Day-ahead battery plan per feeder; export peaks held under the limit; demand map |
| Renewables wind O&M | Expected-vs-actual per turbine; fault vs curtailment vs icing vs sensor; ranked inspection list |
| Residential O&M | Daily top-K work orders ranked by lost kWh and lost flexibility; "not sure" cases go to a human |
| Supply / trading | Hourly committed prosumer flexibility (P10) with a delivery record |

**Headline results (test year 2025, details and run IDs in [Results](#results)):** feeder replay: minutes over the
export limit 6645 → 1905, energy over it 1397 → 94.5 kWh, P10 hit rate 92.0 %, 0 comfort breaches (synthetic RO,
replay#506) · PVNet nMAE 0.015 / 0.036 on seen / unseen sites vs LightGBM 0.041 / 0.045 and pvlib 0.083 / 0.093
(synthetic RO, eval#521) · SiteHealth fault PR-AUC 0.73 / 0.56 vs threshold + CUSUM 0.49 / 0.40 (synthetic RO,
eval#521) · WindNBM precision 0.92, recall 0.89 vs WindMon 0.84 / 0.75 (synthetic RO, train#593/594) · DemandNet
0.039 vs persistence 0.067 on **real Romanian national load** (`make demand`) · on real US PV (PVDAQ, non-RO) PVNet
is only 14 % better than pvlib (eval#521).

---

## How it works

One 15-minute tick (plan §3.1):

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

- **Estimate.** Each model answers one question and returns the same prediction envelope
  (`q10/q50/q90`, `fault_probs`, `ood_score`, `member_std`; [team/contracts/model.md](team/contracts/model.md)).
- **Qualify** ([sentinel/agent/qualify.py](sentinel/agent/qualify.py)). Stale telemetry → excluded. Wide interval,
  ensemble disagreement or out-of-distribution input → **human review** (no truck, out of the pool). Confident PV
  fault → derated. Turbines: fault stop → excluded; curtailment → eligible, logged, no work order; icing → safety
  alert. Thresholds are tuned on validation and frozen in [models/thresholds.json](models/thresholds.json).
- **Decide** ([sentinel/opt/lp.py](sentinel/opt/lp.py)). A per-feeder LP (cvxpy + HiGHS) keeps the *worst plausible*
  export (P90 PV − P10 load, plus the non-participating homes) under the limit, never breaks the customer's
  comfort reserve, and shaves the 17:00–22:00 import peak. Slack is reported, not hidden.
- **Commit** ([sentinel/opt/commit.py](sentinel/opt/commit.py)). ~200 conformal-calibrated scenarios per
  feeder-day → `committed = (1 − haircut) · Q10(deliverable)`. Nominal ≥ available ≥ committed; realised is
  measured afterwards.
- **Act + log** ([sentinel/agent/](sentinel/agent/), [replay.py](replay.py)). Battery schedules, feeder commitments,
  work orders (top-K per crew type, score = P(fault) × (lost kWh × price + lost flex × value) × days to visit),
  alerts, park reports, zone demand. Each action has an `audit_log` row with model version, input hash,
  prediction, threshold and status.

### The app: one map of Romania, four modes

The mode is in the URL (`?mode=feeders`), so a reviewer can be sent straight to it.

| Mode | Shows | Status |
|---|---|---|
| **Feeders** (rețea) | Feeders coloured by export vs limit, baseline/Sentinel toggle, nominal / available / committed / realised | on main (#19, #53) |
| **O&M** (mentenanță) | Sites and turbines by status, alert + work-order queue, human-review queue, audit trail | on main (#32) |
| **Production** (producție) | Wind particles from the real ERA5 100 m wind field, parks and counties by expected MWh | on main (#39, #55, #59) |
| **Demand** (cerere) | Zones by peak-stress score, peak hour, P50 with P10–P90 band | on main (#40) |

The app reads `demo/sentinel.sqlite` (the promoted replay, #29), else `demo/fixtures.sqlite` (mock week from
`python3 demo/fixtures/make_fixtures.py`, labelled synthetic), else an empty in-memory DB. It never runs Python,
torch or the LP: every panel is precomputed by `replay.py`.

---

## Models

All models are trained from scratch in this repo and run in our code (ONNX + onnxruntime at inference, no
external API). Each one has a fallback, so a missing model degrades a panel, never the demo.

| # | Model | Question | Architecture | Tier | Status | Fallback |
|---|---|---|---|---|---|---|
| M1 | PVNet | What should this PV system produce? | PatchTST-style encoder, 0.29 M params, weather + solar geometry + the site's own past-power reference window | MUST | trained on the #11 fleet, 4-seed ensemble (#14 train#383–386), in `models/` | LightGBM quantile (#15) |
| M2 | LoadNet | What will this household consume? | same encoder, separate weights, 0.21 M params | MUST | trained, 4 seeds (#14 train#387–390), in `models/` | persistence |
| M4 | SiteHealth | Is this prosumer site healthy? | 1D-CNN on M1/M2 residuals + telemetry + 14-day summary, focal loss, 8 classes, 56 k params | MUST | trained (#14 train#391), in `models/` | threshold rule + CUSUM (#15) |
| M6 | WindNBM | Is this turbine behaving normally? | GRU normal-behaviour model (power + component temps) + LightGBM classifier on residuals and park context, 0.13 M params | SHOULD | in `models/`: #41 run 135 (stand-in wind data); evaluated on #22 parks (#99 train#591–594) | WindMon rules (#24) |
| M3 | DemandNet | Where and when will demand peak? | v1 sum of LoadNet scenarios per zone; v2 LightGBM quantile per county | SHOULD | on main (#44) | persistence |
| M8 | GridTwin | Is the GIS feeder/phase right? | voltage correlation + spectral clustering | SHOULD | on main (#13) | GIS as-is (τ = 1) |
| M5 | WindPower | What will each park produce? | probabilistic power-curve MLP | OPTIONAL | experiment, not merged (#42) | binned empirical curve |
| M7 | WindFailure | Will a component fail soon? | LightGBM on daily M6 residuals | OPTIONAL | on main (#43 train#379); fails on real CARE data | residual trend, no score |

Not ML: the per-feeder LP, the P10 rollout and the qualification rules.

Model code reaches `main` when its config is chosen (plan R17): `train.py` + `sentinel/model/` came with #14,
`sentinel/model/wind/` with #41, the promoted ONNX weights in `models/` with #29 (runs listed in
[models/README.md](models/README.md)).

### Training

Training runs on the team's GPU workers through `lab`, our shared card board and job runner (run types in
[lab.toml](lab.toml)), never on a laptop for more than a few minutes:

```bash
lab submit 14 --run train --cmd ".venv/bin/python train.py --model pvnet"           # one model
lab submit 14 --run train --cmd ".venv/bin/python train.py --preset {p}" \
              --grid p=pv_now@s0,pv_now@s1,pv_now@s2,pv_now@s3                     # 4-seed ensemble
lab submit 44 --run train_cpu --cmd ".venv/bin/python train.py --model demandnet"   # CPU models
```

Locally: `pip install -r requirements-train.txt` (adds torch), then `python train.py --model
pvnet|loadnet|sitehealth|windnbm|demandnet [--max-minutes 5]`, or `make train TRAIN_LOCAL=1 MODEL=pvnet`. Each run
prints `LAB_METRIC` lines (test-set metrics and every baseline on the same split) and writes ONNX + calibration
JSON to `$LAB_OUTPUT_DIR`. Every model fits a 4 GB GPU (peak < 0.9 GB measured); the final runs used a 12 GB RTX 5070
(1–5 min per run). `requirements.txt` stays torch-free so the app sandbox and CPU runs never pull CUDA.

---

## Reproduce

```bash
make reproduce
```

Needs `make`, Python 3.12 and [uv](https://docs.astral.sh/uv/) (falls back to `python3.12 -m venv` + pip). The only
network access is the first `uv pip install`; all raw data and the trained `models/*.onnx` are committed. Outputs
and the data cache go to `.lab-out/reproduce/` (gitignored); committed files in `demo/` and `models/` are never
overwritten. It ends with a table of every number it printed, the step and card that produced it, and its label.
On a clean checkout it took **553 s** (~9 min) on an Apple-silicon laptop CPU and exited 0; `make test` gives 313
passed.

Two targets regenerate the README's numbers:

- `make eval` = **eval#521** (#110): `eval.py --predictor onnx,baselines --models models --drills`. Every PVNet,
  LoadNet, SiteHealth, pvlib and persistence metric matches the lab run exactly. The LightGBM baseline is retrained
  on each run, so it moves in the third decimal (e.g. PV seen sites 0.040 vs 0.041).
- `make replay-demo` = **replay#506** (#29, `demo/sentinel.sqlite`): builds the data cache, then
  `replay.py --world sim --policy baseline,sentinel` with the committed `models/`. Its KPIs match replay#506 exactly
  (6645 → 1905 minutes, P10 92.0 %, 23.73 lei), also after #99's M6 rule merged. The bytes differ
  (`runs.created_at` is wall-clock time).

`make replay` is different: a 15 s mock-world replay with `MockPredictor`. It checks the pipeline; its KPIs are not
results.

| Target | What it runs | Output | Time* |
|---|---|---|---|
| `make setup` | `.venv` (Python 3.12) + `requirements.txt` | `.venv/` | ~5 s cached |
| `make test` | `pytest tests` (contracts, simulator, faults, LP, qualify, uncertainty, demand, agent, kernels, models) | pass/fail | ~1–2 min |
| `make sim` | RO prosumer fleet, 600 sites / 15 feeders / 2022–2025 (#11) + PV fault injector (#12) | `fleet/`, `faults/` | ~40 s |
| `make eval` | eval#521: committed `models/*.onnx` + every baseline (pvlib, persistence, LightGBM, threshold + CUSUM) + #28's robustness drills, on the test fleet, PVDAQ and real RO series (#16) | `eval/` | ~5 min (builds the test fleet + faults once) |
| `make baselines` | `eval.py --baselines` only; not in `reproduce` (`make eval` prints the same baselines) | `baselines/` | ~5 min |
| `make demand` | M3 DemandNet on real RO load, test 2025 (#44) | `demand/` | ~10 s |
| `make uncertainty` | conformal calibration + P10 commitment on mock forecasts (#27) | `uncertainty.json` | ~3 s |
| `make replay` | baseline vs Sentinel on the mock world: pipeline check (#17) | `sentinel.sqlite` | ~15 s |
| `make replay-demo` | replay#506: #10 windows, #22 wind parks, #13 grid-twin, then the sim-world replay with `models/` (#29) | `demo/sentinel.sqlite` | ~30 s with a warm cache, ~2.5 min after `make eval`, ~9 min from nothing |
| `make app` | `npm ci`, `npm run build`, `npm start` (`APP_DB=.lab-out/reproduce/demo/sentinel.sqlite` to view the rebuilt replay) | http://localhost:3000 | ~40 s build |
| `make train` | prints the lab commands; `TRAIN_LOCAL=1 MODEL=…` trains locally | `train/` | — |
| `make clean` | removes `.lab-out/reproduce/` | | |

\*Apple-silicon laptop CPU, measured on this commit. Useful variables: `DEMO=0` (`reproduce` without
`replay-demo`), `EVAL_ARGS=--quick` (150-site eval fleet, a smoke test), `FULL=1` (fault injector on the full fleet,
~4 min), `SEED`, `PROMOTE=1` (allow writing a committed `demo/sentinel.sqlite`). Not regenerated locally: the wind
model numbers (M6, M7) and train-time numbers; they cite lab train runs.

Data can be re-downloaded with `python -m sentinel.data.download` (no keys; `--only <job>`). Optional Rust kernels
(#48): `sh rust/build.sh build`; Python is the reference and the default when the extension is missing.

---

## Results

Every number carries a label: **synthetic RO** (our simulator, driven by real Romanian weather), **real RO**
(measured Romanian data) or **non-RO** (validation only, never trained on). Splits: train 2022–23, validation 2024,
test 2025, 30 % of sites never seen in training; thresholds frozen on validation. Each row names the lab run it
comes from and the make target that regenerates it. The full claim list with targets and misses is
[team/results.md](team/results.md).

| Claim | Ours | Baselines (same test split) | Label | Source |
|---|---|---|---|---|
| Feeder export peak shaving (5 feeders × 7 days, June 2025) | minutes over limit **1905**, energy over limit **94.5 kWh**, worst overshoot **10.6 kW**; peak reduction 19.1 % (24.1 % on congested feeder-days) | self-consumption rule: 6645 min, 1397 kWh, 29.1 kW | synthetic RO + real OPCOM prices | replay#506 (#29), `make replay-demo` |
| P10 commitment hit rate | **92.0 %** of feeder-steps (1845 kWh committed, 16 kWh short) | — | synthetic RO | replay#506, `make replay-demo` |
| Comfort reserve; customer value | **0** comfort breaches; **23.73 lei/home/month** saved, net of battery wear | default rule: 1165 breaches | synthetic RO + real OPCOM prices | replay#506, `make replay-demo` |
| Wind alerts in the demo week (park WP01, 70 turbines) | **0.114** alerts / turbine-day | WindMon rules (#24): 0.698 (not de-duplicated) | synthetic RO | replay#506, `make replay-demo` |
| PV nowcast nMAE, seen / unseen sites | **0.015 / 0.036**; 10–90 coverage 0.88 / 0.61 | LightGBM 0.041 / 0.045, pvlib 0.083 / 0.093, persistence 0.279 / 0.261 | synthetic RO | eval#521 (#110), `make eval` |
| PV nowcast on real systems | **0.268**; 10–90 coverage 0.10 | LightGBM 0.283, pvlib 0.312 (−14 %), persistence 0.538 | non-RO (NREL PVDAQ) | eval#521, `make eval` |
| Household load nMAE, seen / unseen sites | **0.246 / 0.285** (a tie with LightGBM, no win claimed) | LightGBM 0.256 / 0.284, persistence (yesterday) 0.361 / 0.390 | synthetic RO | eval#521, `make eval` |
| Site fault detection | **PR-AUC 0.73 / 0.56** (seen / unseen); 97.5 % of 40 fault episodes found, median 21.5 h to detect. PVDAQ + injected faults: PR-AUC 0.43 | threshold rule + CUSUM: 0.49 / 0.40; 40 % found, median 35.9 h. PVDAQ + injected: 0.32 | synthetic RO; non-RO (PVDAQ + injected) | eval#521, `make eval` |
| Wind anomalies (held-out park WP2, test 2025) | precision **0.92**, recall **0.89**, 0.070 alerts / turbine-day, curtailment-vs-fault F1 **0.93** (label-free normal selection, mean of 2 seeds) | WindMon rules: 0.84 / 0.75 / 0.077 (0.142 before de-duplication) / F1 0.39 | synthetic RO (ERA5-driven parks, #22) | train#593/594 (#99, #70's command; label-masked variant train#591/592: 0.92 / 0.88 / 0.071 / 0.93) |
| Early failure warning (M7, optional) | lead time **120 h**, event recall 0.98, 0.42 false alarms / turbine-year; held-out park 165 h | fixed residual threshold: 46 h, recall 0.87, 0.39 false alarms / turbine-year | synthetic RO (ERA5-driven parks) | train#379 (#43) |
| Early failure warning on real SCADA | CARE score 0.40, event recall 0: **does not transfer** | fixed threshold: CARE 0.39–0.41 | non-RO (CARE-to-Compare Farm A) | train#379 (#43) |
| National demand, day ahead | **nMAE 0.039**, peak-hour error 0.30 h, 10–90 coverage 0.74 | persistence week 0.067, day 0.069, national-shape scaling 0.070 | **real RO** | `make demand` (#44) |
| GIS feeder/phase recovery (grid-twin) | phase records 83.3 % recovered (precision 69.2 %, 54 corrupted); feeder records 100 % (16 corrupted) | GIS as-is: 0 % | synthetic RO fleet | replay#506 (gridtwin step), `make replay-demo` |
| P10 commitment method | coverage 0.62 → 0.795 after conformal; P10 hit rate 0.938 | — | synthetic (mock truth), method check | `make uncertainty` (#27) |
| Feeder LP | 100-site feeder solved in 1.9 s; slack 0 vs baseline +19.5 kW; evening peak 10.9 vs 52.8 kW | self-consumption rule | synthetic (mock forecasts) | #18 |

M5 WindPower (#42) is an experiment card and not merged, so its numbers are not claimed here.

**Error analysis (eval#521, #14).** The PV 10–90 band is the main weakness: coverage is 0.88 on seen sites but 0.61
on unseen sites and 0.10 on real PVDAQ systems, so the bands are too narrow off-distribution. On PVDAQ PVNet is only
14 % better than pvlib. PVNet is weakest on snow days (train-time nMAE 0.043–0.048 vs 0.021–0.022 without snow,
train#383–386). SiteHealth drops from PR-AUC 0.73 to 0.56 on unseen sites. LoadNet beats persistence but only ties
LightGBM. Under 40 % telemetry gaps PVNet degrades to nMAE 0.058 but stays ahead of LightGBM (0.096; eval#521
`--drills`). DemandNet's interval is too narrow (0.74 coverage vs the 0.78–0.82 target). About 88 % of M7's
synthetic false alarms are drifts the simulator never ended in failure, and on real CARE SCADA M7 does no better than
the threshold (#43 note). Findings are logged in
[team/insights.md](team/insights.md).

## Honest limitations

- **Synthetic circularity (plan R13, R14).** The simulator is built on pvlib physics, so beating pvlib on synthetic
  RO data is partly circular. On real NREL PVDAQ systems (non-RO) the gain shrinks to 14 % and the 10–90 band covers
  only 10 % (eval#521). A load model trained on our own synthetic households mainly shows it learns our generator;
  the real-RO check is DemandNet on national load.
- **Model training data.** M1/M2/M4 in `models/` are trained on the shared #11 fleet + #12 faults (#14
  train#383–392). The promoted M6 (#41 run 135) was trained on #41's own wind stand-in before the #22 parks existed;
  the wind evaluation we quote (#99 train#591–594) retrains the same model on #22 data. The demo replay runs run 135.
  On the stand-in data WindMon looked much weaker (precision 0.53, run 135); that gap is not the one we claim.
- **Strong LightGBM baseline.** LightGBM quantiles (#15) are far better than pvlib on our fleet. PVNet still beats
  them (0.015 vs 0.041 on seen sites), but LoadNet only ties them (0.285 vs 0.284 on unseen sites; eval#521).
- **Two replays.** `make replay` runs the small mock world with the contract `MockPredictor`; it prints
  `FALLBACK components in use` and its KPIs (e.g. 3870 → 2295 minutes) are a pipeline check, not results. The
  results come from `make replay-demo`: the simulated fleet with the committed `models/`, as in replay#506.
- **Control is simulated.** No real battery or turbine is controlled; wind is monitoring and forecasting only.
  Market figures, PPC fleet details and the export remuneration ratio β are inputs to verify, not claims (plan §8).
- **Coordinates.** The Deleni wind park position is approximate (no OSM match).

## Data

All raw inputs are committed as zstd Parquet / GeoJSON in [data/raw/](data/raw/) (31.5 MB); generated data is
rebuilt deterministically. Full table with columns: [data/README.md](data/README.md).

| Source | Used for | Licence | Label |
|---|---|---|---|
| [Open-Meteo Historical](https://open-meteo.com/en/docs/historical-weather-api) (Copernicus ERA5 / ERA5-Land), 2022–2025 | 33 prosumer localities, 3 wind parks, 0.5° grid over Romania | CC BY 4.0; ERA5 Copernicus licence | RO |
| [Energy-Charts](https://api.energy-charts.info) (Fraunhofer ISE, republishing ENTSO-E Transparency) | national load + generation by type, installed capacity | CC BY 4.0 | RO |
| [OPCOM PZU](https://www.opcom.ro) | day-ahead prices 2023–2025 (LP prices, MWh → lei) | public market data, attribution | RO |
| [OpenStreetMap](https://www.openstreetmap.org) via Overpass | wind-turbine positions, park centroids | ODbL 1.0 | RO |
| [geoBoundaries](https://www.geoboundaries.org) gbOpen ROU | county and country outlines | CC BY 4.0 | RO |
| [NREL PVDAQ](https://data.openei.org/submissions/4568) + matching ERA5 | 4 real PV systems: validation only | CC BY 4.0 | non-RO |
| CARE-to-Compare (Zenodo), pandapower IEEE European LV feeder | wind / grid-twin validation, downloaded at run time (#43, #13) | per source | non-RO |

No personal data: prosumers and households are synthetic, placed in real Romanian localities.

## Repository layout

```
sentinel/                 Python package
  contracts/              prediction envelope, schemas, actions, MockPredictor (stdlib only)
  data/download.py        raw data → data/raw/                sim/fleet/, sim/faults/   prosumer simulator + PV faults
  model/                  M1/M2/M4 (+ wind/ M6)               demand/                   M3 DemandNet
  opt/                    per-feeder LP, baseline rules, P10 commitment
  agent/                  qualify, triage, loop, replay, SQLite audit
  eval/uncertainty/       conformal, OOD, thresholds          kernels/                  Python kernels (+ optional Rust, rust/)
train.py  eval.py  replay.py  Makefile
apps/web/                 Next.js app: shell, map, modes      packages/db/  Drizzle schema = the shared SQLite contract
packages/api/             oRPC routers                         demo/         fixtures/ + promoted sentinel.sqlite
data/raw/                 committed raw inputs                 models/       promoted weights + thresholds.json
team/                     brief, plan, decisions, contracts    tests/        pytest
```

Interfaces between parts are frozen in [team/contracts/](team/contracts/) (Python side) and
[team/contracts/app.md](team/contracts/app.md) (SQLite schema + API); what the replay writes where:
[team/contracts/replay.md](team/contracts/replay.md).

## Prior work: WindMon

The wind side builds on **WindMon** by Andrei ([github.com/Andrei9383/Windmon](https://github.com/Andrei9383/Windmon),
PPC HackITall, April 2026): we reuse its PPC SCADA schema (`WP<park>.TID<turbine>.<sensor>`), its quality rules,
its four anomaly rules as the wind baseline, and its dashboard layout as visual reference. None of its code is
ported. The models, data pipeline, optimiser, agent and app in this repo were built during the 24 h.

## Team

Andrei (data, simulation, grid-twin) · Stefan (ML models, baselines, evaluation) · yolooo332 (contracts, optimiser,
agent, uncertainty, demand) · Tudor (app, map, Feeders + Production modes, demo) · Alexandra (fixtures, panels,
O&M + Demand modes). Work was coordinated as cards on a shared board by humans and coding agents; how we work:
[AGENTS.md](AGENTS.md).
