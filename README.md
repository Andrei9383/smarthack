# Prosumer Sentinel

**An energy agent for PPC (Romania).** On each low-voltage feeder it schedules home batteries so midday solar export
stays under the feeder's limit and is released into the evening peak. It watches wind-turbine fleets to separate
real faults from curtailment and sensor noise. It commits only the flexibility that healthy, predictable assets
can actually deliver (P10). Every decision is written to an audit log that names the model prediction behind it.

Built in 24 h at Smarthack XI (PPC theme "24h to Power PPC"). Challenge summary: [team/brief.md](team/brief.md);
binding plan: [team/plan.md](team/plan.md).

```bash
make reproduce                                 # offline after the first install, ~8 min on a laptop CPU
npm ci && npm run build && npm start           # the app on http://localhost:3000
```

| Who at PPC | Gets |
|---|---|
| Distribution / network ops | Day-ahead battery plan per feeder; export peaks held under the limit; demand map |
| Renewables wind O&M | Expected-vs-actual per turbine; fault vs curtailment vs icing vs sensor; ranked inspection list |
| Residential O&M | Daily top-K work orders ranked by lost kWh and lost flexibility; "not sure" cases go to a human |
| Supply / trading | Hourly committed prosumer flexibility (P10) with a delivery record |

**Headline results (test year 2025, details and run IDs in [Results](#results)):** PVNet nMAE 0.118 vs pvlib 0.159
(synthetic RO) · LoadNet 0.20 vs persistence 0.35 (synthetic RO) · SiteHealth fault PR-AUC 0.63 vs rule 0.22
(synthetic RO) · DemandNet 0.039 vs persistence 0.067 on **real Romanian national load** · feeder replay KPIs:
pending (#29).

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
| **O&M** (mentenanță) | Sites and turbines by status, alert + work-order queue, human-review queue, audit trail | pending (#32) |
| **Production** (producție) | Wind particles from the real ERA5 100 m wind field, parks and counties by expected MWh | on main (#39, #55, #59) |
| **Demand** (cerere) | Zones by peak-stress score, peak hour, P50 with P10–P90 band | pending (#40) |

The app reads `demo/sentinel.sqlite` (the promoted replay, #29), else `demo/fixtures.sqlite` (mock week from
`python3 demo/fixtures/make_fixtures.py`, labelled synthetic), else an empty in-memory DB. It never runs Python,
torch or the LP: every panel is precomputed by `replay.py`.

---

## Models

All models are trained from scratch in this repo and run in our code (ONNX + onnxruntime at inference, no
external API). Each one has a fallback, so a missing model degrades a panel, never the demo.

| # | Model | Question | Architecture | Tier | Status | Fallback |
|---|---|---|---|---|---|---|
| M1 | PVNet | What should this PV system produce? | PatchTST-style encoder, 0.29 M params, weather + solar geometry + the site's own past-power reference window | MUST | trained, 4-seed ensemble (#14) | LightGBM quantile (#15) |
| M2 | LoadNet | What will this household consume? | same encoder, separate weights, 0.21 M params | MUST | trained, 4 seeds (#14) | persistence |
| M4 | SiteHealth | Is this prosumer site healthy? | 1D-CNN on M1/M2 residuals + telemetry + 14-day summary, focal loss, 8 classes, 56 k params | MUST | trained (#14) | threshold rule + CUSUM (#15) |
| M6 | WindNBM | Is this turbine behaving normally? | GRU normal-behaviour model (power + component temps) + LightGBM classifier on residuals and park context, 0.13 M params | SHOULD | trained, sweep running (#41) | WindMon rules (#24) |
| M3 | DemandNet | Where and when will demand peak? | v1 sum of LoadNet scenarios per zone; v2 LightGBM quantile per county | SHOULD | on main (#44) | persistence |
| M8 | GridTwin | Is the GIS feeder/phase right? | voltage correlation + spectral clustering | SHOULD | pending (#13) | GIS as-is (τ = 1) |
| M5 | WindPower | What will each park produce? | probabilistic power-curve MLP | OPTIONAL | pending (#42) | binned empirical curve |
| M7 | WindFailure | Will a component fail soon? | LightGBM on daily M6 residuals | OPTIONAL | pending (#43) | residual trend, no score |

Not ML: the per-feeder LP, the P10 rollout and the qualification rules.

Model code reaches `main` when its config is chosen (plan R17): `train.py` + `sentinel/model/` come with #14,
`sentinel/model/wind/` with #41, the promoted ONNX weights in `models/` with #29.

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
network access is the first `uv pip install`; all raw data is committed in [data/raw/](data/raw/). Outputs go to
`.lab-out/reproduce/` (gitignored); committed files in `demo/` and `models/` are never overwritten. It ends with a
table of every number it printed, the step and card that produced it, and its label.

| Target | What it runs | Output | Time* |
|---|---|---|---|
| `make setup` | `.venv` (Python 3.12) + `requirements.txt` | `.venv/` | ~5 s cached |
| `make test` | `pytest tests` (contracts, simulator, faults, LP, qualify, uncertainty, demand, agent, kernels) | pass/fail | ~30 s |
| `make sim` | RO prosumer fleet, 600 sites / 15 feeders / 2022–2025 (#11) + PV fault injector (#12) | `fleet/`, `faults/` | ~40 s |
| `make baselines` | `eval.py --baselines`: pvlib (recorded metadata), persistence, LightGBM quantiles, threshold rule + CUSUM (#15) on the test fleet, PVDAQ and real RO series | `baselines/` | ~5 min (builds the test fleet once) |
| `make eval` | `eval.py`: model test-set metrics grouped by label (#16). Scores `models/*.onnx` (#29); until those are committed it runs the `MockPredictor`, a pipeline check | `eval/` | ~1 min |
| `make demand` | M3 DemandNet on real RO load, test 2025 (#44) | `demand/` | ~8 s |
| `make uncertainty` | conformal calibration + P10 commitment on mock forecasts (#27) | `uncertainty.json` | ~3 s |
| `make replay` | baseline vs Sentinel, same 2025 test week, through the whole agent (#17) | `sentinel.sqlite` | ~15 s |
| `make app` | `npm ci`, `npm run build`, `npm start` (`APP_DB=.lab-out/reproduce/sentinel.sqlite` to view the replay) | http://localhost:3000 | ~40 s build |
| `make train` | prints the lab commands; `TRAIN_LOCAL=1 MODEL=…` trains locally | `train/` | — |
| `make clean` | removes `.lab-out/reproduce/` | | |

\*Apple-silicon laptop. Useful variables: `EVAL_ARGS=--quick` (150-site eval fleet), `REPLAY_WORLD=sim` (replay the simulated 600-site fleet instead of the
mock world), `FULL=1` (fault injector on the full fleet, ~4 min), `SEED`, `PROMOTE=1` (allow writing a committed
`demo/sentinel.sqlite`). Steps whose code has not merged yet print `skipped: … (#card)` and the target stays green.

Data can be re-downloaded with `python -m sentinel.data.download` (no keys; `--only <job>`). Optional Rust kernels
(#48): `sh rust/build.sh build`; Python is the reference and the default when the extension is missing.

---

## Results

Every number carries a label: **synthetic RO** (our simulator, driven by real Romanian weather), **real RO**
(measured Romanian data) or **non-RO** (validation only, never trained on). Splits: train 2022–23, validation 2024,
test 2025, 30 % of sites never seen in training; thresholds frozen on validation. Run IDs are lab runs; the numbers
are the ones reported on the cards.

| Claim | Model | Baselines (same test split) | Label | Source |
|---|---|---|---|---|
| PV nowcast nMAE | **0.118 ± 0.001** (4 seeds), 10–90 coverage 0.79–0.80 | pvlib on recorded metadata 0.159 (−26 %), pvlib rescaled 0.137, persistence 0.53 | synthetic RO | train #120–#123 (#14) |
| PV day-ahead forecast nMAE | **0.227** | pvlib 0.316 | synthetic RO | train #125 (#14) |
| Household load nMAE | **0.20**, coverage 0.80 | persistence (yesterday) 0.35 (−42 %) | synthetic RO | train #126–#129 (#14) |
| Site fault detection | **PR-AUC 0.63**; 81 % of faults detected within 72 h, mean 25.8 h | threshold rule 0.22 / CUSUM 0.21 PR-AUC; 29 % / 46 % detected within 72 h | synthetic RO | train #131 (#14) |
| Wind anomalies (held-out park) | precision **0.83**, recall **0.87**, 0.104 alerts / turbine-day, curtailment-vs-fault F1 **0.99** | WindMon rules: 0.53 / 0.43 / 0.107 / F1 0.31 | synthetic RO | train #113 (#41) |
| National demand, day ahead | **nMAE 0.039**, peak-hour error 0.30 h, 10–90 coverage 0.74 | persistence week 0.067, day 0.069, national-shape scaling 0.070 | **real RO** | `make demand` (#44) |
| P10 commitment method | coverage 0.62 → 0.795 after conformal; P10 hit rate 0.938 | — | synthetic (mock truth), method check | `make uncertainty` (#27) |
| Feeder LP | 100-site feeder solved in 1.9 s; slack 0 vs baseline +19.5 kW; evening peak 10.9 vs 52.8 kW | self-consumption rule | synthetic (mock forecasts) | #18 |
| Feeder peak shaving, minutes over limit, P10 hit rate, comfort breaches, lei/home, alerts/turbine-day | pending | self-consumption rule, tuned per-home threshold rule | synthetic RO | replay run → #29 |
| PV nowcast on real systems | pending: `make eval` once #29 commits the M1 ONNX | pvlib, LightGBM (`make baselines`) | non-RO (NREL PVDAQ) | #16, #29 |
| GIS feeder/phase recovery | pending | GIS as-is | non-RO (IEEE LV feeder) | #13 |

**Error analysis so far (#14).** PVNet is weakest on snow days and on unseen sites (coverage 0.73); SiteHealth finds
soiling and shading hardest (average precision 0.12–0.19); EV households are the hardest loads. DemandNet's
interval is too narrow (0.74 coverage vs the 0.78–0.82 target); conformal widening from #27 is next. Findings are
logged in [team/insights.md](team/insights.md).

## Honest limitations

- **Synthetic circularity (plan R13, R14).** The simulator is built on pvlib physics, so beating pvlib on synthetic
  RO data is partly circular; the real-system check on NREL PVDAQ (non-RO) is pending (#16). A load model trained on
  our own synthetic households mainly shows it learns our generator; the real-RO check is DemandNet on national load.
- **Model training data.** M1/M2/M4 (#14) and M6 (#41) were trained on their cards' own deterministic RO stand-in
  generators (`sentinel/model/synth.py`, `sentinel/model/wind/synth.py`) while the shared windows (#10) and the wind
  simulator (#22, #23) were being built. They are meant to drop in unchanged.
- **Strong LightGBM baseline on the shared fleet.** On the #11 fleet the LightGBM quantile baseline (#15) beats pvlib
  by a wide margin (lab baseline#186: PV nMAE 0.031 vs 0.178, validation 2024; `make baselines` prints the test-2025
  numbers). M1 has not been scored on that fleet yet, so M1 vs LightGBM is an open comparison (#14, #16), not a claim.
- **Mock components in the replay.** Until #29 promotes the ONNX weights, `replay.py` uses the contract
  `MockPredictor` for PV/load/site health and built-in WindMon-style rules for wind. The replay prints
  `FALLBACK components in use` and stores it in `runs.notes`, so its KPIs are a pipeline check, not results.
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
