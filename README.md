<p align="center">
  <img src="apps/web/public/brand/wattsentry-icon.svg" width="72" alt="WattSentry logo">
</p>

<h1 align="center">WattSentry</h1>

<p align="center">
  Forecasting and control for distributed energy: rooftop solar, home batteries, wind farms and the low-voltage grid.
</p>

<p align="center">
  <a href="#the-problem">Problem</a> ·
  <a href="#quick-start">Quick start</a> ·
  <a href="#how-it-works">How it works</a> ·
  <a href="#models">Models</a> ·
  <a href="#ml-quality">ML quality</a> ·
  <a href="#robustness">Robustness</a> ·
  <a href="#explainability">Explainability</a> ·
  <a href="#reproducing-the-numbers">Reproduce</a>
</p>

![WattSentry overview](docs/screenshots/overview.png)

WattSentry watches every prosumer site, wind turbine and feeder on one map of Romania and acts every 15 minutes.
It forecasts solar output, household load and demand with calibrated P10/P50/P90 bands, decides which assets can be
trusted, schedules home batteries so feeders stay under their export limit, and sends repair crews where they save
the most. Every decision is written to an audit log together with the prediction and model version behind it.

It was built for PPC, Romania's largest energy supplier and distributor, at Smarthack XI, and runs entirely on
open data: real Romanian weather, market prices and national load, with a simulated prosumer fleet on top.

## The problem

Rooftop solar and home batteries changed how energy flows, but three PPC teams still decide as if they hadn't.

| PPC Energie · supplier | PPC Renewables · wind farms | Rețele Electrice · grid |
|---|---|---|
| **Paying twice for prosumers' energy** | **False alarms on wind turbines** | **Street cables over their limit** |
| PPC buys tomorrow's energy for its customers on the day-ahead market using a standard profile. Prosumers don't follow it: their solar exports at noon and their batteries cover the evening. PPC buys energy those homes never take, then pays imbalance on the difference. | SCADA rules raise an alarm whenever power is low or a temperature is high. They can't tell a broken turbine from a grid curtailment, icing or a faulty wind sensor, so crews drive out for nothing while slow faults get missed or misnamed. | At midday, solar export pushes low-voltage feeders over their limit. Operators can't plan around it because the grid map (GIS) has the wrong cable or phase for some homes, and nobody knows which. |
| **Agent:** learns each home's battery strategy from its smart meter alone, simulates tomorrow, and buys the quantile that minimises expected imbalance. | **Agent:** a normal-behaviour model per turbine plus a classifier name one of 8 states, then turn them into ranked work orders, safety alerts or "logged, no crew". | **Agent:** schedules home batteries so P90 export stays under each cable's limit, and flags GIS records the voltage data contradicts. |
| **2025:** avoidable imbalance cost 873,110 → **177,397 lei (−79.7 %)**, 600 homes. | **Unseen park, 2025:** precision 0.84 → **0.92**, recall 0.75 → **0.89**; curtailment-vs-fault F1 0.39 → **0.93**. | **Demo week:** minutes over the limit 6,645 → **1,905**; wrong-cable records recovered **100 %**, wrong-phase **83 %**. |

The theme's problem types all appear, and each one feeds a decision:

| ML problem | Model | Decision it drives |
|---|---|---|
| Time-series forecasting with uncertainty | PVNet, LoadNet, DemandNet, WindPower | battery schedules, P10 commitment, day-ahead position |
| Anomaly detection with few labels | WindNBM (label-free normal behaviour), GridTwin | turbine status, GIS review requests |
| Classification on imbalanced data | SiteHealth (8 classes, faults rare), battery strategy (price-driven homes rare), WindFailure | derate, exclude, work order, purchase quantile |
| Clustering | GridTwin (phase centroids) | which phase a home is really on |
| Ranking under capacity | work-order triage, top-K per crew per day | tomorrow's crew list |

## Features

- **Feeder limits.** A per-feeder optimiser plans home batteries a day ahead so the worst plausible midday export
  stays under the cable limit and the stored energy covers the evening peak. Customers' comfort reserve is never
  touched.
- **Committed flexibility.** Flexibility is offered to the market at its P10, computed from conformal-calibrated
  scenarios, so what is promised is what healthy assets can actually deliver.
- **Asset health.** Each site and turbine gets one status every tick: eligible, derated, excluded or human review.
  Uncertain cases go to a person instead of a truck.
- **Wind fleet monitoring.** A normal-behaviour model per turbine separates real faults from curtailment, icing and
  sensor noise, and ranks inspections by expected loss.
- **Work orders.** A daily top-K list per crew type, scored by lost energy, lost flexibility and travel.
- **Demand and imbalance.** County-level peak demand forecasts, and a comparison of what the supplier bought on the
  day-ahead market against what its prosumers actually used.
- **Judge.** A second model grades every prediction against what happened and flags the ones not to trust.
- **Voice.** Optional narrated walkthrough and "Ask WattSentry" Q&A over the current view (ElevenLabs).

| | |
|---|---|
| ![Feeders](docs/screenshots/feeders.png) | ![Production](docs/screenshots/production.png) |
| **Feeders**: export vs. limit for every feeder, with and without WattSentry | **Production**: live wind field, park forecasts and rooftop solar by county |
| ![O&M](docs/screenshots/om.png) | ![Demand](docs/screenshots/demand.png) |
| **O&M**: site and turbine status, alerts, work orders and the review queue | **Demand**: peak stress per county with forecast bands |

## Quick start

Requirements: Node.js 22+, and for the Python side Python 3.12 with [uv](https://docs.astral.sh/uv/).

```bash
git clone https://github.com/Andrei9383/smarthack.git wattsentry
cd wattsentry
npm ci
npm run build
npm start                 # http://localhost:3000
```

The app ships with a recorded replay week (`demo/wattsentry.sqlite`), so it works offline and needs no Python.
Each mode has its own URL, e.g. `/?mode=feeders`.

To run the Python pipeline (simulator, models, replay):

```bash
make setup                # .venv with Python 3.12 and requirements.txt
make test                 # pytest
make replay               # quick end-to-end replay on a small mock world
make reproduce            # every step from raw data to the demo database, ~10 min on a laptop
```

`make help` lists all targets. All raw inputs are committed under `data/raw/`, so everything after the first
`pip install` runs offline.

## How it works

Every 15 minutes the agent runs one tick:

```
perceive ──► estimate ──► qualify ──► decide ──► act ──► log
telemetry    PV, load,    status per   per-feeder  battery     audit row per
weather      site health, asset        LP + P10    schedules,  action: model,
prices       wind, demand              commitment  work orders input hash,
                                                   alerts      prediction
```

- **Estimate.** Every model returns the same prediction envelope: quantiles, fault probabilities, an
  out-of-distribution score and ensemble spread.
- **Qualify** (`wattsentry/agent/qualify.py`). Stale telemetry excludes an asset. A wide interval, ensemble
  disagreement or unfamiliar input sends it to human review. A confident fault derates it. Thresholds are tuned on
  the validation year and frozen in `models/thresholds.json`.
- **Decide** (`wattsentry/opt/lp.py`). A linear program (cvxpy + HiGHS) per feeder keeps P90 solar minus P10 load
  under the limit, respects each battery's comfort reserve and shaves the 17:00–22:00 import peak.
- **Commit** (`wattsentry/opt/commit.py`). About 200 calibrated scenarios per feeder-day give the deliverable
  flexibility; the committed amount is its P10 minus a haircut.
- **Act and log** (`wattsentry/agent/`). Schedules, commitments, work orders and alerts go to SQLite, which the web
  app reads. The app never runs Python: every panel is precomputed by `replay.py`.

## Models

All models are trained from scratch in this repository and served with onnxruntime. Each one has a simple fallback,
so a missing model degrades a panel rather than the app.

| Model | Question | Approach | Fallback |
|---|---|---|---|
| PVNet | What should this PV system produce? | Patch transformer on weather, solar geometry and the site's own history; 4-seed ensemble | LightGBM quantiles |
| LoadNet | What will this household consume? | Same encoder, separate weights | Persistence |
| SiteHealth | Is this prosumer site healthy? | 1D-CNN on forecast residuals and telemetry, 8 fault classes | Threshold + CUSUM |
| WindNBM | Is this turbine behaving normally? | GRU normal-behaviour model + LightGBM on residuals | Rule set |
| WindPower | What will each park produce? | Probabilistic power-curve model, P10/P50/P90 | Binned power curve |
| WindFailure | Will a component fail within 7 / 14 days? | LightGBM on daily normal-behaviour residuals. Optional: validated on simulated parks only | Residual trend |
| DemandNet | Where and when will demand peak? | LightGBM quantiles per county | Persistence |
| Battery strategy | How does this home use its battery? | LightGBM on 28 days of smart-meter features; also estimates size, power and reserve | Standard profile |
| GridTwin | Is the GIS feeder/phase mapping right? | Unsupervised: voltage-step correlation for the cable, GIS-seeded phase centroids on night voltage | GIS as recorded |
| Judge | Can this prediction be trusted? | LightGBM detectors trained on replayed outcomes | Static priors |

Trained weights live in `models/`. Training entry point: `python train.py --model <name>` (needs
`requirements-train.txt` for torch). Evaluation against every baseline on the 2025 test year: `make eval`.

### Results (2025 test year)

| | WattSentry | Baseline |
|---|---|---|
| Avoidable imbalance cost, 600 homes, 2025 | **177,397 lei (−79.7 %)** | 873,110 lei (standard profile); 564,822 lei (forecast without the battery model) |
| Battery strategy F1, unseen homes (2,000-home world, 10 % of batteries price-driven) | **0.971** | – |
| Minutes over feeder export limit, demo week | **1,905** | 6,645 (self-consumption rule) |
| Energy over the limit | **94.5 kWh** | 1,397 kWh |
| P10 commitment hit rate | **92.0 %** | – |
| PV forecast nMAE, seen / unseen sites | **0.015 / 0.036** | 0.041 / 0.045 (LightGBM), 0.083 / 0.093 (pvlib) |
| Site fault detection PR-AUC, seen / unseen | **0.73 / 0.56** | 0.49 / 0.40 (threshold + CUSUM) |
| Site fault episodes found / median time to detect | **97.5 % / 21.5 h** | 40 % / 35.9 h |
| Wind anomaly precision / recall, unseen park | **0.92 / 0.89** | 0.84 / 0.75 (WindMon rules) |
| Wind curtailment-vs-fault F1 | **0.93** | 0.39 |
| Component failure warned / median lead time | **98 % / 120 h** | 87 % / 46 h (fixed residual threshold) |
| Wrong GIS records recovered, cable / phase | **100 % / 83.3 %** | 0 % (GIS as recorded) |
| National demand, day ahead (real data) | **nMAE 0.039** | 0.067 (persistence) |

Feeder, PV, fault and wind figures come from the simulated fleet driven by real Romanian weather; the simulator uses
pvlib physics, so the pvlib comparison flatters the model. On four real US PV systems (NREL PVDAQ) the gain over
pvlib drops to 14 %. National demand is measured on real Romanian load. The wind figures are from the retrained
model (train runs #593/594); the demo database still runs the older r135 model.

## ML quality

### How the data was prepared

- **Real inputs:** weather (ERA5 via Open-Meteo) for every site and park, OPCOM prices, national load and turbine
  positions (see [Data](#data)).
- **Synthetic on top:** the prosumer fleet, its faults and the wind SCADA are simulated, because no public labelled
  Romanian data exists. Faults are injected with known start and end times, which gives exact labels.
- **Realistic noise on purpose:** 20–25 % of homes have wrong panel or battery records, GIS has wrong cables and
  phases, meters drop readings, and some degradation drifts never end in a failure.
- **Features with a reason:** every model input is listed with the fault it is meant to reveal. For example:
  - measured ÷ expected power is a steady 0.5 or 0.67 when a solar string is out;
  - evening export reveals a battery, because solar can't export after sunset;
  - the gap to the park median separates a broken turbine from a park-wide curtailment.

### Data split

| Model | Train | Validation (epoch, thresholds) | Test (touched once) |
|---|---|---|---|
| PVNet, LoadNet, SiteHealth | 2022–2023, 70 % of homes | 2024 | 2025, reported separately for seen and the 30 % unseen homes |
| WindNBM | before 2025, days 1–19 of each month, parks WP1 and WP3 | days 22–31 (2-day gap) | 2025 on park WP2, never seen in training |
| WindFailure | before 2025, 4 time blocks with a 14-day embargo | held-out blocks | 2025, unseen park reported separately |
| Battery strategy | 2022–2023 | 2024 | 2025, unseen homes; meter data only up to D−2 |
| GridTwin | nothing learned | thresholds tuned on the fleet | flags scored against the hidden true cable and phase |

Real-data checks outside Romania: NREL PVDAQ (4 US PV systems, faults injected into real power) and CARE-to-Compare
(a real wind farm with real failures).

### Where the models fail

| Model | Error | Why | What we did |
|---|---|---|---|
| SiteHealth | String outage called "sensor fault" on battery homes | The only "PV off, rest normal" example in training was a sensor fault | Battery-home augmentation in training, plus a test that injects the outage |
| SiteHealth | Soiling called on new homes, only 20–34 % right | A steady gap caused by wrong records looks like dirt | The agent needs P(soiling) ≥ 0.7 instead of 0.6 |
| SiteHealth | PR-AUC drops 0.73 → 0.56 on unseen homes | No history of that home's normal | Reported openly; uncertain cases go to human review |
| SiteHealth | 10.6 % of healthy windows alarm in 2025 vs a 5 % target | Threshold set on 2024; the year changed | The agent acts at 0.6, above the model's own 0.387 |
| WindNBM | Slow faults misnamed (underperformance 27 of 64, overheating 31 of 58) | Both look like "a bit worse for days" | Still detected and sent to inspection |
| WindNBM | 249 of 809 icing episodes missed | Light icing barely cuts power | Icing found is still top priority |
| WindFailure | 0 real failures caught on CARE | Simulated drifts are cleaner than real ones | Marked optional; the app shows the raw trend, never a score, unless the model is marked shipped |
| GridTwin | Homes near the transformer flagged on the wrong phase | Their voltage barely moves with export | Phase flag threshold raised to 0.98; phase precision 0.69 → 0.91 |
| Battery strategy | Price-driven homes are rare (12 of 379 in test) | Few examples | F1 on that class is fragile; low-confidence homes are reported |

## Decision layer

Every model returns the same prediction envelope (`wattsentry/contracts/envelope.py`): quantiles, class
probabilities, an out-of-distribution score and ensemble spread. `wattsentry/agent/qualify.py` turns it into one of
four statuses, first matching rule wins:

| Status | Meaning | Battery pool | Crew |
|---|---|---|---|
| eligible | healthy and predictable | yes | no |
| derated | probably faulty but still producing | yes, at reduced power | work order |
| excluded | offline, stale, sensor fault, battery fault, turbine stopped | no | truck for a real fault, remote check for a sensor |
| human_review | the agent is not sure | no | a person looks first |

Each threshold was chosen on the 2024 validation year, for a stated reason, and then frozen in `models/thresholds.json`:

| Threshold | Value | Why this value |
|---|---|---|
| Act on a site fault | P ≥ 0.6 | Above the model's 0.387 alarm point, which already overshoots its 5 % false-alarm target in 2025 |
| Act on soiling | P ≥ 0.7 | Soiling is the most confused class on homes with wrong records |
| Ensemble spread | > 0.15 → review | Model copies that disagree mean the answer isn't stable |
| Interval width | > 0.8 of the value → review | The forecast behind the gap is too unsure to call a fault |
| OOD score | > 3 → review | Input unlike anything in training |
| Stale telemetry | > 60 min → excluded | One offline window instead of an alarm every 15 minutes |
| Wind classes | 0.35 (stop) to 0.93 (slow faults), with a minimum duration | Picked per class on validation; slow faults need hours of evidence |
| WindFailure alarm | risk ≥ 0.208 | Budget of 0.5 false-alarm episodes per turbine per year |
| GridTwin cable / phase flag | ≥ 0.7 / ≥ 0.98 | Every flag costs a field visit, so phase (the weaker signal) needs more certainty |
| Day-ahead purchase | quantile a / (a + b) of ~200 scenarios | The cost-optimal quantile for asymmetric imbalance prices a and b |

**Capacity.** Crews are limited: 5 residential, 3 wind and 20 remote checks per day (`wattsentry/agent/triage.py`).
Work orders are ranked by P(fault) × lost value per day × days until a crew can get there. What doesn't fit today
goes to a ranked backlog.

**Actions.** Battery schedules, P10 commitments, day-ahead positions, ranked work orders, safety alerts (icing:
keep clear), grid-map review requests and feeder notices. Wind is monitoring only: the agent never stops a turbine.

## Robustness

| Situation | How the agent handles it |
|---|---|
| Missing data | Stale telemetry over 60 min → excluded, one offline window. A data gap or a confident sensor fault gets a remote check, never a truck |
| Rare cases | Focal loss, 50 % of training windows drawn around faults, and logit adjustment back to real frequencies (SiteHealth). Balanced sampling per class (WindNBM). Class weights (battery strategy) |
| Wrong input data | ±30 % noise on site records during training, because 20–25 % of records are wrong. GridTwin finds wrong cable and phase records and lowers that cable's trust, so the optimiser uses a safer export limit |
| Uncertainty | OOD score, ensemble spread and interval width each send a case to human review: "not sure, a person looks" |
| Unseen assets | A learned "unknown turbine" identity (turbine-id dropout) and results reported separately for unseen homes and the unseen park |
| Broken models | Every model has a fallback, so a missing model degrades one panel, not the agent |

`python eval.py --predictor onnx,baselines --drills` replays robustness drills on the test year (`wattsentry/data/drills.py`):
10 % and 40 % missing readings, a missing day, a frozen sensor, power sent in kW instead of W, an unseen site type, a
6-hour telemetry dropout and day-ahead weather error.

## Explainability

- **"Why this decision" in the O&M drawer:** fault class, action, days to visit, energy and lei lost per day,
  confidence, the fault probability and headroom bars, and the evidence the triage wrote down. For a turbine, that
  includes the component temperature trend against its own normal.
- **Review queue reasons:** every human-review case says why, for example "Unseen behaviour (OOD)" or "Wide interval".
- **Plain-language flags:** GridTwin writes messages such as "likely L2 of F07, GIS says L1 of F06, 0.83". Feeder
  notices give the reason and the homes involved.
- **Audit log:** every action is a row with the model, model version, input hash, prediction, threshold and
  resulting status, so any decision can be traced back to the prediction that produced it.
- **Judge:** grades each model's predictions against what happened and marks the ones not to trust.

## Reproducing the numbers

| Result | Command |
|---|---|
| Everything, from raw data to the demo database | `make reproduce` |
| PV, load, site-health and baseline figures, plus the drills | `make eval` |
| Imbalance cost, 600 homes | `python replay.py --mode imbalance` |
| Battery strategy F1 | `python replay.py --mode imbalance --data v2 --share-maxcontrol 0.1` |
| Feeder minutes over the limit, demo week | `make replay-demo` |
| Wind anomaly | `python train.py --model windnbm` |
| Component failure | `python train.py --model windfailure` |
| GridTwin flags | `python -m wattsentry.gridtwin.run` |
| National demand | `make demand` |

## Project layout

```
apps/web/            Next.js app: map, modes, panels, voice
packages/
  api/               oRPC routers the app calls
  db/                Drizzle schema of the replay database
  ui/                shared UI components and styles
  auth/              better-auth setup
  config/            shared TypeScript config
wattsentry/          Python package
  agent/             tick loop, qualification, triage, replay, audit log
  opt/               feeder LP, baseline policies, P10 commitment
  model/             PVNet, LoadNet, SiteHealth, wind models, battery behaviour
  judge/             prediction grading and trust models
  sim/               prosumer fleet, fault injection, wind parks
  data/              loaders and downloaders for data/raw
  eval/              baselines, metrics, uncertainty, imbalance replay
  demand/  gridtwin/ live/  contracts/  kernels/  infra/
rust/                optional Rust versions of the hot kernels
train.py eval.py replay.py
data/raw/            committed raw inputs (Parquet, GeoJSON)
models/              trained weights, calibration and thresholds
demo/                replay databases and grid results the app reads
scripts/             dataset builders, demo database tools, live feed
tests/               pytest suite
```

## Development

```bash
npm run dev               # app with hot reload
npm test                  # TypeScript type checks across the workspace
make test                 # Python tests
sh rust/build.sh build    # optional: build the Rust kernels (Python stays the reference)
```

The app picks its database in this order: `DATABASE_URL`, `demo/wattsentry.sqlite`, `demo/fixtures.sqlite`
(generate with `make fixtures`), then an empty in-memory database. Copy `.env.example` to `.env` for the optional
settings (voice key, live feed path).

## Data

| Source | Used for | Licence |
|---|---|---|
| [Open-Meteo](https://open-meteo.com/en/docs/historical-weather-api) (ERA5 / ERA5-Land), 2022–2025 | weather for 33 localities, 3 wind parks and a 0.5° grid over Romania | CC BY 4.0 |
| [Energy-Charts](https://api.energy-charts.info) (Fraunhofer ISE, ENTSO-E) | national load, generation, installed capacity | CC BY 4.0 |
| [OPCOM](https://www.opcom.ro) | day-ahead and intraday prices | public market data |
| [OpenStreetMap](https://www.openstreetmap.org) | wind turbine positions | ODbL 1.0 |
| [geoBoundaries](https://www.geoboundaries.org) | county and country outlines | CC BY 4.0 |
| [NREL PVDAQ](https://data.openei.org/submissions/4568) | real PV systems, validation only | CC BY 4.0 |

Prosumers and households are synthetic and contain no personal data. `python -m wattsentry.data.download`
re-downloads the raw inputs.

## Acknowledgements

The wind side builds on [WindMon](https://github.com/Andrei9383/Windmon) (PPC HackITall 2026): its SCADA naming,
quality rules and anomaly rules serve as the wind baseline.

Built by Andrei, Stefan, Tudor, Alexandra and yolooo332.
