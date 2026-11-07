# Plan: Prosumer Sentinel

An AI agent that watches every PV + BESS installation, catches faults before they become losses, and keeps the
grid model honest. Core = PPC's O&M challenge; second action = PPC's GIS challenge (droppable module).
Full proposal: the team doc "Prosumer Sentinel — Smarthack XI PPC" (Nov 7). This file is the binding summary.

## The idea

One model, three decisions, four actions. The agent runs on a 15-min tick: perceive → estimate → decide → act → log.

- **Model ("SentinelNet")**: one multi-task time-series model, trained from scratch (~0.5 M params, PatchTST-style
  encoder + weather / solar-geometry covariates). Heads: (1) expected production P10/P50/P90 = the nowcast "what
  should it have made in the weather it had" (pinball loss); a secondary noisy-weather forecast mode for the
  battery planner only; (2) fault class, 8 classes (healthy, soiling, partial shading, string outage, inverter
  derating/clipping, inverter trip/overvoltage, battery underuse/degradation, sensor fault; focal loss);
  (3) embedding for similar past cases and OOD detection. Uncertainty: 4-seed ensemble + split-conformal.
- **Decide**: O&M triage (expected loss = P(fault) × kWh/day × price × days to next visit; pick K visits/day with
  OR-Tools, route-ordered; sensor faults get a remote check, never a truck). Battery schedule (cvxpy + HiGHS MILP
  on P50, P10 when uncertain). "Not sure" rule (ensemble std, conformal width, Mahalanobis OOD) → human queue.
- **Act**: work orders (JSON + PDF), alerts (webhook), battery schedule + plain-language tip, GeoJSON GIS flag
  layer, and an audit log (model version, input hash, prediction, threshold) for every action.
- **Explain**: Integrated Gradients, 3 most similar past cases (FAISS), expected-vs-actual chart with P10–P90 band,
  optional local LLM sentence (never decides).
- **Grid-twin** (`sentinel/gridtwin/`, promote or drop after PPC's presentation): voltage correlation between
  inverters → spectral clustering → compare with GIS feeder/phase → flags with confidence; trust score per LV
  section that decays since last verification.

## Success criteria (every slide number cites a lab run ID)

| Claim | Baseline | Target |
|---|---|---|
| Nowcast nMAE on real PVDAQ (headline) | pvlib with recorded metadata, persistence, LightGBM | beat pvlib; stretch ≥ 20% lower nMAE |
| Interval calibration | n/a | 10–90% coverage in 88–92% |
| Fault detection: PR-AUC + hours-to-detect, synthetic and PVDAQ + injected | threshold rule (< 80% of pvlib for 3 days), CUSUM on pvlib residual | beat both |
| Energy / money recovered per month across the fleet | doing nothing | kWh and lei |
| Battery schedule savings per customer | default self-consumption rule | lei/customer, OPCOM prices |
| Grid-twin (if kept) | n/a | % of corrupted feeder/phase records recovered |

## Demo script (3 minutes)

1. Replay October for the fleet from precomputed predictions (preview CPU runs only decide / act / log).
2. A string fails on day 4: Sentinel flags it within hours; CUSUM in about a day; the threshold rule in 3 days.
3. The work order lands in tomorrow's top-6 with "N lei at risk", with the expected-vs-actual band and similar cases.
4. A cluster of inverter trips on one feeder is routed to the network team; the GIS layer turns that section orange
   with its confidence.
5. A strange new site goes to a human with "not sure".
(Numbers are placeholders until `eval.py` produces them.)

## Scope

- Data: Open-Meteo ERA5 (20–40 RO locations, 2022–2025), pvlib, NREL PVDAQ (a handful of sites), OPCOM PZU
  2023–2025, London Smart Meter (~200 households), pandapower IEEE European LV (grid-twin only). All raw inputs
  committed as compressed Parquet (10–30 MB): no outside APIs during runs.
- Synthetic fleet: 500–1,000 sites × 2–3 years at 15 min, with effects pvlib doesn't model (wrong recorded
  tilt/azimuth on ~20% of sites, inverter curves, degradation, snow, horizon shading). Fault injector with exact
  onset labels (~15% faulty site-days) that also runs on real PVDAQ series.
- Generator: deterministic (fixed seed), < 5 min, cached at `~/.cache/sentinel/data-v{N}`.
- Splits: train 2022–2023 on 70% of sites; val 2024; test 2025 + the 30% unseen sites; PVDAQ clean (nowcast);
  PVDAQ + injected (fault head); a separate calibration slice for conformal.
- Robustness drills: 10–40% gaps, a missing day, frozen sensor, W vs kW, unseen site type → handle or "not sure".

## Amendments from the plan review (Nov 7, hour 0)

1. **Nowcast gets a reference window of the site's own past power** (e.g. 7–14 days before the target day, daily
   or patch summaries), never the target window itself. Without it the model sees the same wrong metadata as
   pvlib and cannot honestly beat it on wrong-tilt sites. Faults are also injected into the reference window
   during training so the model learns to discount it.
2. **Fault head needs multi-day context.** A 24 h window can't see soiling (slow) or degradation, and battery
   classes need SoC / battery power / load channels. Input = 24 h detailed window + ~14-day daily residual
   summary + battery channels.
3. **Headline gate is "beats pvlib on PVDAQ"**; 20% is a stretch. PVDAQ is US data while training is RO synthetic,
   so report the domain gap honestly.
4. **Battery economics**: most Romanian residential prosumers are on a fixed tariff with compensation for exported
   energy, not on PZU. Frame the PZU schedule as the dynamic-tariff / aggregator scenario and also report the
   fixed-tariff case. Ask the O&M mentor.
5. **Grid-twin cost**: `runpp_3ph` on the ~900-bus European LV feeder is slow; time 10 steps first, then pick
   hourly steps over 1–2 weeks if needed. One feeder ⇒ phase identification only; feeder identification needs
   ≥ 2 feeders.
6. **ERA5 is hourly, telemetry is 15-min**: interpolate through the clear-sky index, documented in the data card.
7. **OPCOM download risk**: the data card tries OPCOM exports first, then ENTSO-E (free token) as fallback.
8. **PPC sample data**: PPC offers sample installation and GIS data. Contracts (Parquet schemas) are written so
   their data can be dropped in through an adapter, without changing model or agent code.
9. **lab.toml now**: `[runs.preview]` runs `npm start` but there is no package.json, so every preview fails until
   the skeleton card switches it to Streamlit.

## Not doing

- GPVS-Faults (high-frequency lab signals, not 15-min power).
- DDP / multi-GPU training, Optuna, W&B, any external ML API. LLM is optional and only rewrites text.
- Replacing inverters, EMS or the GIS: read-only ingestion, outputs are work orders, schedules and a GeoJSON layer.
- Raw personal data: customer IDs pseudonymised, public or synthetic data only.

## Fallbacks (cut in this order)

1. LLM sentence. 2. Routing (keep top-K ranking). 3. Grid-twin → one slide with offline results.
If the GPU sweep stalls, LightGBM serves the agent. Never cut: SentinelNet, baseline comparison, decision layer,
work-order action.

## Phases and gates

| Phase | Hours | Gate |
|---|---|---|
| Setup & contracts | 0–2 | plan.md; contract card merged; lab.toml + requirements split; cache probe passed on all 3 workers |
| Data + baselines | 2–7 | raw data committed; simulator + injector cached as data-v1; baseline numbers in `lab top` |
| Model v1 + agent on mock | 7–12 | SentinelNet vs LightGBM on val_nmae reported; agent makes a work order end-to-end from the mock |
| Sweeps + integration (night) | 12–17 | 2–3 `--grid` rounds; 4-seed ensemble; promoted ONNX in `models/`; preview runs the real model |
| Evaluation & robustness | 17–21 | PVDAQ results, CUSUM comparison, error analysis, drills, grid-twin decision |
| Demo freeze | 21–24 | `lab freeze on` at h22; `make reproduce` on a clean checkout; slides from `lab report --out`; 2 rehearsals |

## First-wave cards

Start here (hour 0–2, challenge-agnostic): **#6** repo skeleton (P3), **#7** contracts (P5, review), **#8** cache probe (P1).
Then: P1 #9 raw data, #10 windows/splits · P2 #11 simulator, #12 fault injector, #13 grid-twin ·
P3 #14 SentinelNet + train.py · P4 #15 baselines, #16 eval.py · P5 #17 agent loop, #18 decisions, #19 Streamlit.

## Lab rules for the 24 h

- Every card has `--paths`; agents never touch each other's files. Contract and "promote model" cards are
  `merge=review`; code cards `merge=auto`; training runs are `never` (experiments).
- Contract change = `lab announce` + a line in `team/decisions.md`.
- After every training run: `lab note` + one line in `team/insights.md` (becomes the error-analysis slide).
- Runs: ≤ 20–30 min each; memory-mapped float16 windows; mixed precision; `--compile` only on server-gpu;
  cheapest grid configs go to the 4 GB laptop. Every run prints `LAB_METRIC val_nmae=… val_pinball=…
  val_prauc=… hours_to_detect=…` and writes ONNX + calibration JSON to `$LAB_OUTPUT_DIR`.
- Keep a phone hotspot ready: every submit and worker fetch goes through GitHub.

## Areas and owners

| Area | Paths | Owner |
|---|---|---|
| P1 Data engineering | `sentinel/data/**`, `data/**` | tbd |
| P2 Physics & synthetic, grid-twin | `sentinel/sim/**`, `sentinel/gridtwin/**` | tbd |
| P3 ML lead / training infra | `sentinel/model/**`, `train.py`, `lab.toml`, `requirements*.txt` | Stefan |
| P4 Evaluation, uncertainty, explainability | `sentinel/eval/**`, `eval.py` | tbd |
| P5 Agent, decisions, demo | `sentinel/contracts/**`, `sentinel/agent/**`, `app/**`, `tests/test_contracts.py` | tbd |
| Plan / docs | `team/plan.md`, `team/decisions.md` | yolooo332 |
