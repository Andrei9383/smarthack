# models/ — promoted weights (card #29)

**Status: fleet-trained (stage 2).** M1/M2/M4 are #14's runs **train#383-392** (card #14, worker rtx5070, commit
`54483ca`, `--data lab` = the shared #11 fleet + #12 faults through the #10 windows only; train 2022-23, val 2024,
test 2025; all ten passed with `onnx_parity_ok = 1`). They replace the interim synthetic-trained runs 201-210 of
stage 1 (git history). The demo week (June 2025) lies in the test year. The agreed model comparison for slides is
`eval.py` (#16); the numbers in the last column are #14's train-time **test-split** metrics from `reports/` (same
fleet, same 2025 test year, train.py's own baselines), labelled as such until an eval.py run on these files exists.
Campaign #139 (#202) replaced M2 LoadNet, M1 forecast mode and M6 WindNBM and added M7 WindFailure; see the promotion
record at the end. M1 nowcast and M4 are still #14's runs.

All files are ONNX + calibration JSON (onnxruntime only, no torch); `.pt` checkpoints are not committed. ~18 MB.
Loaded by `wattsentry.model.predictor.load_predictor(model, "models/")` (eval.py) and by the replay
(`wattsentry/agent/predictors.py`; `$WATTSENTRY_MODELS` overrides the directory).

| file | model | lab run | worker | commit | data label | test split 2025 (train.py, synthetic RO fleet) |
|---|---|---|---|---|---|---|
| `pvnet.s0.onnx` + `.calibration.json` | M1 PVNet nowcast, seed 0 | #14 train#383 | rtx5070 | 54483ca | synthetic RO (#10 windows) | nMAE 0.024 vs LightGBM 0.045 vs pvlib 0.073; 10-90 coverage 0.78 |
| `pvnet.s1.*` | M1 seed 1 | #14 train#384 | rtx5070 | 54483ca | synthetic RO (#10 windows) | nMAE 0.023 vs LightGBM 0.046; coverage 0.78 |
| `pvnet.s2.*` | M1 seed 2 | #14 train#385 | rtx5070 | 54483ca | synthetic RO (#10 windows) | nMAE 0.022 vs LightGBM 0.044; coverage 0.79 |
| `pvnet.s3.*` | M1 seed 3 | #14 train#386 | rtx5070 | 54483ca | synthetic RO (#10 windows) | nMAE 0.023 vs LightGBM 0.045; coverage 0.76 |
| `loadnet.s0.*` | M2 LoadNet, seed 0 | #141 train#680 | rtx5070 | 72e1337 | synthetic RO (#10 windows) | nMAE 0.2205 vs LightGBM 0.259 vs persistence 0.360; coverage 0.81 |
| `loadnet.s1.*` | M2 seed 1 | #141 train#681 | server-gpu | 72e1337 | synthetic RO (#10 windows) | nMAE 0.2199 vs LightGBM 0.260; coverage 0.81 |
| `loadnet.s2.*` | M2 seed 2 | #141 train#682 | desktop | 72e1337 | synthetic RO (#10 windows) | nMAE 0.2193 vs LightGBM 0.260; coverage 0.82 |
| `loadnet.s3.*` | M2 seed 3 | #141 train#683 | rtx5070 | 72e1337 | synthetic RO (#10 windows) | nMAE 0.2201 vs LightGBM 0.260; coverage 0.81 |
| `sitehealth.onnx` + `.calibration.json` | M4 SiteHealth (preset `sh`) | #14 train#391 | rtx5070 | 54483ca | synthetic RO (#10 windows + #12 faults) | PR-AUC 0.706 (seen 0.73 / unseen 0.67) |
| `pvnet_fc/pvnet.onnx` + `.calibration.json` | M1 PVNet **forecast** mode (day-ahead, noisy weather) | #141 train#684 | laptop | 72e1337 | synthetic RO (#10 windows) | nMAE 0.1052 vs LightGBM 0.119 vs pvlib 0.169; coverage 0.80 |
| `windnbm.onnx`, `windnbm_clf.txt`, `windnbm_thresholds.json`, `windnbm.priors.json` | M6 WindNBM (GRU NBM + LightGBM classes; label-free normal selection, seed 0) | #143 train#674 | server-gpu | 72e1337 | synthetic RO parks (#22, ERA5-driven; the replay's wind data) | held-out park WP2, test 2025: precision 0.913 vs WindMon 0.835, recall 0.898 vs 0.747, alerts/turbine-day 0.074 vs 0.077, curtail-vs-fault F1 0.936 vs 0.388 |
| `windfailure/` (`windfailure_7d.txt`, `_14d.txt`, `_thresholds.json`, `_nbm.onnx`) | M7 WindFailure (LightGBM risk 7/14 d on NBM temperature residuals; `shipped: true`) | #144 train#778 | laptop | 3b624f5 | synthetic RO parks (#22) | test 2025: lead time 124.8 h vs fixed threshold 46.3 h, event recall 0.982 vs 0.873, false alarms 0.42 vs 0.37 /turbine-year; held-out park WP2 164.8 h vs 64.9 h |
| `judge.calibration.json`, `judge.forecast.s0-4.txt`, `judge.detector.s0-4.txt` | J3 Judge (LightGBM, 5 seeds; trained on 2024 replay weeks) | #148 train#908 | rtx5070 | — | 2024 replay weeks (drift + faults) | 2025: tau_use precision below the 0.90 target; drifted rows 0.66 vs ledger-only 0.81 (#148 note) |
| `thresholds.json` | qualify thresholds (#26 / #27 defaults) | — | — | — | — | — |
| `reports/r<run>-*.json` | train-time manifest / report / metrics of each run above | | | | | |

Seen vs unseen sites (PVNet s0): nMAE 0.020 vs 0.035, calibrated coverage 0.84 vs 0.63 — unseen sites get too
narrow PV bands. Train run 391 also trained its own m1/m2 helper models for SiteHealth's inputs; they are not
promoted: the replay feeds SiteHealth from the `pvnet.s*` / `loadnet.s*` ensembles above (as in stage 1).
Earlier, the interim recipe on #14's own synthetic data scored worse on this fleet in eval.py (#67 run #228: PVNet
nMAE 0.080 vs LightGBM 0.041) — the reason for the fleet retrain.

Members: every `pvnet.*.onnx` / `loadnet.*.onnx` in this directory is one ensemble member (q = member mean,
`member_std` = spread of q50). `pvnet_fc/` is kept in a subdirectory so it does not join the nowcast ensemble.
SiteHealth takes its expected-power input (m1) from the `pvnet.s*` ensemble and m2 from `loadnet.s*`.

Source of the files: the workers' `/var/lib/lab-gpu/work/outputs/<run>/` (not reachable from the PCs). #202 fetched them through the run log: `lab submit <card> --run train --cmd 'bash scripts/lab_fetch_outputs.sh {job}' --grid job=...`, then `python scripts/lab_decode_dump.py <saved lab logs> <dir>` (sha256 checked).


## Promotion record: campaign #139, T7b (#202)

Files fetched from the workers by train#1135-1138 (`scripts/lab_fetch_outputs.sh`; every file sha256-checked; the
recovered metrics.json equal the original runs' LAB_METRICs). Gate: the 2025 test-split headline metric must beat the
file it replaces (train-time numbers from the cards' notes; the eval.py numbers in team/results.md #4-#6 are eval#521 on
the older files).

| model | before | after | headline (test 2025) | decision |
|---|---|---|---|---|
| M2 LoadNet (4 seeds) | #14 train#387-390 | #141 train#680-683 | nMAE mean 0.2203 -> 0.2199 (per seed 0.2206/0.2195/0.2204/0.2206 -> 0.2205/0.2199/0.2193/0.2201); calibrated pinball 0.0741 -> 0.0740; coverage 0.81 -> 0.81 | promoted (better, within seed noise) |
| M1 PVNet forecast | #14 train#392 | #141 train#684 | nMAE 0.1064 -> 0.1052; calibrated pinball 0.0340 -> 0.0336; coverage 0.81 -> 0.80 | promoted |
| M6 WindNBM | #41 train#135 (stand-in wind, pre-#22) | #143 train#674 (#22 parks, seed 0) | held-out WP2: P/R 0.913/0.898, F1 curtail-vs-fault 0.936 (WindMon 0.835/0.747/0.388). #135's 0.854/0.819 are on the stand-in data (#99: not comparable); #674 is the first M6 trained on the parks the replay uses | promoted |
| M7 WindFailure | none | #144 train#778 (`--residuals nbm`) | lead time 124.8 h vs threshold 46.3 h, recall 0.982 vs 0.873, wf_ship=1 | promoted (first M7); not wired into the replay yet (`WindFailurePredictor("models/windfailure")`) |

Not changed: M1 nowcast #383-386 and M4 SiteHealth #391 (#145: not better). M3 #663 and M8 #664 reproduce the current
numbers exactly.

Judge priors: the LoadNet calibration JSONs keep the eval#699 `priors` block of the members they replace (labelled
"carried over", same recipe). `windnbm.priors.json` is rebuilt from `reports/r674-windnbm.report.json` with
`wattsentry.judge.priors.wind_priors` (alarm precision 0.873, was 0.763 from r135). `reports/r135-windnbm.*` stay because
`wattsentry/judge/priors.py` still reads r135 by name; rerunning `python -m wattsentry.judge.priors` would bring back the
r135 wind priors until that path is updated.

## T7 promotion (#145)

- **Promoted:** J3 Judge #148 train#908 (first trained Judge; installed with `python -m wattsentry.judge.train --apply`
  from the run log). The slot now passes `witness` as `model` to `Judge.predict` (it raised KeyError once weights existed).
- **Kept (not better):** M4 SiteHealth #391 (#142 train#746 PR-AUC 0.696 < 0.706); M1 PVNet nowcast #383-386
  (#141 #676-679 within noise). The other campaign models were promoted by #202 (above).
