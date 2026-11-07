# models/ — promoted weights (card #29)

**Status: fleet-trained (stage 2).** M1/M2/M4 are #14's runs **train#383-392** (card #14, worker rtx5070, commit
`54483ca`, `--data lab` = the shared #11 fleet + #12 faults through the #10 windows only; train 2022-23, val 2024,
test 2025; all ten passed with `onnx_parity_ok = 1`). They replace the interim synthetic-trained runs 201-210 of
stage 1 (git history). The demo week (June 2025) lies in the test year. The agreed model comparison for slides is
`eval.py` (#16); the numbers in the last column are #14's train-time **test-split** metrics from `reports/` (same
fleet, same 2025 test year, train.py's own baselines), labelled as such until an eval.py run on these files exists.

All files are ONNX + calibration JSON (onnxruntime only, no torch); `.pt` checkpoints are not committed. ~18 MB.
Loaded by `sentinel.model.predictor.load_predictor(model, "models/")` (eval.py) and by the replay
(`sentinel/agent/predictors.py`; `$SENTINEL_MODELS` overrides the directory).

| file | model | lab run | worker | commit | data label | test split 2025 (train.py, synthetic RO fleet) |
|---|---|---|---|---|---|---|
| `pvnet.s0.onnx` + `.calibration.json` | M1 PVNet nowcast, seed 0 | #14 train#383 | rtx5070 | 54483ca | synthetic RO (#10 windows) | nMAE 0.024 vs LightGBM 0.045 vs pvlib 0.073; 10-90 coverage 0.78 |
| `pvnet.s1.*` | M1 seed 1 | #14 train#384 | rtx5070 | 54483ca | synthetic RO (#10 windows) | nMAE 0.023 vs LightGBM 0.046; coverage 0.78 |
| `pvnet.s2.*` | M1 seed 2 | #14 train#385 | rtx5070 | 54483ca | synthetic RO (#10 windows) | nMAE 0.022 vs LightGBM 0.044; coverage 0.79 |
| `pvnet.s3.*` | M1 seed 3 | #14 train#386 | rtx5070 | 54483ca | synthetic RO (#10 windows) | nMAE 0.023 vs LightGBM 0.045; coverage 0.76 |
| `loadnet.s0.*` | M2 LoadNet, seed 0 | #14 train#387 | rtx5070 | 54483ca | synthetic RO (#10 windows) | nMAE 0.221 vs LightGBM 0.259 vs persistence 0.360; coverage 0.81 |
| `loadnet.s1.*` | M2 seed 1 | #14 train#388 | rtx5070 | 54483ca | synthetic RO (#10 windows) | nMAE 0.220 vs LightGBM 0.260; coverage 0.82 |
| `loadnet.s2.*` | M2 seed 2 | #14 train#389 | rtx5070 | 54483ca | synthetic RO (#10 windows) | nMAE 0.220 vs LightGBM 0.260; coverage 0.81 |
| `loadnet.s3.*` | M2 seed 3 | #14 train#390 | rtx5070 | 54483ca | synthetic RO (#10 windows) | nMAE 0.221 vs LightGBM 0.260; coverage 0.81 |
| `sitehealth.onnx` + `.calibration.json` | M4 SiteHealth (preset `sh`) | #14 train#391 | rtx5070 | 54483ca | synthetic RO (#10 windows + #12 faults) | PR-AUC 0.706 (seen 0.73 / unseen 0.67) |
| `pvnet_fc/pvnet.onnx` + `.calibration.json` | M1 PVNet **forecast** mode (day-ahead, noisy weather) | #14 train#392 | rtx5070 | 54483ca | synthetic RO (#10 windows) | nMAE 0.106 vs LightGBM 0.119 vs pvlib 0.169; coverage 0.81 |
| `windnbm.onnx`, `windnbm_clf.txt`, `windnbm_thresholds.json` | M6 WindNBM (GRU NBM + LightGBM classes) | #41 train#135 | rtx5070 | 8268c35b | synthetic RO (synth-v1 wind stand-in, pre-#22) | held-out park WP2 (#41 note): precision 0.854 vs WindMon 0.530, recall 0.819 vs 0.440, alerts/turbine-day 0.101 vs 0.114 |
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

Source of the files: `ssh rtx5070:/var/lib/lab-gpu/work/outputs/<run>/` (read-only copy).
