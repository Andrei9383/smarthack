# models/ — promoted weights (card #29)

**Status: INTERIM.** M1/M2/M4 here were trained on #14's *own* synthetic RO generator (`sentinel/model/synth.py`),
not on the shared #11 fleet + #12 faults (#10 windows). They are what the stage-1 demo replay
(`demo/sentinel.sqlite`) runs on. They are replaced by #14's fleet-retrained runs once those pass (`lab runs 14`).
The only agreed model comparison for slides is `eval.py` (#16) on the shared fleet test split (team/insights.md);
the train-time metrics inside `reports/` are on #14's synthetic split and are **not** results to cite.

All files are ONNX + calibration JSON (onnxruntime only, no torch); `.pt` checkpoints are not committed. ~18 MB.
Loaded by `sentinel.model.predictor.load_predictor(model, "models/")` (eval.py) and by the replay
(`sentinel/agent/predictors.py`; `$SENTINEL_MODELS` overrides the directory).

| file | model | lab run | worker | commit | data label | headline (fleet, eval.py) |
|---|---|---|---|---|---|---|
| `pvnet.s0.onnx` + `.calibration.json` | M1 PVNet nowcast, seed 0 | #29 train#201 | rtx5070 | 22324281 | synthetic RO (#14 synth) | not scored on the fleet yet¹ |
| `pvnet.s1.*` | M1 seed 1 | #29 train#202 | rtx5070 | 22324281 | synthetic RO (#14 synth) | ¹ |
| `pvnet.s2.*` | M1 seed 2 | #29 train#203 | rtx5070 | 22324281 | synthetic RO (#14 synth) | ¹ |
| `pvnet.s3.*` | M1 seed 3 | #29 train#204 | rtx5070 | 22324281 | synthetic RO (#14 synth) | ¹ |
| `loadnet.s0.*` | M2 LoadNet, seed 0 | #29 train#205 | rtx5070 | 22324281 | synthetic RO (#14 synth) | ¹ |
| `loadnet.s1.*` | M2 seed 1 | #29 train#206 | rtx5070 | 22324281 | synthetic RO (#14 synth) | ¹ |
| `loadnet.s2.*` | M2 seed 2 | #29 train#207 | rtx5070 | 22324281 | synthetic RO (#14 synth) | ¹ |
| `loadnet.s3.*` | M2 seed 3 | #29 train#208 | rtx5070 | 22324281 | synthetic RO (#14 synth) | ¹ |
| `sitehealth.onnx` + `.calibration.json` | M4 SiteHealth (preset `sh`) | #29 train#209 | rtx5070 | 22324281 | synthetic RO (#14 synth) | ¹ |
| `pvnet_fc/pvnet.onnx` + `.calibration.json` | M1 PVNet **forecast** mode (day-ahead, noisy weather) | #29 train#210 | rtx5070 | 22324281 | synthetic RO (#14 synth) | ¹ |
| `windnbm.onnx`, `windnbm_clf.txt`, `windnbm_thresholds.json` | M6 WindNBM (GRU NBM + LightGBM classes) | #41 train#135 | rtx5070 | 8268c35b | synthetic RO (synth-v1 wind stand-in, pre-#22) | held-out park WP2 (#41 note): precision 0.854 vs WindMon 0.530, recall 0.819 vs 0.440, alerts/turbine-day 0.101 vs 0.114 |
| `thresholds.json` | qualify thresholds (#26 / #27 defaults) | — | — | — | — | — |
| `reports/r<run>-*.json` | train-time manifest / report / metrics of each run above | | | | | train split only, not results |

¹ The same recipe (presets `pv_now` / `load` / `sh`, #14 synthetic data, shorter budgets) was scored by `eval.py` on
the shared fleet test split in **#67 run #228**: PVNet nMAE 0.080 vs LightGBM 0.041 vs pvlib 0.083; LoadNet nMAE
0.319 vs LightGBM 0.256, 10-90 coverage 0.44; SiteHealth PR-AUC 0.53 vs threshold+CUSUM 0.50, 97 % vs 37 % of
episodes found, median detection 43 h vs 36 h; PVDAQ (non-RO): PVNet failed (240 predict failures). Those are other
weights of the same recipe, so they bound, not measure, the files above.

Members: every `pvnet.*.onnx` / `loadnet.*.onnx` in this directory is one ensemble member (q = member mean,
`member_std` = spread of q50). `pvnet_fc/` is kept in a subdirectory so it does not join the nowcast ensemble.
SiteHealth takes its expected-power input (m1) from the `pvnet.s*` ensemble and m2 from `loadnet.s*`.

Source of the files: `ssh rtx5070:/var/lib/lab-gpu/work/outputs/<run>/` (read-only copy).
