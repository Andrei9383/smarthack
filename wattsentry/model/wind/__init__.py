"""M6 WindNBM: wind normal-behaviour model + anomaly classifier vs WindMon rules.

Modules (torch only in nbm.py / train.py):
- physics.py: shear, air density, density-normalised wind, generic class power curves.
- data.py: WindData wide view, PPC long-format <-> wide, loader (simulator output, else synth), splits.
- synth.py: deterministic stand-in for the simulator (`python -m wattsentry.model.wind.synth --out DIR`).
- windmon_rules.py: re-export of WindMon's four rules (baseline). metrics.py: event-level metrics shared by both.
- features.py: NBM inputs, chunked inference, classifier features, decision smoothing.
- nbm.py + train.py: `python -m wattsentry.model.wind.train` / `train.py --model windnbm` → ONNX + calibration.
- predictor.py: onnxruntime-only `WindNBMPredictor` returning the contract Prediction.
"""
