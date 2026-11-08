"""M6 WindNBM (card #41): wind normal-behaviour model + anomaly classifier vs WindMon rules.

Modules (torch only in nbm.py / train.py):
- physics.py: shear, air density, density-normalised wind (plan §3.5), generic class power curves.
- data.py: WindData wide view, PPC long-format <-> wide, loader (#22/#23 output, else synth), splits.
- synth.py: deterministic stand-in for #22 + #23 in the PPC schema (`python -m wattsentry.model.wind.synth --out DIR`).
- windmon_rules.py: WindMon's four rules (baseline). metrics.py: event-level metrics shared by both.
- features.py: NBM inputs, chunked inference, classifier features, decision smoothing.
- nbm.py + train.py: `python -m wattsentry.model.wind.train` / `train.py --model windnbm` → ONNX + calibration.
- predictor.py: onnxruntime-only `WindNBMPredictor` returning the contract Prediction.
"""
