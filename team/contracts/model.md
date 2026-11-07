# Model contract: prediction envelope and `predict()`

Code: `sentinel/contracts/envelope.py`, `sentinel/contracts/mock.py`. Implemented by #14 (M1/M2/M4), #15
(baselines), #41 (M6), #44 (M3) and `MockPredictor`.

## Interface
```python
class Predictor(Protocol):
    model: str            # ModelId value, e.g. "pvnet"
    model_version: str    # run id, e.g. "pvnet-r123"; mocks use "mock-1"
    def predict(self, asset: Mapping, window: Window) -> Prediction: ...
```
`asset` = a metadata row (`sites` / `turbines` / zone dict). `window` = `Window(t0, horizon=96, weather,
solar_geometry, reference, residual_summary, battery, calendar, history, park_context, meta)`.
PV window = 24 h weather + solar geometry (no target power) + 7–14 days of the site's own past power +
14-day residual summary + battery channels. Forecast mode uses noisy weather, never ERA5 actuals.

## Prediction envelope
| field | type | meaning |
|---|---|---|
| `asset_id` | str | site / turbine / zone / feeder id |
| `model`, `model_version` | str | registry id + run id |
| `t0` | UTC ISO str | start of step 0 |
| `horizon` | int | steps (96 = day ahead at 15 min) |
| `q10`, `q50`, `q90` | float[horizon] | in `unit` (kW; zones MW); monotone q10 ≤ q50 ≤ q90 |
| `ood_score` | float ≥ 0 | 0 = in-distribution |
| `member_std` | float ≥ 0 | ensemble spread of q50 |
| `fault_probs?` | {PvFault: p} | M4 only; sums to 1 |
| `anomaly_class?`, `anomaly_score?` | WindAnomaly, 0..1 | M6 only |
| `unit`, `step_minutes` | str, int | defaults `kW`, 15 |
| `extra` | dict | model-specific; consumers never require it |

Detectors (M4, M6) also fill q10/q50/q90 with the expected power they measure the fault against.
Helpers: `p_fault` (= 1 − P(healthy)), `top_fault`, `interval_width()`. `Prediction.validate()` raises on bad shapes.

## Registry ids (`ModelId`)
`pvnet` M1 · `loadnet` M2 · `demandnet` M3 · `sitehealth` M4 · `windpower` M5 · `windnbm` M6 · `windfailure` M7 ·
`gridtwin` M8 · baselines `pvlib`, `persistence`, `lgbm`, `threshold_cusum`, `windmon_rules`, `binned_curve` ·
`mock`.

## Classes
- `PvFault` (M4, #12): `healthy, soiling, shading, string_outage, clipping, overvoltage_trip, battery_fade,
  sensor_fault`.
- `WindAnomaly` (M6, #23): `normal, fault_stop, curtailment, icing, underperformance, overheating,
  anemometer_fault, data_gap`.

## Artifacts
`train.py` writes ONNX weights + a calibration JSON (`{model, model_version, quantile offsets per horizon bucket,
thresholds: {ood, width, member_std, p_fault}}`) as a release asset; #29 promotes the chosen ones into `models/`.

**Judge priors (#147, judge-design §6.2).** A calibration JSON may carry an optional `priors` key: what that model
version scored on its 2024 validation year, read by the Judge's trust ledger when there is no live record
("no record → trust the model"). Shape: `{"version": 1, "split": "val-2024", "source_run", "source":
"validation|report|default", "scores": {score_name: {bucket_key: {"prior": 0..1, "n": asset-days}}}}`. Score names
follow §5.3 (`calibration`, `sharpness`, `alarm_precision`, `miss_score`, `hit`, `flag`); bucket keys are
`sentinel/judge/buckets.py` (`kind|seen|battery|season|quality|ood`, every ancestor present, root `"*"`). The ledger
uses the most specific covered key and falls back towards `"*"`. Ensemble members (`pvnet.s*`, `loadnet.s*`) share
the ensemble's priors. Models without a calibration JSON (M6 `windnbm.priors.json`) and the substitutes / ledger-only
models (M3, M7, M8) keep the same block in `models/judge_priors.json` under `models.<id>`. Written by
`python -m sentinel.judge.priors --out DIR` (lab `eval` run); consumers must tolerate the key being absent.

## Mock
`MockPredictor(kind)` with kind `pv | load | sitehealth | wind | zone`; same `(asset, t0)` → same output on every
machine. PV follows RO solar geometry and a per-day cloud factor; load has morning + evening peaks (heat in winter
for heating archetypes, AC in summer); ~15% of sites get a fault; ~12% of turbine-days an anomaly; zones are MW.
`model` is the real id it stands in for, `model_version = "mock-1"`, so consumers route the same way.
```python
from sentinel.contracts import MockPredictor
p = MockPredictor("pv").predict({"site_id": "S0001", "kwp_true": 6, "lat": 44.2}, t0="2025-06-21T00:00:00Z")
```
