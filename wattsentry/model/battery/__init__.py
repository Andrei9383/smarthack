"""Battery behaviour model for data-v2 prosumers (card #174, spec I step 5).

PPC does not see a prosumer's battery, only its meter. This package learns how each home's battery behaves and
forecasts the home's net import per 15 minutes for the next day, including that behaviour:

- ``features``   expected PV / load (THE M1/M2 seam ``expected_pv_load``), 28-day strategy features, per-step rows
- ``classifier`` StrategyClassifier: P(none / self_consumption / price_driven / enrolled) + confidence, and
                 battery-parameter estimates (cap_kwh, p_kw, reserve_frac, k_top)
- ``net_model``  (a) NetQuantileModel, LightGBM quantile 0.1/0.5/0.9 on net import (fallback);
                 (b) simulate_guess / mix: the guessed strategy through the K8 kernel; portfolio_behaviour for I5
- ``train``      python train.py --model battery (lab: train_cpu --needs gpu, grid over --seed)
- ``data``       data-v2 loading (I2 #172); hidden files only via ``labels`` (targets) and ``truth_chunk`` (scoring)

Predict API (I5 #175 / I6 #176):

    from wattsentry.model.battery import BatteryModel
    bm = BatteryModel.load("<run out>/battery")
    fc = bm.forecast(data_dir, home_ids, years=[2025])
    # fc["net_kwh"] (n, T) two-stage P50 net import kWh/15 min, fc["q_kwh"] (n, T, 3) quantile LightGBM,
    # fc["guess"] DataFrame per home x day (strategy, confidence, p_*, cap_kwh, p_kw, reserve_frac, k_top),
    # fc["ts"] (T,) UTC; fc["behaviour"] = callable for wattsentry.eval.portfolio.set_behaviour (one day).
"""

from __future__ import annotations

from .api import BatteryModel

__all__ = ["BatteryModel"]
