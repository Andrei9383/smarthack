"""Battery behaviour model for data-v2 prosumers: PPC sees only the meter, so this package learns how each home's
battery behaves and forecasts the home's net import per 15 minutes for the next day.

- ``features``   expected PV / load, 28-day strategy features, per-step rows
- ``classifier`` StrategyClassifier: P(none / self_consumption / price_driven / enrolled) + confidence, and
                 battery-parameter estimates (cap_kwh, p_kw, reserve_frac, k_top)
- ``net_model``  NetQuantileModel (LightGBM quantiles on net import) and simulate / mix: the guessed strategy
                 through the battery-strategies kernel
- ``train``      python train.py --model battery
- ``data``       data-v2 loading; hidden files only via ``labels`` (targets) and ``truth_chunk`` (scoring)
- ``api``        BatteryModel.load("<run out>/battery") for the imbalance evaluation
"""

from __future__ import annotations

from .api import BatteryModel

__all__ = ["BatteryModel"]
