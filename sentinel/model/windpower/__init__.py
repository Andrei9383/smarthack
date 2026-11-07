"""M5 WindPower (card #164): per-turbine P10/P50/P90 of power, 15-min steps, and park totals.

Predict API: `WindPowerModel` (sentinel/model/windpower/model.py). Training: `python train.py --model windpower`
(sentinel/model/windpower/train.py). Fallback: binned empirical curve per turbine class (binned.py)."""

from .model import WindPowerModel, park_aggregate

__all__ = ["WindPowerModel", "park_aggregate"]
