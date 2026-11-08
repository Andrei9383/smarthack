"""M5 WindPower: per-turbine P10/P50/P90 of power on 15-min steps, and park totals.

Training: `python train.py --model windpower` (train.py); predict API: `WindPowerModel` (model.py); baseline: binned
empirical curve per turbine class (binned.py)."""

from .model import WindPowerModel, park_aggregate

__all__ = ["WindPowerModel", "park_aggregate"]
