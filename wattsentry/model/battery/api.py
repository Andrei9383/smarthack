"""Loads a trained battery model (strategy classifier + optional net quantile model + sim mode) from its run dir."""

from __future__ import annotations

import json
from pathlib import Path


from .classifier import StrategyClassifier
from .net_model import NetQuantileModel


class BatteryModel:
    def __init__(self, clf: StrategyClassifier, netq: NetQuantileModel | None, mode: str = "sim_soft"):
        self.clf, self.netq, self.mode = clf, netq, mode

    @classmethod
    def load(cls, model_dir: str | Path) -> "BatteryModel":
        d = Path(model_dir)
        netq = NetQuantileModel.load(d) if (d / "battery_net.json").exists() else None
        mode = "sim_soft"
        rep = d / "battery_report.json"
        if rep.exists():
            mode = json.loads(rep.read_text()).get("sim_mode", mode)
        return cls(StrategyClassifier.load(d), netq, mode)
