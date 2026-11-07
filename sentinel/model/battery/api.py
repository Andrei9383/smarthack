"""Predict API of the battery model (card #174) for the portfolio (I5 #175) and the replay (I6 #176)."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Sequence

import numpy as np
import pandas as pd

from . import data as D
from .classifier import PARAMS, STRATEGIES, StrategyClassifier
from .net_model import NetQuantileModel, guessed_battery_arrays, portfolio_behaviour


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

    def forecast(self, data: str | Path | D.World, home_ids: Sequence[str], years: Sequence[int],
                 chunk: int = 250) -> dict:
        """Day-ahead forecasts (each day uses meter data up to D-2 only) for ``home_ids`` over local ``years``.
        Returns ts (T,), home_ids, net_kwh (n, T) two-stage P50, q_kwh (n, T, 3) quantile LightGBM (if trained),
        fo_kwh (n, T) forecast_only, guess (DataFrame home x day), and the arrays behaviour_for(day) needs."""
        from .train import Chunk, forecast_chunk

        world = data if isinstance(data, D.World) else D.load_world(data)
        homes = world.homes.set_index("home_id").loc[list(home_ids)].reset_index()
        parts = []
        for hc in D.chunks(homes["home_id"], chunk):
            ch = Chunk(world, homes[homes.home_id.isin(hc)].reset_index(drop=True))
            parts.append((ch.homes["home_id"].tolist(), forecast_chunk(world, ch, years, self.clf, self.netq)))
        f0 = parts[0][1]
        cat = lambda k: np.concatenate([p[k] for _, p in parts], axis=0)  # noqa: E731
        ids = [h for hs, _ in parts for h in hs]
        days = f0["days"]
        day_ts = world.ts[world.day_start[days]].tz_convert(D.TZ).strftime("%Y-%m-%d")
        probs = cat("probs")
        g = {"home_id": np.repeat(ids, len(days)), "day": np.tile(np.asarray(day_ts), len(ids)),
             "strategy": np.asarray(STRATEGIES)[cat("code").reshape(-1)], "confidence": cat("conf").reshape(-1)}
        for i, s in enumerate(STRATEGIES):
            g[f"p_{s}"] = probs[..., i].reshape(-1)
        for k in PARAMS:
            g[k] = np.concatenate([p["params"][k] for _, p in parts], axis=0).reshape(-1)
        out = {"ts": world.ts[f0["steps"]], "home_ids": ids, "net_kwh": cat(self.mode), "fo_kwh": cat("fo"),
               "guess": pd.DataFrame(g), "world": world, "steps": f0["steps"]}
        if "lgbm_q" in f0:
            out["q_kwh"] = cat("lgbm_q")
        return out

    @staticmethod
    def behaviour_for(fc: dict, day: str):
        """``behaviour(pv, load, battery, soc0)`` for one local day 'YYYY-MM-DD' (I5's set_behaviour), using each
        home's most likely strategy and parameters guessed for that day."""
        world, steps = fc["world"], fc["steps"]
        g = fc["guess"][fc["guess"]["day"] == day].set_index("home_id").loc[fc["home_ids"]]
        code = g["strategy"].map({s: i for i, s in enumerate(STRATEGIES)}).to_numpy(np.int8)
        loc = world.ts[steps].tz_convert(D.TZ).strftime("%Y-%m-%d")
        st = steps[np.asarray(loc) == day]
        return portfolio_behaviour(code, {k: g[k].to_numpy() for k in PARAMS}, world.price[st], world.rank[st],
                                   world.med[st], world.cal["day_index"][st])

    @classmethod
    def portfolio_inputs(cls, fc: dict, day: str, region_key: str = "county"):
        """(Fleet, behaviour) for I5 on data-v2 for one local day: the Fleet's battery arrays are the GUESSED ones
        (PPC does not know the real battery), and behaviour = behaviour_for(fc, day). Usage:
            fleet, fn = BatteryModel.portfolio_inputs(fc, "2025-06-02"); portfolio.set_behaviour(fn, "battery-i4")
            portfolio.portfolio_scenarios(pv_q, load_q, fleet, ...)"""
        from sentinel.eval.portfolio import BatteryArrays, Fleet

        world = fc["world"]
        g = fc["guess"][fc["guess"]["day"] == day].set_index("home_id").loc[fc["home_ids"]]
        code = g["strategy"].map({s: i for i, s in enumerate(STRATEGIES)}).to_numpy(np.int8)
        bat = BatteryArrays(*guessed_battery_arrays(code, {k: g[k].to_numpy() for k in PARAMS}))
        h = world.homes.set_index("home_id").loc[fc["home_ids"]]
        names = h[region_key].astype(str).to_numpy()
        regions = sorted(set(names))
        rc = {r: i for i, r in enumerate(regions)}
        fleet = Fleet(np.asarray(fc["home_ids"], dtype=object), np.array([rc[r] for r in names], np.int64), regions,
                      bat, h["inverter_kw"].to_numpy(float))
        return fleet, cls.behaviour_for(fc, day)
