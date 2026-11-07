"""Stage 1 of spec I step 5b: which battery strategy a home runs, from 28 days of its meter data (card #174).

LightGBM multiclass over ``features.CLS_FEATURES`` -> P(none / self_consumption / price_driven / enrolled) + the
confidence (max probability), and LightGBM regressors for the battery parameters the simulation needs (capacity,
power, reserve, k_top; trained on battery homes only). Labels and parameters come from the hidden homes table and
are TARGETS only. In a real pilot they come from what PPC knows (MaxControl contracts, declared batteries) and from
enrolled customers; here: synthetic RO.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Mapping

import numpy as np

from .features import CLS_FEATURES

STRATEGIES = ("none", "self_consumption", "price_driven", "enrolled")   # = sentinel.kernels.strategies codes 0..3
PARAMS = ("cap_kwh", "p_kw", "reserve_frac", "k_top")
# Simulation fallbacks = the middle of I2's synthetic ranges (config.py): cap 5-15, P 3-5, reserve 10-40 %, k 8-20.
PRIOR = {"cap_kwh": 10.0, "p_kw": 4.0, "reserve_frac": 0.25, "k_top": 14.0, "eta": 0.95}
BOUNDS = {"cap_kwh": (2.0, 20.0), "p_kw": (1.0, 8.0), "reserve_frac": (0.05, 0.5), "k_top": (4.0, 32.0)}


def _lgb():
    import lightgbm as lgb

    return lgb


class StrategyClassifier:
    def __init__(self, cls_booster=None, param_boosters: Mapping[str, object] | None = None,
                 meta: dict | None = None):
        self.cls = cls_booster
        self.params = dict(param_boosters or {})
        self.meta = meta or {}

    # ------------------------------------------------------------------------------------------------ training
    @classmethod
    def fit(cls, X: np.ndarray, y: np.ndarray, P: Mapping[str, np.ndarray] | None = None, *,
            X_val: np.ndarray | None = None, y_val: np.ndarray | None = None, P_val: Mapping | None = None,
            rounds: int = 400, lr: float = 0.05, leaves: int = 31, min_leaf: int = 50, seed: int = 0,
            class_weight: str = "sqrt", threads: int = 0, log=print) -> "StrategyClassifier":
        """X (N, F) rows of CLS_FEATURES, y (N,) codes 0..3. P: name -> (N,) parameter targets (NaN = no battery)."""
        lgb = _lgb()
        base = dict(learning_rate=lr, num_leaves=leaves, min_data_in_leaf=min_leaf, feature_fraction=0.9,
                    bagging_fraction=0.8, bagging_freq=1, lambda_l2=1.0, seed=seed, verbose=-1,
                    num_threads=threads)
        counts = np.bincount(y, minlength=len(STRATEGIES)).astype(float)
        w_cls = np.where(counts > 0, counts.sum() / np.maximum(counts, 1) / len(STRATEGIES), 0.0)
        w_cls = {"balanced": w_cls, "sqrt": np.sqrt(w_cls), "none": np.ones_like(w_cls)}[class_weight]
        dtr = lgb.Dataset(X, y, weight=w_cls[y], feature_name=list(CLS_FEATURES), free_raw_data=False)
        valid, cb = [], []
        if X_val is not None and len(X_val):
            valid = [lgb.Dataset(X_val, y_val, weight=w_cls[y_val], reference=dtr)]
            cb = [lgb.early_stopping(40, verbose=False)]
        booster = lgb.train({**base, "objective": "multiclass", "num_class": len(STRATEGIES),
                             "metric": "multi_logloss"}, dtr, rounds, valid_sets=valid, callbacks=cb)
        log(f"[battery] strategy classifier: {booster.best_iteration or booster.current_iteration()} rounds, "
            f"class counts {dict(zip(STRATEGIES, counts.astype(int)))}")
        params = {}
        if P is not None:
            for name in PARAMS:
                t = np.asarray(P[name], float)
                m = np.isfinite(t) & (y > 0)
                if m.sum() < 20:
                    continue
                d = lgb.Dataset(X[m], t[m], feature_name=list(CLS_FEATURES))
                vs, cbs = [], []
                if X_val is not None and P_val is not None:
                    tv = np.asarray(P_val[name], float)
                    mv = np.isfinite(tv) & (y_val > 0)
                    if mv.sum() >= 20:
                        vs = [lgb.Dataset(X_val[mv], tv[mv], reference=d)]
                        cbs = [lgb.early_stopping(40, verbose=False)]
                params[name] = lgb.train({**base, "objective": "regression_l1", "metric": "l1"}, d,
                                         max(rounds // 2, 50), valid_sets=vs, callbacks=cbs)
        meta = {"features": list(CLS_FEATURES), "strategies": list(STRATEGIES), "class_weight": class_weight,
                "class_counts": counts.astype(int).tolist(), "seed": seed}
        return cls(booster, params, meta)

    # ------------------------------------------------------------------------------------------------ inference
    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        return np.asarray(self.cls.predict(X, num_threads=0), np.float64).reshape(len(X), len(STRATEGIES))

    def predict_params(self, X: np.ndarray) -> dict[str, np.ndarray]:
        out = {}
        for name in PARAMS:
            lo, hi = BOUNDS[name]
            b = self.params.get(name)
            v = np.full(len(X), PRIOR[name]) if b is None else np.asarray(b.predict(X), float)
            out[name] = np.clip(v, lo, hi)
        out["k_top"] = np.rint(out["k_top"]).astype(np.int32)
        return out

    def guess(self, X: np.ndarray) -> dict[str, np.ndarray]:
        """-> strategy (str), code (int8), confidence (max prob), probs (N, 4), and the parameter estimates."""
        pr = self.predict_proba(X)
        code = pr.argmax(1).astype(np.int8)
        return {"code": code, "strategy": np.asarray(STRATEGIES)[code], "confidence": pr.max(1), "probs": pr,
                **self.predict_params(X)}

    def importance(self) -> dict[str, float]:
        g = self.cls.feature_importance("gain")
        g = g / max(g.sum(), 1e-9)
        return dict(sorted(zip(CLS_FEATURES, map(float, g)), key=lambda kv: -kv[1]))

    # ------------------------------------------------------------------------------------------------ io
    def save(self, d: str | Path) -> None:
        d = Path(d)
        d.mkdir(parents=True, exist_ok=True)
        self.cls.save_model(str(d / "battery_strategy.txt"))
        for k, b in self.params.items():
            b.save_model(str(d / f"battery_param_{k}.txt"))
        (d / "battery_strategy.json").write_text(json.dumps({**self.meta, "params": sorted(self.params)}, indent=1))

    @classmethod
    def load(cls, d: str | Path) -> "StrategyClassifier":
        lgb = _lgb()
        d = Path(d)
        meta = json.loads((d / "battery_strategy.json").read_text())
        params = {k: lgb.Booster(model_file=str(d / f"battery_param_{k}.txt")) for k in meta.get("params", [])}
        return cls(lgb.Booster(model_file=str(d / "battery_strategy.txt")), params, meta)


def macro_f1(y_true: np.ndarray, y_pred: np.ndarray, labels=range(len(STRATEGIES))) -> tuple[float, dict]:
    """Macro F1 over the classes present in y_true (sklearn's definition restricted to present labels)."""
    from sklearn.metrics import f1_score

    present = [k for k in labels if (y_true == k).any()]
    per = f1_score(y_true, y_pred, labels=present, average=None, zero_division=0)
    return float(np.mean(per)) if len(per) else float("nan"), {STRATEGIES[k]: float(v) for k, v in zip(present, per)}
