"""The Judge validity model (team/judge-design.md §7.4): LightGBM binary classifiers per family (forecast / detector),
5 seeds averaged, isotonic calibration on held-out assets, thresholds tau_use / tau_widen.

Artifacts in a models dir (what `load_judge` reads; written by sentinel.judge.train):
    judge.<family>.s<k>.txt      LightGBM text model, one per seed
    judge.calibration.json       {judge_version, tau_use, tau_widen, widen_factor, families: {family: {features,
                                  categorical, seeds, isotonic: {x, y}, tau_use, tau_widen, widen_factor, ...}},
                                  ledger_only: [...], metrics}

Inference (J4, #149):
    judge = load_judge(models_dir)            # None when the weights are missing -> ledger-only mode (§11)
    out = judge.predict(rows)                 # rows = raw rows (sentinel.judge.features.RAW_COLUMNS)
    out.p_valid, out.p_valid_sd, out.reasons  # calibrated 5-seed mean, seed disagreement, up to 3 plain reasons
Rows of M7 / M8 (and any witness without a family model) get p_valid = trust_main (ledger only, §7.4).
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Optional, Sequence

import numpy as np
import pandas as pd

from sentinel.judge import features as F

CAL_FILE = "judge.calibration.json"
PRECISION_USE = 0.90  # §7.5: tau_use = lowest P(valid) at which >= 90 % of "use" predictions are valid
WIDEN_CAL = 0.80  # §7.5: tau_widen = lowest P(valid) at which widened bands reach 0.80 calibration (§5.3)
MIN_USE = 30  # a threshold needs at least this many held-out predictions above it
DEFAULT_PARAMS = {"objective": "binary", "learning_rate": 0.05, "num_leaves": 31, "min_data_in_leaf": 40,
                  "feature_fraction": 0.8, "bagging_fraction": 0.8, "bagging_freq": 1, "lambda_l2": 1.0,
                  "max_bin": 255, "verbose": -1}


# ============================================================================================ calibration
@dataclass
class Isotonic:
    """Monotone piecewise-linear map fitted with sklearn's IsotonicRegression, stored as knots (JSON friendly)."""
    x: list = field(default_factory=lambda: [0.0, 1.0])
    y: list = field(default_factory=lambda: [0.0, 1.0])

    @classmethod
    def fit(cls, p: np.ndarray, y: np.ndarray, w: Optional[np.ndarray] = None) -> "Isotonic":
        from sklearn.isotonic import IsotonicRegression
        ir = IsotonicRegression(y_min=0.0, y_max=1.0, out_of_bounds="clip", increasing=True)
        ir.fit(np.asarray(p, float), np.asarray(y, float), sample_weight=w)
        xs, ys = np.asarray(ir.X_thresholds_, float), np.asarray(ir.y_thresholds_, float)
        if len(xs) < 2:
            xs, ys = np.array([0.0, 1.0]), np.array([float(ys[0]) if len(ys) else 0.5] * 2)
        return cls([round(float(v), 6) for v in xs], [round(float(v), 6) for v in ys])

    def __call__(self, p: np.ndarray) -> np.ndarray:
        return np.interp(np.asarray(p, float), self.x, self.y)


# ============================================================================================ thresholds
def tau_for_precision(p: np.ndarray, y: np.ndarray, target: float = PRECISION_USE, min_n: int = MIN_USE) -> float:
    """Lowest threshold t such that the predictions with p >= t are valid at least `target` of the time (and at
    least `min_n` of them). 1.01 (never use on the model's say) when no threshold meets it."""
    p, y = np.asarray(p, float), np.asarray(y, float)
    ok = np.isfinite(p) & np.isfinite(y)
    p, y = p[ok], y[ok]
    if len(p) == 0:
        return 1.01
    o = np.argsort(-p, kind="stable")
    ps, ys = p[o], y[o]
    cum = np.cumsum(ys) / np.arange(1, len(ys) + 1)  # precision of the top-k
    # only cut between distinct p values (ties share a verdict)
    last_of_tie = np.r_[ps[1:] != ps[:-1], True]
    good = last_of_tie & (cum >= target) & (np.arange(1, len(ps) + 1) >= min_n)
    if not good.any():
        return 1.01
    k = np.max(np.nonzero(good)[0])  # the lowest p (largest k) that still meets the precision rule
    return float(ps[k])


def widened_hit(hit_m: dict[float, np.ndarray], band_scale: np.ndarray, factor: float) -> np.ndarray:
    """Hit rate of each row's band widened by band_scale x factor, interpolated over the graded multipliers."""
    mults = np.array(sorted(hit_m))
    if not len(mults):
        return np.zeros(0)
    H = np.stack([np.asarray(hit_m[m], float) for m in mults], 1)  # [N, M]
    m_eff = np.clip(np.nan_to_num(np.asarray(band_scale, float), nan=1.0) * factor, mults[0], mults[-1])
    j = np.clip(np.searchsorted(mults, m_eff, side="right") - 1, 0, len(mults) - 2) if len(mults) > 1 else \
        np.zeros(len(m_eff), int)
    if len(mults) == 1:
        return H[:, 0]
    lo, hi = mults[j], mults[j + 1]
    w = np.clip((m_eff - lo) / (hi - lo), 0, 1)
    r = np.arange(len(m_eff))
    return H[r, j] * (1 - w) + H[r, j + 1] * w


def calibration_score(hit: np.ndarray) -> np.ndarray:
    """§5.3 calibration of an 80 % band: 1 - |hit - 0.8| / 0.8 (1 at the nominal rate, 0.75 for a band that never
    misses)."""
    return 1.0 - np.abs(np.asarray(hit, float) - 0.8) / 0.8


def widen_choice(p: np.ndarray, hit_m: dict[float, np.ndarray], band_scale: np.ndarray, tau_use: float,
                 target: float = WIDEN_CAL, factors: Sequence[float] = (1.0, 1.25, 1.5, 2.0, 2.5, 3.0),
                 min_n: int = MIN_USE) -> tuple[float, float]:
    """(tau_widen, widen_factor) for forecasts (§7.5 rule 5). A widened band = band x band_scale x factor. For each
    factor, tau_widen(factor) = lowest t < tau_use such that rows with t <= p < tau_use reach a mean §5.3
    calibration score of the widened band >= target (0.80). Over-wide bands (hit ~1) score 0.75, so widening cannot
    rescue a forecast by making it useless. The factor reaching the lowest tau wins (ties: the smaller factor)."""
    p = np.asarray(p, float)
    zone = np.isfinite(p) & (p < tau_use)
    best = (tau_use, 1.0)
    if not hit_m:
        return best
    for f in factors:
        score = calibration_score(widened_hit(hit_m, band_scale, f))
        z = zone & np.isfinite(score)
        if z.sum() < min_n:
            continue
        o = np.argsort(-p[z], kind="stable")
        ps, hs = p[z][o], score[z][o]
        cum = np.cumsum(hs) / np.arange(1, len(hs) + 1)
        last_of_tie = np.r_[ps[1:] != ps[:-1], True]
        good = last_of_tie & (cum >= target) & (np.arange(1, len(ps) + 1) >= min_n)
        if good.any():
            t = float(ps[np.max(np.nonzero(good)[0])])
            if t < best[0] - 1e-9:
                best = (t, float(f))
    return best


# ============================================================================================ verdict metrics
def verdict_metrics(p: np.ndarray, y: np.ndarray, tau_use: float, use_rule3: Optional[np.ndarray] = None,
                    tau_widen: Optional[float] = None) -> dict[str, float]:
    """Brier, precision of 'use' (p >= tau_use, or rule 3 when given), recall of 'don't use' on invalid rows
    (dont_use = below tau_widen for forecasts, below tau_use otherwise), recall_flagged = invalid rows not used as
    they are (widen or dont_use), shares of each verdict."""
    p, y = np.asarray(p, float), np.asarray(y, float)
    use = p >= tau_use
    if use_rule3 is not None:
        use = use | np.asarray(use_rule3, bool)
    low = tau_widen if tau_widen is not None else tau_use
    dont = ~use & (p < low)
    widen = ~use & ~dont
    inv = y < 0.5
    nan = float("nan")
    return {"brier": float(np.mean((p - y) ** 2)) if len(p) else nan,
            "precision_use": float(y[use].mean()) if use.any() else nan,
            "recall_dont_use": float(dont[inv].mean()) if inv.any() else nan,
            "recall_flagged": float((dont | widen)[inv].mean()) if inv.any() else nan,
            "share_use": float(use.mean()) if len(p) else nan, "share_widen": float(widen.mean()) if len(p) else nan,
            "share_dont_use": float(dont.mean()) if len(p) else nan,
            "valid_rate": float(y.mean()) if len(y) else nan, "n": float(len(p))}


# ============================================================================================ family model
@dataclass
class FamilyModel:
    family: str
    boosters: list  # lightgbm.Booster per seed
    features: list
    iso: Isotonic
    tau_use: float = 1.01
    tau_widen: float = 1.01
    widen_factor: float = 1.0
    info: dict = field(default_factory=dict)

    def raw(self, X: pd.DataFrame) -> np.ndarray:
        """[seeds, N] uncalibrated probabilities."""
        return np.stack([b.predict(X[self.features], num_threads=1 if len(X) < 2000 else 0)
                         for b in self.boosters])

    def predict(self, X: pd.DataFrame) -> tuple[np.ndarray, np.ndarray]:
        r = self.raw(X)
        return self.iso(r.mean(0)), r.std(0)  # sd = seed disagreement on the raw scale (isotonic has plateaus)

    def contrib(self, X: pd.DataFrame) -> np.ndarray:
        return np.mean([b.predict(X[self.features], pred_contrib=True) for b in self.boosters], 0)


def _lgb_dataset(X: pd.DataFrame, y: np.ndarray, cat: Sequence[str], ref=None):
    import lightgbm as lgb
    return lgb.Dataset(X, label=y, categorical_feature=[c for c in cat if c in X.columns], reference=ref,
                       free_raw_data=False)


def fit_family(X: pd.DataFrame, y: np.ndarray, groups: np.ndarray, family: str, seeds: Sequence[int] = range(5),
               params: Optional[dict] = None, rounds: int = 600, threads: int = 0, device: str = "cpu",
               early_stop: int = 50, log=print) -> tuple[list, dict]:
    """Train one booster per seed on (X, y). 15 % of the `groups` (at least one; the trainer passes sampled weeks)
    are held out for early stopping. Returns (boosters, info)."""
    import lightgbm as lgb
    groups = np.asarray(groups)
    uniq = np.unique(groups)
    rng = np.random.default_rng(0)
    es_assets = set(rng.choice(uniq, max(1, int(round(0.15 * len(uniq)))), replace=False).tolist())
    es = np.array([g in es_assets for g in groups])
    cat = [c for c in F.CATEGORICAL if c in X.columns]
    dtr = _lgb_dataset(X[~es], y[~es], cat)
    dva = _lgb_dataset(X[es], y[es], cat, ref=dtr)
    boosters, iters = [], []
    for s in seeds:
        p = {**DEFAULT_PARAMS, **(params or {}), "seed": int(s), "bagging_seed": int(s), "feature_fraction_seed": int(s),
             "num_threads": int(threads), "device_type": device}
        b = lgb.train(p, dtr, num_boost_round=rounds, valid_sets=[dva],
                      callbacks=[lgb.early_stopping(early_stop, verbose=False), lgb.log_evaluation(0)])
        boosters.append(b)
        iters.append(int(b.best_iteration or rounds))
    return boosters, {"best_iterations": iters, "n_train": int((~es).sum()), "n_early_stop": int(es.sum())}


# ============================================================================================ Judge
class Judge:
    """Per-family validity models + thresholds; `predict(rows)` -> p_valid, p_valid_sd, reasons."""

    def __init__(self, families: dict[str, FamilyModel], calibration: dict):
        self.families = families
        self.calibration = calibration
        self.judge_version = str(calibration.get("judge_version", "judge"))
        self.tau_use = float(calibration.get("tau_use", 1.01))
        self.tau_widen = float(calibration.get("tau_widen", 1.01))
        self.widen_factor = float(calibration.get("widen_factor", 1.0))

    def thresholds(self, model: str) -> dict[str, float]:
        fm = self.families.get(F.family_of(model) or "")
        if fm is None:
            return {"tau_use": self.tau_use, "tau_widen": self.tau_widen, "widen_factor": self.widen_factor}
        return {"tau_use": fm.tau_use, "tau_widen": fm.tau_widen, "widen_factor": fm.widen_factor}

    def predict(self, rows: pd.DataFrame, reasons: bool = True) -> pd.DataFrame:
        rows = rows.reset_index(drop=True)
        out = pd.DataFrame({"p_valid": np.nan, "p_valid_sd": np.nan, "family": "", "tau_use": np.nan,
                            "tau_widen": np.nan, "judge_version": self.judge_version}, index=rows.index)
        out["reasons"] = [[] for _ in range(len(rows))]
        fam = rows["model"].map(lambda m: F.family_of(str(m)) or "ledger_only")
        for name, fm in self.families.items():
            idx = np.nonzero((fam == name).to_numpy())[0]
            if not len(idx):
                continue
            sub = rows.iloc[idx]
            X = F.build(sub, name)
            p, sd = fm.predict(X)
            out.loc[idx, "p_valid"], out.loc[idx, "p_valid_sd"] = p, sd
            out.loc[idx, "family"] = name
            out.loc[idx, "tau_use"], out.loc[idx, "tau_widen"] = fm.tau_use, fm.tau_widen
            if reasons:
                rs = F.reasons_for(fm.contrib(X), fm.features, sub, p)
                for j, r in zip(idx, rs):
                    out.at[j, "reasons"] = r
        rest = np.nonzero(out["family"].eq("").to_numpy())[0]
        if len(rest):  # M7 / M8 and anything without a family model: ledger trust is P(valid) (§7.4, §11)
            t = pd.to_numeric(rows.iloc[rest].get("trust_main", pd.Series(np.nan, index=rows.index[rest])),
                              errors="coerce").to_numpy(float)
            out.loc[rest, "p_valid"] = t
            out.loc[rest, "family"] = "ledger_only"
            out.loc[rest, "tau_use"], out.loc[rest, "tau_widen"] = self.tau_use, self.tau_widen
            for j, tj in zip(rest, t):
                out.at[j, "reasons"] = ([f"ledger only: trust {tj:.2f}"] if np.isfinite(tj) else
                                        ["ledger only: no trust value"])
        return out

    # -- persistence -------------------------------------------------------------------------------------------
    def save(self, out_dir: Path) -> list[Path]:
        out_dir = Path(out_dir)
        out_dir.mkdir(parents=True, exist_ok=True)
        files = []
        cal = dict(self.calibration)
        fams = {}
        for name, fm in self.families.items():
            seeds = []
            for k, b in enumerate(fm.boosters):
                p = out_dir / f"judge.{name}.s{k}.txt"
                b.save_model(str(p), num_iteration=b.best_iteration or None)
                files.append(p)
                seeds.append(p.name)
            fams[name] = {"features": fm.features, "categorical": [c for c in F.CATEGORICAL if c in fm.features],
                          "seeds": seeds, "isotonic": {"x": fm.iso.x, "y": fm.iso.y}, "tau_use": fm.tau_use,
                          "tau_widen": fm.tau_widen, "widen_factor": fm.widen_factor, **fm.info}
        cal["families"] = fams
        cal.setdefault("ledger_only", list(F.LEDGER_ONLY))
        p = out_dir / CAL_FILE
        p.write_text(json.dumps(cal, indent=1, default=float) + "\n")
        files.append(p)
        return files


def load_judge(models_dir: Path | str) -> Optional[Judge]:
    """The Judge from `models_dir`, or None when judge.calibration.json or any of its LightGBM files is missing
    (the loop then runs ledger-only, §11)."""
    d = Path(models_dir)
    p = d / CAL_FILE
    if not p.exists():
        return None
    try:
        import lightgbm as lgb
        cal = json.loads(p.read_text())
        fams = {}
        for name, spec in (cal.get("families") or {}).items():
            paths = [d / s for s in spec.get("seeds", [])]
            if not paths or not all(x.exists() for x in paths):
                return None
            boosters = [lgb.Booster(model_file=str(x)) for x in paths]
            iso = Isotonic(**spec.get("isotonic", {})) if spec.get("isotonic") else Isotonic()
            fams[name] = FamilyModel(name, boosters, list(spec["features"]), iso, float(spec.get("tau_use", 1.01)),
                                     float(spec.get("tau_widen", 1.01)), float(spec.get("widen_factor", 1.0)))
        if not fams:
            return None
        return Judge(fams, cal)
    except Exception as e:  # a corrupt artifact must not take the loop down
        print(f"[judge] could not load Judge from {d}: {type(e).__name__}: {e}")
        return None
