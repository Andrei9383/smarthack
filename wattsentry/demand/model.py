"""M3 DemandNet v2: LightGBM quantile regression (alpha 0.1/0.5/0.9) per zone (county / RO) on hourly load.

Features (all known at a day-ahead decision made at local midnight): Europe/Bucharest hour, day of week, month,
season (doy sin/cos), RO public holidays incl. Orthodox Easter (+ day before/after, "punte" bridge days),
temperature with heating/cooling degree hours (hourly and 24 h mean), the national load shape for that
(month, day type, hour), and the zone's own load 24 h and 168 h earlier.

Target = load / the zone's mean load over the previous 7 local days (trees cannot extrapolate a level trend, and
one model then fits any zone size); predictions are multiplied back. The 10-90 % interval is split-conformal
(CQR) calibrated on the validation year so that it covers ~80 % (plan decision 13:52: 78-82 %).

Baselines (plan §2 row 9): persistence day (y[t−24h]), persistence week (y[t−168h]) and national-shape scaling
(shape(t) × the zone's level over the previous 7 local days).
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Optional

import numpy as np

from wattsentry.contracts import ModelId, Prediction

from .calendar import calendar_features, degree_hours, parse_ts, utc_iso
from .data import LoadSeries

QUANTILES = (0.1, 0.5, 0.9)
FEATURES = (
    "hour", "dow", "month", "doy_sin", "doy_cos", "weekend", "holiday", "pre_holiday", "post_holiday", "bridge",
    "temp_c", "hdd", "cdd", "temp_24h", "hdd_24h", "cdd_24h", "national_shape", "lag_24h", "lag_168h",
)
DEFAULT_PARAMS = {
    "learning_rate": 0.05,
    "num_leaves": 31,
    "min_data_in_leaf": 40,
    "feature_fraction": 0.9,
    "bagging_fraction": 0.9,
    "bagging_freq": 1,
    "lambda_l2": 1.0,
    "verbose": -1,
    "num_threads": 4,
}


# --- national shape -------------------------------------------------------------------------------------

@dataclass
class NationalShape:
    """Mean normalised national load by (month, day type, local hour): table[12, 3, 24], mean = 1."""

    table: np.ndarray

    @classmethod
    def fit(cls, series: LoadSeries, mask: Optional[np.ndarray] = None) -> "NationalShape":
        cal = calendar_features(series.ts)
        y = series.load_mw
        ok = np.isfinite(y) & (mask if mask is not None else True)
        norm = y[ok] / np.mean(y[ok])
        idx = ((cal["month"][ok] - 1) * 3 + cal["daytype"][ok]) * 24 + cal["hour"][ok]
        s = np.bincount(idx, weights=norm, minlength=12 * 3 * 24)
        n = np.bincount(idx, minlength=12 * 3 * 24)
        tab = np.where(n > 0, s / np.maximum(n, 1), np.nan).reshape(12, 3, 24)
        # unseen cells (short series): fall back to the (day type, hour) mean, then the hour mean, then 1
        by_dt_h = np.nanmean(tab, axis=0) if np.isfinite(tab).any() else np.ones((3, 24))
        tab = np.where(np.isfinite(tab), tab, by_dt_h[None])
        by_h = np.nanmean(tab, axis=(0, 1)) if np.isfinite(tab).any() else np.ones(24)
        tab = np.where(np.isfinite(tab), tab, by_h[None, None])
        return cls(np.nan_to_num(tab, nan=1.0))

    def __call__(self, cal: dict[str, np.ndarray]) -> np.ndarray:
        return self.table[cal["month"] - 1, cal["daytype"], np.minimum(cal["hour"], 23)]


# --- features --------------------------------------------------------------------------------------------

def _lag(y: np.ndarray, k: int) -> np.ndarray:
    out = np.full_like(y, np.nan, dtype=float)
    if k < len(y):
        out[k:] = y[:-k]
    return out


def _rolling_mean_causal(x: np.ndarray, w: int) -> np.ndarray:
    x = np.nan_to_num(np.asarray(x, dtype=float), nan=np.nanmean(x) if np.isfinite(x).any() else 0.0)
    c = np.concatenate([[0.0], np.cumsum(x)])
    n = np.minimum(np.arange(1, len(x) + 1), w)
    return (c[1:] - c[np.maximum(0, np.arange(1, len(x) + 1) - w)]) / n


def trailing_level(y: np.ndarray, local_day: np.ndarray, days_back: int = 7, min_hours: int = 72) -> np.ndarray:
    """Per row: mean of y over the previous `days_back` local days (strictly before the row's day); NaN if thin."""
    y = np.asarray(y, dtype=float)
    d_idx = (local_day - local_day.min()).astype(int)
    nd = d_idx.max() + 1
    ok = np.isfinite(y)
    cy = np.concatenate([[0.0], np.cumsum(np.bincount(d_idx[ok], weights=y[ok], minlength=nd))])
    cn = np.concatenate([[0.0], np.cumsum(np.bincount(d_idx[ok], minlength=nd))])
    lo = np.maximum(0, np.arange(nd) - days_back)
    num, cnt = cy[np.arange(nd)] - cy[lo], cn[np.arange(nd)] - cn[lo]
    with np.errstate(invalid="ignore", divide="ignore"):
        lvl = np.where(cnt >= min_hours, num / np.maximum(cnt, 1), np.nan)
    return lvl[d_idx]


def build_features(series: LoadSeries, shape: NationalShape) -> tuple[np.ndarray, dict[str, np.ndarray]]:
    """X [n, len(FEATURES)] for every hour of `series` (regular hourly UTC grid) + the calendar dict.

    `cal["level"]` = trailing 7-day mean load (MW), the scale the target and the lag features are divided by.
    """
    cal = calendar_features(series.ts)
    temp = np.asarray(series.temp_c, dtype=float)
    hdd, cdd = degree_hours(temp)
    t24 = _rolling_mean_causal(temp, 24)
    hdd24, cdd24 = degree_hours(t24)
    y = np.asarray(series.load_mw, dtype=float)
    level = trailing_level(y, cal["local_day"])
    cal["level"] = level
    cols = {
        **{k: cal[k].astype(float) for k in ("hour", "dow", "month", "doy_sin", "doy_cos", "weekend", "holiday",
                                             "pre_holiday", "post_holiday", "bridge")},
        "temp_c": temp, "hdd": hdd, "cdd": cdd, "temp_24h": t24, "hdd_24h": hdd24, "cdd_24h": cdd24,
        "national_shape": shape(cal), "lag_24h": _lag(y, 24) / level, "lag_168h": _lag(y, 168) / level,
    }
    return np.column_stack([cols[f] for f in FEATURES]), cal


# --- baselines -------------------------------------------------------------------------------------------

def persistence(y: np.ndarray, hours: int) -> np.ndarray:
    return _lag(np.asarray(y, dtype=float), hours)


def shape_scaling(y: np.ndarray, shape_vals: np.ndarray, local_day: np.ndarray, days_back: int = 7) -> np.ndarray:
    """shape(t) × level, level = Σ y / Σ shape over the previous `days_back` local days (data before the day)."""
    y = np.asarray(y, dtype=float)
    d_idx = (local_day - local_day.min()).astype(int)
    nd = d_idx.max() + 1
    ok = np.isfinite(y)
    sy = np.bincount(d_idx[ok], weights=y[ok], minlength=nd)
    ss = np.bincount(d_idx[ok], weights=shape_vals[ok], minlength=nd)
    cy, cs = np.concatenate([[0.0], np.cumsum(sy)]), np.concatenate([[0.0], np.cumsum(ss)])
    lo = np.maximum(0, np.arange(nd) - days_back)
    num, den = cy[np.arange(nd)] - cy[lo], cs[np.arange(nd)] - cs[lo]
    with np.errstate(invalid="ignore", divide="ignore"):
        level = np.where((np.arange(nd) >= days_back) & (den > 0), num / den, np.nan)
    return shape_vals * level[d_idx]


# --- metrics ---------------------------------------------------------------------------------------------

def nmae(y: np.ndarray, p: np.ndarray) -> float:
    ok = np.isfinite(y) & np.isfinite(p)
    return float(np.sum(np.abs(y[ok] - p[ok])) / max(np.sum(np.abs(y[ok])), 1e-9))


def pinball(y: np.ndarray, q: np.ndarray, alpha: float) -> float:
    ok = np.isfinite(y) & np.isfinite(q)
    d = y[ok] - q[ok]
    return float(np.mean(np.maximum(alpha * d, (alpha - 1) * d)))


def coverage(y: np.ndarray, lo: np.ndarray, hi: np.ndarray) -> float:
    ok = np.isfinite(y) & np.isfinite(lo) & np.isfinite(hi)
    return float(np.mean((y[ok] >= lo[ok]) & (y[ok] <= hi[ok])))


def peak_hour_error(y: np.ndarray, p: np.ndarray, local_day: np.ndarray, hour: np.ndarray,
                    min_hours: int = 23) -> float:
    """Mean |local hour of the predicted daily peak − actual daily peak| (hours) over complete local days."""
    errs = []
    for d in np.unique(local_day):
        m = (local_day == d) & np.isfinite(y) & np.isfinite(p)
        if m.sum() < min_hours:
            continue
        hy, hp = hour[m][np.argmax(y[m])], hour[m][np.argmax(p[m])]
        errs.append(abs(int(hy) - int(hp)))
    return float(np.mean(errs)) if errs else float("nan")


# --- the model -------------------------------------------------------------------------------------------

def default_split(ts: np.ndarray) -> dict[str, np.ndarray]:
    """Plan §4: train 2022–2023, validate 2024, test 2025 when the data covers them; else 60/20/20 by time."""
    years = ts.astype("datetime64[Y]").astype(int) + 1970
    tr, va, te = years <= 2023, years == 2024, years == 2025
    if tr.sum() > 24 * 60 and va.sum() > 24 * 30 and te.sum() > 24 * 30:
        return {"train": tr, "valid": va, "test": te, "scheme": "years 2022-23 / 2024 / 2025"}
    n = len(ts)
    days = ts.astype("datetime64[D]")
    cut1, cut2 = days[int(0.6 * n)], days[int(0.8 * n)]
    return {"train": days < cut1, "valid": (days >= cut1) & (days < cut2), "test": days >= cut2,
            "scheme": "chronological 60/20/20 by day"}


@dataclass
class DemandNet:
    """One LightGBM booster per (zone, quantile). `predict()` returns MW quantiles, monotone by construction."""

    params: dict[str, Any] = field(default_factory=lambda: dict(DEFAULT_PARAMS))
    num_boost_round: int = 600
    early_stopping_rounds: int = 40
    seed: int = 0
    model_version: str = "demandnet-v2"
    shape: Optional[NationalShape] = None
    boosters: dict[str, dict[float, Any]] = field(default_factory=dict)
    best_iters: dict[str, dict[float, int]] = field(default_factory=dict)
    cqr_offset: dict[str, float] = field(default_factory=dict)  # added to q90, subtracted from q10 (ratio units)
    target_coverage: float = 0.8
    model: str = ModelId.DEMANDNET.value

    def fit_zone(self, zone_id: str, X: np.ndarray, y: np.ndarray, level: np.ndarray,
                 X_val: Optional[np.ndarray] = None, y_val: Optional[np.ndarray] = None,
                 level_val: Optional[np.ndarray] = None) -> "DemandNet":
        """Fit the 3 quantile boosters on y / level; early-stop and CQR-calibrate on the validation slice."""
        import lightgbm as lgb

        y = np.asarray(y, dtype=float) / level
        if y_val is not None:
            y_val = np.asarray(y_val, dtype=float) / level_val
        ok = np.isfinite(y)
        dtrain = lgb.Dataset(X[ok], y[ok], feature_name=list(FEATURES), free_raw_data=False)
        valid_sets, cbs = [], []
        if X_val is not None and y_val is not None and np.isfinite(y_val).sum() > 48:
            okv = np.isfinite(y_val)
            valid_sets = [lgb.Dataset(X_val[okv], y_val[okv], reference=dtrain)]
            cbs = [lgb.early_stopping(self.early_stopping_rounds, verbose=False)]
        self.boosters[zone_id], self.best_iters[zone_id] = {}, {}
        for a in QUANTILES:
            params = {**self.params, "objective": "quantile", "alpha": a, "metric": "quantile", "seed": self.seed}
            b = lgb.train(params, dtrain, num_boost_round=self.num_boost_round, valid_sets=valid_sets or None,
                          callbacks=cbs or None)
            self.boosters[zone_id][a] = b
            self.best_iters[zone_id][a] = int(b.best_iteration or b.current_iteration())
        self.cqr_offset[zone_id] = 0.0
        if valid_sets:
            R = self._raw(zone_id, X_val[okv])
            e = np.maximum(R[:, 0] - y_val[okv], y_val[okv] - R[:, 2])  # CQR conformity score
            n = len(e)
            self.cqr_offset[zone_id] = float(np.quantile(e, min(1.0, self.target_coverage * (n + 1) / n)))
        return self

    def _raw(self, zone_id: str, X: np.ndarray) -> np.ndarray:
        bs = self.boosters[zone_id]
        P = np.column_stack([bs[a].predict(X, num_iteration=self.best_iters[zone_id][a]) for a in QUANTILES])
        return np.sort(P, axis=1)

    def predict(self, zone_id: str, X: np.ndarray, level: np.ndarray) -> np.ndarray:
        """[n, 3] MW quantiles (q10, q50, q90): ratio × level, CQR-widened, sorted so they never cross."""
        P = self._raw(zone_id, X)
        off = self.cqr_offset.get(zone_id, 0.0)
        P[:, 0] -= off
        P[:, 2] += off
        P = np.sort(np.maximum(P, 0.0), axis=1)
        return P * np.asarray(level, dtype=float)[:, None]

    def predict_day(self, series: LoadSeries, t0: str, horizon: int = 96, step_minutes: int = 15) -> Prediction:
        """Envelope for `series.zone_id` from `t0` (UTC): hourly quantiles held over each 15-min step, MW.

        `series` must hold the zone's load up to t0 (for the 24 h / 168 h lags) and temperature for the window.
        """
        X, cal = build_features(series, self.shape)
        start = parse_ts(t0)
        times = start + np.arange(horizon) * np.timedelta64(step_minutes, "m")
        pos = ((times.astype("datetime64[h]") - series.ts[0].astype("datetime64[h]")).astype(int))
        if pos.min() < 0 or pos.max() >= len(series.ts):
            raise ValueError("series does not cover the prediction window (append the window with load = NaN)")
        P = self.predict(series.zone_id, X[pos], cal["level"][pos])
        return Prediction(asset_id=series.zone_id, model=self.model, model_version=self.model_version,
                          t0=utc_iso(start), horizon=horizon, q10=P[:, 0], q50=P[:, 1], q90=P[:, 2], unit="MW",
                          step_minutes=step_minutes, extra={"label": series.label, "hourly": True})

    def feature_importance(self, zone_id: str) -> dict[str, float]:
        b = self.boosters[zone_id][0.5]
        g = b.feature_importance(importance_type="gain")
        tot = float(g.sum()) or 1.0
        return {f: round(float(v) / tot, 4) for f, v in sorted(zip(FEATURES, g), key=lambda kv: -kv[1])}

    def save(self, out_dir: str | Path) -> list[str]:
        out = Path(out_dir)
        out.mkdir(parents=True, exist_ok=True)
        files = []
        for z, bs in self.boosters.items():
            for a, b in bs.items():
                p = out / f"demandnet_{z.replace('/', '_')}_q{int(round(a * 100)):02d}.txt"
                b.save_model(str(p), num_iteration=self.best_iters[z][a])
                files.append(p.name)
        meta = {"model": self.model, "model_version": self.model_version, "features": list(FEATURES),
                "quantiles": list(QUANTILES), "zones": list(self.boosters), "files": files,
                "cqr_offset": self.cqr_offset, "target_coverage": self.target_coverage,
                "national_shape": self.shape.table.round(5).tolist() if self.shape is not None else None}
        (out / "demandnet_meta.json").write_text(json.dumps(meta))
        return files + ["demandnet_meta.json"]

    @classmethod
    def load(cls, out_dir: str | Path) -> "DemandNet":
        import lightgbm as lgb

        out = Path(out_dir)
        meta = json.loads((out / "demandnet_meta.json").read_text())
        m = cls(model_version=meta["model_version"])
        m.shape = NationalShape(np.array(meta["national_shape"])) if meta.get("national_shape") else None
        m.cqr_offset = {k: float(v) for k, v in meta.get("cqr_offset", {}).items()}
        for z in meta["zones"]:
            m.boosters[z], m.best_iters[z] = {}, {}
            for a in meta["quantiles"]:
                b = lgb.Booster(model_file=str(out / f"demandnet_{z.replace('/', '_')}_q{int(round(a * 100)):02d}.txt"))
                m.boosters[z][a] = b
                m.best_iters[z][a] = b.current_iteration()
        return m


def evaluate_zone(series: LoadSeries, model: DemandNet, *, shape: Optional[NationalShape] = None,
                  split: Optional[dict[str, np.ndarray]] = None) -> dict[str, Any]:
    """Fit DemandNet on the zone's train slice (early stopping on valid) and score it + baselines on test."""
    split = split or default_split(series.ts)
    if shape is None:
        shape = NationalShape.fit(series, split["train"])
    if model.shape is None:
        model.shape = shape
    X, cal = build_features(series, shape)
    y = np.asarray(series.load_mw, dtype=float)
    tr, va, te = split["train"], split["valid"], split["test"]
    lvl = cal["level"]
    model.fit_zone(series.zone_id, X[tr], y[tr], lvl[tr], X[va], y[va], lvl[va])
    P = model.predict(series.zone_id, X[te], lvl[te])
    yt, day, hour = y[te], cal["local_day"][te], cal["hour"][te]
    base = {
        "persistence_day": persistence(y, 24)[te],
        "persistence_week": persistence(y, 168)[te],
        "shape_scaling": shape_scaling(y, shape(cal), cal["local_day"])[te],
    }
    # score everyone on the same rows (all baselines defined)
    ok = np.isfinite(yt) & np.all(np.isfinite(P), axis=1)
    for b in base.values():
        ok &= np.isfinite(b)
    res: dict[str, Any] = {
        "zone_id": series.zone_id,
        "label": series.label,
        "temp_label": series.temp_label,
        "source": series.source,
        "split": split.get("scheme"),
        "n_train": int(np.isfinite(y[tr]).sum()),
        "n_valid": int(np.isfinite(y[va]).sum()),
        "n_test": int(ok.sum()),
        "best_iters": {str(a): i for a, i in model.best_iters[series.zone_id].items()},
        "cqr_offset": model.cqr_offset[series.zone_id],
        "demand_nmae": nmae(yt[ok], P[ok, 1]),
        "demand_peak_hour_err": peak_hour_error(yt[ok], P[ok, 1], day[ok], hour[ok]),
        "demand_pinball_q10": pinball(yt[ok], P[ok, 0], 0.1),
        "demand_pinball_q50": pinball(yt[ok], P[ok, 1], 0.5),
        "demand_pinball_q90": pinball(yt[ok], P[ok, 2], 0.9),
        "demand_coverage_10_90": coverage(yt[ok], P[ok, 0], P[ok, 2]),
        "feature_importance": model.feature_importance(series.zone_id),
    }
    for name, b in base.items():
        res[f"baseline_{name}_nmae"] = nmae(yt[ok], b[ok])
        res[f"baseline_{name}_peak_hour_err"] = peak_hour_error(yt[ok], b[ok], day[ok], hour[ok])
    best = min(base, key=lambda k: res[f"baseline_{k}_nmae"])
    res["best_baseline"] = best
    res["skill_vs_best_baseline"] = 1 - res["demand_nmae"] / max(res[f"baseline_{best}_nmae"], 1e-9)
    # error analysis: where is it wrong? (holidays vs normal days, by season, worst days)
    hol = cal["holiday"][te][ok] == 1
    month = cal["month"][te][ok]
    err = np.abs(yt[ok] - P[ok, 1])
    res["error_analysis"] = {
        "nmae_holidays": float(err[hol].sum() / max(yt[ok][hol].sum(), 1e-9)) if hol.any() else None,
        "nmae_non_holidays": float(err[~hol].sum() / max(yt[ok][~hol].sum(), 1e-9)),
        "nmae_by_season": {s: float(err[m].sum() / max(yt[ok][m].sum(), 1e-9))
                           for s, m in (("winter", np.isin(month, [12, 1, 2])), ("spring", np.isin(month, [3, 4, 5])),
                                        ("summer", np.isin(month, [6, 7, 8])), ("autumn", np.isin(month, [9, 10, 11])))
                           if m.any()},
        "worst_days": _worst_days(day[ok], err, yt[ok]),
    }
    return res


def _worst_days(day: np.ndarray, err: np.ndarray, y: np.ndarray, k: int = 5) -> list[dict[str, Any]]:
    u, inv = np.unique(day, return_inverse=True)
    e = np.bincount(inv, weights=err) / np.maximum(np.bincount(inv, weights=y), 1e-9)
    order = np.argsort(-e)[:k]
    return [{"local_day": str(u[i]), "nmae": round(float(e[i]), 4)} for i in order]
