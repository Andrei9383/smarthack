"""LightGBM quantile baselines (P10/P50/P90) for PV and household load, and the M1/M2 fallback predictor.

    from sentinel.eval.baselines.pv.lgbm import LGBMPredictor
    pred = LGBMPredictor.load("lgbm_pv.json").predict(site_row, window)     # -> contracts Prediction

Training works on the wide arrays from data.py; prediction on one contract Window (model.md), with the same
feature code (features.py). PV targets are kW per recorded kWp, load targets are kW / the household's 7-day mean;
night PV is 0 by construction. Quantiles are sorted after prediction so q10 <= q50 <= q90 always holds.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterator, Mapping

import lightgbm as lgb
import numpy as np
import pandas as pd

from sentinel.contracts.envelope import Prediction, Window, parse_utc

from .data import D, FleetArrays, day_grid
from .features import LOAD_FEATURES, PV_FEATURES, REF_DAYS, calendar, load_features, pv_features, stack
from .physical import clearsky_power_kw, pv_power_kw, recorded_meta, solar_geometry

QUANTILES = (0.1, 0.5, 0.9)
NIGHT_ZENITH = 90.0
PARAMS = dict(objective="quantile", learning_rate=0.05, num_leaves=63, min_data_in_leaf=200,
              feature_fraction=0.9, bagging_fraction=0.8, bagging_freq=1, lambda_l2=1.0, verbose=-1)
WEATHER_COLS = ("ghi", "dni", "dhi", "temp_c", "wind10_ms", "snow_cm")
# Default channel order of an array window.weather = #14's physics.WEATHER_CHANNELS (window.meta["weather_channels"])
ARRAY_CHANNELS = ("ghi", "dni", "dhi", "temp", "cloud", "wind", "snow")
_ALIASES = {"ghi_wm2": "ghi", "dni_wm2": "dni", "dhi_wm2": "dhi", "temp_air": "temp_c", "wind_speed": "wind10_ms",
            "snow_depth_cm": "snow_cm", "temp": "temp_c", "wind": "wind10_ms", "snow": "snow_cm"}


def weather_channels(window: Window) -> list[str] | None:
    meta = window.meta or {}
    return meta.get("weather_channels") or meta.get("weather_cols")


def power_scale(window: Window) -> float:
    """Divisor that turns window power series (reference / history / telemetry) into kW."""
    return 1.0 if (window.meta or {}).get("power_unit", "W") == "kW" else 1000.0


@dataclass
class QuantileBundle:
    kind: str                                # "pv" | "load"
    features: list[str]
    models: dict[float, lgb.Booster]
    version: str = "lgbm-local"
    info: dict[str, Any] = field(default_factory=dict)

    def predict(self, X: np.ndarray) -> np.ndarray:
        """[rows, F] -> [3, rows] sorted quantiles, in target units."""
        q = np.stack([self.models[a].predict(X, num_threads=0) for a in QUANTILES])
        return np.sort(q, axis=0)

    def save(self, path: Path) -> Path:
        path = Path(path)
        path.write_text(json.dumps({"kind": self.kind, "features": self.features, "version": self.version,
                                    "info": self.info,
                                    "models": {str(a): m.model_to_string() for a, m in self.models.items()}}))
        return path

    @classmethod
    def load(cls, path: Path) -> "QuantileBundle":
        d = json.loads(Path(path).read_text())
        models = {float(a): lgb.Booster(model_str=s) for a, s in d["models"].items()}
        return cls(d["kind"], d["features"], models, d.get("version", "lgbm-local"), d.get("info", {}))


def fit_quantiles(X: np.ndarray, y: np.ndarray, names: list[str], kind: str, rounds: int = 300, seed: int = 0,
                  version: str = "lgbm-local") -> QuantileBundle:
    ds = lgb.Dataset(X, y, feature_name=names, free_raw_data=False)
    models = {a: lgb.train({**PARAMS, "alpha": a, "seed": seed}, ds, num_boost_round=rounds) for a in QUANTILES}
    return QuantileBundle(kind, list(names), models, version, {"rows": int(len(y)), "rounds": rounds})


# --------------------------------------------------------------------------------------- bulk (wide arrays)
def _chunks(days_idx: np.ndarray, size: int = 45) -> Iterator[np.ndarray]:
    days_idx = days_idx[days_idx >= REF_DAYS]
    for i in range(0, len(days_idx), size):
        yield days_idx[i:i + size]


def _ref(arr: np.ndarray, di: np.ndarray) -> np.ndarray:
    """[n, d, 96] -> [n, len(di), 7, 96]: the 7 days before each day in di."""
    return arr[:, di[:, None] + np.arange(-REF_DAYS, 0)[None, :]]


def pv_bulk_features(a: FleetArrays, di: np.ndarray) -> dict[str, np.ndarray]:
    s = a.sites
    return pv_features(
        weather={k: a.weather[k][:, di] for k in WEATHER_COLS},
        geom={k: a.geom[k][:, di] for k in a.geom},
        pvlib_kw=a.pvlib_kw[:, di], pvlib_cs_kw=a.pvlib_cs_kw[:, di],
        ref_kw=_ref(a.pv_kw, di), ref_cs_kw=_ref(a.pvlib_cs_kw, di),
        kwp=s.kwp_recorded.to_numpy()[:, None], tilt=s.tilt_recorded.to_numpy()[:, None],
        azimuth=s.azimuth_recorded.to_numpy()[:, None], lat=s.lat.to_numpy()[:, None])


def load_bulk_features(a: FleetArrays, di: np.ndarray) -> dict[str, np.ndarray]:
    s = a.sites
    ts = day_grid(a.days[di])
    cal = {k: v.reshape(1, len(di), D) for k, v in calendar(ts).items()}
    site = {k: s[k].to_numpy()[:, None] for k in ("archetype", "heating", "has_ac", "has_ev",
                                                   "has_electric_boiler", "works_from_home")}
    return load_features(temp_c=a.weather["temp_c"][:, di], ghi=a.weather["ghi"][:, di], cal=cal,
                         hist_kw=_ref(a.load_kw, di), hist_temp_c=_ref(a.weather["temp_c"], di), site=site)


def _rows(kind: str, a: FleetArrays, di: np.ndarray, site_mask: np.ndarray, healthy_only: bool):
    f = pv_bulk_features(a, di) if kind == "pv" else load_bulk_features(a, di)
    names = PV_FEATURES if kind == "pv" else LOAD_FEATURES
    X = stack(f, names)
    if kind == "pv":
        kwp = a.sites.kwp_recorded.to_numpy()[:, None, None]
        y = a.pv_kw[:, di] / kwp
        keep = a.geom["zenith"][:, di] < NIGHT_ZENITH
    else:
        y = a.load_kw[:, di] / f["scale"]
        keep = np.ones_like(y, bool)
    keep = keep & np.isfinite(y) & site_mask[:, None, None]
    if healthy_only:
        keep &= a.fault[:, di] == 0
    keep = keep.ravel()
    return X[keep], y.ravel()[keep]


def train(kind: str, a: FleetArrays, day_mask: np.ndarray, site_mask: np.ndarray, max_rows: int = 400_000,
          rounds: int = 300, seed: int = 0, log=print) -> QuantileBundle:
    """Fit P10/P50/P90 on healthy (unlabelled-fault) steps of the training days and sites."""
    rng = np.random.default_rng(seed)
    Xs, ys = [], []
    di_all = np.flatnonzero(day_mask)
    per_chunk = max(1, int(max_rows / max(1, len(di_all)) * 45 * 1.2))
    for di in _chunks(di_all):
        X, y = _rows(kind, a, di, site_mask, healthy_only=True)
        if len(y) > per_chunk:
            pick = rng.choice(len(y), per_chunk, replace=False)
            X, y = X[pick], y[pick]
        Xs.append(X)
        ys.append(y)
    X, y = np.concatenate(Xs), np.concatenate(ys)
    log(f"lgbm {kind}: training on {len(y):,} rows x {X.shape[1]} features")
    names = PV_FEATURES if kind == "pv" else LOAD_FEATURES
    return fit_quantiles(X, y, names, kind, rounds=rounds, seed=seed, version=f"lgbm-{kind}-local")


def predict_bulk(b: QuantileBundle, a: FleetArrays, day_mask: np.ndarray) -> np.ndarray:
    """[3, n, d, 96] kW for the masked days (NaN elsewhere)."""
    out = np.full((3,) + a.pv_kw.shape, np.nan)
    for di in _chunks(np.flatnonzero(day_mask)):
        f = pv_bulk_features(a, di) if b.kind == "pv" else load_bulk_features(a, di)
        q = b.predict(stack(f, b.features)).reshape((3, a.n, len(di), D))
        if b.kind == "pv":
            q = np.clip(q, 0, None) * a.sites.kwp_recorded.to_numpy()[None, :, None, None]
            q = np.where(a.geom["zenith"][None, :, di] < NIGHT_ZENITH, q, 0.0)
        else:
            q = np.clip(q, 0, None) * f["scale"][None]
        out[:, :, di] = q
    return out


# --------------------------------------------------------------------------------------- one window (contract)
def _to_steps(x: np.ndarray, H: int) -> np.ndarray:
    x = np.asarray(x, np.float64)
    if len(x) == H:
        return x
    if len(x) * 4 == H:  # hourly -> 15 min, same value for the 4 steps (ERA5 / forecasts are hourly)
        return np.repeat(x, 4)
    raise ValueError(f"weather has {len(x)} rows, horizon is {H} (need {H} or {H // 4})")


def parse_weather(weather: Any, H: int, cols: list[str] | None = None) -> dict[str, np.ndarray]:
    """DataFrame / dict with data.md or short names, or an array [T, C] in `cols` order (default ARRAY_CHANNELS)."""
    if weather is None:
        raise ValueError("window.weather is required")
    if isinstance(weather, pd.DataFrame):
        d = {c: weather[c].to_numpy() for c in weather.columns}
    elif isinstance(weather, Mapping):
        d = dict(weather)
    else:
        arr = np.asarray(weather, np.float64)
        cols = list(cols or ARRAY_CHANNELS)
        d = {c: arr[:, i] for i, c in enumerate(cols[: arr.shape[1]])}
    d = {_ALIASES.get(k, k): v for k, v in d.items()}
    out = {}
    for c in WEATHER_COLS:
        out[c] = _to_steps(d[c], H) if c in d else np.full(H, 0.0 if c == "snow_cm" else np.nan)
    if np.isnan(out["dni"]).all() or np.isnan(out["dhi"]).all():  # GHI only: split with Erbs later
        out["_need_split"] = np.ones(1)
    return out


def _past_days(ref: Any, H: int, unit_div: float) -> np.ndarray:
    """Past series ending at t0 (15-min) -> [7, H] aligned by time of day, NaN-padded if short."""
    r = np.asarray(ref if ref is not None else [], np.float64) / unit_div
    need = REF_DAYS * D
    if len(r) < need:
        r = np.concatenate([np.full(need - len(r), np.nan), r])
    return r[-need:].reshape(REF_DAYS, D)[:, :H]


class LGBMPredictor:
    """Implements contracts.Predictor for PV (M1 fallback) or load (M2 fallback); kind from the bundle."""

    model = "lgbm"

    def __init__(self, bundle: QuantileBundle):
        self.bundle = bundle
        self.kind = bundle.kind
        self.model_version = bundle.version

    @classmethod
    def load(cls, path: Path) -> "LGBMPredictor":
        return cls(QuantileBundle.load(path))

    def predict(self, asset: Mapping[str, Any], window: Window) -> Prediction:
        H = int(window.horizon)
        if H > D:
            raise ValueError("LGBMPredictor forecasts at most one day (96 steps) per window")
        t0 = pd.Timestamp(parse_utc(window.t0))
        ts = pd.date_range(t0, periods=H, freq="15min")
        wx = parse_weather(window.weather, H, weather_channels(window))
        if self.kind == "pv":
            q, extra = self._pv(asset, window, t0, ts, wx, H)
        else:
            q, extra = self._load(asset, window, t0, ts, wx, H)
        return Prediction(asset_id=str(asset.get("site_id", asset.get("asset_id", "?"))), model=self.model,
                          model_version=self.model_version, t0=window.t0, horizon=H, q10=q[0], q50=q[1], q90=q[2],
                          ood_score=extra.pop("ood_score", 0.0), member_std=0.0, extra={"kind": self.kind, **extra})

    def _pv(self, asset, window, t0, ts, wx, H):
        meta = recorded_meta(asset)
        g = solar_geometry(ts, meta["lat"], meta["lon"])
        if "_need_split" in wx:
            import pvlib
            sp = pvlib.irradiance.erbs(wx["ghi"], g["zenith"], g["doy"])
            wx["dni"], wx["dhi"] = np.asarray(sp["dni"]), np.asarray(sp["dhi"])
        pv = pv_power_kw(meta, g, wx["ghi"], wx["dni"], wx["dhi"], wx["temp_c"], wx["wind10_ms"])["ac_kw"]
        cs = clearsky_power_kw(meta, g)
        ref_ts = pd.date_range(t0 - pd.Timedelta(days=REF_DAYS), periods=REF_DAYS * D, freq="15min")
        gr = solar_geometry(ref_ts, meta["lat"], meta["lon"])
        ref_cs = clearsky_power_kw(meta, gr).reshape(REF_DAYS, D)[:, :H]
        ref = _past_days(window.reference, H, power_scale(window))  # pv_ac_w, W by default
        f = pv_features(weather={k: wx[k][None] for k in WEATHER_COLS}, geom={k: v[None] for k, v in g.items()},
                        pvlib_kw=pv[None], pvlib_cs_kw=cs[None], ref_kw=ref[None], ref_cs_kw=ref_cs[None],
                        kwp=[meta["kwp"]], tilt=[meta["tilt"]], azimuth=[meta["azimuth"]], lat=[meta["lat"]])
        q = self.bundle.predict(stack(f, self.bundle.features))
        q = np.clip(q, 0, None) * meta["kwp"]
        q = np.where(g["zenith"][None] < NIGHT_ZENITH, q, 0.0)
        day = cs > 0.02 * meta["kwp"]
        ref_day = np.broadcast_to(day[None], ref.shape) if ref.shape[1] == len(day) else np.ones_like(ref, bool)
        missing = float(np.isnan(ref[ref_day]).mean()) if ref_day.any() else 1.0
        return np.nan_to_num(q), {"ood_score": missing, "pvlib_kw": np.round(pv, 4).tolist()}

    def _load(self, asset, window, t0, ts, wx, H):
        hist = _past_days(window.history, H, power_scale(window))  # load_w, W by default
        hist_temp = (window.meta or {}).get("history_temp_c")
        ht = _past_days(hist_temp, H, 1.0) if hist_temp is not None else np.repeat(wx["temp_c"][None], 7, 0)
        cal = calendar(ts)
        site = {k: [asset.get(k, d)] for k, d in (("archetype", "base"), ("heating", "gas"), ("has_ac", False),
                                                  ("has_ev", False), ("has_electric_boiler", False),
                                                  ("works_from_home", False))}
        f = load_features(temp_c=wx["temp_c"][None], ghi=wx["ghi"][None], cal={k: v[None] for k, v in cal.items()},
                          hist_kw=hist[None], hist_temp_c=ht[None], site=site)
        q = np.clip(self.bundle.predict(stack(f, self.bundle.features)), 0, None) * f["scale"].ravel()[None]
        return np.nan_to_num(q), {"ood_score": float(np.isnan(hist).mean())}
