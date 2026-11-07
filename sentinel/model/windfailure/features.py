"""Daily per turbine-component features from 15-min temperature residuals (numpy only, causal).

For day d, turbine j, component c (all windows end at the end of day d, so nothing looks ahead):
- residual stats of the day: mean, std, p90, mean at high load (load > 0.5), coverage;
- the turbine's OWN baseline: median of the daily mean residual over days [d−60, d−21] (lagged 21 days, the longest
  degradation drift, so a drift in progress is not absorbed); `dev` = today's mean − baseline, `z` = dev / MAD;
- EWMA of dev (half-life 3 and 7 days), OLS slopes of dev over 3 / 7 / 14 days, 3- and 7-day means;
- exceedance counts: 15-min steps with residual − baseline > max(1 °C, 3 × the baseline daily std), per day and
  summed over 3 / 7 / 14 days;
- context: dev minus the park median dev of the same component that day (common-mode model error), the max dev of
  the turbine's other components, daily load / ambient / producing fraction, start and stop counts (1 and 7 days),
  turbine class, component id.
`dev` alone is the fixed-threshold baseline's score.
"""

from __future__ import annotations

import warnings

import numpy as np

from . import COMPONENTS

STEPS = 96
BASE_LAG_D = 21
BASE_WIN_D = 60
BASE_MIN_D = 7
FEATURES = [
    "comp_id", "is_large",
    "r_mean", "r_std", "r_p90", "r_hi", "cov",
    "base", "dev", "dev_hi", "z",
    "ewm3", "ewm7", "slope3", "slope7", "slope14", "dev_m3", "dev_m7",
    "exc1", "exc3", "exc7", "exc14",
    "dev_park", "dev_other",
    "load", "amb", "prod_frac", "starts", "stops", "stops7",
]
BASELINE_SCORE = "dev"


def _roll_sum(x: np.ndarray, w: int) -> np.ndarray:
    """Causal rolling sum along axis 0 (NaN treated as 0)."""
    c = np.cumsum(np.nan_to_num(x), axis=0)
    c = np.concatenate([np.zeros((1, *x.shape[1:])), c])
    i = np.arange(1, x.shape[0] + 1)
    return c[i] - c[np.clip(i - w, 0, None)]


def _roll_mean(x: np.ndarray, w: int) -> np.ndarray:
    n = _roll_sum(np.isfinite(x).astype(np.float64), w)
    with np.errstate(invalid="ignore", divide="ignore"):
        return np.where(n > 0, _roll_sum(x, w) / np.maximum(n, 1), np.nan)


def _roll_slope(y: np.ndarray, w: int) -> np.ndarray:
    """Causal OLS slope per day over the last w days (NaN-aware; NaN with < 3 points)."""
    ok = np.isfinite(y).astype(np.float64)
    x = np.arange(y.shape[0], dtype=np.float64).reshape(-1, *([1] * (y.ndim - 1)))
    x = x - x.mean()
    yy = np.nan_to_num(y)
    n = _roll_sum(ok, w)
    sx = _roll_sum(ok * x, w)
    sy = _roll_sum(yy, w)
    sxy = _roll_sum(yy * x, w)
    sxx = _roll_sum(ok * x * x, w)
    den = n * sxx - sx * sx
    with np.errstate(invalid="ignore", divide="ignore"):
        return np.where((n >= 3) & (den > 1e-9), (n * sxy - sx * sy) / np.where(den > 1e-9, den, 1), np.nan)


def _ewm(x: np.ndarray, halflife: float) -> np.ndarray:
    a = 1 - 0.5 ** (1 / halflife)
    out = np.full(x.shape, np.nan)
    s = np.full(x.shape[1:], np.nan)
    for d in range(x.shape[0]):
        v = x[d]
        s = np.where(np.isfinite(v), np.where(np.isfinite(s), (1 - a) * s + a * v, v), s)
        out[d] = s
    return out


def _lagged_median(x: np.ndarray, lag: int = BASE_LAG_D, win: int = BASE_WIN_D, min_n: int = BASE_MIN_D):
    """Median over days [d − win, d − lag] per element (NaN with < min_n finite days)."""
    out = np.full(x.shape, np.nan)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        for d in range(lag, x.shape[0]):
            sl = x[max(0, d - win) : d - lag + 1]
            n = np.isfinite(sl).sum(0)
            out[d] = np.where(n >= min_n, np.nanmedian(sl, axis=0), np.nan)
    return out


def daily_features(r: np.ndarray, power: np.ndarray, amb: np.ndarray, rated: np.ndarray, park_of: np.ndarray,
                   is_large: np.ndarray) -> np.ndarray:
    """r [T, N, C] residuals (°C), power [T, N] kW, amb [T, N] °C → X [D, N, C, F] float32 (F = len(FEATURES))."""
    T, N, C = r.shape
    D = T // STEPS
    R = r[: D * STEPS].reshape(D, STEPS, N, C).astype(np.float64)
    p = power[: D * STEPS].reshape(D, STEPS, N).astype(np.float64) / np.asarray(rated, np.float64)[None, None]
    A = amb[: D * STEPS].reshape(D, STEPS, N).astype(np.float64)
    out = {}
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        fin = np.isfinite(R)
        out["cov"] = fin.mean(1)
        rm = np.nanmean(R, 1)
        out["r_mean"] = rm
        out["r_std"] = np.nanstd(R, 1)
        out["r_p90"] = np.nanpercentile(R, 90, axis=1)
        hi = (p > 0.5)[..., None]
        rhi = np.nanmean(np.where(hi, R, np.nan), 1)
        out["r_hi"] = rhi
        base = _lagged_median(rm)
        base_hi = _lagged_median(rhi)
        base_sd = _lagged_median(out["r_std"])
        absdev = np.abs(rm - _lagged_median(rm, lag=0, win=BASE_WIN_D))  # spread of the daily mean around its median
        mad = np.maximum(1.4826 * _lagged_median(absdev), 0.1)
        dev = rm - base
        out["base"] = base
        out["dev"] = dev
        out["dev_hi"] = rhi - base_hi
        out["z"] = dev / mad
        out["ewm3"] = _ewm(dev, 3)
        out["ewm7"] = _ewm(dev, 7)
        out["slope3"] = _roll_slope(dev, 3)
        out["slope7"] = _roll_slope(dev, 7)
        out["slope14"] = _roll_slope(dev, 14)
        out["dev_m3"] = _roll_mean(dev, 3)
        out["dev_m7"] = _roll_mean(dev, 7)
        thr = np.maximum(1.0, 3 * np.nan_to_num(base_sd, nan=1.0 / 3))
        exc = ((R - base[:, None]) > thr[:, None]).sum(1).astype(np.float64)
        exc[~np.isfinite(base)] = np.nan
        out["exc1"] = exc
        for w in (3, 7, 14):
            out[f"exc{w}"] = _roll_sum(exc, w)
        park_med = np.full(dev.shape, np.nan)
        for pk in np.unique(park_of):
            cols = park_of == pk
            park_med[:, cols] = np.nanmedian(dev[:, cols], axis=1, keepdims=True)
        out["dev_park"] = dev - park_med
        other = np.full(dev.shape, np.nan)
        for c in range(C):
            other[..., c] = np.nanmax(np.delete(dev, c, axis=2), axis=2)
        out["dev_other"] = other
        load = np.clip(p, 0, 1.01)
        out["load"] = np.nanmean(load, 1)[..., None]
        out["amb"] = np.nanmean(A, 1)[..., None]
        prod = p > 0.02
        okp = np.isfinite(p)
        out["prod_frac"] = np.nanmean(np.where(okp, prod, np.nan), 1)[..., None]
        pp = np.concatenate([prod.reshape(-1, N)[:1], prod.reshape(-1, N)])  # step before the first = itself
        ok2 = np.concatenate([okp.reshape(-1, N)[:1], okp.reshape(-1, N)])
        trans_ok = ok2[1:] & ok2[:-1]
        starts = (pp[1:] & ~pp[:-1] & trans_ok).reshape(D, STEPS, N).sum(1).astype(np.float64)
        stops = (~pp[1:] & pp[:-1] & trans_ok).reshape(D, STEPS, N).sum(1).astype(np.float64)
        out["starts"] = starts[..., None]
        out["stops"] = stops[..., None]
        out["stops7"] = _roll_sum(stops, 7)[..., None]
    out["comp_id"] = np.broadcast_to(np.arange(C, dtype=np.float64)[None, None], (D, N, C))
    out["is_large"] = np.broadcast_to(np.asarray(is_large, np.float64)[None, :, None], (D, N, C))
    X = np.empty((D, N, C, len(FEATURES)), np.float32)
    for k, name in enumerate(FEATURES):
        X[..., k] = np.broadcast_to(out[name], (D, N, C))
    return X


def feature_index(name: str) -> int:
    return FEATURES.index(name)


__all__ = ["FEATURES", "BASELINE_SCORE", "daily_features", "feature_index", "COMPONENTS"]
