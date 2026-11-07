"""K4 causal NaN-aware rolling mean along axis 0 (M6 WindNBM classifier features, card #95).

    from sentinel.kernels import rolling_mean
    m = rolling_mean(x, w)    # x (T, ...) -> float64 (T, ...): mean of the finite values in x[t-w+1 : t+1], 0 if none

The reference is the ``rmean`` of ``sentinel.model.wind.features`` (cumsum differences) moved here verbatim; the
Rust copy ``sentinel._core.rolling_mean`` (rust/sentinel_core/src/rolling.rs) runs the same prefix sums in one
pass and matches it bit for bit. ``clf_features`` calls it ~40 times per park (>50 % of the stage, #95 profile).
"""

from __future__ import annotations

from typing import Any

import numpy as np


def _prep(x: Any, w: int) -> tuple[np.ndarray, int]:
    x = np.asarray(x, dtype=np.float64)
    if x.ndim < 1:
        raise ValueError("x must have at least one axis (time)")
    w = int(w)
    if w < 0:
        raise ValueError(f"w must be >= 0, got {w}")
    return x, w


def rolling_mean_py(x: Any, w: int) -> np.ndarray:
    x, w = _prep(x, w)
    ok = np.isfinite(x)
    cs = np.cumsum(np.where(ok, x, 0.0), axis=0)
    cn = np.cumsum(ok, axis=0)
    cs = np.concatenate([np.zeros((1, *x.shape[1:])), cs])
    cn = np.concatenate([np.zeros((1, *x.shape[1:])), cn])
    idx = np.arange(1, x.shape[0] + 1)
    lo = np.clip(idx - w, 0, None)
    s = cs[idx] - cs[lo]
    n = cn[idx] - cn[lo]
    return np.where(n > 0, s / np.maximum(n, 1), 0.0)


def rolling_mean_rust(x: Any, w: int) -> np.ndarray:
    from sentinel import kernels

    x, w = _prep(x, w)
    x2 = np.ascontiguousarray(x.reshape(x.shape[0], int(np.prod(x.shape[1:]))))
    return kernels._load_core().rolling_mean(x2, w).reshape(x.shape)
