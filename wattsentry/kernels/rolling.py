"""Causal NaN-aware rolling mean along axis 0 (M6 WindNBM classifier features).

    from wattsentry.kernels import rolling_mean
    m = rolling_mean(x, w)    # x (T, ...) -> float64 (T, ...): mean of the finite values in x[t-w+1 : t+1], 0 if none

The reference ``rolling_mean_py`` uses cumsum differences; the Rust copy ``wattsentry._core.rolling_mean``
(rust/wattsentry_core/src/rolling.rs) runs the same prefix sums in one pass and matches it bit for bit.
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
    from wattsentry import kernels

    x, w = _prep(x, w)
    x2 = np.ascontiguousarray(x.reshape(x.shape[0], int(np.prod(x.shape[1:]))))
    return kernels._load_core().rolling_mean(x2, w).reshape(x.shape)
