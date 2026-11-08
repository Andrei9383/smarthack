"""Exact medians for the WindNBM label-free normal selection (card #136).

    from wattsentry.kernels import rolling_median, binned_median_mad
    r = rolling_median(x, window, min_periods)             # x (T, ...) -> float64 (T, ...), centred, NaN/inf skipped
    med, mad = binned_median_mad(b, x, ok, n_bins, min_n)  # b/x/ok (T, ...) -> (n_bins, ...) each

Both work down axis 0 (time), independently for every column.
``rolling_median_py`` is pandas ``DataFrame.rolling(window, center=True, min_periods=...).median()`` (what
``normal_select`` called once per turbine); ``binned_median_mad_py`` is ``normal_select``'s per-bin median + MAD of
``x[ok]`` grouped by the bin index ``b`` (NaN where a bin has fewer than ``min_n`` values), looped over columns.
The Rust copies (rust/wattsentry_core/src/median.rs) keep a sorted window per column / counting-sort the values by bin,
in parallel over blocks of columns, reading the row-major input in place. They pick the same order statistics and
average the two middle values the same way, so the outputs are bit-identical.
#136 profile of ``select_normal`` (117 turbines x 70,176 steps): pandas rolling medians 75 %, binned medians 5 %.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd


def _as2d(x: np.ndarray) -> np.ndarray:
    return x.reshape(x.shape[0], int(np.prod(x.shape[1:])))


def _prep_roll(x: Any, window: int, min_periods: int) -> tuple[np.ndarray, int, int]:
    x = np.asarray(x, dtype=np.float64)
    if x.ndim < 1:
        raise ValueError("x must have at least one axis (time)")
    window, min_periods = int(window), int(min_periods)
    if window < 1:
        raise ValueError(f"window must be >= 1, got {window}")
    if not 0 <= min_periods <= window:
        raise ValueError(f"min_periods must be in [0, window], got {min_periods}")
    return x, window, min_periods


def rolling_median_py(x: Any, window: int, min_periods: int) -> np.ndarray:
    x, window, min_periods = _prep_roll(x, window, min_periods)
    out = pd.DataFrame(_as2d(x)).rolling(window, center=True, min_periods=min_periods).median().to_numpy()
    return out.reshape(x.shape)


def rolling_median_rust(x: Any, window: int, min_periods: int) -> np.ndarray:
    from wattsentry import kernels

    x, window, min_periods = _prep_roll(x, window, min_periods)
    out = kernels._load_core().rolling_median(np.ascontiguousarray(_as2d(x)), window, min_periods)
    return out.reshape(x.shape)


def _prep_bins(b: Any, x: Any, ok: Any, n_bins: int, min_n: int):
    b, x, ok = np.asarray(b, dtype=np.int64), np.asarray(x, dtype=np.float64), np.asarray(ok, dtype=bool)
    if b.ndim < 1 or b.shape != x.shape or b.shape != ok.shape:
        raise ValueError(f"b, x, ok must have the same shape (T, ...), got {b.shape}, {x.shape}, {ok.shape}")
    n_bins, min_n = int(n_bins), int(min_n)
    if n_bins < 0:
        raise ValueError("n_bins must be >= 0")
    return b, x, ok, n_bins, max(min_n, 0)


def _binned_1d(b: np.ndarray, x: np.ndarray, ok: np.ndarray, n_bins: int, min_n: int):
    """Per-bin median and MAD of x over the steps in ok. b, x, ok are 1-D; returns (med, mad), NaN when < min_n."""
    med = np.full(n_bins, np.nan)
    mad = np.full(n_bins, np.nan)
    bb, xx = b[ok], x[ok]
    if bb.size == 0:
        return med, mad
    order = np.argsort(bb, kind="stable")
    bb, xx = bb[order], xx[order]
    cuts = np.flatnonzero(np.diff(bb)) + 1
    for seg_b, seg_x in zip(np.split(bb, cuts), np.split(xx, cuts)):
        if seg_x.size >= min_n:
            m = np.median(seg_x)
            med[seg_b[0]] = m
            mad[seg_b[0]] = np.median(np.abs(seg_x - m))
    return med, mad


def binned_median_mad_py(b: Any, x: Any, ok: Any, n_bins: int, min_n: int) -> tuple[np.ndarray, np.ndarray]:
    b, x, ok, n_bins, min_n = _prep_bins(b, x, ok, n_bins, min_n)
    if (ok & ((b < 0) | (b >= n_bins))).any():
        raise ValueError(f"b must be in [0, {n_bins}) where ok")
    b2, x2, ok2 = (np.ascontiguousarray(_as2d(a).T) for a in (b, x, ok))  # one contiguous row per column
    med = np.full((n_bins, b2.shape[0]), np.nan)
    mad = np.full((n_bins, b2.shape[0]), np.nan)
    for j in range(b2.shape[0]):
        med[:, j], mad[:, j] = _binned_1d(b2[j], x2[j], ok2[j], n_bins, min_n)
    shape = (n_bins, *b.shape[1:])
    return med.reshape(shape), mad.reshape(shape)


def binned_median_mad_rust(b: Any, x: Any, ok: Any, n_bins: int, min_n: int) -> tuple[np.ndarray, np.ndarray]:
    from wattsentry import kernels

    b, x, ok, n_bins, min_n = _prep_bins(b, x, ok, n_bins, min_n)
    c = np.ascontiguousarray
    med, mad = kernels._load_core().binned_median_mad(c(_as2d(b)), c(_as2d(x)), c(_as2d(ok)), n_bins, min_n)
    shape = (n_bins, *b.shape[1:])
    return med.reshape(shape), mad.reshape(shape)
