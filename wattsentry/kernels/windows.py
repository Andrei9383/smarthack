"""Window gather: copy fixed-length time windows out of a memmapped [rows, T(, C)] array (the window cache).

Reference (source of truth) = the numpy fancy-index gather of the model data path (``features.gather``)::

    np.asarray(arr[rows[:, None], starts[:, None] + np.arange(offset, offset + length)[None]], dtype)

The Rust copy (``wattsentry._core``) copies each window as one contiguous slice, in parallel, converting
float16 -> float32 bit-identically to numpy, optionally straight into ``out``. It handles dtype == arr.dtype and
float16 -> float32; other conversions fall back to the reference. Out-of-range windows raise ValueError in both
backends (numpy would wrap negative indices).
"""

from __future__ import annotations

from typing import Any

import numpy as np


def _prep(arr: Any, rows: Any, starts: Any, offset: int, length: int, dtype: Any, out: Any):
    arr = arr if isinstance(arr, np.ndarray) else np.asarray(arr)
    if arr.ndim not in (2, 3):
        raise ValueError(f"arr must be (R, T) or (R, T, C), got shape {arr.shape}")
    rows = np.asarray(rows, dtype=np.int64).reshape(-1)
    starts = np.asarray(starts, dtype=np.int64).reshape(-1)
    if rows.shape != starts.shape:
        raise ValueError(f"rows {rows.shape} and starts {starts.shape} must have the same length")
    offset, length = int(offset), int(length)
    if length < 0:
        raise ValueError("length must be >= 0")
    R, T = arr.shape[:2]
    if len(rows):
        if rows.min() < 0 or rows.max() >= R:
            raise ValueError(f"rows must be in [0, {R})")
        lo, hi = starts.min() + offset, starts.max() + offset + length
        if lo < 0 or hi > T:
            raise ValueError(f"windows [{lo}, {hi}) fall outside the time axis [0, {T})")
    dtype = arr.dtype if dtype is None else np.dtype(dtype)
    shape = (len(rows), length, *arr.shape[2:])
    if out is not None and (not isinstance(out, np.ndarray) or out.shape != shape or out.dtype != dtype):
        raise ValueError(f"out must be an ndarray of shape {shape} and dtype {dtype}")
    return arr, rows, starts, offset, length, dtype, shape


def window_gather_py(arr: Any, rows: Any, starts: Any, offset: int, length: int, dtype: Any = None,
                     out: np.ndarray | None = None) -> np.ndarray:
    arr, rows, starts, offset, length, dtype, _ = _prep(arr, rows, starts, offset, length, dtype, out)
    res = np.asarray(arr[rows[:, None], starts[:, None] + np.arange(offset, offset + length)[None]], dtype)
    if out is None:
        return res
    out[...] = res
    return out


def window_gather_rust(arr: Any, rows: Any, starts: Any, offset: int, length: int, dtype: Any = None,
                       out: np.ndarray | None = None) -> np.ndarray:
    from wattsentry import kernels

    arr, rows, starts, offset, length, dtype, shape = _prep(arr, rows, starts, offset, length, dtype, out)
    f16_to_f32 = arr.dtype == np.float16 and dtype == np.float32
    if (dtype != arr.dtype and not f16_to_f32) or not arr.flags.c_contiguous or arr.dtype.hasobject:
        return window_gather_py(arr, rows, starts, offset, length, dtype, out)
    res = out if out is not None and out.flags.c_contiguous else np.empty(shape, dtype)
    if res.size:
        R, T = arr.shape[:2]
        w = int(np.prod(arr.shape[2:], dtype=np.int64))  # elements per time step
        core = kernels._load_core()
        if f16_to_f32:
            core.window_gather_f16_f32(arr.reshape(R, T * w).view(np.uint16), rows, starts, w, offset * w,
                                       length * w, res.reshape(len(rows), -1))
        else:
            step = w * arr.dtype.itemsize  # bytes per time step
            core.window_gather_bytes(arr.reshape(R, T * w).view(np.uint8), rows, starts, step, offset * step,
                                     length * step, res.reshape(len(rows), -1).view(np.uint8))
    if out is not None and res is not out:
        out[...] = res
        return out
    return res
