"""K2 window gather: copy fixed-length time windows out of a memmapped [rows, T(, C)] array (team/infra.md §2).

The source of truth is the numpy fancy-index gather the model data path uses on #10's window cache
(`sentinel-windows-v1`, written by ``sentinel.data.windows``), e.g. #14's ``features.gather``::

    idx = starts[:, None] + np.arange(offset, offset + length)[None]
    np.asarray(arr[rows[:, None], idx], dtype)

The Python reference here is exactly that expression. The Rust copy (``sentinel._core``) copies each window as one
contiguous row slice (numpy does not know the index is contiguous, so it builds a [B, length] int64 index and
gathers element by element), runs the windows in parallel with rayon, converts float16 -> float32 with a table
that reproduces numpy's conversion bit for bit (all 65536 values are tested), and can write straight into a
caller's buffer or memmap.

    from sentinel.kernels import window_gather
    hist = window_gather(src.array("pv_ac"), sites, starts, -14 * 96, 14 * 96, np.float32)   # [B, 1344]
    wx   = window_gather(src.array("weather_act"), loc, starts, 0, 96, np.float32)          # [B, 96, 7]

arr (R, T) or (R, T, C), any dtype, C-contiguous (np.load(..., mmap_mode="r") arrays are); rows, starts (B,) int;
the window for item b is arr[rows[b], starts[b] + offset : starts[b] + offset + length]. dtype None = arr.dtype.
Out-of-range windows raise ValueError (numpy would wrap negative indices; we refuse them in both backends).
The Rust path handles a dtype equal to arr.dtype (any dtype, byte copy) and float16 -> float32; any other
conversion falls back to the reference.
"""

from __future__ import annotations

from typing import Any

import numpy as np


def _prep(arr: Any, rows: Any, starts: Any, offset: int, length: int, dtype: Any, out: Any):
    if not isinstance(arr, np.ndarray):
        arr = np.asarray(arr)
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
        lo = starts.min() + offset
        hi = starts.max() + offset + length
        if lo < 0 or hi > T:
            raise ValueError(f"windows [{lo}, {hi}) fall outside the time axis [0, {T})")
    dtype = arr.dtype if dtype is None else np.dtype(dtype)
    shape = (len(rows), length, *arr.shape[2:])
    if out is not None:
        if not isinstance(out, np.ndarray) or out.shape != shape or out.dtype != dtype:
            raise ValueError(f"out must be an ndarray of shape {shape} and dtype {dtype}")
    return arr, rows, starts, offset, length, dtype, shape


def window_gather_py(arr: Any, rows: Any, starts: Any, offset: int, length: int, dtype: Any = None,
                     out: np.ndarray | None = None) -> np.ndarray:
    """Reference: ``np.asarray(arr[rows[:, None], starts[:, None] + offset + arange(length)], dtype)``."""
    arr, rows, starts, offset, length, dtype, _ = _prep(arr, rows, starts, offset, length, dtype, out)
    idx = starts[:, None] + np.arange(offset, offset + length)[None]
    res = np.asarray(arr[rows[:, None], idx], dtype)
    if out is None:
        return res
    out[...] = res
    return out


def window_gather_rust(arr: Any, rows: Any, starts: Any, offset: int, length: int, dtype: Any = None,
                       out: np.ndarray | None = None) -> np.ndarray:
    from sentinel import kernels

    arr, rows, starts, offset, length, dtype, shape = _prep(arr, rows, starts, offset, length, dtype, out)
    f16_to_f32 = arr.dtype == np.float16 and dtype == np.float32
    if (dtype != arr.dtype and not f16_to_f32) or not arr.flags.c_contiguous or arr.dtype.hasobject:
        return window_gather_py(arr, rows, starts, offset, length, dtype, out)
    res = out
    if res is None or not res.flags.c_contiguous:
        res = np.empty(shape, dtype)
    if res.size:
        R, T = arr.shape[:2]
        w = int(np.prod(arr.shape[2:], dtype=np.int64))  # elements per time step
        core = kernels._load_core()
        if f16_to_f32:
            src = arr.reshape(R, T * w).view(np.uint16)
            core.window_gather_f16_f32(src, rows, starts, w, offset * w, length * w, res.reshape(len(rows), -1))
        else:
            step = w * arr.dtype.itemsize  # bytes per time step
            src = arr.reshape(R, T * w).view(np.uint8)
            core.window_gather_bytes(src, rows, starts, step, offset * step, length * step,
                                     res.reshape(len(rows), -1).view(np.uint8))
    if out is not None and res is not out:
        out[...] = res
        return out
    return res
