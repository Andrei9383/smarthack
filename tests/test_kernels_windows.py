"""K2 window gather (#51): Python reference == numpy fancy-index gather on a #10-layout memmap; Rust == Python
byte for byte."""

import numpy as np
import pytest

from wattsentry import kernels
from wattsentry.kernels.windows import window_gather_py

H, HH = 96, 14 * 96


def fancy(arr, rows, starts, offset, length, dtype):
    """The gather the model data path runs on #10's cache (#14 features.gather)."""
    idx = np.asarray(starts)[:, None] + np.arange(offset, offset + length)[None]
    return np.asarray(arr[np.asarray(rows)[:, None], idx], dtype)


@pytest.fixture(scope="module")
def cache(tmp_path_factory):
    """Small wattsentry-windows-v1-style arrays written as .npy and reopened as read-only memmaps."""
    d = tmp_path_factory.mktemp("win")
    rng = np.random.default_rng(0)
    S, L, T = 9, 3, 40 * H
    arrays = {"pv_ac": (rng.random((S, T)) * 8).astype(np.float16),
              "inv_status": rng.integers(0, 4, (S, T)).astype(np.int8),
              "weather_act": (rng.random((L, T, 7)) * 900).astype(np.float16),
              "geom": rng.random((L, T, 3)).astype(np.float32), "calendar": rng.random((T, 8)).astype(np.float32)}
    for k, a in arrays.items():
        np.save(d / f"{k}.npy", a)
    return {k: np.load(d / f"{k}.npy", mmap_mode="r") for k in arrays}


def batch(rng, n_rows, B=50, days=40):
    return rng.integers(0, n_rows, B), rng.integers(14, days, B) * H


def test_reference_is_fancy_gather(cache):
    assert kernels.get("window_gather", "python") is window_gather_py
    rng = np.random.default_rng(1)
    s, st = batch(rng, 9)
    for name, off, ln, dt in (("pv_ac", -HH, HH, np.float32), ("pv_ac", 0, H, None), ("inv_status", 0, H, None)):
        got = window_gather_py(cache[name], s, st, off, ln, dt)
        assert got.tobytes() == fancy(cache[name], s, st, off, ln, dt).tobytes()
    loc = s % 3
    got = window_gather_py(cache["weather_act"], loc, st, 0, H, np.float32)
    assert got.shape == (50, H, 7) and got.tobytes() == fancy(cache["weather_act"], loc, st, 0, H, np.float32).tobytes()


def test_reference_bounds_and_out(cache):
    a = cache["pv_ac"]
    for args in (([0], [10], -HH, HH), ([0], [a.shape[1] - 10], 0, H), ([9], [HH], 0, H)):  # numpy would wrap
        with pytest.raises(ValueError):
            window_gather_py(a, *args)
    out = np.zeros((2, H), np.float32)
    r = window_gather_py(a, [1, 2], [HH, 2 * HH], 0, H, np.float32, out=out)
    assert r is out and out.tobytes() == fancy(a, [1, 2], [HH, 2 * HH], 0, H, np.float32).tobytes()
    assert window_gather_py(a, [], [], 0, H, np.float32).shape == (0, H)


rust = pytest.mark.skipif(not kernels.rust_available(), reason="wattsentry._core not built")


def _both(*args, **kw):
    py, rs = (kernels.get("window_gather", b)(*args, **kw) for b in ("python", "rust"))
    assert py.shape == rs.shape and py.dtype == rs.dtype and py.tobytes() == rs.tobytes()
    return rs


@rust
def test_rust_parity_on_memmap(cache):
    rng = np.random.default_rng(2)
    s, st = batch(rng, 9, B=300)
    for name in ("pv_ac", "inv_status"):
        for off, ln in ((-HH, HH), (0, H), (-HH, HH + H), (-7 * H, 7 * H), (0, 0)):
            for dt in (None, np.float32) if name == "pv_ac" else (None,):
                _both(cache[name], s, st, off, ln, dt)
    loc = s % 3
    for name in ("weather_act", "geom"):
        _both(cache[name], loc, st, 0, H, None)
    _both(cache["weather_act"], loc, st, 0, H, np.float32)
    _both(cache["calendar"][None], np.zeros_like(st), st, 0, H, None)  # (T, 8) as one row


@rust
def test_rust_all_float16_bit_patterns():
    """float16 -> float32 must match numpy for every value: zeros, subnormals, inf, NaN payloads."""
    bits = np.arange(1 << 16, dtype=np.uint32).astype(np.uint16).reshape(64, 1024).view(np.float16)
    rows = np.repeat(np.arange(64), 4)
    starts = np.tile(np.arange(4) * 256, 64)
    rs = _both(bits, rows, starts, 0, 256, np.float32)
    assert rs.view(np.uint32).reshape(-1).tolist() == bits.reshape(-1).astype(np.float32).view(np.uint32).tolist()


@rust
def test_rust_out_buffer_and_errors(cache, tmp_path):
    a = cache["pv_ac"]
    rng = np.random.default_rng(3)
    s, st = batch(rng, 9, B=20)
    out = np.lib.format.open_memmap(tmp_path / "o.npy", mode="w+", dtype=np.float32, shape=(20, HH))
    g = kernels.get("window_gather", "rust")
    assert g(a, s, st, -HH, HH, np.float32, out=out) is out
    assert np.asarray(out).tobytes() == fancy(a, s, st, -HH, HH, np.float32).tobytes()
    with pytest.raises(ValueError):
        g(a, [0], [5], -HH, HH, np.float32)
    with pytest.raises(ValueError):
        kernels._load_core().window_gather_f16_f32(  # bounds are checked in Rust too
            a.view(np.uint16), np.array([0]), np.array([-HH]), 1, 0, H, np.zeros((1, H), np.float32))
    # a non-C-contiguous source falls back to the reference
    _both(np.asarray(a)[:, ::2], s, st // 2, -7 * H, 7 * H, np.float32)
    # float32 -> float16 is not a Rust path; still dispatches and matches
    _both(cache["geom"], s % 3, st, 0, H, np.float16)
