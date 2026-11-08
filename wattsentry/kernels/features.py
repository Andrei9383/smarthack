"""Fused feature build: the heavy part of ``wattsentry.model.features.pv_inputs`` / ``load_inputs`` / ``site_inputs``.

The source of truth is the numpy code in ``wattsentry.model.features`` (``pv_arrays``, ``load_arrays``,
``site_arrays``); the Python references here call it. The Rust copy (``wattsentry._core``) makes one pass per window
over its 14-day history and 96 target steps, windows in parallel. ``site_features`` without telemetry history
(degraded mode) always uses the reference.

Parity: sums replicate numpy's float32 reductions (pairwise along a row, sequential across days) and max/min/clip
propagate NaN like numpy, so every channel is bit-identical except the trig ones (cos/sin of zenith/azimuth, the
hour from atan2), which may differ by 1 ulp (numpy has its own SIMD float32 sin/cos).

    from wattsentry.kernels import load_features, pv_features, site_features
    seq, ref, summ, night = pv_features(wx, geom, pvclear, pvlib, hist_pv, kwp)  # [B,96,17] [B,168] [B,14,5] [B,96]
    seq, ref, summ, scale = load_features(wx, cal, hist_load)                     # [B,96,14] [B,168] [B,14,4] [B]
    seq, summ = site_features(raw)                                                # [B,96,18] [B,14,8]

wx [B,96,7], geom [B,96,3], cal [B,96,8], pvclear / pvlib [B,96], hist_pv [B, >=14*96], hist_load [B, 14*96], kwp [B];
any float dtype (cast to float32). Other shapes fall back to the reference.
"""

from __future__ import annotations

from typing import Any

import numpy as np

H, HH = 96, 14 * 96


def _f32(*a: Any) -> list[np.ndarray]:
    return [np.ascontiguousarray(x, np.float32) for x in a]


def pv_features_py(wx: Any, geom: Any, pvclear: Any, pvlib: Any, hist_pv: Any, kwp: Any):
    """Reference: ``wattsentry.model.features.pv_arrays``."""
    from wattsentry.model.features import pv_arrays

    return pv_arrays(*(np.asarray(x) for x in (wx, geom, pvclear, pvlib, hist_pv, kwp)))


def load_features_py(wx: Any, cal: Any, hist_load: Any):
    """Reference: ``wattsentry.model.features.load_arrays``."""
    from wattsentry.model.features import load_arrays

    return load_arrays(*(np.asarray(x) for x in (wx, cal, hist_load)))


def pv_features_rust(wx: Any, geom: Any, pvclear: Any, pvlib: Any, hist_pv: Any, kwp: Any):
    from wattsentry import kernels

    wx, geom, pvclear, pvlib, hist_pv, kwp = _f32(wx, geom, pvclear, pvlib, hist_pv, kwp)
    B = len(kwp)
    if (wx.shape, geom.shape, pvclear.shape, pvlib.shape) != ((B, H, 7), (B, H, 3), (B, H), (B, H)) \
            or hist_pv.ndim != 2 or hist_pv.shape[0] != B or hist_pv.shape[1] < HH or not B:
        return pv_features_py(wx, geom, pvclear, pvlib, hist_pv, kwp)
    seq, ref = np.empty((B, H, 17), np.float32), np.empty((B, 168), np.float32)
    summ, night = np.empty((B, 14, 5), np.float32), np.empty((B, H), bool)
    kernels._load_core().pv_features(wx.reshape(B, -1), geom.reshape(B, -1), pvclear, pvlib, hist_pv, kwp,
                                     seq.reshape(B, -1), ref, summ.reshape(B, -1), night.view(np.uint8))
    return seq, ref, summ, night


def load_features_rust(wx: Any, cal: Any, hist_load: Any):
    from wattsentry import kernels

    wx, cal, hist_load = _f32(wx, cal, hist_load)
    B = len(hist_load)
    if (wx.shape, cal.shape, hist_load.shape) != ((B, H, 7), (B, H, 8), (B, HH)) or not B:
        return load_features_py(wx, cal, hist_load)
    seq, ref = np.empty((B, H, 14), np.float32), np.empty((B, 168), np.float32)
    summ, scale = np.empty((B, 14, 4), np.float32), np.empty(B, np.float32)
    kernels._load_core().load_features(wx.reshape(B, -1), cal.reshape(B, -1), hist_load, seq.reshape(B, -1), ref,
                                       summ.reshape(B, -1), scale)
    return seq, ref, summ, scale


def site_features_py(raw: dict):
    """Reference: ``wattsentry.model.features.site_arrays`` (M4 SiteHealth) -> seq [B,96,18], summ [B,14,8]."""
    from wattsentry.model.features import site_arrays

    return site_arrays(raw)


def _i8(a: Any, hist: bool) -> np.ndarray:
    """Status codes as int8 with the reference's comparisons kept: seq casts to int64 first, history compares raw."""
    a = np.asarray(a)
    if a.dtype == np.int8:
        return np.ascontiguousarray(a)
    return np.where(a == 3, 3, 0).astype(np.int8) if hist else np.clip(a.astype(np.int64), -1, 4).astype(np.int8)


def site_features_rust(raw: dict):
    from wattsentry import kernels

    B = len(raw["kwp_recorded"])
    if raw.get("hist_volt") is None or not B:  # degraded mode (no telemetry history): reference
        return site_features_py(raw)
    pv, m2, pvlib, clear, volt, soc, batt, load = _f32(*(raw[k] for k in ("pv", "m2", "pvlib", "pvclear", "volt",
                                                                          "soc", "batt", "load")))
    m1, geom, hpv, hload, hv, hso, hb, kwp, bkw, bkwh, hasb = _f32(*(raw[k] for k in (
        "m1", "geom", "hist_pv", "hist_load", "hist_volt", "hist_soc", "hist_batt", "kwp_recorded", "battery_kw",
        "battery_kwh", "has_batt")))
    valid = _f32(raw.get("valid", np.ones((B, H), np.float32)))[0]
    st, hst = _i8(raw["status"], False), _i8(raw["hist_status"], True)
    step = (pv, m2, pvlib, clear, volt, st, soc, batt, load, valid)
    if any(a.shape != (B, H) for a in step) or (m1.shape, geom.shape) != ((B, H, 3), (B, H, 3)) \
            or any(a.ndim != 2 or a.shape[0] != B or a.shape[1] < HH for a in (hpv, hv)) \
            or any(a.shape != hv.shape for a in (hst, hso, hb)) or hload.ndim != 2 or hload.shape[0] != B \
            or not hload.shape[1] or any(a.shape != (B,) for a in (kwp, bkw, bkwh, hasb)):
        return site_features_py(raw)
    seq, summ = np.empty((B, H, 18), np.float32), np.empty((B, 14, 8), np.float32)
    kernels._load_core().site_features(pv, m1.reshape(B, -1), m2, geom.reshape(B, -1), pvlib, clear, volt, st, soc,
                                       batt, load, valid, hpv, hload, hv, hst, hso, hb, kwp, bkw, bkwh, hasb,
                                       seq.reshape(B, -1), summ.reshape(B, -1))
    return seq, summ
