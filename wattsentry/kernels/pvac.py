"""pv_ac_batch: pvlib AC power for many sites under one locality's weather, for the window cache build.

Source of truth: ``wattsentry.model.physics`` (``poa`` + ``ac_power_kw``, the code the ONNX predictor runs). The
reference ``pv_ac_batch_py`` is exactly ``physics.pvlib_expected_kw`` / ``pvlib_clear_kw`` with the site metadata as
[n, 1] columns broadcast against the [T] weather, which gives the same bytes as one call per site. The Rust copy
``wattsentry._core.pv_ac_batch`` fuses the ~25 numpy passes into one loop per site (rayon over sites).

    kw = pv_ac_batch(kwp, tilt, azimuth, inverter_kw, ghi, dni, dhi, temp, wind, zenith, sun_azimuth)

Per-site arrays are [n] (kW, degrees); per-step arrays are [T] (W/m2, degC, m/s, degrees). ``temp`` may be a scalar
and ``wind`` None (no wind cooling), as in ``pvlib_clear_kw``. Returns float64 [n, T] kW.
"""

from __future__ import annotations

from typing import Any

import numpy as np


def _site(a: Any) -> np.ndarray:
    return np.ascontiguousarray(np.asarray(a, dtype=np.float64).reshape(-1))


def pv_ac_batch_py(kwp: Any, tilt: Any, azimuth: Any, inverter_kw: Any, ghi: Any, dni: Any, dhi: Any, temp: Any,
                   wind: Any, zenith: Any, sun_azimuth: Any) -> np.ndarray:
    from wattsentry.model import physics as ph

    kwp, tilt, azimuth, inverter_kw = (_site(a)[:, None] for a in (kwp, tilt, azimuth, inverter_kw))
    g, _ = ph.poa(tilt, azimuth, np.asarray(zenith, np.float64), np.asarray(sun_azimuth, np.float64),
                  np.asarray(dni, np.float64), np.asarray(ghi, np.float64), np.asarray(dhi, np.float64))
    t = temp if np.ndim(temp) == 0 else np.asarray(temp, np.float64)
    return np.asarray(ph.ac_power_kw(g, t, kwp, inverter_kw, wind=wind), dtype=np.float64)


def pv_ac_batch_rust(kwp: Any, tilt: Any, azimuth: Any, inverter_kw: Any, ghi: Any, dni: Any, dhi: Any, temp: Any,
                     wind: Any, zenith: Any, sun_azimuth: Any) -> np.ndarray:
    from pvlib import tools

    from wattsentry import kernels
    from wattsentry.model import physics as ph

    tilt, azimuth, kwp, inverter_kw = (_site(a) for a in (tilt, azimuth, kwp, inverter_kw))
    ghi, dni, dhi, zen, saz = (np.ascontiguousarray(np.asarray(a, dtype=np.float64).reshape(-1))
                               for a in (ghi, dni, dhi, zenith, sun_azimuth))
    T = len(ghi)
    temp = np.full(T, float(temp)) if np.ndim(temp) == 0 else np.ascontiguousarray(np.asarray(temp, np.float64))
    wind = None if wind is None else np.ascontiguousarray(np.asarray(wind, np.float64))
    # trig of the per-site and per-step angles exactly as pvlib computes it (numpy); only the per-element
    # cos(sun_az - az), arccos and cos(aoi) run in Rust
    return kernels._load_core().pv_ac_batch(
        tools.cosd(tilt), tools.sind(tilt), azimuth, kwp, inverter_kw, ghi, dni, dhi, temp,
        np.ascontiguousarray(tools.cosd(zen)), np.ascontiguousarray(tools.sind(zen)), saz, wind,
        0.2, ph.DEFAULT_LOSSES, ph.DEFAULT_GAMMA, ph.NOCT_RISE, ph.INV_ETA)
