"""Kernel dispatch: pure-numpy reference ``<name>_py`` (source of truth) + optional Rust copy in ``wattsentry._core``.

WATTSENTRY_KERNELS=auto (default: Rust if ``wattsentry._core`` imports, else Python, logged once) | python | rust
(raise ImportError if the extension is missing).
"""

from __future__ import annotations

import logging
import os
from typing import Any, Callable

import numpy as np

from wattsentry.kernels.battery_resim import battery_resim_py, battery_resim_rust
from wattsentry.kernels.cusum import cusum_two_sided_py, cusum_two_sided_rust
from wattsentry.kernels.features import (load_features_py, load_features_rust, pv_features_py, pv_features_rust,
                                       site_features_py, site_features_rust)
from wattsentry.kernels.median import (binned_median_mad_py, binned_median_mad_rust, rolling_median_py,
                                     rolling_median_rust)
from wattsentry.kernels.pvac import pv_ac_batch_py, pv_ac_batch_rust
from wattsentry.kernels.rolling import rolling_mean_py, rolling_mean_rust
from wattsentry.kernels.rollout import battery_rollout_py, battery_rollout_rust
from wattsentry.kernels.scatter import scatter_telemetry_py, scatter_telemetry_rust
from wattsentry.kernels.selfcons import self_consumption_py, self_consumption_rust
from wattsentry.kernels.strategies import battery_strategies_py, battery_strategies_rust
from wattsentry.kernels.windows import window_gather_py, window_gather_rust

log = logging.getLogger("wattsentry.kernels")
_MODES = ("auto", "python", "rust")
_core: Any = None
_core_error: str | None = None
_core_loaded = _fallback_logged = False
_REGISTRY: dict[str, tuple[Callable[..., Any], Callable[..., Any] | None]] = {}


def _load_core() -> Any:
    global _core, _core_error, _core_loaded
    if not _core_loaded:
        _core_loaded = True
        try:
            from wattsentry import _core as core  # type: ignore[attr-defined]
            _core = core
        except Exception as e:  # ImportError, or a broken build
            _core_error = f"{type(e).__name__}: {e}"
    return _core


def rust_available() -> bool:
    return _load_core() is not None


def backend() -> str:
    """The backend ``get`` dispatches to right now: ``"rust"`` or ``"python"``."""
    global _fallback_logged
    mode = os.environ.get("WATTSENTRY_KERNELS", "auto").strip().lower() or "auto"
    if mode not in _MODES:
        raise ValueError(f"WATTSENTRY_KERNELS={mode!r}; expected one of {_MODES}")
    if mode == "python":
        return "python"
    if rust_available():
        return "rust"
    if mode == "rust":
        raise ImportError(f"WATTSENTRY_KERNELS=rust but wattsentry._core is not available ({_core_error})")
    if not _fallback_logged:
        _fallback_logged = True
        log.info("wattsentry._core not available (%s); kernels use the Python reference", _core_error)
    return "python"


def register(name: str, python: Callable[..., Any], rust: Callable[..., Any] | None = None) -> None:
    """``rust`` is a wrapper with the same signature as ``python`` (None = Python only)."""
    _REGISTRY[name] = (python, rust)


def names() -> list[str]:
    return sorted(_REGISTRY)


def get(name: str, backend_: str | None = None) -> Callable[..., Any]:
    """Dispatched callable for ``name``; ``backend_`` forces "python" or "rust" (tests, bench)."""
    if name not in _REGISTRY:
        raise KeyError(f"unknown kernel {name!r}; registered: {names()}")
    py, rs = _REGISTRY[name]
    be = backend_ or backend()
    if be == "python":
        return py
    if be != "rust":
        raise ValueError(f"backend must be 'python' or 'rust', got {be!r}")
    if rs is None or not rust_available():
        raise ImportError(f"no Rust implementation of {name!r} available ({_core_error or 'not built'})")
    return rs


register("cusum_two_sided", cusum_two_sided_py, cusum_two_sided_rust)
register("battery_rollout", battery_rollout_py, battery_rollout_rust)
register("window_gather", window_gather_py, window_gather_rust)
register("rolling_mean", rolling_mean_py, rolling_mean_rust)
register("scatter_telemetry", scatter_telemetry_py, scatter_telemetry_rust)
register("pv_features", pv_features_py, pv_features_rust)
register("load_features", load_features_py, load_features_rust)
register("site_features", site_features_py, site_features_rust)
register("battery_strategies", battery_strategies_py, battery_strategies_rust)
register("self_consumption", self_consumption_py, self_consumption_rust)
register("pv_ac_batch", pv_ac_batch_py, pv_ac_batch_rust)
register("rolling_median", rolling_median_py, rolling_median_rust)
register("binned_median_mad", binned_median_mad_py, binned_median_mad_rust)
register("battery_resim", battery_resim_py, battery_resim_rust)


# ---- dispatched entry points (each module documents its reference and shapes)

def cusum_two_sided(x: Any, k: float, h: float,
                    cap: float | None = None) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Two-sided CUSUM; see ``cusum.cusum_two_sided_py``."""
    return get("cusum_two_sided")(x, k, h, cap)


def battery_rollout(req: Any, soc0: Any, cap: Any, pmax: Any, eta_c: Any, eta_d: Any, e_lo: Any, e_hi: Any,
                    dt_h: float = 0.25) -> tuple[np.ndarray, np.ndarray]:
    """Battery rollout with clipping; see ``rollout`` (reference = wattsentry.opt.commit.rollout)."""
    return get("battery_rollout")(req, soc0, cap, pmax, eta_c, eta_d, e_lo, e_hi, dt_h)


def window_gather(arr: Any, rows: Any, starts: Any, offset: int, length: int, dtype: Any = None,
                  out: np.ndarray | None = None) -> np.ndarray:
    """Fixed-length time-window gather; see ``windows`` (reference = numpy fancy-index gather)."""
    return get("window_gather")(arr, rows, starts, offset, length, dtype, out)


def rolling_mean(x: Any, w: int) -> np.ndarray:
    """Causal NaN-aware rolling mean along axis 0; see ``rolling``."""
    return get("rolling_mean")(x, w)


def scatter_telemetry(ti: Any, tj: Any, sk: Any, v: Any, is_code: Any, scale: Any,
                      T: int) -> tuple[np.ndarray, np.ndarray]:
    """Long telemetry -> [sensor, step, turbine] bin means + last code per bin; see ``scatter``."""
    return get("scatter_telemetry")(ti, tj, sk, v, is_code, scale, T)


def pv_features(wx: Any, geom: Any, pvclear: Any, pvlib: Any, hist_pv: Any, kwp: Any) -> tuple[np.ndarray, ...]:
    """PV feature build -> (seq, ref, summ, night); reference = wattsentry.model.features.pv_arrays."""
    return get("pv_features")(wx, geom, pvclear, pvlib, hist_pv, kwp)


def load_features(wx: Any, cal: Any, hist_load: Any) -> tuple[np.ndarray, ...]:
    """Load feature build -> (seq, ref, summ, scale); reference = wattsentry.model.features.load_arrays."""
    return get("load_features")(wx, cal, hist_load)


def site_features(raw: dict) -> tuple[np.ndarray, np.ndarray]:
    """M4 site feature build -> (seq, summ); reference = wattsentry.model.features.site_arrays."""
    return get("site_features")(raw)


def battery_strategies(*args: Any, **kw: Any) -> tuple[np.ndarray, np.ndarray]:
    """Per-home battery strategies; see ``strategies``."""
    return get("battery_strategies")(*args, **kw)


def self_consumption(p: Any, pv_w: Any, load_w: Any, dt_h: float,
                     e0_kwh: Any = None) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Fleet battery rule -> (batt_w, soc, e_end_kwh); reference =
    wattsentry.sim.fleet.battery.simulate_self_consumption (same signature)."""
    return get("self_consumption")(p, pv_w, load_w, dt_h, e0_kwh)


def pv_ac_batch(kwp: Any, tilt: Any, azimuth: Any, inverter_kw: Any, ghi: Any, dni: Any, dhi: Any, temp: Any,
                wind: Any, zenith: Any, sun_azimuth: Any) -> np.ndarray:
    """pvlib AC kW [n, T] for n sites under one weather series; reference = wattsentry.model.physics."""
    return get("pv_ac_batch")(kwp, tilt, azimuth, inverter_kw, ghi, dni, dhi, temp, wind, zenith, sun_azimuth)


def rolling_median(x: Any, window: int, min_periods: int) -> np.ndarray:
    """Centred NaN/inf-skipping rolling median along axis 0; see ``median`` (reference = pandas)."""
    return get("rolling_median")(x, window, min_periods)


def binned_median_mad(b: Any, x: Any, ok: Any, n_bins: int, min_n: int) -> tuple[np.ndarray, np.ndarray]:
    """Per-column, per-bin median + MAD of x[ok] grouped by b; see ``median.binned_median_mad_py``."""
    return get("binned_median_mad")(b, x, ok, n_bins, min_n)


def battery_resim(surplus: Any, soc0: Any, batt0: Any, cf: Any, pf: Any, start: Any, e0: Any, cap: Any, pmax: Any,
                  eta_c: Any, eta_d: Any, soc_min: Any, soc_max: Any, t0: int,
                  dt_h: float = 0.25) -> tuple[np.ndarray, np.ndarray]:
    """Fault-injector battery re-simulation; see ``battery_resim``."""
    return get("battery_resim")(surplus, soc0, batt0, cf, pf, start, e0, cap, pmax, eta_c, eta_d, soc_min, soc_max,
                                t0, dt_h)


__all__ = [
    "backend", "get", "names", "register", "rust_available",
    "battery_resim", "battery_resim_py", "battery_rollout", "battery_rollout_py", "battery_strategies",
    "binned_median_mad", "binned_median_mad_py", "cusum_two_sided", "cusum_two_sided_py", "load_features",
    "load_features_py", "pv_features", "pv_features_py", "rolling_mean", "rolling_mean_py", "rolling_median",
    "rolling_median_py", "scatter_telemetry", "scatter_telemetry_py", "self_consumption", "self_consumption_py",
    "site_features", "site_features_py", "window_gather", "window_gather_py",
]
