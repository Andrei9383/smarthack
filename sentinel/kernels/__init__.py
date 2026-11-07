"""Kernel dispatch: pure-numpy reference ``<name>_py`` (source of truth) + optional Rust copy in ``sentinel._core``.

SENTINEL_KERNELS=auto (default: Rust if ``sentinel._core`` imports, else Python, logged once) | python | rust
(raise ImportError if the extension is missing). See team/infra.md §1-4.
"""

from __future__ import annotations

import logging
import os
from typing import Any, Callable

import numpy as np

log = logging.getLogger("sentinel.kernels")
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
            from sentinel import _core as core  # type: ignore[attr-defined]
            _core = core
        except Exception as e:  # ImportError, or a broken build
            _core_error = f"{type(e).__name__}: {e}"
    return _core


def rust_available() -> bool:
    return _load_core() is not None


def backend() -> str:
    """The backend ``get`` dispatches to right now: ``"rust"`` or ``"python"``."""
    global _fallback_logged
    mode = os.environ.get("SENTINEL_KERNELS", "auto").strip().lower() or "auto"
    if mode not in _MODES:
        raise ValueError(f"SENTINEL_KERNELS={mode!r}; expected one of {_MODES}")
    if mode == "python":
        return "python"
    if rust_available():
        return "rust"
    if mode == "rust":
        raise ImportError(f"SENTINEL_KERNELS=rust but sentinel._core is not available ({_core_error})")
    if not _fallback_logged:
        _fallback_logged = True
        log.info("sentinel._core not available (%s); kernels use the Python reference", _core_error)
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


from sentinel.kernels.cusum import cusum_two_sided_py, cusum_two_sided_rust  # noqa: E402
from sentinel.kernels.rollout import battery_rollout_py, battery_rollout_rust  # noqa: E402
from sentinel.kernels.windows import window_gather_py, window_gather_rust  # noqa: E402

register("cusum_two_sided", cusum_two_sided_py, cusum_two_sided_rust)
register("battery_rollout", battery_rollout_py, battery_rollout_rust)
register("window_gather", window_gather_py, window_gather_rust)


def cusum_two_sided(x: Any, k: float, h: float,
                    cap: float | None = None) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """K3, dispatched; see ``cusum.cusum_two_sided_py``."""
    return get("cusum_two_sided")(x, k, h, cap)


def battery_rollout(req: Any, soc0: Any, cap: Any, pmax: Any, eta_c: Any, eta_d: Any, e_lo: Any, e_hi: Any,
                    dt_h: float = 0.25) -> tuple[np.ndarray, np.ndarray]:
    """K1, dispatched; see ``rollout`` (reference = sentinel.opt.commit.rollout)."""
    return get("battery_rollout")(req, soc0, cap, pmax, eta_c, eta_d, e_lo, e_hi, dt_h)


def window_gather(arr: Any, rows: Any, starts: Any, offset: int, length: int, dtype: Any = None,
                  out: np.ndarray | None = None) -> np.ndarray:
    """K2, dispatched; see ``windows`` (reference = numpy fancy-index gather)."""
    return get("window_gather")(arr, rows, starts, offset, length, dtype, out)


from sentinel.kernels.rolling import rolling_mean_py, rolling_mean_rust  # noqa: E402

register("rolling_mean", rolling_mean_py, rolling_mean_rust)


def rolling_mean(x: Any, w: int) -> np.ndarray:
    """K4, dispatched causal NaN-aware rolling mean along axis 0; see ``rolling`` (the reference)."""
    return get("rolling_mean")(x, w)


# --------------------------------------------------------------------------- K5: telemetry scatter (scatter.py)
from sentinel.kernels.scatter import scatter_telemetry_py, scatter_telemetry_rust  # noqa: E402

register("scatter_telemetry", scatter_telemetry_py, scatter_telemetry_rust)


def scatter_telemetry(ti: Any, tj: Any, sk: Any, v: Any, is_code: Any, scale: Any,
                      T: int) -> tuple[np.ndarray, np.ndarray]:
    """Dispatched long-telemetry scatter to [sensor, step, turbine] bin means + last code per bin;
    see ``sentinel.kernels.scatter`` (the reference)."""
    return get("scatter_telemetry")(ti, tj, sk, v, is_code, scale, T)


# --------------------------------------------------------------------------- K6: fused feature build (features.py)
from sentinel.kernels.features import (load_features_py, load_features_rust, pv_features_py,  # noqa: E402
                                       pv_features_rust)

register("pv_features", pv_features_py, pv_features_rust)
register("load_features", load_features_py, load_features_rust)


def pv_features(wx: Any, geom: Any, pvclear: Any, pvlib: Any, hist_pv: Any, kwp: Any) -> tuple[np.ndarray, ...]:
    """Dispatched PV feature build -> (seq, ref, summ, night); reference = sentinel.model.features.pv_arrays."""
    return get("pv_features")(wx, geom, pvclear, pvlib, hist_pv, kwp)


def load_features(wx: Any, cal: Any, hist_load: Any) -> tuple[np.ndarray, ...]:
    """Dispatched load feature build -> (seq, ref, summ, scale); reference = sentinel.model.features.load_arrays."""
    return get("load_features")(wx, cal, hist_load)


__all__ = ["backend", "battery_rollout", "battery_rollout_py", "cusum_two_sided", "cusum_two_sided_py", "get",
           "load_features", "load_features_py", "names", "pv_features", "pv_features_py", "register", "rolling_mean",
           "rolling_mean_py", "rust_available", "scatter_telemetry", "scatter_telemetry_py", "window_gather",
           "window_gather_py"]
