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


__all__ = ["backend", "battery_rollout", "battery_rollout_py", "cusum_two_sided", "cusum_two_sided_py", "get",
           "names", "register", "rolling_mean", "rolling_mean_py", "rust_available", "window_gather",
           "window_gather_py"]
