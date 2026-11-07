"""Kernel dispatch: Python reference (source of truth) + optional Rust copy in ``sentinel._core``.

See team/infra.md §1-4. Every kernel registered here has a pure-numpy reference ``<name>_py``. The Rust
extension is opt-in and never required: if it is missing, the dispatcher falls back to Python and logs it once.

    SENTINEL_KERNELS=auto    (default) Rust if ``sentinel._core`` imports, else Python
    SENTINEL_KERNELS=python  always Python
    SENTINEL_KERNELS=rust    Rust, raise ImportError if the extension is missing

Usage::

    from sentinel import kernels
    kernels.backend()                       # "rust" | "python"
    cusum = kernels.get("cusum_two_sided")  # dispatched callable
    s_pos, s_neg, alarm = cusum(x, k=0.5, h=5.0)            # cap=None: sums not capped
"""

from __future__ import annotations

import logging
import os
from dataclasses import dataclass
from typing import Any, Callable

import numpy as np

log = logging.getLogger("sentinel.kernels")

_MODES = ("auto", "python", "rust")


# --------------------------------------------------------------------------- Rust extension loading

_core: Any = None
_core_error: str | None = None
_core_loaded = False
_fallback_logged = False


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
    """True if the compiled extension ``sentinel._core`` imports."""
    return _load_core() is not None


def _mode() -> str:
    mode = os.environ.get("SENTINEL_KERNELS", "auto").strip().lower() or "auto"
    if mode not in _MODES:
        raise ValueError(f"SENTINEL_KERNELS={mode!r}; expected one of {_MODES}")
    return mode


def backend() -> str:
    """The backend ``get`` dispatches to right now: ``"rust"`` or ``"python"``."""
    global _fallback_logged
    mode = _mode()
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


# --------------------------------------------------------------------------- registry


@dataclass(frozen=True)
class Kernel:
    name: str
    python: Callable[..., Any]
    rust: Callable[..., Any] | None  # wrapper that calls into sentinel._core; None = Python only


_REGISTRY: dict[str, Kernel] = {}


def register(name: str, python: Callable[..., Any], rust: Callable[..., Any] | None = None) -> None:
    """Register a kernel. ``rust`` is a wrapper with the same signature as ``python``."""
    _REGISTRY[name] = Kernel(name, python, rust)


def names() -> list[str]:
    return sorted(_REGISTRY)


def get(name: str, backend_: str | None = None) -> Callable[..., Any]:
    """Dispatched callable for ``name``. ``backend_`` forces "python" or "rust" (used by tests and bench)."""
    try:
        k = _REGISTRY[name]
    except KeyError:
        raise KeyError(f"unknown kernel {name!r}; registered: {names()}") from None
    be = backend_ or backend()
    if be == "python":
        return k.python
    if be != "rust":
        raise ValueError(f"backend must be 'python' or 'rust', got {be!r}")
    if k.rust is None or not rust_available():
        raise ImportError(f"no Rust implementation of {name!r} available ({_core_error or 'not built'})")
    return k.rust


# --------------------------------------------------------------------------- K3: two-sided CUSUM (cusum.py)
from sentinel.kernels.cusum import cusum_two_sided_py, cusum_two_sided_rust  # noqa: E402

register("cusum_two_sided", cusum_two_sided_py, cusum_two_sided_rust)


def cusum_two_sided(x: Any, k: float, h: float,
                    cap: float | None = None) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Dispatched two-sided CUSUM; see ``sentinel.kernels.cusum.cusum_two_sided_py`` (the reference)."""
    return get("cusum_two_sided")(x, k, h, cap)


# --------------------------------------------------------------------------- K1: battery rollout (rollout.py)
from sentinel.kernels.rollout import battery_rollout_py, battery_rollout_rust  # noqa: E402

register("battery_rollout", battery_rollout_py, battery_rollout_rust)


def battery_rollout(req: Any, soc0: Any, cap: Any, pmax: Any, eta_c: Any, eta_d: Any, e_lo: Any, e_hi: Any,
                    dt_h: float = 0.25) -> tuple[np.ndarray, np.ndarray]:
    """Dispatched battery rollout; see ``sentinel.kernels.rollout`` (reference = sentinel.opt.commit.rollout)."""
    return get("battery_rollout")(req, soc0, cap, pmax, eta_c, eta_d, e_lo, e_hi, dt_h)


# --------------------------------------------------------------------------- K2: window gather (windows.py)
from sentinel.kernels.windows import window_gather_py, window_gather_rust  # noqa: E402

register("window_gather", window_gather_py, window_gather_rust)


def window_gather(arr: Any, rows: Any, starts: Any, offset: int, length: int, dtype: Any = None,
                  out: np.ndarray | None = None) -> np.ndarray:
    """Dispatched window gather; see ``sentinel.kernels.windows`` (reference = numpy fancy-index gather)."""
    return get("window_gather")(arr, rows, starts, offset, length, dtype, out)


__all__ = [
    "backend",
    "battery_rollout",
    "battery_rollout_py",
    "cusum_two_sided",
    "cusum_two_sided_py",
    "get",
    "names",
    "register",
    "rust_available",
    "window_gather",
    "window_gather_py",
]
