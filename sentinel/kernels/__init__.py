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
    s_pos, s_neg, alarm = cusum(x, k=0.5, h=5.0)
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


# --------------------------------------------------------------------------- K3: two-sided CUSUM
# Python reference. Card #49 moves this to sentinel/kernels/cusum.py.


def _cusum_prep(x: Any, k: float, h: float) -> tuple[np.ndarray, bool, float, float]:
    x = np.asarray(x, dtype=np.float64)
    if x.ndim not in (1, 2):
        raise ValueError(f"x must be 1-D (n,) or 2-D (series, n), got shape {x.shape}")
    k, h = float(k), float(h)
    if not (np.isfinite(k) and np.isfinite(h)):
        raise ValueError("k and h must be finite")
    one_d = x.ndim == 1
    x2 = np.ascontiguousarray(x.reshape(1, -1) if one_d else x)
    return x2, one_d, k, h


def cusum_two_sided_py(x: Any, k: float, h: float) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Two-sided CUSUM over the last axis of ``x`` (shape (n,) or (series, n)).

    For each step, with sums starting at 0::

        x[t] not finite (NaN/inf = gap): s_pos = s_neg = 0, alarm = False
        otherwise: s_pos = max(s_pos + x[t] - k, 0)
                   s_neg = max(s_neg - x[t] - k, 0)
                   alarm = s_pos > h or s_neg > h

    ``x`` should already be standardised (target mean removed, scaled); ``k`` is the allowance (slack) and
    ``h`` the decision threshold, in the same units. Sums are not reset after an alarm.
    Returns ``(s_pos, s_neg, alarm)`` with the shape of ``x`` (float64, float64, bool).
    """
    x2, one_d, k, h = _cusum_prep(x, k, h)
    m, n = x2.shape
    xt = np.ascontiguousarray(x2.T)  # (n, m): one contiguous row per time step, vectorised over series
    sp = np.empty((n, m))
    sn = np.empty((n, m))
    cp = np.zeros(m)
    cn = np.zeros(m)
    finite = np.isfinite(xt)
    with np.errstate(invalid="ignore"):
        for t in range(n):
            v = xt[t]
            ok = finite[t]
            cp = np.where(ok, np.maximum(cp + v - k, 0.0), 0.0)
            cn = np.where(ok, np.maximum(cn - v - k, 0.0), 0.0)
            sp[t] = cp
            sn[t] = cn
    sp = np.ascontiguousarray(sp.T)
    sn = np.ascontiguousarray(sn.T)
    alarm = (sp > h) | (sn > h)
    if one_d:
        return sp[0], sn[0], alarm[0]
    return sp, sn, alarm


def _cusum_two_sided_rust(x: Any, k: float, h: float) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    x2, one_d, k, h = _cusum_prep(x, k, h)
    sp, sn, alarm = _load_core().cusum_two_sided(x2, k, h)
    if one_d:
        return sp[0], sn[0], alarm[0]
    return sp, sn, alarm


register("cusum_two_sided", cusum_two_sided_py, _cusum_two_sided_rust)


def cusum_two_sided(x: Any, k: float, h: float) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Dispatched two-sided CUSUM; see ``cusum_two_sided_py`` (the reference) for the definition."""
    return get("cusum_two_sided")(x, k, h)


# --------------------------------------------------------------------------- K1: battery rollout (rollout.py)
from sentinel.kernels.rollout import battery_rollout_py, battery_rollout_rust  # noqa: E402

register("battery_rollout", battery_rollout_py, battery_rollout_rust)


def battery_rollout(req: Any, soc0: Any, cap: Any, pmax: Any, eta_c: Any, eta_d: Any, e_lo: Any, e_hi: Any,
                    dt_h: float = 0.25) -> tuple[np.ndarray, np.ndarray]:
    """Dispatched battery rollout; see ``sentinel.kernels.rollout`` (reference = sentinel.opt.commit.rollout)."""
    return get("battery_rollout")(req, soc0, cap, pmax, eta_c, eta_d, e_lo, e_hi, dt_h)


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
]
