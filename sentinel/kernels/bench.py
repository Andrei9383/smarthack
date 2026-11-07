"""Bench Python reference vs Rust for each registered kernel.

    python -m sentinel.kernels.bench [--scale 1.0] [--repeat 3] [--only cusum_two_sided]

Prints ``LAB_METRIC`` lines (numeric only), e.g.::

    LAB_METRIC rust_available=1
    LAB_METRIC kernel_backend_rust=1      # what sentinel.kernels.backend() picks in this environment
    LAB_METRIC t_py_cusum_s=...
    LAB_METRIC t_rust_cusum_s=...
    LAB_METRIC speedup_cusum=...
    LAB_METRIC parity_cusum=1             # Rust output == Python output (bit-exact) on a slice of the bench input

Without the extension it still runs the Python timings and reports ``rust_available=0``.
"""

from __future__ import annotations

import argparse
import os
import sys
import time
from dataclasses import dataclass
from typing import Any, Callable

import numpy as np

from sentinel import kernels


def metric(name: str, value: float) -> None:
    print(f"LAB_METRIC {name}={value:.6g}", flush=True)


@dataclass
class Case:
    kernel: str  # registry name
    key: str  # short name used in metric names
    make: Callable[[float, np.random.Generator], tuple[Any, ...]]  # scale, rng -> inputs
    run: Callable[[Callable[..., Any], tuple[Any, ...]], None]  # full-size run, outputs discarded
    sample: Callable[[tuple[Any, ...]], tuple[Any, ...]]  # smaller slice of the inputs for the parity check
    desc: str


# ---- cusum: 2,000 series x 35,040 steps (one year of 15-min data), standardised residuals with gaps.
_CUSUM_CHUNK = 250  # series per call; keeps peak memory well under 2 GB for the Python reference


def _cusum_make(scale: float, rng: np.random.Generator) -> tuple[Any, ...]:
    m = max(1, int(round(2000 * scale)))
    n = max(2, int(round(35040 * scale)))
    x = rng.standard_normal((m, n))
    x[:, n // 2 :] += 0.3  # a drift in the second half so alarms fire
    gaps = rng.random((m, n)) < 0.002
    x[gaps] = np.nan
    return (x, 0.5, 8.0)


def _cusum_cap_make(scale: float, rng: np.random.Generator) -> tuple[Any, ...]:
    x, k, h = _cusum_make(scale, rng)
    return (x, k, h, h)  # cap = h, as the PV fault baseline uses it


def _cusum_run(fn: Callable[..., Any], args: tuple[Any, ...]) -> None:
    x, *rest = args
    for i in range(0, x.shape[0], _CUSUM_CHUNK):
        fn(x[i : i + _CUSUM_CHUNK], *rest)


def _cusum_sample(args: tuple[Any, ...]) -> tuple[Any, ...]:
    x, *rest = args
    return (x[:_CUSUM_CHUNK], *rest)


# ---- rollout: one P10 commitment = 200 scenarios x 50 sites x 96 steps; a replay week does many of these.
_ROLLOUT_CALLS = 20  # feeder-days per timed run


def _rollout_make(scale: float, rng: np.random.Generator) -> tuple[Any, ...]:
    n = max(1, int(round(200 * scale)))
    s = max(1, int(round(50 * scale)))
    t = 96
    has = rng.random(s) < 0.6  # ~40 % of sites have no battery
    cap = np.where(has, rng.uniform(5, 15, s), 0.0)
    pmax = np.where(has, rng.uniform(3, 5, s), 0.0)
    eta = np.where(has, 0.95, 1.0)
    e_lo = 0.1 * cap + np.where(has, rng.uniform(0, 2, s), 0.0)
    e_hi = 0.95 * cap
    req = rng.normal(0, 3, (n, s, t))
    soc0 = rng.uniform(0.05, 0.9, (n, s))
    return (req, soc0, cap, pmax, eta, eta, e_lo, e_hi, 0.25)


def _rollout_run(fn: Callable[..., Any], args: tuple[Any, ...]) -> None:
    for _ in range(_ROLLOUT_CALLS):
        fn(*args)


def _rollout_sample(args: tuple[Any, ...]) -> tuple[Any, ...]:
    return args


CASES = [
    Case("cusum_two_sided", "cusum", _cusum_make, _cusum_run, _cusum_sample, "2000 series x 35040 steps, f64, 0.2% NaN gaps"),
    Case("cusum_two_sided", "cusum_cap", _cusum_cap_make, _cusum_run, _cusum_sample,
         "same input, sums capped at h (PV fault baseline)"),
    Case("battery_rollout", "rollout", _rollout_make, _rollout_run, _rollout_sample,
         f"{_ROLLOUT_CALLS} feeder-days x 200 scenarios x 50 sites x 96 steps"),
]


def _time(fn: Callable[[], Any], repeat: int) -> tuple[float, Any]:
    best, out = float("inf"), None
    for _ in range(repeat):
        t0 = time.perf_counter()
        out = fn()
        best = min(best, time.perf_counter() - t0)
    return best, out


def _same(a: tuple[np.ndarray, ...], b: tuple[np.ndarray, ...]) -> bool:
    return all(
        x.shape == y.shape and x.dtype == y.dtype and np.array_equal(x, y, equal_nan=x.dtype.kind == "f")
        for x, y in zip(a, b, strict=True)
    )


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--scale", type=float, default=1.0, help="shrink sizes (both axes) for a quick run")
    p.add_argument("--repeat", type=int, default=3, help="best-of-N for Rust (Python runs once)")
    p.add_argument("--only", default=None, help="comma-separated kernel names")
    p.add_argument("--seed", type=int, default=0)
    a = p.parse_args(argv)

    have_rust = kernels.rust_available()
    try:
        be = kernels.backend()
    except ImportError as e:
        print(f"bench: {e}", file=sys.stderr)
        be = "python"
    print(f"bench: SENTINEL_KERNELS={os.environ.get('SENTINEL_KERNELS', 'auto')} backend={be}", flush=True)
    if have_rust:
        core = kernels._load_core()
        print(f"bench: sentinel._core {getattr(core, '__version__', '?')}, rayon threads={core.num_threads()}")
        metric("rust_threads", core.num_threads())
    else:
        print(f"bench: Rust extension not available ({kernels._core_error}); timing Python only", flush=True)
    metric("rust_available", int(have_rust))
    metric("kernel_backend_rust", int(be == "rust"))

    only = set(a.only.split(",")) if a.only else None
    rng = np.random.default_rng(a.seed)
    for case in CASES:
        if only and case.kernel not in only and case.key not in only:
            continue
        args = case.make(a.scale, rng)
        print(f"bench: {case.kernel}: {case.desc} (scale={a.scale})", flush=True)
        t_py, _ = _time(lambda: case.run(kernels.get(case.kernel, "python"), args), 1)
        metric(f"t_py_{case.key}_s", t_py)
        if not have_rust:
            continue
        rust_fn = kernels.get(case.kernel, "rust")
        case.run(rust_fn, args)  # warm-up (thread pool start)
        t_rs, _ = _time(lambda: case.run(rust_fn, args), max(1, a.repeat))
        metric(f"t_rust_{case.key}_s", t_rs)
        metric(f"speedup_{case.key}", t_py / t_rs if t_rs > 0 else float("nan"))
        small = case.sample(args)
        ok = _same(kernels.get(case.kernel, "python")(*small), rust_fn(*small))
        metric(f"parity_{case.key}", int(ok))
        if not ok:
            print(f"bench: PARITY FAILURE for {case.kernel}: Rust output differs from the Python reference",
                  file=sys.stderr)
            return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
