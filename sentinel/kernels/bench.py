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


# ---- window gather: training batches the way the model data path reads #10's cache (600 sites x 4 years of
# 15-min steps, float16): per batch of 1024 (site, day) windows, 14-day history + 24 h target of PV and load,
# and 24 h of locality weather [L, T, 7], all converted to float32.
_WIN_BATCHES = 30
_WIN_B = 1024
_HIST = 14 * 96


def _windows_make(scale: float, rng: np.random.Generator) -> tuple[Any, ...]:
    s = max(1, int(round(600 * scale)))
    days = max(16, int(round(4 * 365 * scale)))
    t = days * 96
    pv = rng.random((s, t), dtype=np.float32).astype(np.float16)
    load = rng.random((s, t), dtype=np.float32).astype(np.float16)
    wx = rng.random((20, t, 7), dtype=np.float32).astype(np.float16)
    sites = rng.integers(0, s, (_WIN_BATCHES, _WIN_B))
    loc = sites % 20
    starts = rng.integers(14, days, (_WIN_BATCHES, _WIN_B)) * 96
    return (pv, load, wx, sites, loc, starts)


def _windows_run(fn: Callable[..., Any], args: tuple[Any, ...]) -> None:
    pv, load, wx, sites, loc, starts = args
    for k in range(sites.shape[0]):
        s, st = sites[k], starts[k]
        fn(pv, s, st, -_HIST, _HIST, np.float32)
        fn(load, s, st, -_HIST, _HIST, np.float32)
        fn(pv, s, st, 0, 96, np.float32)
        fn(wx, loc[k], st, 0, 96, np.float32)


def _windows_sample(args: tuple[Any, ...]) -> tuple[Any, ...]:
    pv, _, _, sites, _, starts = args
    return (pv, sites[0, :256], starts[0, :256], -_HIST, _HIST + 96, np.float32)


# ---- rolling mean: M6 clf_features for one park over the full span (70 turbines x 70,176 steps, 2 years of
# 15-min data); one call per window size it uses (4, 16, 96, 672, 2016), 10% NaN.
_ROLL_W = (4, 16, 96, 672, 2016)


def _rolling_make(scale: float, rng: np.random.Generator) -> tuple[Any, ...]:
    x = rng.standard_normal((max(2, int(round(70176 * scale))), max(1, int(round(70 * scale)))))
    x[rng.random(x.shape) < 0.1] = np.nan
    return (x, 96)


def _rolling_run(fn: Callable[..., Any], args: tuple[Any, ...]) -> None:
    for w in _ROLL_W:
        fn(args[0], w)


def _rolling_sample(args: tuple[Any, ...]) -> tuple[Any, ...]:
    return (args[0][:5000], 672)


# ---- telemetry scatter: WindNBM load of WP1 (70 turbines x 70,176 steps x 8 sensors, one row per bin on average; 2%
# codes, 1% NaN and some rows off the grid / unknown tags).
def _scatter_make(scale: float, rng: np.random.Generator) -> tuple[Any, ...]:
    S, T, N = 8, max(2, int(round(70176 * scale))), max(1, int(round(70 * scale)))
    n = S * T * N
    ti, tj, sk = rng.integers(-2, T + 2, n), rng.integers(-1, N, n), rng.integers(-1, S, n)
    is_code = rng.random(n) < 0.02
    v = np.where(is_code, rng.integers(0, 9, n), rng.standard_normal(n))
    v[rng.random(n) < 0.01] = np.nan
    sc = np.ones((S, N))
    sc[0] = rng.random(N) + 0.5
    return (np.sort(ti), tj, sk, v, is_code, sc, T)


def _scatter_run(fn: Callable[..., Any], args: tuple[Any, ...]) -> None:
    fn(*args)


def _scatter_sample(args: tuple[Any, ...]) -> tuple[Any, ...]:
    return (*(a[:200_000] for a in args[:5]), args[5], args[6])


CASES = [
    Case("cusum_two_sided", "cusum", _cusum_make, _cusum_run, _cusum_sample, "2000 series x 35040 steps, f64, 0.2% NaN gaps"),
    Case("cusum_two_sided", "cusum_cap", _cusum_cap_make, _cusum_run, _cusum_sample,
         "same input, sums capped at h (PV fault baseline)"),
    Case("battery_rollout", "rollout", _rollout_make, _rollout_run, _rollout_sample,
         f"{_ROLLOUT_CALLS} feeder-days x 200 scenarios x 50 sites x 96 steps"),
    Case("window_gather", "windows", _windows_make, _windows_run, _windows_sample,
         f"{_WIN_BATCHES} batches x {_WIN_B} windows: 14 d hist + 24 h target, f16 -> f32, from 600 x 140160"),
    Case("rolling_mean", "rolling", _rolling_make, _rolling_run, _rolling_sample,
         f"70176 steps x 70 turbines, f64, 10% NaN, w in {_ROLL_W} (M6 clf_features)"),
    Case("scatter_telemetry", "scatter", _scatter_make, _scatter_run, _scatter_sample,
         "39 M long rows (time-sorted) -> 8 sensors x 70176 steps x 70 turbines, 2% codes, 1% NaN (WindNBM load)"),
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
