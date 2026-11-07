"""Bench Python reference vs Rust for each registered kernel.

    python -m sentinel.kernels.bench [--scale 1.0] [--repeat 3] [--only cusum_two_sided]

Prints LAB_METRIC rust_available, kernel_backend_rust, and per case t_py_<key>_s, t_rust_<key>_s, speedup_<key>,
parity_<key> (1 = Rust output bit-exact to Python on a slice of the input). Without the extension: Python only.
"""

from __future__ import annotations

import argparse
import os
import sys
import time

import numpy as np

from sentinel import kernels

CUSUM_CHUNK = 250  # series per call; keeps peak memory well under 2 GB for the Python reference
ROLLOUT_CALLS = 20  # feeder-days per timed run
WIN_BATCHES, WIN_B, HIST = 30, 1024, 14 * 96


def metric(name: str, value: float) -> None:
    print(f"LAB_METRIC {name}={value:.6g}", flush=True)


def _sz(n: int, scale: float, lo: int = 1) -> int:
    return max(lo, int(round(n * scale)))


def cusum_make(scale, rng):
    """One year of 15-min standardised residuals, 0.2 % gaps, a drift in the second half so alarms fire."""
    m, n = _sz(2000, scale), _sz(35040, scale, 2)
    x = rng.standard_normal((m, n))
    x[:, n // 2:] += 0.3
    x[rng.random((m, n)) < 0.002] = np.nan
    return (x, 0.5, 8.0)


def cusum_run(fn, args):
    x, *rest = args
    for i in range(0, x.shape[0], CUSUM_CHUNK):
        fn(x[i:i + CUSUM_CHUNK], *rest)


def rollout_make(scale, rng):
    """One P10 commitment: 200 scenarios x 50 sites x 96 steps, ~40 % of sites without a battery."""
    n, s = _sz(200, scale), _sz(50, scale)
    has = rng.random(s) < 0.6
    cap = np.where(has, rng.uniform(5, 15, s), 0.0)
    pmax = np.where(has, rng.uniform(3, 5, s), 0.0)
    eta = np.where(has, 0.95, 1.0)
    e_lo = 0.1 * cap + np.where(has, rng.uniform(0, 2, s), 0.0)
    return (rng.normal(0, 3, (n, s, 96)), rng.uniform(0.05, 0.9, (n, s)), cap, pmax, eta, eta, e_lo, 0.95 * cap, 0.25)


def windows_make(scale, rng):
    """#10's cache (600 sites x 4 years, float16); batches of 1024 (site, day) windows, as the model reads it."""
    s, days = _sz(600, scale), _sz(4 * 365, scale, 16)
    pv, load = (rng.random((s, days * 96), dtype=np.float32).astype(np.float16) for _ in range(2))
    wx = rng.random((20, days * 96, 7), dtype=np.float32).astype(np.float16)
    sites = rng.integers(0, s, (WIN_BATCHES, WIN_B))
    return (pv, load, wx, sites, sites % 20, rng.integers(14, days, (WIN_BATCHES, WIN_B)) * 96)


def windows_run(fn, args):
    pv, load, wx, sites, loc, starts = args
    for s, lc, st in zip(sites, loc, starts):
        fn(pv, s, st, -HIST, HIST, np.float32)
        fn(load, s, st, -HIST, HIST, np.float32)
        fn(pv, s, st, 0, 96, np.float32)
        fn(wx, lc, st, 0, 96, np.float32)


# (kernel, key, make(scale, rng), run(fn, args), parity sample(args), description)
CASES = [
    ("cusum_two_sided", "cusum", cusum_make, cusum_run, lambda a: (a[0][:CUSUM_CHUNK], *a[1:]),
     "2000 series x 35040 steps, f64, 0.2% NaN gaps"),
    ("cusum_two_sided", "cusum_cap", lambda sc, rng: (*cusum_make(sc, rng), 8.0), cusum_run,
     lambda a: (a[0][:CUSUM_CHUNK], *a[1:]), "same input, sums capped at h (PV fault baseline)"),
    ("battery_rollout", "rollout", rollout_make, lambda fn, a: [fn(*a) for _ in range(ROLLOUT_CALLS)],
     lambda a: a, f"{ROLLOUT_CALLS} feeder-days x 200 scenarios x 50 sites x 96 steps"),
    ("window_gather", "windows", windows_make, windows_run,
     lambda a: (a[0], a[3][0, :256], a[5][0, :256], -HIST, HIST + 96, np.float32),
     f"{WIN_BATCHES} batches x {WIN_B} windows: 14 d hist + 24 h target, f16 -> f32, from 600 x 140160"),
]


def _best(fn, repeat: int) -> float:
    ts = []
    for _ in range(repeat):
        t0 = time.perf_counter()
        fn()
        ts.append(time.perf_counter() - t0)
    return min(ts)


def _same(a, b) -> bool:
    return all(x.shape == y.shape and x.dtype == y.dtype and np.array_equal(x, y, equal_nan=x.dtype.kind == "f")
               for x, y in zip(a, b, strict=True))


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--scale", type=float, default=1.0, help="shrink sizes (both axes) for a quick run")
    p.add_argument("--repeat", type=int, default=3, help="best-of-N for Rust (Python runs once)")
    p.add_argument("--only", default=None, help="comma-separated kernel names or keys")
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
    for kernel, key, make, run, sample, desc in CASES:
        if only and kernel not in only and key not in only:
            continue
        args = make(a.scale, rng)
        print(f"bench: {kernel}: {desc} (scale={a.scale})", flush=True)
        py_fn = kernels.get(kernel, "python")
        t_py = _best(lambda: run(py_fn, args), 1)
        metric(f"t_py_{key}_s", t_py)
        if not have_rust:
            continue
        rs_fn = kernels.get(kernel, "rust")
        run(rs_fn, args)  # warm-up (thread pool start)
        t_rs = _best(lambda: run(rs_fn, args), max(1, a.repeat))
        metric(f"t_rust_{key}_s", t_rs)
        metric(f"speedup_{key}", t_py / t_rs if t_rs > 0 else float("nan"))
        ok = _same(py_fn(*sample(args)), rs_fn(*sample(args)))
        metric(f"parity_{key}", int(ok))
        if not ok:
            print(f"bench: PARITY FAILURE for {kernel}: Rust output differs from the Python reference", file=sys.stderr)
            return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
