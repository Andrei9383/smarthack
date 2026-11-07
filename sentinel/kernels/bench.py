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


ROLL_W = (4, 16, 96, 672, 2016)  # window sizes M6 clf_features uses


def rolling_make(scale, rng):
    """M6 clf_features for one park: 2 years of 15-min data x 70 turbines, 10 % NaN."""
    x = rng.standard_normal((_sz(70176, scale, 2), _sz(70, scale)))
    x[rng.random(x.shape) < 0.1] = np.nan
    return (x,)


# ---- telemetry scatter: WindNBM load of WP1 (70 turbines x 70,176 steps x 8 sensors, one row per bin on average; 2%
# codes, 1% NaN and some rows off the grid / unknown tags).
def scatter_make(scale: float, rng: np.random.Generator) -> tuple[Any, ...]:
    S, T, N = 8, max(2, int(round(70176 * scale))), max(1, int(round(70 * scale)))
    n = S * T * N
    ti, tj, sk = rng.integers(-2, T + 2, n), rng.integers(-1, N, n), rng.integers(-1, S, n)
    is_code = rng.random(n) < 0.02
    v = np.where(is_code, rng.integers(0, 9, n), rng.standard_normal(n))
    v[rng.random(n) < 0.01] = np.nan
    sc = np.ones((S, N))
    sc[0] = rng.random(N) + 0.5
    return (np.sort(ti), tj, sk, v, is_code, sc, T)


# ---- feature build: the model data path's per-batch numpy features (#94) on batches of 256 windows, the way
# features.build sees them after the gather (float32 [B, 96, C] target-day arrays + 14-day history).
FEAT_BATCHES, FEAT_B = 40, 256


def feat_raw(scale: float, rng: np.random.Generator) -> list[dict]:
    b = max(1, int(round(FEAT_B * scale)))
    out = []
    for _ in range(FEAT_BATCHES):
        kwp = rng.uniform(3, 12, b).astype(np.float32)
        tod = np.arange(96) / 96 * 2 * np.pi
        prof = np.clip(np.sin(tod - np.pi / 2), 0, None).astype(np.float32)
        geom = np.stack([rng.uniform(0, 180, (b, 96)), rng.uniform(0, 360, (b, 96)), rng.uniform(0, 1000, (b, 96))],
                        -1).astype(np.float32)
        hist = (prof[None, None] * kwp[:, None, None] * rng.uniform(0, 1, (b, 14, 1))).reshape(b, -1)
        hist[:, ::7] = hist[:, 1::7]  # some stuck (frozen) readings
        cal = np.tile(np.stack([np.sin(tod), np.cos(tod)], -1), (b, 1, 4)).astype(np.float32)
        out.append({"wx": rng.random((b, 96, 7), np.float32) * 500, "geom": geom, "cal": cal,
                    "pvclear": prof[None] * kwp[:, None], "pvlib": prof[None] * kwp[:, None] * 0.8,
                    "hist": hist.astype(np.float32), "kwp": kwp})
    return out


def site_raw(scale: float, rng: np.random.Generator) -> list[dict]:
    """M4 site windows (features.site_inputs after gather): feat_raw + assessed-day telemetry and its 14-day history."""
    out = []
    for r in feat_raw(scale, rng):
        b, kwp = len(r["kwp"]), r["kwp"]
        f = lambda *s: rng.random(s, np.float32)  # noqa: E731
        bkwh = np.where(rng.random(b) < 0.5, rng.uniform(5, 15, b), 0.0)
        out.append({"kwp_recorded": kwp.astype(np.float64), "battery_kw": bkwh / 2, "battery_kwh": bkwh,
                    "has_batt": (bkwh > 0).astype(np.float32), "geom": r["geom"], "pvclear": r["pvclear"],
                    "pvlib": r["pvlib"], "hist_pv": r["hist"], "hist_load": f(b, 1344), "pv": r["pvlib"] * f(b, 1),
                    "m1": np.sort(f(b, 96, 3), -1) * kwp[:, None, None], "m2": f(b, 96), "volt": 230 + 5 * f(b, 96),
                    "status": rng.integers(0, 4, (b, 96), dtype=np.int8), "soc": f(b, 96), "batt": f(b, 96) - 0.5,
                    "load": f(b, 96), "valid": rng.random((b, 96)) > 0.02, "hist_volt": 230 + 5 * f(b, 1344),
                    "hist_status": rng.integers(0, 4, (b, 1344), dtype=np.int8), "hist_soc": f(b, 1344),
                    "hist_batt": f(b, 1344) - 0.5})
    return out


def pv_args(r: dict) -> tuple[Any, ...]:
    return r["wx"], r["geom"], r["pvclear"], r["pvlib"], r["hist"], r["kwp"]


def load_args(r: dict) -> tuple[Any, ...]:
    return r["wx"], r["cal"], r["hist"]


def strategies_make(scale, rng):
    """K8: one data-v2 chunk-year, 500 homes x 35040 steps (all four strategies), as the generator calls it."""
    from sentinel.kernels.strategies import day_rank_median

    n, days = _sz(500, scale), _sz(365, scale)
    t = days * 96
    hod = np.tile(np.arange(96) / 4, days)
    pv = np.clip(np.sin((hod - 6) / 12 * np.pi), 0, None)[None] * rng.uniform(2, 8, (n, 1)) * rng.uniform(0.3, 1, (n, t))
    strat = (np.arange(n) % 4).astype(np.int8)
    cap = np.where(strat > 0, rng.uniform(5, 15, n), 0.0)
    day = np.repeat(np.arange(days), 96).astype(np.int32)
    price = 400 + 300 * np.sin((hod - 13) / 12 * np.pi) + rng.normal(0, 40, t)
    rank, med = day_rank_median(price, day)
    return (pv, rng.uniform(0.2, 2.5, (n, t)), strat, cap, np.where(strat > 0, 4.0, 0.0), rng.uniform(0.1, 0.4, n),
            np.full(n, 0.95), rng.integers(8, 21, n).astype(np.int32), 0.5 * cap, price, rank, med, day,
            rng.random((n, days)) > 0.1)

# ---- fleet self-consumption (#135): the simulator's battery rule over one year, 600 homes, 45 % with a battery.
def selfcons_make(scale: float, rng: np.random.Generator) -> tuple:
    from sentinel.sim.fleet.battery import BatteryParams

    n, t = _sz(600, scale), _sz(35040, scale, 2)
    has = rng.random(n) < 0.45
    cap = np.where(has, rng.choice([5.0, 10.0, 13.5], n), 0.0)
    p = BatteryParams(cap, np.where(has, cap / 2, 0.0), np.full(n, 0.95), np.full(n, 0.95), np.where(has, 0.1, 0.0),
                      np.where(has, 0.95, 0.0), np.zeros(n))
    day = np.clip(np.sin(np.arange(t) / 96 * 2 * np.pi - np.pi / 2), 0, None)
    pv = (rng.uniform(3000, 9000, (n, 1)) * day[None] * rng.uniform(0.3, 1, (n, t))).astype(np.float32)
    return (p, pv, rng.exponential(700, (n, t)).astype(np.float32), 0.25)


def selfcons_sample(a: tuple) -> tuple:
    from sentinel.sim.fleet.battery import BatteryParams

    p, pv, load, dt = a
    k = slice(0, 50)
    return (BatteryParams(*(getattr(p, f)[k] for f in p.__dataclass_fields__)), pv[k], load[k], dt)


PVAC_CHUNK = 64  # sites per call, as sentinel.data.windows does per locality (#133)


def pvac_make(scale: float, rng: np.random.Generator) -> tuple[Any, ...]:
    """#133: 600 sites x 1 year of 15-min steps under one locality's weather (pvlib + clear-sky channels)."""
    import pandas as pd

    from sentinel.model import physics as ph

    T, n = _sz(35040, scale, 96), _sz(600, scale)
    times = pd.date_range("2025-01-01", periods=T, freq="15min", tz="UTC")
    g = ph.solar_geometry(times, 45.0, 25.0).astype(np.float32).astype(np.float64)
    ghi = np.clip(g[:, 2] * rng.uniform(0.1, 1.05, T), 0, None)
    dni, dhi = ph.decompose(ghi, g[:, 0], times.day_of_year.to_numpy())
    site = (rng.uniform(2, 12, n), rng.uniform(0, 60, n), rng.uniform(90, 270, n), rng.uniform(2, 10, n))
    return site, (ghi, dni, dhi, rng.normal(15, 8, T), rng.gamma(2, 2, T), g[:, 0], g[:, 1])


def pvac_run(fn, a):
    site, (ghi, dni, dhi, temp, wind, zen, saz) = a
    for k in range(0, len(site[0]), PVAC_CHUNK):
        s = [x[k:k + PVAC_CHUNK] for x in site]
        fn(*s, ghi, dni, dhi, temp, wind, zen, saz)
        fn(*s, ghi, dni, dhi, 20.0, None, zen, saz)

def resim_make(scale: float, rng: np.random.Generator) -> tuple[Any, ...]:
    """K7 (#134): one fault-injector batch re-simulated: 48 battery sites with a fault x 1 year of 15-min steps,
    onsets spread over the year, battery_fade capacity ramps and underuse steps, 0.1 % NaN surplus."""
    m, T = _sz(48, scale), _sz(35040, scale, 2)
    tod = np.sin(np.arange(T) / 96 * 2 * np.pi)
    surplus = 4000 * tod[None] + rng.normal(0, 3000, (m, T))
    surplus[rng.random((m, T)) < 0.001] = np.nan
    cf, pf = np.ones((m, T)), np.ones((m, T))
    for r in range(0, m, 4):
        i0 = int(rng.integers(0, T))
        cf[r, i0:] = np.linspace(1, 0.6, T - i0)
        pf[r + 1 if r + 1 < m else r, i0:i0 + T // 8] = 0.2
    start = rng.integers(0, T, m)
    cap = rng.uniform(5, 15, m)
    soc0 = rng.uniform(0, 1, (m, T))
    t0 = int(start.min())
    return (surplus, soc0, rng.normal(0, 2000, (m, T)), cf, pf, start, soc0[:, max(t0 - 1, 0)] * cap, cap,
            rng.uniform(2, 6, m), np.full(m, 0.95), np.full(m, 0.95), np.full(m, 0.1), np.full(m, 0.95), t0, 0.25)


SCEN_DAYS = 40


def scen_make(scale: float, rng: np.random.Generator) -> tuple[Any, ...]:
    """K9 (#187): W2 forecast stage, per day x 3 parks x 6 policies: 100 scenarios x 96 steps x 39 turbines."""
    S, H, N = _sz(100, scale), 96, 117
    rated = np.full(N, 3000.0)
    p50 = rng.uniform(0, 1, (H, N)) * rated
    q = np.stack([0.7 * p50, p50, np.minimum(1.3 * p50 + 50, rated)], -1).astype(np.float32)
    u = rng.uniform(0, 1, (S, H))
    pu, der = rng.uniform(0, 1, (H, N)), rng.uniform(0.5, 1, (H, N))
    return q, rated, [np.arange(p, N, 3) for p in range(3)], rng.uniform(0.9, 1, H), u, pu, der


def scen_run(fn, a):
    q, rated, cols, r, u, pu, der = a
    for d in range(SCEN_DAYS):
        for c in cols:
            for kw in ({}, dict(p_up=pu, derate=der), dict(p_up=pu, derate=der, bernoulli=True), dict(p_up=pu),
                       dict(p_up=pu), dict(p_up=pu)):
                fn(q, rated, c, r, u, **kw, rng=np.random.default_rng(d))


# (kernel, key, make(scale, rng), run(fn, args), parity sample(args), description[, parity atol])
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
    ("rolling_mean", "rolling", rolling_make, lambda fn, a: [fn(a[0], w) for w in ROLL_W],
     lambda a: (a[0][:5000], 672), f"70176 steps x 70 turbines, f64, 10% NaN, w in {ROLL_W} (M6 clf_features)"),
    ("scatter_telemetry", "scatter", scatter_make, lambda fn, a: fn(*a),
     lambda a: (*(x[:200_000] for x in a[:5]), a[5], a[6]),
     "39 M long rows (time-sorted) -> 8 sensors x 70176 steps x 70 turbines, 2% codes, 1% NaN (WindNBM load)"),
    ("pv_features", "features_pv", lambda sc, rng: (feat_raw(sc, rng),), lambda fn, a: [fn(*pv_args(r)) for r in a[0]],
     lambda a: pv_args(a[0][0]), f"{FEAT_BATCHES} batches x {FEAT_B} PV windows (features.pv_inputs)", 1e-6),  # cos/sin: 1 ulp
    ("load_features", "features_load", lambda sc, rng: (feat_raw(sc, rng),), lambda fn, a: [fn(*load_args(r)) for r in a[0]],
     lambda a: load_args(a[0][0]), f"{FEAT_BATCHES} batches x {FEAT_B} load windows (features.load_inputs)"),
    ("site_features", "features_site", lambda sc, rng: (site_raw(sc, rng),), lambda fn, a: [fn(r) for r in a[0]],
     lambda a: (a[0][0],), f"{FEAT_BATCHES} batches x {FEAT_B} M4 site windows (features.site_inputs)", 1e-6),  # cos_zen
    ("self_consumption", "selfcons", selfcons_make, lambda fn, a: fn(*a), selfcons_sample,
     "fleet battery rule: 600 homes x 35040 steps (one year), f32 pv/load (#135)"),


    ("pv_ac_batch", "pvac", pvac_make, pvac_run,
     lambda a: (*(x[:PVAC_CHUNK] for x in a[0]), *a[1]),
     "600 sites x 35040 steps, measured + clear-sky channel, 64 sites per call (windows.py pvlib channels)",
     1e-9),  # bit-exact where numpy's f64 cos/arccos are libm's; a few ulp on AVX-512 SVML CPUs
    ("battery_resim", "battery_resim", resim_make, lambda fn, a: fn(*a), lambda a: a,
     "fault injector battery resim (#134): 48 sites x 35040 steps, fade/underuse, 0.1% NaN"),
    ("battery_strategies", "strategies", strategies_make, lambda fn, a: fn(*a),
     lambda a: (a[0][:40], a[1][:40], *(x[:40] for x in a[2:9]), *a[9:13], a[13][:40]),
     "500 homes x 35040 steps, 4 strategies (one data-v2 chunk-year, #172)"),

    ("park_scenarios", "park_scenarios", scen_make, scen_run,
     lambda a: (a[0], a[1], a[2][1], a[3], a[4], a[5], a[6], True, None),  # rng None: same draws per call
     f"W2 forecast (#187): {SCEN_DAYS} days x 3 parks x 6 policies, 100 scen x 96 steps x 39 turbines, f32 q"),
]


# ---- exact medians (#136): WindNBM label-free normal selection, 121 turbines x 70,176 steps; residuals with ~20 % NaN
MED_W = ((96, 12), (1344, 96), (192, 24))  # (window, min_periods) select_normal uses


def median_make(scale: float, rng: np.random.Generator) -> tuple[Any, ...]:
    T, N = _sz(70176, scale, 2), _sz(121, scale)
    x = np.round(rng.standard_normal((T, N)) * 3, 2)  # rounded: ties, as in real telemetry
    x[rng.random((T, N)) < 0.2] = np.nan
    return x, rng.integers(0, 60, (T, N)), rng.random((T, N)) < 0.3


CASES += [
    ("rolling_median", "rolling_median", median_make, lambda fn, a: [fn(a[0], w, mp) for w, mp in MED_W],
     lambda a: (a[0][:6000], 1344, 96), f"121 turbines x 70176 steps, f64, 20% NaN, centred, (w, min_periods) {MED_W}"),
    ("binned_median_mad", "binned_median", median_make, lambda fn, a: [fn(a[1], a[0], a[2], 60, 30) for _ in range(4)],
     lambda a: (a[1], a[0], a[2], 60, 30), "4 x per-turbine median + MAD in 60 wind bins, 121 x 70176, 30% kept"),
]


def _best(fn, repeat: int) -> float:
    ts = []
    for _ in range(repeat):
        t0 = time.perf_counter()
        fn()
        ts.append(time.perf_counter() - t0)
    return min(ts)


def _same(a, b, atol: float = 0.0) -> bool:
    return all(x.shape == y.shape and x.dtype == y.dtype and (
        np.allclose(x, y, rtol=0, atol=atol, equal_nan=True) if atol and x.dtype.kind == "f"
        else np.array_equal(x, y, equal_nan=x.dtype.kind == "f")) for x, y in zip(a, b, strict=True))


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
    for kernel, key, make, run, sample, desc, *tol in CASES:
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
        ok = _same(py_fn(*sample(args)), rs_fn(*sample(args)), *tol)
        metric(f"parity_{key}", int(ok))
        if not ok:
            print(f"bench: PARITY FAILURE for {kernel}: Rust output differs from the Python reference", file=sys.stderr)
            return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
