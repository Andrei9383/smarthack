# ML infra + Rust kernels (Stefan)

Owner: Stefan. Covers how the ML code in `sentinel/` runs on the lab, and where Rust is used and how.
Amends plan R21 (see `team/decisions.md`). Contracts (#7, #38) are not changed by anything here.

## 1. Principles

1. **Python is the reference, Rust is a faster copy.** Every Rust kernel has a pure-Python (numpy) twin with the
   same signature. Parity tests compare them. If they disagree, the Python version is correct.
2. **Rust is never on the critical path.** `check`, `preview` and `make reproduce` work without a Rust toolchain.
   If the extension is missing or fails to build, the dispatcher falls back to Python and logs it.
3. **Every claim about speed comes from a logged run.** Each run logs per-stage wall time as `LAB_METRIC
   t_<stage>_s`, and `bench` runs log the speedup per kernel. A kernel is enabled by default only if its bench
   shows it is faster on the lab workers.
4. **Native code we don't write stays native.** torch, LightGBM, HiGHS, polars and onnxruntime are already C/C++/Rust;
   we don't rewrite them. Rust targets **stateful per-timestep loops** that numpy can't vectorise.

## 2. Where Rust pays off (kernel list)

Sizes are from plan §3.5 and §3.2. Priority = how much time it is likely to save × how sure we are that it's a loop.

| ID | Kernel | Why it's a loop | Size | Python reference (owner) | Priority |
|---|---|---|---|---|---|
| K1 | `rollout`: battery schedule replay with clipping (SoC bounds, power limits, η_c/η_d, reserve R) | each step's SoC depends on the last | ~200 scenarios × 96 steps × sites × 10–20 feeders per day, × days in replay and drills | battery model #11 (Andrei), P10 commitment #27 (yolooo332) | 1 |
| K2 | `windows`: gather float16 windows (24 h + 7–14-day reference + 14-day residual summary) into a memmap | random gather + several resolutions per sample; numpy makes big temporary copies | all sites × all days × channels, rebuilt on every data change | #10 (Andrei) | 2 |
| K3 | `cusum` + rolling residual stats (two-sided CUSUM, EWMA, gap-aware rolling mean/std) | recurrence over time, resets on gaps | every site and turbine, every tick, for #15 baselines and M4/M6 features | #15, #24 (Stefan) | 3 |
| K4 | `windmon_rules`: rule state machines over SCADA (persistence counters, error-code runs) | counters with hysteresis | ~120 turbines × 5-min series | #24 (Stefan) | 4, only if a bench shows polars is slow |

Not Rust: the LP (HiGHS), model training and inference (torch/ONNX), Parquet I/O and group-bys (polars/duckdb), the
binned power curve.

## 3. Layout

```
rust/sentinel_core/            Cargo crate (PyO3 + numpy + rayon), built with maturin as sentinel._core
  Cargo.toml  pyproject.toml   abi3-py312, release profile with lto = "fat", codegen-units = 1
  src/lib.rs                   #[pymodule] registers kernels
  src/{rollout,windows,cusum}.rs
sentinel/kernels/              dispatch + Python reference for each kernel
  __init__.py                  backend() → "rust" | "python"; SENTINEL_KERNELS=auto|python|rust
  rollout.py  windows.py  cusum.py   each: def <name>_py(...), def <name>(...) dispatches
  bench.py                     python -m sentinel.kernels.bench → LAB_METRIC speedup_<kernel>=…
sentinel/infra/                run harness (Python only)
tests/test_kernels.py          parity: Rust vs Python on random + edge inputs (skipped if no _core)
```

Kernel API rules: inputs are C-contiguous numpy arrays (float32/float64, int32 indexes) plus plain scalars.
Outputs are newly allocated arrays, or written into a numpy/memmap buffer passed in by the caller. No Python objects
inside loops. The GIL is released (`py.allow_threads`), and parallel over the outer axis (scenarios, sites) with rayon.

## 4. Build on the lab

- Local machines and workers may not have a toolchain. Setup for runs that want Rust (`bench`, and `train`,
  `baseline`, `replay` when `SENTINEL_RUST=1`): use `cargo` if present, otherwise install a minimal rustup into
  `$HOME/.cargo` (cached on the worker after the first run), then `uv pip install maturin && maturin develop
  --release -m rust/sentinel_core/Cargo.toml`. A failed build only prints a warning, and the run continues in Python.
- `check` stays pure Python, so merges never depend on cargo. A separate `rustcheck` run builds the crate and runs
  the parity tests. It doesn't gate merges, but kernel cards must pass it before they merge.
- First task on the Rust card: run a probe on each worker (does it have cargo, can it reach rustup, how long does a
  build take), noted on #8.

## 5. Python ML infra (`sentinel/infra/`)

| Piece | What it does | Used by |
|---|---|---|
| `run.py` `RunContext` | seeds (python/numpy/torch), device + AMP dtype pick (bf16 if supported, else fp16), `$LAB_OUTPUT_DIR`, config dump | train.py, eval.py, replay.py |
| `metrics.py` | `emit(name, value)` → `LAB_METRIC name=value` line + `$LAB_METRICS` JSON; prefix per model | all runs |
| `timing.py` | `with stage("windows"):` → `t_windows_s`; peak RSS and peak CUDA memory per stage | all runs |
| `manifest.py` | git sha, data hashes (Parquet files), config, kernel backend, package versions → `manifest.json` next to outputs; also feeds the `audit_record.model_version` | train, replay, #29 promote |
| `cache.py` | memmapped float16 window cache keyed by (data hash, window config); rebuilt in a deterministic way, stored under `$LAB_CACHE_DIR` or `.cache/` | #10, #14, #15 |
| `export.py` | torch → ONNX (opset 17) + onnxruntime parity check + calibration JSON next to it | #14, #41 |

Training conventions (for #14/#41): models fit in 4 GB at batch size 1; `--grid` keys map directly to config
fields; one ONNX file per model with a model card JSON. `val_nmae` is the leaderboard metric (lower is better).
`torch.compile` is off by default (compile time is too long for 20–30-minute runs); a bench can turn it on.

## 6. Cards and order

| Order | Card | Paths | Depends on |
|---|---|---|---|
| 1 | #6 repo skeleton (run types, requirements, package dirs) | as on the card | — |
| 2 | #9 RO data (critical path for everyone) | as on the card | ENTSO-E token |
| 3 | ML infra harness (§5) | `sentinel/infra/**`, `tests/test_infra.py` | #6 |
| 4 | Rust core: crate + dispatch + build steps + `bench`/`rustcheck` runs + worker probe | `rust/**`, `sentinel/kernels/__init__.py`, `sentinel/kernels/bench.py`, `lab.toml [runs.bench,runs.rustcheck]`, `tests/test_kernels.py` | #6 |
| 5 | K3 cusum kernel (with the Python reference that #15 uses) | `sentinel/kernels/cusum.py`, `rust/sentinel_core/src/cusum.rs` | Rust core |
| 6 | K1 rollout kernel | `sentinel/kernels/rollout.py`, `rust/sentinel_core/src/rollout.rs` | Rust core; battery model from #11 / #27 (the Python reference is copied from theirs, which stays the source of truth) |
| 7 | K2 window kernel | `sentinel/kernels/windows.py`, `rust/sentinel_core/src/windows.rs` | Rust core; #10 window spec |

Cards 3–4 run in parallel with #9 (agents work on them while data downloads). Cards 5–7 need a Python reference
first. Other people's modules call `sentinel.kernels.<name>` only once the kernel has passed parity and has a bench
result. Until then they keep their own code.

## 7. What "done" looks like

- `bench` run on main: one `speedup_<kernel>` metric per kernel, measured on the same worker type that runs the work.
- Plan-level numbers for the slide: replay wall time and P10 scenario count per feeder-day, before and after the
  kernels. Only measured numbers go on the slide.

## Kernel table

Status of every kernel registered in `sentinel/kernels` (#179). Parity = Rust output equals the Python reference:
bit-exact unless a tolerance is named. It is checked by each kernel's own `tests/test_kernels_<name>.py`, by
`tests/test_kernels_backend_equivalence.py` (every bench case, 3 seeds × 3 sizes, python vs rust, plus "every
registered kernel has a bench case and a Rust copy"), and by `parity_<key>` in `bench`. Speedup = `speedup_<key>` from
`bench`/`rustcheck` on the lab server (20 cores, rayon 20 threads, Python reference timed once, Rust best of 3).
"On main" is rustcheck#759 on main's code (#179); the range in brackets is min–max over that run and the card
runs listed below the table. Python timings move a lot with server load, so read the ranges, not the digits.

| Kernel (`register` name) | Bench key | Parity | Speedup on main (range) | Called from |
|---|---|---|---|---|
| K3 `cusum_two_sided` | `cusum`, `cusum_cap` | bit-exact | 23.3× (14–25×); capped 22.4× (16–30×) | `sentinel/eval/baselines/pv/cusum.py` → `faults.ThresholdCusumPredictor` (M4 fallback PV fault baseline) |
| K1 `battery_rollout` | `rollout` | bit-exact | 18.7× (5–19×) | `sentinel/opt/commit.py` `_net_export` (P10 commitment, replay, drills) |
| K2 `window_gather` | `windows` | bit-exact | 23.0× (13–23×) | `sentinel/model/features.py` (model data path: M1/M2/M4 train, eval) |
| K4 `rolling_mean` | `rolling` | bit-exact | 9.0× (5–13×) | `sentinel/model/wind/features.py` (M6 WindNBM classifier features) |
| K5 `scatter_telemetry` | `scatter` | bit-exact | 14.8× (11–15×) | `sentinel/model/wind/data.py` (WindNBM telemetry load) |
| K6 `pv_features` | `features_pv` | atol 1e-6 (cos/sin, 1 ulp) | 17.1× (6–23×) | `sentinel/model/features.py` `pv_inputs` (M1 PVNet) |
| K6 `load_features` | `features_load` | bit-exact | 11.8× (5–12×) | `sentinel/model/features.py` `load_inputs` (M2 LoadNet) |
| K6 `site_features` | `features_site` | atol 1e-6 (cos_zen) | 13.4× (4–13×) | `sentinel/model/features.py` `site_inputs` (M4 SiteHealth) |

Arriving (passed `rustcheck` on their card, not on main yet; the harness covers them as soon as they merge, because
it reads `bench.CASES`):

| Kernel | Card | Parity | Speedup | Called from |
|---|---|---|---|---|
| `pv_ac_batch` (+ `sitebatch.SiteReader`, not a kernel) | #133 R2 | bit-exact | 8.9× | `sentinel/data/windows.py` pvlib channels |
| `battery_resim` | #134 R3 | bit-exact | 224× | `sentinel/sim/faults/` fault injector battery re-simulation |
| `self_consumption` | #135 R4 | bit-exact (raw bits) | 27.6× | `sentinel/sim/fleet/generate.py` home batteries |
| `rolling_median`, `binned_median_mad` | #136 R5 | bit-exact | 18.0×, 7.2× | `sentinel/model/wind/normal_select.py`, `eval/baselines/wind/windmon_rules.py` |

#146, #147: no kernel in their diffs so far. #172: branch not pushed yet.

Source runs: rustcheck #759 (main, #179), #689 (#135), #692 (#133), #702 (#136), #705 (#134). All have
`rust_build_ok=1` and every `parity_*=1`.

**Check runtime (#179).** `check` (pure Python) took 80–130 s when it had the server mostly to itself
(check#645/#650/#653, same tests). Under load from concurrent runs, check#688/#691/#704/#716 and merge#701 stalled at
about 88% until the 900 s timeout.
All of them stopped in the same place: the `trained` fixture of `tests/test_windfailure.py`, which runs
`windfailure/train.py`. That trains LightGBM with `num_threads=0` (every core), and with several runs sharing the
server the OpenMP workers oversubscribe the CPU. Fix: `tests/conftest.py` sets `OMP_NUM_THREADS=2` unless the caller
set it, and registers a `slow` marker. The two tests that use the fixture are `@pytest.mark.slow`. `check` runs
`-m 'not slow'` with `faulthandler_timeout=300`, so a future hang prints a traceback, and `--durations=15`.
`rustcheck` also runs `-m slow`, so the coverage moves and nothing is dropped. Locally (4 cores, loaded) the fixture
takes 137 s with all threads and 67 s with one thread. After the fix, check#758 took 1m49s in total (pytest 84 s,
slowest test 6.9 s), and rustcheck#759 ran the fixture in 8.8 s with 2 threads.
