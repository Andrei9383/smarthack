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
