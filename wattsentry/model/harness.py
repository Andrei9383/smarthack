"""Small self-contained run harness for #14 (mirrors team/infra.md §5 until wattsentry/infra lands).

RunContext: seeds, device + AMP dtype (bf16 on CUDA when supported, else fp16; fp32 on CPU), $LAB_OUTPUT_DIR
(default ./.lab-out), `emit()` -> `LAB_METRIC name=value` + metrics.json, `stage()` -> `LAB_METRIC t_<stage>_s`
with peak CUDA memory, manifest.json (git sha, config, versions, data digest), ONNX export + onnxruntime parity.
"""

from __future__ import annotations

import contextlib
import json
import math
import os
import platform
import random
import subprocess
import sys
import time
from pathlib import Path

import numpy as np


def _git_sha() -> str:
    for k in ("LAB_COMMIT", "LAB_SHA", "GIT_COMMIT"):
        if os.environ.get(k):
            return os.environ[k]
    try:
        return subprocess.check_output(["git", "rev-parse", "HEAD"], stderr=subprocess.DEVNULL, text=True).strip()
    except Exception:
        return "unknown"


def _versions() -> dict:
    out = {"python": sys.version.split()[0], "platform": platform.platform()}
    for m in ("numpy", "torch", "onnx", "onnxruntime", "pvlib", "pandas", "scipy", "sklearn"):
        try:
            out[m] = __import__(m).__version__
        except Exception:
            pass
    return out


class RunContext:
    def __init__(self, model: str, config: dict, seed: int = 0, device: str = "auto", prefix: str = ""):
        self.model, self.config, self.seed, self.prefix = model, dict(config), seed, prefix
        self.t0 = time.time()
        self.out = Path(os.environ.get("LAB_OUTPUT_DIR") or ".lab-out")
        self.out.mkdir(parents=True, exist_ok=True)
        self.metrics: dict[str, float] = {}
        self.stages: dict[str, dict] = {}
        self.sha = _git_sha()
        run_id = os.environ.get("LAB_RUN_ID") or os.environ.get("LAB_RUN") or self.sha[:7]
        self.run_id = run_id
        self.model_version = f"{model}-{run_id}-s{seed}"
        self._seed_all(seed)
        self.torch = None
        try:
            import torch

            self.torch = torch
        except ImportError:
            pass
        self.device = self._pick_device(device)
        self.amp_dtype = None
        if self.device == "cuda":
            t = self.torch
            self.amp_dtype = t.bfloat16 if t.cuda.is_bf16_supported() else t.float16
        print(f"[run] {self.model_version} device={self.device} amp={self.amp_dtype} out={self.out}", flush=True)

    def _seed_all(self, seed: int):
        random.seed(seed)
        np.random.seed(seed)
        try:
            import torch

            torch.manual_seed(seed)
            if torch.cuda.is_available():
                torch.cuda.manual_seed_all(seed)
        except ImportError:
            pass

    def _pick_device(self, device: str) -> str:
        if device != "auto":
            return device
        if self.torch is not None and self.torch.cuda.is_available():
            return "cuda"
        return "cpu"

    def autocast(self):
        if self.amp_dtype is None:
            return contextlib.nullcontext()
        return self.torch.autocast("cuda", dtype=self.amp_dtype)

    def elapsed_min(self) -> float:
        return (time.time() - self.t0) / 60

    # -- metrics ------------------------------------------------------------------------------------------
    def emit(self, name: str, value) -> None:
        try:
            v = float(value)
        except (TypeError, ValueError):
            return
        if math.isnan(v) or math.isinf(v):
            print(f"[metric] {name}=nan (not emitted)", flush=True)
            return
        name = self.prefix + name
        self.metrics[name] = v
        print(f"LAB_METRIC {name}={v:.6g}", flush=True)

    def get(self, name: str, default=float("nan")) -> float:
        return self.metrics.get(self.prefix + name, default)

    @contextlib.contextmanager
    def stage(self, name: str):
        t = time.time()
        cuda = self.device == "cuda" and self.torch is not None
        if cuda:
            self.torch.cuda.reset_peak_memory_stats()
        print(f"[stage] {name} ...", flush=True)
        try:
            yield
        finally:
            dt = time.time() - t
            rec = {"seconds": round(dt, 2)}
            if cuda:
                rec["peak_cuda_mb"] = round(self.torch.cuda.max_memory_allocated() / 2**20, 1)
            self.stages[name] = rec
            self.emit(f"t_{name}_s", dt)
            if cuda:
                self.emit(f"mem_{name}_mb", rec["peak_cuda_mb"])

    # -- artifacts ----------------------------------------------------------------------------------------
    def write_json(self, name: str, obj) -> Path:
        p = self.out / name
        p.write_text(json.dumps(obj, indent=1, default=_json_default))
        return p

    def finish(self, extra: dict | None = None) -> None:
        self.emit("t_total_s", time.time() - self.t0)
        name = self.model + (f".{self.prefix.rstrip('_')}" if self.prefix else "")
        lab_env = {k: v for k, v in os.environ.items() if k.startswith("LAB_")
                   and not any(s in k for s in ("TOKEN", "SECRET", "KEY", "PASS", "INVITE"))}
        manifest = {
            "model": self.model, "model_version": self.model_version, "run_id": self.run_id, "git_sha": self.sha,
            "seed": self.seed, "config": self.config, "device": self.device,
            "gpu": (self.torch.cuda.get_device_name(0) if self.device == "cuda" else None),
            "amp": str(self.amp_dtype), "versions": _versions(), "stages": self.stages, "metrics": self.metrics,
            "lab_env": lab_env, "argv": sys.argv, **(extra or {}),
        }
        self.write_json(f"{name}.manifest.json", manifest)
        mpath = os.environ.get("LAB_METRICS")
        allm = self.out / "metrics.json"
        prev = json.loads(allm.read_text()) if allm.exists() else {}
        prev.update(self.metrics)
        allm.write_text(json.dumps(prev, indent=1))
        if mpath:
            try:
                Path(mpath).write_text(json.dumps(prev, indent=1))
            except OSError:
                pass


class GpuSampler:
    """Background GPU utilisation sampler (#140): torch.cuda.utilization() (pynvml) if it works, else nvidia-smi.
    `mean()` = mean % busy over the samples taken between start() and stop(); NaN when no GPU / no tool."""

    def __init__(self, interval: float = 0.5):
        import threading

        self.interval, self.samples, self._stop = interval, [], threading.Event()
        self._thread = threading.Thread(target=self._loop, daemon=True)
        self._read = None

    def _probe(self):
        try:
            import torch

            if not torch.cuda.is_available():
                return None
            torch.cuda.utilization()
            return lambda: float(torch.cuda.utilization())
        except Exception:
            pass
        vis = os.environ.get("CUDA_VISIBLE_DEVICES", "0").split(",")[0].strip() or "0"
        cmd = ["nvidia-smi", "--query-gpu=utilization.gpu", "--format=csv,noheader,nounits", "-i", vis]

        def smi():
            return float(subprocess.check_output(cmd, stderr=subprocess.DEVNULL, text=True, timeout=5).split()[0])

        try:
            smi()
            return smi
        except Exception:
            return None

    def _loop(self):
        while not self._stop.wait(self.interval):
            try:
                self.samples.append(self._read())
            except Exception:
                return

    def start(self) -> "GpuSampler":
        self._read = self._probe()
        if self._read is not None:
            self._thread.start()
        return self

    def stop(self) -> float:
        self._stop.set()
        if self._thread.is_alive():
            self._thread.join(timeout=2)
        return self.mean()

    def mean(self) -> float:
        return float(np.mean(self.samples)) if self.samples else float("nan")


class Throughput:
    """Train-loop accounting (#140): data wait (blocked on the loader) vs compute (H2D + forward/backward/step) vs
    per-epoch validation, samples/s and GPU utilisation. With sync=True (--profile) each compute block ends with a
    cuda synchronize, so the split is exact (costs a little throughput); otherwise compute is the CPU-side time and
    the GPU tail shows up in the next wait/val block.

        tp = Throughput(emit, device, sync=args.profile)
        with tp.wait(): batch = next_batch()
        with tp.compute(len(batch)): ...step...
        with tp.val(): ...validation...
        tp.finish()  -> LAB_METRIC samples_per_s, gpu_util_mean, t_data_wait_s, t_compute_s, t_val_s, data_wait_frac
    """

    def __init__(self, emit, device: str = "cpu", sync: bool = False):
        self.emit, self.cuda, self.sync = emit, str(device).startswith("cuda"), sync
        self.t = {"data_wait": 0.0, "compute": 0.0, "val": 0.0}
        self.samples, self.steps = 0, 0
        self.warm_s, self.warm_n, self.warm_steps = 0.0, 0, 0
        self.gpu = GpuSampler().start() if self.cuda else None
        self.t0 = time.perf_counter()

    def _sync(self):
        if self.cuda:
            import torch

            torch.cuda.synchronize()

    @contextlib.contextmanager
    def _block(self, key: str, sync: bool):
        t = time.perf_counter()
        try:
            yield
        finally:
            if sync:
                self._sync()
            self.t[key] += time.perf_counter() - t

    def wait(self):
        return self._block("data_wait", self.sync)

    def compute(self, n: int):
        self.samples += int(n)
        self.steps += 1
        return self._block("compute", self.sync)

    def val(self):
        return self._block("val", True)

    def warmup(self, seconds: float, n: int, steps: int = 1) -> None:
        """Leave a one-off block (e.g. the torch.compile first step) out of samples_per_s / ms_per_step."""
        self.warm_s, self.warm_n, self.warm_steps = self.warm_s + seconds, self.warm_n + n, self.warm_steps + steps

    def finish(self, prefix: str = "") -> dict:
        wall = time.perf_counter() - self.t0
        util = self.gpu.stop() if self.gpu else float("nan")
        train_wall = max(wall - self.t["val"] - self.warm_s, 1e-9)
        out = {"samples_per_s": (self.samples - self.warm_n) / train_wall, "gpu_util_mean": util,
               "t_data_wait_s": self.t["data_wait"], "t_compute_s": self.t["compute"], "t_val_s": self.t["val"],
               "data_wait_frac": self.t["data_wait"] / train_wall,
               "ms_per_step": 1000 * train_wall / max(self.steps - self.warm_steps, 1)}
        for k, v in out.items():
            self.emit(prefix + k, v)
        return out


def resolve_compile(mode: str | None) -> str:
    """auto -> reduce-overhead on CUDA with compute capability >= 7.5 (triton + CUDA graphs), else off."""
    if mode in (None, "", "off"):
        return "off"
    if mode != "auto":
        return mode
    try:
        import torch

        if torch.cuda.is_available() and torch.cuda.get_device_capability() >= (7, 5):
            return "reduce-overhead"
    except Exception:
        pass
    return "off"


def compile_net(net, mode: str = "off"):
    """torch.compile(net) (a module or a bound forward) for the train forward only; eval / export / state_dict keep
    using `net` (shared weights).
    Falls back to eager (and says so) if the compiler is unavailable on this worker."""
    mode = resolve_compile(mode)
    if mode == "off":
        return net
    import torch

    try:
        import torch._dynamo

        torch._dynamo.config.suppress_errors = True  # compile errors (e.g. no triton) fall back to eager per frame
        return torch.compile(net, mode=None if mode == "default" else mode, dynamic=False)
    except Exception as e:  # pragma: no cover - depends on the worker's toolchain
        print(f"[compile] unavailable ({type(e).__name__}: {e}); training eagerly", flush=True)
        return net


def pin(t):
    """Pinned host copy (when CUDA is up) so .to(dev, non_blocking=True) really overlaps."""
    try:
        return t.pin_memory()
    except Exception:
        return t


def _json_default(o):
    if isinstance(o, (np.floating,)):
        return float(o)
    if isinstance(o, (np.integer,)):
        return int(o)
    if isinstance(o, np.ndarray):
        return o.tolist()
    if isinstance(o, Path):
        return str(o)
    return str(o)


def export_onnx(module, inputs: dict, path: Path, output_names: list[str], atol: float = 2e-3) -> dict:
    """Export an eval-mode fp32 module (opset 17, dynamic batch) and check onnxruntime parity on `inputs`."""
    import torch

    import copy

    module = copy.deepcopy(module).float().eval().cpu()  # never move the caller's (possibly CUDA) module
    names = list(inputs)
    args = tuple(torch.as_tensor(inputs[k]).float() for k in names)
    dyn = {k: {0: "batch"} for k in names + output_names}
    with torch.no_grad():
        ref = module(*args)
    ref = ref if isinstance(ref, (tuple, list)) else (ref,)
    import warnings

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        torch.onnx.export(module, args, str(path), input_names=names, output_names=output_names,
                          dynamic_axes=dyn, opset_version=17, dynamo=False, do_constant_folding=True)
    import onnxruntime as ort

    sess = ort.InferenceSession(str(path), providers=["CPUExecutionProvider"])
    got = sess.run(output_names, {k: np.asarray(inputs[k], np.float32) for k in names})
    diffs = {n: float(np.abs(g - r.numpy()).max()) for n, g, r in zip(output_names, got, ref)}
    ok = all(d <= atol * max(1.0, float(np.abs(r.numpy()).max())) for d, r in zip(diffs.values(), ref))
    print(f"[onnx] {path.name} parity max|diff|={diffs} ok={ok}", flush=True)
    return {"path": str(path), "max_abs_diff": diffs, "parity_ok": bool(ok), "opset": 17}
