"""Small self-contained run harness for #14 (mirrors team/infra.md §5 until sentinel/infra lands).

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
