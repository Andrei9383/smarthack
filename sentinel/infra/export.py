"""torch → ONNX (opset 17) + onnxruntime parity check + calibration JSON next to the model."""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Sequence

import numpy as np


def export_onnx(model: Any, inputs: Sequence[Any], path: str | Path, input_names: Sequence[str],
                output_names: Sequence[str], dynamic_axes: dict | None = None, check_inputs: Sequence[Any] | None = None,
                calibration: dict | None = None, atol: float = 1e-3, opset: int = 17) -> float:
    """Exports, runs torch and onnxruntime on `check_inputs` (default `inputs`), returns max |diff|; raises above
    `atol`. Writes `<stem>.calibration.json` when `calibration` is given."""
    import onnxruntime as ort
    import torch

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    model = model.eval()
    torch.onnx.export(model, tuple(inputs), str(path), input_names=list(input_names), output_names=list(output_names),
                      dynamic_axes=dynamic_axes, opset_version=opset, dynamo=False)
    xs = tuple(check_inputs if check_inputs is not None else inputs)
    with torch.no_grad():
        ref = model(*xs)
    ref = [r.numpy() for r in (ref if isinstance(ref, (tuple, list)) else (ref,))]
    got = ort.InferenceSession(str(path), providers=["CPUExecutionProvider"]).run(
        None, {n: x.numpy() for n, x in zip(input_names, xs)})
    parity = max(float(np.abs(a - b).max()) for a, b in zip(ref, got))
    if parity > atol:
        raise ValueError(f"ONNX parity check failed for {path.name}: {parity:.3g} > {atol}")
    if calibration is not None:
        path.with_suffix(".calibration.json").write_text(json.dumps(calibration, indent=1, default=float))
    return parity
