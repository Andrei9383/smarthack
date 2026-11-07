"""python -m sentinel.sim.prosumers_v2 [--quick] [--out DIR] [--force] [--n-homes N] [--share-battery X] ..."""

from __future__ import annotations

import argparse
import dataclasses
import os
import shutil
from pathlib import Path

from .config import QUICK, ProsumerConfig
from .generate import generate


def main() -> None:
    ap = argparse.ArgumentParser(prog="python -m sentinel.sim.prosumers_v2", description=__doc__)
    ap.add_argument("--quick", action="store_true", help="small world, one year (smoke test)")
    ap.add_argument("--out", type=Path, help="output dir (default $SENTINEL_CACHE/prosumers-v2/v1-<digest>)")
    ap.add_argument("--force", action="store_true", help="regenerate even if cached")
    for f in dataclasses.fields(ProsumerConfig):
        if f.type in ("int", "float", "str"):
            ap.add_argument(f"--{f.name.replace('_', '-')}", type={"int": int, "float": float, "str": str}[f.type])
    a = ap.parse_args()
    kw = dict(QUICK) if a.quick else {}
    for f in dataclasses.fields(ProsumerConfig):
        v = getattr(a, f.name, None)
        if v is not None:
            kw[f.name] = v
    out = generate(ProsumerConfig(**kw), out_dir=a.out, force=a.force)
    lab_out = os.environ.get("LAB_OUTPUT_DIR")
    if lab_out:  # small files for the run page; the dataset itself stays in the worker cache
        Path(lab_out).mkdir(parents=True, exist_ok=True)
        shutil.copy(out / "manifest.json", Path(lab_out) / "prosumers_v2_manifest.json")
        print(f"data-v2 at {out} (copied manifest to $LAB_OUTPUT_DIR)")


if __name__ == "__main__":
    main()
