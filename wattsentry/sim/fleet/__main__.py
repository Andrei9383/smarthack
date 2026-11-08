"""python -m wattsentry.sim.fleet [--quick] [--out DIR] [--force] [--no-plots] [--sites N] [--feeders N] ..."""

from __future__ import annotations

import argparse
import dataclasses
from pathlib import Path

from .config import QUICK, FleetConfig
from .generate import generate


def main() -> None:
    ap = argparse.ArgumentParser(prog="python -m wattsentry.sim.fleet", description=__doc__)
    ap.add_argument("--quick", action="store_true", help="small world, one year (smoke test)")
    ap.add_argument("--out", type=Path, help="output dir (default ~/.cache/wattsentry/data-vN/fleet-<digest>)")
    ap.add_argument("--force", action="store_true", help="regenerate even if cached")
    ap.add_argument("--no-plots", action="store_true")
    for f in dataclasses.fields(FleetConfig):
        if f.type in ("int", "float", "str"):
            ap.add_argument(f"--{f.name.replace('_', '-')}", type={"int": int, "float": float, "str": str}[f.type])
    a = ap.parse_args()
    kw = dict(QUICK) if a.quick else {}
    for f in dataclasses.fields(FleetConfig):
        v = getattr(a, f.name, None)
        if v is not None:
            kw[f.name] = v
    generate(FleetConfig(**kw), out_dir=a.out, force=a.force, plots=not a.no_plots)


if __name__ == "__main__":
    main()
