"""python -m sentinel.sim.faults [--quick] [--fleet DIR] [--out DIR] [--force] [--seed N] [--share 0.15] [--plots]"""

from __future__ import annotations

import argparse
import dataclasses
from pathlib import Path

from ..fleet import FleetConfig, generate
from ..fleet.config import QUICK
from .dataset import inject_dataset
from .inject import FaultConfig


def main() -> None:
    ap = argparse.ArgumentParser(prog="python -m sentinel.sim.faults", description=__doc__)
    ap.add_argument("--quick", action="store_true", help="use the quick fleet (120 sites, 2024)")
    ap.add_argument("--fleet", type=Path, help="fleet dir (default: generate/cached default fleet)")
    ap.add_argument("--out", type=Path)
    ap.add_argument("--force", action="store_true")
    ap.add_argument("--seed", type=int, default=FaultConfig.seed)
    ap.add_argument("--share", type=float, default=FaultConfig.faulty_share, help="target faulty site-day share")
    ap.add_argument("--plots", type=Path, nargs="?", const=Path(__file__).parent / "plots",
                    help="write one explainer plot per fault class (default: sentinel/sim/faults/plots)")
    a = ap.parse_args()
    fleet = a.fleet or generate(FleetConfig(**(QUICK if a.quick else {})), plots=False)
    cfg = dataclasses.replace(FaultConfig(), seed=a.seed, faulty_share=a.share)
    out = inject_dataset(fleet, cfg, out_dir=a.out, force=a.force)
    if a.plots:
        from .plots import explainer_plots
        explainer_plots(fleet, out, a.plots)


if __name__ == "__main__":
    main()
