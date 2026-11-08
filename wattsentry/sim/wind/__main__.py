"""python -m wattsentry.sim.wind [--out DIR] [--force] [--no-plots]: build (or reuse) the cached wind dataset."""

from __future__ import annotations

import argparse
from pathlib import Path

from .build import build


def main(argv=None) -> None:
    ap = argparse.ArgumentParser(prog="python -m wattsentry.sim.wind", description=__doc__)
    ap.add_argument("--out", type=Path,
                    help="output dir (default $WATTSENTRY_WIND_DIR or $WATTSENTRY_CACHE/data-vN/wind)")
    ap.add_argument("--force", action="store_true")
    ap.add_argument("--no-plots", action="store_true")
    a = ap.parse_args(argv)
    build(out_dir=a.out, force=a.force, plots=not a.no_plots)


if __name__ == "__main__":
    main()
