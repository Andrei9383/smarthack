"""python -m sentinel.sim.wind [--out DIR] [--force] [--no-faults] [--no-plots] [--start 2024-01-01 --end 2026-01-01]
[--max-turbines-per-park N] [--seed 22] [--fault-seed 23]

Deterministic: same config + same data/raw inputs → same files (cached by digest; --force rebuilds).
"""

from __future__ import annotations

import argparse
import dataclasses
from pathlib import Path

from .build import build
from .config import WindSimConfig


def main(argv=None) -> None:
    from sentinel.sim.wind_faults import FaultConfig

    ap = argparse.ArgumentParser(prog="python -m sentinel.sim.wind", description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out", type=Path, help="output dir (default $SENTINEL_WIND_DIR or ~/.cache/sentinel/data-vN/wind)")
    ap.add_argument("--force", action="store_true")
    ap.add_argument("--no-faults", action="store_true", help="clean telemetry only (no #23 injector, no labels)")
    ap.add_argument("--no-plots", action="store_true")
    ap.add_argument("--no-quality", action="store_true", help="skip the quality-rule summary in the manifest")
    ap.add_argument("--start")
    ap.add_argument("--end")
    ap.add_argument("--seed", type=int)
    ap.add_argument("--fault-seed", type=int)
    ap.add_argument("--max-turbines-per-park", type=int)
    ap.add_argument("--shear", choices=["fit", "fixed"])
    ap.add_argument("--no-cf-calibration", action="store_true")
    a = ap.parse_args(argv)
    kw = {k: v for k, v in {"start": a.start, "end": a.end, "seed": a.seed, "shear": a.shear,
                            "max_turbines_per_park": a.max_turbines_per_park}.items() if v is not None}
    if a.no_cf_calibration:
        kw["cf_calibration"] = False
    cfg = dataclasses.replace(WindSimConfig(), **kw)
    fcfg = FaultConfig() if a.fault_seed is None else FaultConfig(seed=a.fault_seed)
    build(cfg, fcfg, out_dir=a.out, force=a.force, faults=not a.no_faults, plots=not a.no_plots,
          quality=not a.no_quality)


if __name__ == "__main__":
    main()
