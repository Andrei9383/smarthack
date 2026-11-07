"""Time the window cache build (sentinel.data.windows) on an N-home fleet and hash its outputs (card #133).

    python -m sentinel.kernels.profile_windows [--homes 5000] [--backends rust,python] [--old OLD_WINDOWS_PY]

Builds (or reuses from the cache) the fleet + faults with scripts/build_dataset.py's national config, then runs
build_windows once per backend (SENTINEL_KERNELS) and, with --old, once with an older copy of windows.py (e.g.
``git show <sha>:sentinel/data/windows.py > old.py``). Prints ``LAB_METRIC windows_gen_seconds_<label>`` and
``windows_hash_equal`` (1 = every run wrote the same bytes). Each output is deleted after hashing (~3.7 GB at 5000).
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
import os
import shutil
import sys
import time
from pathlib import Path

import numpy as np

COUNTY_WEIGHTS = {  # scripts/build_dataset.py (card #118), national zones
    "Ilfov": 3.0, "Timiș": 2.19, "Bihor": 1.78, "Dolj": 1.75, "Iași": 1.50, "Cluj": 1.4, "Brașov": 1.3,
    "Constanța": 1.2, "Prahova": 1.0, "Sibiu": 1.0, "Argeș": 0.8, "Galați": 0.5, "Bacău": 0.5,
}


def digest(out: Path) -> str:
    h = hashlib.sha256()
    for f in sorted(out.glob("*.npy")):
        a = np.load(f, mmap_mode="r")
        h.update(f.name.encode())
        for i in range(0, a.shape[0], 256):
            h.update(np.ascontiguousarray(a[i:i + 256]).tobytes())
    for name in ("sites.json", "episodes.json"):
        h.update((out / name).read_bytes())
    return h.hexdigest()[:16]


def main(argv: list[str] | None = None) -> int:
    from sentinel.data.windows import build_windows
    from sentinel.sim.faults import FaultConfig, inject_dataset
    from sentinel.sim.fleet import FleetConfig, generate
    from sentinel.sim.fleet.weather import weather_key

    p = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    p.add_argument("--homes", type=int, default=5000)
    p.add_argument("--backends", default="rust,python")
    p.add_argument("--old", type=Path, help="an older windows.py to time on the same data")
    p.add_argument("--out", type=Path, default=Path(os.environ.get("LAB_OUTPUT_DIR", ".")) / "windows-profile")
    a = p.parse_args(argv)
    feeders = round(a.homes / 25)
    cfg = FleetConfig(n_sites=a.homes, n_feeders=feeders, n_congested=round(feeders * 0.25), start_year=2025,
                      end_year=2025, battery_share=0.20, county_weights=COUNTY_WEIGHTS)
    fleet = generate(cfg, out_dir=cfg.out_dir(weather_key=weather_key(cfg.weather_source)), plots=False)
    faults = inject_dataset(fleet, FaultConfig())

    runs = [(be, build_windows, be) for be in a.backends.split(",") if be]
    if a.old:
        spec = importlib.util.spec_from_file_location("windows_old", a.old)
        mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(mod)  # type: ignore[union-attr]
        runs.insert(0, ("old", mod.build_windows, "python"))
    hashes = {}
    for label, fn, be in runs:
        os.environ["SENTINEL_KERNELS"] = be
        t0 = time.perf_counter()
        out = fn(fleet_dir=fleet, fault_dir=faults, out=a.out, force=True, log=print)
        dt = time.perf_counter() - t0
        hashes[label] = digest(out)
        print(f"LAB_METRIC windows_gen_seconds_{label}={dt:.1f}", flush=True)
        print(f"[profile_windows] {label}: {dt:.1f}s hash {hashes[label]}", flush=True)
        shutil.rmtree(out)
    print(f"LAB_METRIC windows_hash_equal={int(len(set(hashes.values())) == 1)}", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
