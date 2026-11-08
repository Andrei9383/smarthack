"""Time the window cache build (wattsentry.data.windows) on an N-home fleet and hash its outputs (card #133).

    python -m wattsentry.kernels.profile_windows [--homes 2000] [--backends rust,python] [--old-rev SHA | --old FILE]

Builds (or reuses from the cache) the fleet + faults with scripts/build_dataset.py's national config, then runs
build_windows once per backend (WATTSENTRY_KERNELS) and, with --old-rev / --old, once with an older copy of
windows.py. Prints ``LAB_METRIC windows_gen_seconds_<label>`` and
``rss_windows_<label>_mb`` (peak RSS of that build, each in its own process), ``rss_data_mb`` and
``windows_hash_equal`` (1 = every run wrote the same bytes). Each output is deleted after hashing (~1.5 GB at 2000).
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
import multiprocessing as mp
import os
import resource
import shutil
import subprocess
import sys
import tempfile
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


def _rss_mb() -> float:
    """Peak RSS of this process image: VmHWM resets on exec (ru_maxrss would carry the parent's peak into the
    spawned child); falls back to ru_maxrss where /proc is missing."""
    try:
        for line in Path("/proc/self/status").read_text().splitlines():
            if line.startswith("VmHWM:"):
                return int(line.split()[1]) / 1024
    except OSError:
        pass
    return resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024  # KiB on Linux


def _build(old: str | None, backend: str, fleet: str, faults: str, out: str, q) -> None:
    """One build in a fresh (spawned) process, so ru_maxrss is that build's peak alone."""
    os.environ["WATTSENTRY_KERNELS"] = backend
    if old:
        spec = importlib.util.spec_from_file_location("windows_old", old)
        mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(mod)  # type: ignore[union-attr]
        fn = mod.build_windows
    else:
        from wattsentry.data.windows import build_windows as fn
    t0 = time.perf_counter()
    o = fn(fleet_dir=Path(fleet), fault_dir=Path(faults), out=Path(out), force=True, log=print)
    dt, rss = time.perf_counter() - t0, _rss_mb()  # before hashing, which touches the mmapped outputs
    q.put((dt, rss, digest(o)))
    shutil.rmtree(o)


def main(argv: list[str] | None = None) -> int:
    from wattsentry.sim.faults import FaultConfig, inject_dataset
    from wattsentry.sim.fleet import FleetConfig, generate
    from wattsentry.sim.fleet.weather import weather_key

    p = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    p.add_argument("--homes", type=int, default=2000)
    p.add_argument("--backends", default="rust,python")
    p.add_argument("--old", type=Path, help="an older windows.py to time on the same data")
    p.add_argument("--old-rev", help="git revision to take the older wattsentry/data/windows.py from (instead of --old)")
    p.add_argument("--out", type=Path, default=Path(os.environ.get("LAB_OUTPUT_DIR", ".")) / "windows-profile")
    a = p.parse_args(argv)
    from wattsentry.sim.fleet.localities import LOCALITIES

    # main's generator allows one feeder per locality (card #118 lifts that); the windows build scales with homes
    feeders = min(round(a.homes / 25), len(LOCALITIES))
    cfg = FleetConfig(n_sites=a.homes, n_feeders=feeders, n_congested=round(feeders * 0.25), start_year=2025,
                      end_year=2025, battery_share=0.20, county_weights=COUNTY_WEIGHTS)
    fleet = generate(cfg, out_dir=cfg.out_dir(weather_key=weather_key(cfg.weather_source)), plots=False)
    faults = inject_dataset(fleet, FaultConfig())

    if a.old_rev and not a.old:
        r = subprocess.run(["git", "show", f"{a.old_rev}:wattsentry/data/windows.py"], capture_output=True, text=True)
        if r.returncode == 0:
            a.old = Path(tempfile.mkdtemp()) / "windows_old.py"
            a.old.write_text(r.stdout)
        else:
            print(f"[profile_windows] no old windows.py at {a.old_rev}: {r.stderr.strip()}", flush=True)
    print(f"LAB_METRIC rss_data_mb={_rss_mb():.0f}", flush=True)  # fleet + faults, this process
    runs = [(be, None, be) for be in a.backends.split(",") if be]
    if a.old:
        runs.insert(0, ("old", str(a.old), "python"))
    ctx = mp.get_context("spawn")
    hashes = {}
    for label, old, be in runs:
        q = ctx.Queue()
        pr = ctx.Process(target=_build, args=(old, be, str(fleet), str(faults), str(a.out), q))
        pr.start()
        pr.join()
        if pr.exitcode != 0:
            print(f"[profile_windows] {label}: build failed (exit {pr.exitcode})", flush=True)
            return 1
        dt, rss, hashes[label] = q.get()
        print(f"LAB_METRIC windows_gen_seconds_{label}={dt:.1f}", flush=True)
        print(f"LAB_METRIC rss_windows_{label}_mb={rss:.0f}", flush=True)
        print(f"[profile_windows] {label}: {dt:.1f}s peak {rss:.0f} MB hash {hashes[label]}", flush=True)
    print(f"LAB_METRIC windows_hash_equal={int(len(set(hashes.values())) == 1)}", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
