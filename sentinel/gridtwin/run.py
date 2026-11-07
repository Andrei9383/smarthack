"""python -m sentinel.gridtwin.run [--fleet DIR] [--out DIR] [--start 2024-05-01] [--days 60]

Runs the grid-twin on the fleet's inverter voltages, scores it against the simulator's hidden truth and writes
flags.geojson (review requests), feeders.json (tau, g_kw per feeder) and sites.parquet to the output dir."""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

from sentinel.sim.fleet import FleetConfig, generate
from sentinel.sim.fleet.config import DATA_VERSION, default_cache_root

from .twin import load_voltages, run_twin, to_geojson


def default_out() -> Path:
    return default_cache_root() / f"data-v{DATA_VERSION}" / "gridtwin"


def score(res, sites) -> dict:
    """Recall / precision of the flags against the simulator truth (feeder_id, phase are physical)."""
    s = res.sites.merge(sites[["site_id", "feeder_id", "phase"]], on="site_id")
    wrong_phase = (s.gis_phase != s.phase) & (s.gis_feeder_id == s.feeder_id)
    wrong_feeder = s.gis_feeder_id != s.feeder_id
    pf, ff = s.phase_flag, s.feeder_flag
    ok_phase = pf & (s.likely_phase == s.phase)
    ok_feeder = ff & (s.likely_feeder_id == s.feeder_id)
    div = lambda a, b: round(float(a) / float(b), 3) if b else 1.0  # noqa: E731
    return {
        "phase_recovery": div((ok_phase & wrong_phase).sum(), wrong_phase.sum()),
        "phase_precision": div(ok_phase.sum(), pf.sum()),
        "feeder_recovery": div((ok_feeder & wrong_feeder).sum(), wrong_feeder.sum()),
        "feeder_precision": div(ok_feeder.sum(), ff.sum()),
        "n_wrong_phase": int(wrong_phase.sum()), "n_wrong_feeder": int(wrong_feeder.sum()),
        "n_flags": int((pf | ff).sum()), "tau_mean": round(float(res.feeders.tau.mean()), 3),
    }


def main(argv=None) -> None:
    ap = argparse.ArgumentParser(prog="python -m sentinel.gridtwin.run", description=__doc__.split("\n\n")[0])
    ap.add_argument("--fleet", type=Path)
    ap.add_argument("--out", type=Path)
    ap.add_argument("--start", default="2024-05-01")
    ap.add_argument("--days", type=int, default=60)
    ap.add_argument("--threshold", type=float, default=0.7)
    a = ap.parse_args(argv)
    t0 = time.perf_counter()
    fleet = a.fleet or generate(FleetConfig(), plots=False)
    import pandas as pd

    sites, v = load_voltages(fleet, a.start, a.days)
    feeders = pd.read_parquet(Path(fleet) / "feeders.parquet")
    res = run_twin(sites, v, feeders, a.threshold)
    m = score(res, sites)
    out = a.out or default_out()
    out.mkdir(parents=True, exist_ok=True)
    (out / "flags.geojson").write_text(json.dumps(to_geojson(res), ensure_ascii=False))
    (out / "feeders.json").write_text(json.dumps(res.feeders.to_dict("records")))
    res.sites.to_parquet(out / "sites.parquet", index=False)
    (out / "metrics.json").write_text(json.dumps({**m, "fleet_dir": str(fleet), "window": [a.start, a.days]}))
    print(res.feeders.to_string(index=False))
    for r in res.flags.head(8).itertuples():
        print(f"  {r.site_id}: {r.message}")
    for k, val in m.items():
        print(f"LAB_METRIC {k}={val}")
    print(f"LAB_METRIC gridtwin_seconds={time.perf_counter() - t0:.1f}")
    print(f"LAB_METRIC t_total_s={time.perf_counter() - t0:.1f}")
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
