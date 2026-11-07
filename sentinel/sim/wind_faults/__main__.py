"""python -m sentinel.sim.wind_faults [DIR]: fault-label summary of a wind dataset (default: the wind cache).

Injection itself runs inside `python -m sentinel.sim.wind` (it needs the clean simulator state)."""

from __future__ import annotations

import json
import sys
from pathlib import Path


def main(argv=None) -> None:
    import polars as pl

    from sentinel.model.wind.data import external_wind_dir

    argv = sys.argv[1:] if argv is None else argv
    d = Path(argv[0]) if argv else external_wind_dir()
    if d is None or not (d / "fault_labels.parquet").exists():
        raise SystemExit("no wind dataset with fault_labels.parquet; run `python -m sentinel.sim.wind` first")
    fl = pl.read_parquet(d / "fault_labels.parquet")
    kinds = fl.with_columns(pl.col("params").map_elements(lambda p: json.loads(p)["kind"], return_dtype=pl.String)
                            .alias("kind"),
                            ((pl.col("end") - pl.col("onset")).dt.total_minutes() / 60).alias("hours"))
    print(f"{d}: {fl.height} episodes on {fl['asset_id'].n_unique()} turbines")
    print(kinds.group_by("fault", "kind").agg(pl.len().alias("episodes"), pl.col("hours").median().alias("median_h"))
          .sort("fault", "kind"))
    m = d / "manifest.json"
    if m.exists():
        print(json.dumps(json.loads(m.read_text()).get("fault_stats"), indent=1))


if __name__ == "__main__":
    main()
