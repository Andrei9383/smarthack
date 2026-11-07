"""Run the injector over a whole fleet dataset (sentinel.sim.fleet output) in batches of sites, all years at once
(the battery is continuous across years), and write the faulty copy + fault_labels.parquet."""

from __future__ import annotations

import json
import shutil
import time
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq

from ..fleet.config import DATA_VERSION
from .inject import TELEMETRY_COLS, FaultConfig, inject_batch, labels_frame

PQ = dict(compression="zstd", compression_level=6)
ROUND = {"pv_ac_w": 1, "inv_voltage_v": 0.1, "soc": 0.001, "batt_w": 1, "load_w": 1, "grid_w": 1}


def default_out(fleet_dir: Path, cfg: FaultConfig) -> Path:
    import hashlib

    key = hashlib.sha256(json.dumps({"fleet": fleet_dir.name, **cfg.to_json()}, sort_keys=True).encode()).hexdigest()
    return fleet_dir.parent / f"faults-{key[:12]}"


def inject_dataset(fleet_dir: Path, cfg: FaultConfig = FaultConfig(), out_dir: Path | None = None,
                   batch: int = 64, force: bool = False, log=print) -> Path:
    fleet_dir = Path(fleet_dir)
    out = Path(out_dir) if out_dir else default_out(fleet_dir, cfg)
    if (out / "manifest.json").exists() and not force:
        log(f"cached: {out}")
        return out
    tmp = out.with_name(out.name + ".tmp")
    for p in (out, tmp):
        if p.exists():
            shutil.rmtree(p)
    (tmp / "telemetry").mkdir(parents=True)
    t0 = time.perf_counter()
    sites = pd.read_parquet(fleet_dir / "sites.parquet")
    files = sorted((fleet_dir / "telemetry").glob("year=*.parquet"))
    writers: dict[Path, pq.ParquetWriter] = {}
    labels: list[dict] = []
    try:
        for b0 in range(0, len(sites), batch):
            sb = sites.iloc[b0:b0 + batch].reset_index(drop=True)
            ids = list(sb.site_id)
            parts, lengths = [], []
            for f in files:
                df = pq.read_table(f, filters=[("site_id", "in", ids)]).to_pandas()
                df["site_id"] = df.site_id.astype(str)
                df = df.sort_values(["site_id", "ts"], kind="stable")
                df = df.set_index("site_id").loc[ids].reset_index()  # batch order
                parts.append(df)
                lengths.append(len(df) // len(ids))
            ts = pd.DatetimeIndex(np.concatenate([p.ts.to_numpy()[:n] for p, n in zip(parts, lengths)]))
            arrays = {c: np.concatenate([p[c].to_numpy().reshape(len(ids), n) for p, n in zip(parts, lengths)], 1)
                      .astype(np.int16 if c == "inv_status" else np.float64) for c in TELEMETRY_COLS}
            labels += inject_batch(ts, arrays, sb, cfg)
            o = 0
            for f, p, n in zip(files, parts, lengths):
                cols = {"ts": pa.array(p.ts.to_numpy(), pa.timestamp("ns", tz="UTC")), "site_id": pa.array(p.site_id)}
                for c in TELEMETRY_COLS:
                    v = arrays[c][:, o:o + n].ravel()
                    if c in ROUND:
                        v = (np.round(v / ROUND[c]) * ROUND[c]).astype(np.float32)
                        cols[c] = pa.array(v, from_pandas=True)
                    else:
                        cols[c] = pa.array(v.astype(np.int16))
                table = pa.table(cols)
                path = tmp / "telemetry" / f.name
                if path not in writers:
                    writers[path] = pq.ParquetWriter(path, table.schema, **PQ)
                writers[path].write_table(table, row_group_size=n * 32)
                o += n
            log(f"sites {b0 + len(ids)}/{len(sites)} ({time.perf_counter() - t0:.0f}s)")
    finally:
        for w in writers.values():
            w.close()

    lab = labels_frame(labels)
    lab.to_parquet(tmp / "fault_labels.parquet", index=False, **PQ)
    for name in ("sites.parquet", "feeders.parquet", "weather.parquet"):
        if (fleet_dir / name).exists():
            shutil.copy2(fleet_dir / name, tmp / name)
    stats = summary(lab, sites, ts)
    (tmp / "manifest.json").write_text(json.dumps({
        "data_version": DATA_VERSION, "fleet_dir": str(fleet_dir), "config": cfg.to_json(),
        "n_episodes": int(len(lab)), "seconds": round(time.perf_counter() - t0, 1), **stats,
        "note": "feeder_ts is not recomputed: feeder flows in the fleet dir are pre-fault",
    }, indent=2))
    tmp.rename(out)
    log(lab.groupby("fault").agg(episodes=("asset_id", "size"), sites=("asset_id", "nunique"),
                                 severity=("severity", "median")).to_string())
    log(f"LAB_METRIC faulty_site_day_share={stats['faulty_site_day_share']}")
    log(f"wrote {out}")
    return out


def summary(lab: pd.DataFrame, sites: pd.DataFrame, ts: pd.DatetimeIndex) -> dict:
    end = lab.end.fillna(ts[-1] + pd.Timedelta(minutes=15))
    faulty_days = float(((end - lab.onset).dt.total_seconds() / 86400).sum())
    total_days = len(sites) * (ts[-1] - ts[0]).total_seconds() / 86400
    return {"faulty_site_day_share": round(faulty_days / total_days, 4),
            "episodes_by_fault": lab.fault.value_counts().to_dict()}
