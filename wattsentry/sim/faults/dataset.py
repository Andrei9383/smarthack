"""Run the injector over a whole fleet dataset (wattsentry.sim.fleet output) in batches of sites, all years at once
(the battery is continuous across years), and write the faulty copy + fault_labels.parquet."""

from __future__ import annotations

import hashlib
import json
import shutil
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.compute as pc
import pyarrow.parquet as pq

from ..fleet.config import DATA_VERSION
from .inject import TELEMETRY_COLS, FaultConfig, inject_batch, labels_frame

PQ = dict(compression="zstd", compression_level=6)
ROUND = {"pv_ac_w": 1, "inv_voltage_v": 0.1, "soc": 0.001, "batt_w": 1, "load_w": 1, "grid_w": 1}


def _read_sites(f: Path, ids: list[str]) -> pa.Table:
    """Rows of `ids` from one telemetry file, ordered by (position in ids, ts). The fleet files are already in that
    order, so usually nothing is reordered. Only row groups whose site_id min/max can hold one of `ids` are decoded
    (read_table(filters=...) decodes about the whole file for every batch, quadratic in fleet size)."""
    pf = pq.ParquetFile(f)
    col = pf.schema_arrow.get_field_index("site_id")
    lo, hi = min(ids), max(ids)
    stats = [pf.metadata.row_group(i).column(col).statistics for i in range(pf.metadata.num_row_groups)]
    rgs = [i for i, st in enumerate(stats) if st is None or not st.has_min_max or (st.min <= hi and st.max >= lo)]
    value_set = pa.array(ids, pa.string())
    t = pf.read_row_groups(rgs)
    t = t.filter(pc.is_in(t.column("site_id"), value_set=value_set))
    pos = pc.index_in(t.column("site_id"), value_set=value_set).to_numpy(zero_copy_only=False)
    tsi = t.column("ts").cast(pa.int64()).to_numpy()
    dp, dt = np.diff(pos), np.diff(tsi)
    if np.all((dp > 0) | ((dp == 0) & (dt >= 0))):  # already in order (O(n) check before an O(n log n) sort)
        return t
    return t.take(np.lexsort((tsi, pos)))


def default_out(fleet_dir: Path, cfg: FaultConfig) -> Path:
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

    def write(tables: list[tuple[Path, pa.Table, int]]) -> None:
        for path, table, n in tables:
            if path not in writers:
                writers[path] = pq.ParquetWriter(path, table.schema, **PQ)
            writers[path].write_table(table, row_group_size=n * 32)

    # One writer thread: zstd encoding releases the GIL, so batch k is written while batch k+1 is read and injected
    # (same tables, same order, so the files are byte-identical to writing inline). A second thread reads batch k+1
    # while batch k is injected.
    pool, reader, pending = ThreadPoolExecutor(1), ThreadPoolExecutor(1), None
    read = lambda b: [_read_sites(f, list(sites.site_id.iloc[b:b + batch])) for f in files]  # noqa: E731
    nxt = reader.submit(read, 0)
    try:
        for b0 in range(0, len(sites), batch):
            sb = sites.iloc[b0:b0 + batch].reset_index(drop=True)
            ids = list(sb.site_id)
            parts = nxt.result()
            if b0 + batch < len(sites):
                nxt = reader.submit(read, b0 + batch)
            lengths = [p.num_rows // len(ids) for p in parts]
            ts = pd.DatetimeIndex(
                np.concatenate([p.column("ts").slice(0, n).to_numpy() for p, n in zip(parts, lengths)])
            ).tz_localize("UTC")
            arrays = {
                c: np.concatenate([p.column(c).to_numpy().reshape(len(ids), n) for p, n in zip(parts, lengths)], 1)
                .astype(np.int16 if c == "inv_status" else np.float64)
                for c in TELEMETRY_COLS}
            labels += inject_batch(ts, arrays, sb, cfg)
            o, tables = 0, []
            for f, p, n in zip(files, parts, lengths):
                # large_string keeps the files identical to the earlier pandas-based writer
                cols = {"ts": p.column("ts").cast(pa.timestamp("ns", tz="UTC")),
                        "site_id": p.column("site_id").cast(pa.large_string())}
                for c in TELEMETRY_COLS:
                    v = arrays[c][:, o:o + n].ravel()
                    if c in ROUND:
                        v = (np.round(v / ROUND[c]) * ROUND[c]).astype(np.float32)
                        cols[c] = pa.array(v, from_pandas=True)
                    else:
                        cols[c] = pa.array(v.astype(np.int16))
                tables.append((tmp / "telemetry" / f.name, pa.table(cols), n))
                o += n
            if pending is not None:
                pending.result()
            pending = pool.submit(write, tables)
            log(f"sites {b0 + len(ids)}/{len(sites)} ({time.perf_counter() - t0:.0f}s)")
        if pending is not None:
            pending.result()
    finally:
        reader.shutdown(wait=True)
        pool.shutdown(wait=True)
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
    log(f"LAB_METRIC inject_seconds={time.perf_counter() - t0:.1f}")
    log(f"wrote {out}")
    return out


def summary(lab: pd.DataFrame, sites: pd.DataFrame, ts: pd.DatetimeIndex) -> dict:
    end = lab.end.fillna(ts[-1] + pd.Timedelta(minutes=15))
    faulty_days = float(((end - lab.onset).dt.total_seconds() / 86400).sum())
    total_days = len(sites) * (ts[-1] - ts[0]).total_seconds() / 86400
    return {"faulty_site_day_share": round(faulty_days / total_days, 4),
            "episodes_by_fault": lab.fault.value_counts().to_dict()}
