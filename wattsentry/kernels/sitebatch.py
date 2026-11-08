"""Fast per-batch Parquet reads for the window cache build (``wattsentry.data.windows``); not a dispatched kernel.

``SiteReader.read(ids, cols, i0, i1)`` returns ``{col: [len(ids), i1 - i0]}``, rows in ``ids`` order and by ts. It reads
only the row groups whose site_id statistics overlap the batch, selects the batch's rows via the dictionary codes of
site_id, and sorts with a stable lexsort only when the rows are not already in (site, ts) order (the usual case).
Values keep their Parquet dtype (float32 / int16, nulls -> NaN).
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq


class SiteReader:
    def __init__(self, files: list[Path]):
        self.files = [pq.ParquetFile(f, read_dictionary=["site_id"]) for f in files]
        self.ranges = [self._ranges(pf) for pf in self.files]

    @staticmethod
    def _ranges(pf: pq.ParquetFile) -> list[tuple[str, str] | None]:
        md = pf.metadata
        names = [md.schema.column(j).path for j in range(md.num_columns)]
        j = names.index("site_id")
        out = []
        for k in range(md.num_row_groups):
            st = md.row_group(k).column(j).statistics
            out.append((st.min, st.max) if st is not None and st.has_min_max else None)
        return out

    def read(self, ids: list[str], cols: list[str], i0: int, i1: int) -> dict[str, np.ndarray]:
        lo, hi = min(ids), max(ids)
        pos = {sid: i for i, sid in enumerate(ids)}
        enc, ts, vals = [], [], {c: [] for c in cols}
        for pf, rng in zip(self.files, self.ranges):
            rgs = [k for k, r in enumerate(rng) if r is None or not (r[1] < lo or r[0] > hi)]
            if not rgs:
                continue
            tb = pf.read_row_groups(rgs, columns=["ts", "site_id", *cols])
            # site_id is read dictionary-encoded: map each chunk's few dictionary strings to their position in ids
            # (-1 = not in this batch), then gather the int codes; no per-row string compares.
            e = np.concatenate([np.array([pos.get(x, -1) for x in ch.dictionary.to_pylist()] or [-1], np.int64)
                                [ch.indices.to_numpy(zero_copy_only=False)] for ch in tb.column("site_id").chunks])
            keep = np.flatnonzero(e >= 0) if (e < 0).any() else None
            sel = (lambda a: a) if keep is None else (lambda a: a[keep])
            enc.append(sel(e))
            ts.append(sel(tb.column("ts").cast(pa.int64()).to_numpy()))
            for c in cols:
                vals[c].append(sel(tb.column(c).to_numpy()))
        e = np.concatenate(enc) if enc else np.zeros(0, np.int32)
        t = np.concatenate(ts) if ts else np.zeros(0, np.int64)
        n, rem = divmod(len(e), len(ids))
        if rem or np.bincount(e, minlength=len(ids)).min(initial=n) != n:
            raise ValueError(f"batch sites do not all have the same number of rows ({len(e)} rows, {len(ids)} sites)")
        de, dt = np.diff(e), np.diff(t)
        order = None if np.all((de > 0) | ((de == 0) & (dt >= 0))) else np.lexsort((t, e))
        out = {}
        for c in cols:
            v = np.concatenate(vals[c])
            if order is not None:
                v = v[order]
            out[c] = v.reshape(len(ids), n)[:, i0:i1]
        return out
