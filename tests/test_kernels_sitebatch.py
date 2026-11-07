"""#133 SiteReader == the old pandas batch read in sentinel.data.windows (filters -> sort -> .loc -> reshape)."""

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq

from sentinel.kernels.sitebatch import SiteReader

COLS = ["pv_ac_w", "inv_status"]


def old_read(files, ids, cols, i0, i1):
    parts = []
    for f in files:
        d = pq.read_table(f, columns=["ts", "site_id", *cols], filters=[("site_id", "in", ids)]).to_pandas()
        d["site_id"] = d.site_id.astype(str)
        parts.append(d)
    d = pd.concat(parts).sort_values(["site_id", "ts"], kind="stable").set_index("site_id").loc[ids]
    n = len(d) // len(ids)
    return {c: d[c].to_numpy().reshape(len(ids), n)[:, i0:i1] for c in cols}


def write_years(tmp_path, rng, n_sites=40, steps=50, shuffle=False):
    sites = [f"S{i + 1:04d}" for i in range(n_sites)]
    files = []
    for y, t0 in enumerate(["2024-01-01", "2025-01-01"]):
        ts = pd.date_range(t0, periods=steps, freq="15min", tz="UTC")
        df = pd.DataFrame({"ts": np.tile(ts, n_sites), "site_id": np.repeat(sites, steps),
                           "pv_ac_w": rng.uniform(0, 5000, n_sites * steps).astype(np.float32),
                           "inv_status": rng.integers(0, 4, n_sites * steps).astype(np.int16)})
        df.loc[rng.random(len(df)) < 0.05, "pv_ac_w"] = np.nan
        if shuffle:
            df = df.sample(frac=1.0, random_state=y)
        f = tmp_path / f"year={2024 + y}.parquet"
        pq.write_table(pa.Table.from_pandas(df, preserve_index=False), f, row_group_size=steps * 8)
        files.append(f)
    return sites, files


def check(files, sites, batches, i0, i1):
    r = SiteReader(files)
    for ids in batches:
        a, b = old_read(files, ids, COLS, i0, i1), r.read(ids, COLS, i0, i1)
        for c in COLS:
            assert a[c].dtype == b[c].dtype and a[c].shape == b[c].shape
            assert np.array_equal(a[c], b[c], equal_nan=True)


def test_sorted_multi_year(tmp_path):
    sites, files = write_years(tmp_path, np.random.default_rng(0))
    check(files, sites, [sites[0:16], sites[16:32], sites[32:], sites[5:21]], 3, 97)


def test_shuffled_rows_and_unsorted_batch(tmp_path):
    sites, files = write_years(tmp_path, np.random.default_rng(1), shuffle=True)
    check(files, sites, [sites[0:16], sites[::-3], [sites[7], sites[2], sites[30]]], 0, 100)
