"""15-min PZU prices for the data-v2 simulation (what MaxControl homes react to).

Source: data/raw/prices.parquet (OPCOM PZU, real RO, 2023-2025; hourly until 30 Sep 2025 -> repeated x4, 15-min
after). Local days the file does not cover (2022) take the price of the same local weekday and quarter-hour 53
weeks later (371 days), labelled ``proxy_shift_371d``. Strategies only use the daily rank and median, so the
2023 shape stands in for 2022 (the 2022 crisis price level is not reproduced: stated in the manifest).

Self-contained on purpose (deterministic for the cache digest); the shared 15-min module is X1 #158
(sentinel/data/prices.py) and gives the same pzu values for 2023-2025.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

REPO = Path(__file__).resolve().parents[3]
PRICES = REPO / "data" / "raw" / "prices.parquet"
TZ = "Europe/Bucharest"
SHIFT_DAYS = 371


def prices_key(path: Path = PRICES) -> str:
    return f"{path.stem}-{path.stat().st_size}" if path.exists() else "missing"


def _raw_15min(path: Path) -> pd.Series:
    df = pd.read_parquet(path, columns=["ts", "price_lei_mwh", "resolution"])
    df["ts"] = pd.to_datetime(df.ts, utc=True).dt.as_unit("ns")
    q = df[df.resolution == "PT15M"].set_index("ts").price_lei_mwh
    h = df[df.resolution != "PT15M"].set_index("ts").price_lei_mwh
    parts = [q]
    if len(h):
        idx = pd.date_range(h.index.min(), h.index.max() + pd.Timedelta(minutes=45), freq="15min")
        hx = h.reindex(idx.floor("h")).set_axis(idx).dropna()
        parts.append(hx[~hx.index.isin(q.index)])
    return pd.concat(parts).sort_index()


def pzu_15min(ts_utc: pd.DatetimeIndex, path: Path = PRICES) -> pd.DataFrame:
    """ts (UTC, 15-min) -> DataFrame(ts, pzu_lei_mwh, price_source) with no gaps (proxy fill, see module doc)."""
    if not path.exists():
        raise FileNotFoundError(f"{path}: OPCOM PZU prices (#9) are needed for data-v2")
    raw = _raw_15min(path)
    ts_utc = pd.DatetimeIndex(ts_utc).tz_convert("UTC").as_unit("ns")
    p = raw.reindex(ts_utc).to_numpy(dtype=np.float64, copy=True)
    src = np.where(np.isnan(p), "", np.where(ts_utc >= pd.Timestamp("2025-09-30T22:00Z"), "opcom_pzu_15min",
                                              "opcom_pzu_hourly_x4")).astype(object)
    miss = np.isnan(p)
    if miss.any():
        # same local wall-clock quarter-hour, SHIFT_DAYS later (falls back to earlier if later is missing too)
        loc = ts_utc.tz_convert(TZ)
        key = pd.DataFrame({"d": loc.tz_localize(None).normalize(), "q": loc.hour * 4 + loc.minute // 15})
        rl = raw.index.tz_convert(TZ)
        table = pd.Series(raw.to_numpy(), index=pd.MultiIndex.from_arrays(
            [rl.tz_localize(None).normalize(), rl.hour * 4 + rl.minute // 15]))
        table = table[~table.index.duplicated()]
        for shift in (SHIFT_DAYS, -SHIFT_DAYS):
            m = np.isnan(p)
            if not m.any():
                break
            k = pd.MultiIndex.from_arrays([key.d[m] + pd.Timedelta(days=shift), key.q[m]])
            got = table.reindex(k).to_numpy()
            # DST: a quarter-hour missing on the shifted day takes the hour before it
            k2 = pd.MultiIndex.from_arrays([key.d[m] + pd.Timedelta(days=shift), (key.q[m] - 4).clip(lower=0)])
            got = np.where(np.isnan(got), table.reindex(k2).to_numpy(), got)
            idx = np.flatnonzero(m)
            p[idx] = got
            src[idx[~np.isnan(got)]] = f"proxy_shift_{shift}d"
    if np.isnan(p).any():
        raise ValueError(f"{int(np.isnan(p).sum())} quarter-hours without a PZU price, even after the proxy fill")
    return pd.DataFrame({"ts": ts_utc, "pzu_lei_mwh": p, "price_source": src})
