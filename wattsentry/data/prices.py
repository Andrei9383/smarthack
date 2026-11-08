"""15-min day-ahead (PZU) and intraday (PI) prices, the one price table Wind and Imbalance read (card #158).

Source: real RO, OPCOM.
- PZU: ``data/raw/prices.parquet`` (#9). OPCOM moved PZU to 15-min products on 2025-10-01 (local); before that the
  file is hourly and **each hourly price is repeated on its 4 quarter-hours** (``pzu_res = 'PT60M_x4'``). Rows are in
  UTC, so DST days just have 92 / 100 quarter-hours.
- PI: ``data/raw/prices_pi.parquet`` from I1 (#171), picked up automatically when present. Until it lands, or for
  quarter-hours with no PI trade, ``pi = pzu`` (``pi_src = 'pzu'``), which makes the intraday term of
  ``wattsentry.eval.imbalance.avoidable_cost`` zero.

    from wattsentry.data.prices import load_prices, prices_at
    df = load_prices("2025-01-01", "2026-01-01")      # polars: ts, pzu_lei_mwh, pi_lei_mwh, pzu_res, pi_src
    pzu, pi = prices_at(ts)                           # numpy, aligned to any array of 15-min UTC timestamps
"""

from __future__ import annotations

from datetime import datetime, timezone
from functools import lru_cache
from pathlib import Path

import numpy as np
import polars as pl

RAW = Path(__file__).resolve().parents[2] / "data" / "raw"
PZU_PATH = RAW / "prices.parquet"
PI_PATH = RAW / "prices_pi.parquet"
STEP = "15m"
# Column names I1 may use for the 15-min PI price, first match wins.
PI_COLUMNS = ("pi_vwap_lei_mwh", "pi_lei_mwh", "pi_vwap", "price_lei_mwh")


def _utc(x) -> datetime | None:
    if x is None:
        return None
    if isinstance(x, str):
        x = datetime.fromisoformat(x)
    if isinstance(x, np.datetime64):
        x = datetime.fromtimestamp(int(x.astype("datetime64[s]").astype(np.int64)), timezone.utc)
    return x.replace(tzinfo=timezone.utc) if x.tzinfo is None else x.astimezone(timezone.utc)


def _pzu_15min(path: Path) -> pl.DataFrame:
    p = (pl.read_parquet(path)
         .filter(pl.col("market") == "opcom_pzu")
         .select(pl.col("ts").dt.convert_time_zone("UTC").dt.cast_time_unit("us"), "price_lei_mwh", "resolution"))
    hourly = p.filter(pl.col("resolution") == "PT60M")
    quarter = p.filter(pl.col("resolution") != "PT60M")
    # Hourly -> 4 quarter-hours with the same price.
    rep = (hourly.with_columns(pl.lit([0, 15, 30, 45]).alias("_m")).explode("_m")
           .with_columns(pl.col("ts") + pl.duration(minutes=pl.col("_m")))
           .drop("_m"))
    out = pl.concat([rep.with_columns(pl.lit("PT60M_x4").alias("pzu_res")),
                     quarter.with_columns(pl.lit("PT15M").alias("pzu_res"))])
    # A native 15-min price wins over a repeated hourly one at the same quarter-hour.
    out = (out.sort("ts", "pzu_res", descending=[False, True]).unique("ts", keep="first", maintain_order=True)
           .select("ts", pl.col("price_lei_mwh").alias("pzu_lei_mwh"), "pzu_res"))
    return out


def _pi_15min(path: Path) -> pl.DataFrame | None:
    if not path.exists():
        return None
    p = pl.read_parquet(path)
    col = next((c for c in PI_COLUMNS if c in p.columns), None)
    if col is None or "ts" not in p.columns:
        raise ValueError(f"{path}: need a 'ts' column and one of {PI_COLUMNS}, got {p.columns}")
    if "market" in p.columns and p["market"].n_unique() > 1:
        p = p.filter(pl.col("market").str.contains("pi"))
    return (p.select(pl.col("ts").dt.convert_time_zone("UTC").dt.cast_time_unit("us"),
                     pl.col(col).cast(pl.Float64).alias("_pi"))
            .drop_nulls().unique("ts", keep="last").sort("ts"))


@lru_cache(maxsize=4)
def _table(pzu_path: str, pi_path: str) -> pl.DataFrame:
    df = _pzu_15min(Path(pzu_path))
    pi = _pi_15min(Path(pi_path))
    if pi is None:
        return df.with_columns(pl.col("pzu_lei_mwh").alias("pi_lei_mwh"), pl.lit("pzu").alias("pi_src")) \
                 .select("ts", "pzu_lei_mwh", "pi_lei_mwh", "pzu_res", "pi_src")
    df = df.join(pi, on="ts", how="left")
    return df.select(
        "ts", "pzu_lei_mwh",
        pl.coalesce("_pi", "pzu_lei_mwh").alias("pi_lei_mwh"),
        "pzu_res",
        pl.when(pl.col("_pi").is_null()).then(pl.lit("pzu")).otherwise(pl.lit("opcom_pi")).alias("pi_src"))


def load_prices(start=None, end=None, *, pzu_path: Path | str = PZU_PATH,
                pi_path: Path | str = PI_PATH) -> pl.DataFrame:
    """15-min prices in ``[start, end)`` (UTC; naive datetimes / ISO strings are read as UTC).

    Columns: ``ts`` (UTC, step start), ``pzu_lei_mwh``, ``pi_lei_mwh``, ``pzu_res`` ('PT15M' | 'PT60M_x4'),
    ``pi_src`` ('opcom_pi' | 'pzu').
    """
    df = _table(str(pzu_path), str(pi_path))
    lo, hi = _utc(start), _utc(end)
    if lo is not None:
        df = df.filter(pl.col("ts") >= lo)
    if hi is not None:
        df = df.filter(pl.col("ts") < hi)
    return df


def prices_at(ts, *, pzu_path: Path | str = PZU_PATH, pi_path: Path | str = PI_PATH,
              strict: bool = True) -> tuple[np.ndarray, np.ndarray]:
    """``(pzu, pi)`` lei/MWh as float64 arrays aligned to ``ts`` (array-like of UTC 15-min step starts:
    numpy datetime64, a polars/pandas datetime series, or epoch seconds). Steps without a price raise when
    ``strict``, else come back NaN."""
    if isinstance(ts, pl.Series):
        sec = ts.dt.convert_time_zone("UTC").dt.epoch("s").to_numpy() if ts.dtype.time_zone else \
            ts.dt.epoch("s").to_numpy()
    else:
        a = np.asarray(ts)
        if a.dtype.kind == "M":
            sec = a.astype("datetime64[s]").astype(np.int64)
        elif a.dtype.kind in "iuf":
            sec = a.astype(np.int64)
        else:  # pandas tz-aware / python datetimes
            sec = np.array([int(_utc(x).timestamp()) for x in a.ravel()], dtype=np.int64).reshape(a.shape)
    df = _table(str(pzu_path), str(pi_path))
    key = df["ts"].dt.epoch("s").to_numpy()
    idx = np.searchsorted(key, sec)
    idx_c = np.clip(idx, 0, len(key) - 1)
    ok = key[idx_c] == sec
    if strict and not ok.all():
        bad = np.asarray(sec)[~ok].ravel()[:3]
        raise KeyError(f"no 15-min price for {[datetime.fromtimestamp(int(s), timezone.utc).isoformat() for s in bad]}"
                       f" ({int((~ok).sum())} steps missing)")
    pzu = np.where(ok, df["pzu_lei_mwh"].to_numpy()[idx_c], np.nan)
    pi = np.where(ok, df["pi_lei_mwh"].to_numpy()[idx_c], np.nan)
    return pzu, pi


if __name__ == "__main__":
    t = load_prices()
    print(t.group_by("pzu_res", "pi_src").agg(pl.len(), pl.col("ts").min().alias("from"), pl.col("ts").max().alias("to"),
                                             pl.col("pzu_lei_mwh").mean().alias("pzu_mean")).sort("from"))
