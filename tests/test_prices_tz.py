"""OPCOM PZU intervals are CET/CEST, not Romanian time (#183).

Three checks: the parser starts delivery day D at 00:00 CET; the committed data/raw/prices.parquet matches an
independent UTC source (Energy-Charts RO day-ahead price, EUR/MWh) on two DST days; and PZU lines up with
ENTSO-E residual load at lag 0 h, not +1 h.
"""
from __future__ import annotations

import datetime as dt
from pathlib import Path

import numpy as np
import polars as pl
import pytest

from sentinel.data.download import _parse_opcom_pzu

RAW = Path(__file__).resolve().parents[1] / "data" / "raw"
UTC = dt.timezone.utc

# Energy-Charts RO day-ahead price (EUR/MWh, UTC interval start), fetched 2026-11-08 from
# api.energy-charts.info/price?bzn=RO. 30 Mar 2025 (spring DST at 01:00 UTC), hourly.
EC_2025_03_30 = {
    "2025-03-29T20:00": 108.31, "2025-03-29T21:00": 86.98, "2025-03-29T22:00": 69.58,
    "2025-03-29T23:00": 46.24, "2025-03-30T00:00": 15.88, "2025-03-30T01:00": 5.09,
}
# 26 Oct 2025 (autumn DST at 01:00 UTC), 15-min.
EC_2025_10_26 = {
    "2025-10-25T21:00": 121.03, "2025-10-25T21:15": 105.13, "2025-10-25T21:30": 96.33, "2025-10-25T21:45": 91.58,
    "2025-10-25T22:00": 93.75, "2025-10-25T22:15": 89.32, "2025-10-25T22:30": 92.2, "2025-10-25T22:45": 87.05,
    "2025-10-25T23:00": 79.96, "2025-10-25T23:15": 76.6, "2025-10-25T23:30": 73.51, "2025-10-25T23:45": 56.91,
    "2025-10-26T00:00": 72.54, "2025-10-26T00:15": 68.3, "2025-10-26T00:30": 56.61, "2025-10-26T00:45": 50.61,
    "2025-10-26T01:00": 63.22, "2025-10-26T01:15": 64.26, "2025-10-26T01:30": 60.48, "2025-10-26T01:45": 55.67,
    "2025-10-26T02:00": 55.24, "2025-10-26T02:15": 52.9, "2025-10-26T02:30": 55.78, "2025-10-26T02:45": 60.28,
}


def _csv(n: int, product: str) -> str:
    head = "Zona de tranzactionare,Interval,Pret,Volum,x,y,Rezolutie\n"
    return head + "".join(f"Romania,{i},{100 + i}.0,{1000 + i}.0,0,0,{product}\n" for i in range(1, n + 1))


@pytest.mark.parametrize("day,n,product,first,last", [
    (dt.date(2025, 1, 15), 24, "PT60M", "2025-01-14T23:00", "2025-01-15T22:00"),  # CET = UTC+1
    (dt.date(2025, 7, 15), 24, "PT60M", "2025-07-14T22:00", "2025-07-15T21:00"),  # CEST = UTC+2
    (dt.date(2025, 3, 30), 23, "PT60M", "2025-03-29T23:00", "2025-03-30T21:00"),  # 23-hour day
    (dt.date(2025, 10, 26), 100, "PT15M", "2025-10-25T22:00", "2025-10-26T22:45"),  # 25-hour day at 15 min
])
def test_parser_starts_delivery_day_at_cet_midnight(day, n, product, first, last):
    rows = _parse_opcom_pzu(day, _csv(n, product))
    ts = [r["ts"] for r in rows]
    assert len(ts) == n
    assert ts[0] == dt.datetime.fromisoformat(first).replace(tzinfo=UTC)
    assert ts[-1] == dt.datetime.fromisoformat(last).replace(tzinfo=UTC)
    step = dt.timedelta(minutes=15 if product == "PT15M" else 60)
    assert all(b - a == step for a, b in zip(ts, ts[1:]))
    assert rows[0]["resolution"] == product


@pytest.fixture(scope="module")
def prices() -> pl.DataFrame:
    return pl.read_parquet(RAW / "prices.parquet").filter(pl.col("market") == "opcom_pzu")


@pytest.mark.parametrize("ref", [EC_2025_03_30, EC_2025_10_26], ids=["2025-03-30", "2025-10-26"])
def test_committed_prices_match_utc_reference_on_dst_days(prices, ref):
    """lei/EUR must be one constant (that day's exchange rate) at every interval; a 1 h shift breaks it badly."""
    p = dict(zip(prices["ts"].to_list(), prices["price_lei_mwh"].to_list()))
    ratio = []
    for k, eur in ref.items():
        t = dt.datetime.fromisoformat(k).replace(tzinfo=UTC)
        assert t in p, f"no PZU price at {k}"
        ratio.append(p[t] / eur)
    ratio = np.asarray(ratio)
    assert 4.9 < np.median(ratio) < 5.2  # RON per EUR in 2025
    assert np.max(np.abs(ratio / np.median(ratio) - 1)) < 0.01


def test_pzu_vs_residual_load_lag_peaks_at_zero(prices):
    rl = (pl.scan_parquet(RAW / "entsoe_ro.parquet").filter(pl.col("series") == "residual_load")
            .filter(pl.col("ts").dt.year() == 2025).collect())

    def hourly(df: pl.DataFrame, col: str) -> pl.DataFrame:
        h = df.group_by(pl.col("ts").dt.truncate("1h")).agg(pl.col(col).mean())
        day = pl.col("ts").dt.convert_time_zone("Europe/Bucharest").dt.date()
        return h.select("ts", (pl.col(col) - pl.col(col).mean().over(day)).alias(col))

    ph, rh = hourly(prices, "price_lei_mwh"), hourly(rl, "mw")
    corr = {}
    for lag in (-1, 0, 1):
        j = ph.with_columns(pl.col("ts") + dt.timedelta(hours=lag)).join(rh, on="ts").drop_nulls()
        j = j.filter(pl.col("ts").dt.year() == 2025)
        corr[lag] = float(np.corrcoef(j["price_lei_mwh"].to_numpy(), j["mw"].to_numpy())[0, 1])
    assert max(corr, key=corr.get) == 0, corr
