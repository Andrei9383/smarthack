"""OPCOM PI parser + aggregation (#171). Offline: a synthetic page in OPCOM's (malformed) table layout."""

import datetime as dt
from pathlib import Path

import polars as pl
import pytest

from sentinel.data import opcom_pi as o

ROOT = Path(__file__).resolve().parents[1]


def page(day: str, blocks: list[tuple[str, list[tuple[str, str, str]]]]) -> str:
    """OPCOM layout: the first row of a block opens <tr> with a rowspan label; the rest only close </tr>."""
    out = ['<table class="border_table"><caption><p><b>Tranzactii incheiate in data ', day, '</b></p></caption>',
           '<tr><td class="color2">Tranzactii orare/bloc</td><td class="color2">Tip</td>'
           '<td class="color2">Pre&#355;</td><td class="color2">Volum</td></tr></div>']
    for label, rows in blocks:
        for k, (side, price, vol) in enumerate(rows):
            head = f'<tr> <td rowspan="{len(rows)}" class="color">{label}</td>' if k == 0 else ""
            out.append(f'{head} <td class="alb">{side}</td> <td class="alb">{price}</td> <td class="alb">{vol}</td> </tr>')
    out.append("</table><table><tr><td>cumparare</td><td>1,0</td><td>1,0</td></tr></table>")  # outside: ignored
    return "".join(out)


def test_ro_number():
    assert o.ro_number("1.465,78") == 1465.78
    assert o.ro_number("-12,5") == -12.5
    with pytest.raises(ValueError):
        o.ro_number("1e3")


def test_parse_page_rowspan_blocks():
    html = page("12.11.2025", [("02-03_XB", [("cumparare", "1.000,00", "1,0")]),
                               ("T03-T04_XB", [("vanzare", "100,00", "2,0"), ("cumparare", "200,00", "3,0")])])
    caption, trades = o.parse_page(html)
    assert o.page_date(caption) == dt.date(2025, 11, 12)
    assert [(t.label, t.side, t.price, t.volume) for t in trades] == [
        ("02-03_XB", "buy", 1000.0, 1.0), ("T03-T04_XB", "sell", 100.0, 2.0), ("T03-T04_XB", "buy", 200.0, 3.0)]


def test_parse_label():
    d = dt.date(2025, 3, 12)
    assert o.parse_label("14-15_XB", d) == (dt.datetime(2025, 3, 12, 14), 60)
    assert o.parse_label("23-T00_XB", d) == (dt.datetime(2025, 3, 12, 23), 60)
    assert o.parse_label("T23:45-TT00:00_XB", d) == (dt.datetime(2025, 3, 13, 23, 45), 15)
    assert o.parse_label("Base", d) is None


def test_trades_frame_utc_and_caption_check():
    html = page("12.11.2025", [("T00:00-T00:15_XB", [("cumparare", "10,00", "1,0")])])
    df, unknown = o.trades_frame(html, dt.date(2025, 11, 12), "PT15M")
    assert unknown == [] and df["ts"][0] == dt.datetime(2025, 11, 12, 22, 0, tzinfo=dt.timezone.utc)  # EET = UTC+2
    with pytest.raises(ValueError):
        o.trades_frame(html, dt.date(2025, 11, 13), "PT15M")


def test_dst_fallback_repeated_label_gets_second_fold():
    # 26 Oct 2025: local 03:00-04:00 happens twice (EEST then EET); OPCOM repeats the label
    html = page("26.10.2025", [("03-04_XB", [("cumparare", "10,00", "1,0")]), ("03-04_XB", [("cumparare", "20,00", "1,0")])])
    df, _ = o.trades_frame(html, dt.date(2025, 10, 26), "PT60M")
    assert df["ts"].to_list() == [dt.datetime(2025, 10, 26, 0, tzinfo=dt.timezone.utc),
                                  dt.datetime(2025, 10, 26, 1, tzinfo=dt.timezone.utc)]


def test_aggregate_vwap_and_fallback():
    utc = dt.timezone.utc
    q0 = dt.datetime(2025, 6, 1, 10, 0, tzinfo=utc)
    trades = pl.DataFrame(dict(
        ts=[q0, q0, q0 + dt.timedelta(hours=1)], minutes=[15, 15, 60], side=["buy", "sell", "buy"],
        price=[100.0, 200.0, 50.0], volume=[1.0, 3.0, 2.0], product=["PT15M", "PT15M", "PT60M"]),
        schema_overrides=dict(ts=pl.Datetime("us", "UTC"), minutes=pl.Int32))
    df = o.aggregate(trades, 2025)
    assert df.height == 35040 and df["ts"].is_unique().all() and df.columns == o.SCHEMA_COLS
    r = df.filter(pl.col("ts") == q0).row(0, named=True)
    assert r["pi_vwap_lei_mwh"] == pytest.approx(175.0) and r["pi_product"] == "PT15M"
    assert r["pi_volume_mwh"] == pytest.approx(1.0) and r["pi_trades"] == 2
    h = df.filter(pl.col("ts").is_between(q0 + dt.timedelta(hours=1), q0 + dt.timedelta(minutes=105)))
    assert h["pi_product"].to_list() == ["PT60M"] * 4 and h["pi_vwap_lei_mwh"].to_list() == [50.0] * 4
    assert h["pi_volume_mwh"].sum() == pytest.approx(2.0)  # 2 MW for one hour = 2 MWh over its 4 quarters
    empty = df.filter(pl.col("pi_product").is_null())
    assert empty["pi_vwap_lei_mwh"].null_count() == empty.height and (empty["pi_volume_mwh"] == 0).all()


def test_committed_parquet_shape():
    path = ROOT / "data" / "raw" / "prices_pi.parquet"
    if not path.exists():
        pytest.skip("prices_pi.parquet not committed yet")
    df = pl.read_parquet(path)
    assert df.columns == o.SCHEMA_COLS and df.height == 35040 and df["ts"].is_unique().all()
    assert df["ts"].dtype == pl.Datetime("us", "UTC")
    traded = df.filter(pl.col("pi_product").is_not_null())
    assert traded["pi_vwap_lei_mwh"].null_count() == 0 and (traded["pi_volume_mwh"] > 0).all()
