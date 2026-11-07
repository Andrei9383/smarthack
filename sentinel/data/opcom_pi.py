"""OPCOM intraday (PI) trades -> volume-weighted price per 15-min delivery interval (#171, spec I step 1b).

    python -m sentinel.data.opcom_pi                    # test year 2025 -> data/raw/prices_pi.parquet + MANIFEST
    python -m sentinel.data.opcom_pi --offline          # rebuild from the response cache only, no network
    python -m sentinel.data.opcom_pi --cache DIR        # keep raw responses somewhere else (default data/.cache)

OPCOM has no export for PI, so this reads the public "Tranzactii PI" page (one form POST per trading date and
product, `tip` 15 or 60 min). The committed Parquet is the product: `make reproduce` never runs this.
Terms (opcom.ro/disclaimer/133/ro, checked 2026-11-07): public data may be downloaded for internal,
non-commercial use with the source credited (OPCOM); republishing or redistributing it needs OPCOM's written
consent. robots.txt allows every path. So: polite (>= 1 s between requests, a descriptive User-Agent, every
response cached so the year is downloaded once), and the data stays inside the private repo.

The page lists the trades *concluded on* the chosen date. A label without a prefix (`14-15_XB`,
`14:15-14:30_XB`) is delivery on that date; `T14-T15` is the next day and `TT00` the boundary after it. So a
delivery day D needs the pages of D-1 and D. Labels are Romanian local time (Europe/Bucharest), like PZU.
Each row is one RO participant's side of a trade (`cumparare` = buy, `vanzare` = sell), price lei/MWh.

Volume: OPCOM labels the column MWh, but both products trade in 0.1 steps (the SIDC MW granularity), so the
number is read as MW of contracted power: energy in one quarter = volume x 0.25 MWh for either product.
VWAP is unaffected by this choice; only the `*_volume_mwh` columns are.

The HTML is treated as untrusted data: it is tokenised with the stdlib HTMLParser (no execution, no eval) and
every number is validated. The table is malformed (continuation rows close `</tr>` without opening one), so
cells are read in document order instead of row by row.
"""

from __future__ import annotations

import argparse
import datetime as dt
import gzip
import http.cookiejar
import re
import sys
import time
import urllib.error
import urllib.parse
import urllib.request
import zoneinfo
from dataclasses import dataclass
from html.parser import HTMLParser
from pathlib import Path

import polars as pl

REPO = Path(__file__).resolve().parents[2]
RAW = REPO / "data" / "raw"
OUT = RAW / "prices_pi.parquet"
CACHE = REPO / "data" / ".cache" / "opcom_pi"
URL = "https://www.opcom.ro/rapoarte-pi-rezultate-pi/ro"
UA = "sentinel-research/0.1 (SmartHack 2026; research use)"
RO_TZ = zoneinfo.ZoneInfo("Europe/Bucharest")
UTC = dt.timezone.utc
TEST_YEAR = 2025
MIN_GAP_S = 1.5  # between requests (card: >= 1 s)
SIDES = {"cumparare": "buy", "vanzare": "sell"}
LABEL = re.compile(r"^(T*)(\d{1,2})(?::(\d{2}))?-(T*)(\d{1,2})(?::(\d{2}))?(?:_(\w+))?$")
SCHEMA_COLS = ["ts", "pi_vwap_lei_mwh", "pi_volume_mwh", "pi_trades", "pi_product",
               "pi15_vwap_lei_mwh", "pi15_volume_mwh", "pi60_vwap_lei_mwh", "pi60_volume_mwh"]


# ---------------------------------------------------------------- parsing (pure, no network)

@dataclass(frozen=True)
class Trade:
    label: str
    side: str  # buy | sell
    price: float  # lei/MWh
    volume: float  # as reported (read as MW, see module doc)
    block: int = 0  # index of the rowspan block on the page (a repeated DST hour is a second block, same label)


class _Cells(HTMLParser):
    """Collects (rowspan, text) for every <td> of the results table (class border_table) and its caption."""

    def __init__(self) -> None:
        super().__init__(convert_charrefs=True)
        self.cells: list[tuple[int | None, str]] = []
        self.caption = ""
        self._depth = 0  # >0 while inside the results table
        self._td: tuple[int | None, list[str]] | None = None
        self._in_caption = False

    def handle_starttag(self, tag, attrs):
        a = dict(attrs)
        if tag == "table":
            if self._depth or "border_table" in (a.get("class") or "").split():
                self._depth += 1
        elif not self._depth:
            return
        elif tag == "td":
            self._close_td()
            rs = a.get("rowspan")
            self._td = (int(rs) if rs and rs.isdigit() else None, [])
        elif tag == "caption":
            self._in_caption = True

    def handle_endtag(self, tag):
        if not self._depth:
            return
        if tag == "td":
            self._close_td()
        elif tag == "caption":
            self._in_caption = False
        elif tag == "table":
            self._close_td()
            self._depth -= 1

    def handle_data(self, data):
        if self._td is not None:
            self._td[1].append(data)
        elif self._in_caption:
            self.caption += data

    def _close_td(self):
        if self._td is not None:
            self.cells.append((self._td[0], " ".join("".join(self._td[1]).split())))
            self._td = None


def ro_number(s: str) -> float:
    """'1.465,78' -> 1465.78 (RO thousands '.' and decimal ',')."""
    s = s.strip().replace("\xa0", "").replace(" ", "")
    if not re.fullmatch(r"-?[\d.]+(,\d+)?", s):
        raise ValueError(f"not a number: {s!r}")
    return float(s.replace(".", "").replace(",", "."))


def parse_page(html: str) -> tuple[str, list[Trade]]:
    """Returns (caption, trades). A cell with rowspan opens a delivery block; then (side, price, volume) triples."""
    p = _Cells()
    p.feed(html)
    p.close()
    trades: list[Trade] = []
    label, block = None, -1
    cells = p.cells
    i = 0
    while i < len(cells):
        rowspan, text = cells[i]
        if rowspan is not None:
            label, block = text, block + 1
            i += 1
            continue
        if text in SIDES and label and i + 2 < len(cells):
            try:
                price, vol = ro_number(cells[i + 1][1]), ro_number(cells[i + 2][1])
            except ValueError:
                i += 1
                continue
            trades.append(Trade(label, SIDES[text], price, vol, block))
            i += 3
            continue
        i += 1
    return " ".join(p.caption.split()), trades


def page_date(caption: str) -> dt.date | None:
    m = re.search(r"(\d{2})\.(\d{2})\.(\d{4})", caption)
    return dt.date(int(m[3]), int(m[2]), int(m[1])) if m else None


def parse_label(label: str, trading_day: dt.date) -> tuple[dt.datetime, int] | None:
    """'T14:15-T14:30_XB' on trading day d -> (local naive start on d+1 at 14:15, 15 minutes). None if unknown."""
    m = LABEL.match(label)
    if not m:
        return None
    t1, h1, m1, t2, h2, m2, _ = m.groups()
    start = dt.datetime.combine(trading_day + dt.timedelta(days=len(t1)), dt.time()) \
        + dt.timedelta(hours=int(h1), minutes=int(m1 or 0))
    end = dt.datetime.combine(trading_day + dt.timedelta(days=len(t2)), dt.time()) \
        + dt.timedelta(hours=int(h2), minutes=int(m2 or 0))
    if not t2 and int(h2) < int(h1):  # '23-T00' style is covered by the T prefix; tolerate a bare wrap too
        end += dt.timedelta(days=1)
    minutes = int((end - start).total_seconds() // 60)
    return (start, minutes) if minutes in (15, 60) else None


def to_utc(local: dt.datetime, fold: int) -> dt.datetime:
    return local.replace(tzinfo=RO_TZ, fold=fold).astimezone(UTC)


def trades_frame(html: str, trading_day: dt.date, product: str) -> tuple[pl.DataFrame, list[str]]:
    """Trades of one page with their UTC delivery start. Returns (frame, unknown labels).

    DST fall-back: the repeated local hour appears as two blocks with the same label; the first is the
    summer-time one (fold 0), the second winter time (fold 1)."""
    caption, trades = parse_page(html)
    got = page_date(caption)
    if trades and got != trading_day:
        raise ValueError(f"page caption {caption!r} is not trading day {trading_day}")
    rows, unknown = [], []
    fold: dict[int, int] = {}  # block index -> 0 for the first block with its label, 1 for a repeat
    blocks_of: dict[str, int] = {}  # label -> blocks seen so far
    for t in trades:
        parsed = parse_label(t.label, trading_day)
        if parsed is None:
            unknown.append(t.label)
            continue
        start, minutes = parsed
        if t.block not in fold:
            fold[t.block] = min(blocks_of.get(t.label, 0), 1)
            blocks_of[t.label] = blocks_of.get(t.label, 0) + 1
        rows.append(dict(ts=to_utc(start, fold[t.block]), minutes=minutes, side=t.side,
                         price=t.price, volume=t.volume))
    schema = dict(ts=pl.Datetime("us", "UTC"), minutes=pl.Int32, side=pl.Utf8, price=pl.Float64, volume=pl.Float64)
    df = pl.DataFrame(rows, schema=schema).with_columns(product=pl.lit(product))
    return df, unknown


def quarter_grid(year: int) -> pl.DataFrame:
    """Every 15-min delivery interval of the local year, UTC (35,040 rows in a non-leap year)."""
    a = dt.datetime(year, 1, 1, tzinfo=RO_TZ).astimezone(UTC)
    b = dt.datetime(year + 1, 1, 1, tzinfo=RO_TZ).astimezone(UTC)
    return pl.DataFrame({"ts": pl.datetime_range(a, b, "15m", closed="left", time_unit="us", time_zone="UTC",
                                                 eager=True)})


def aggregate(trades: pl.DataFrame, year: int) -> pl.DataFrame:
    """VWAP per 15-min interval: the 15-min product if that quarter traded, else the containing hour's hourly
    product. pi_vwap_lei_mwh stays null without trades (the loader fills pi = pzu)."""
    grid = quarter_grid(year)
    t = trades.filter(pl.col("volume") > 0)

    def per(product: str, minutes: int) -> pl.DataFrame:
        g = (t.filter((pl.col("product") == product) & (pl.col("minutes") == minutes))
              .group_by("ts")
              .agg(vwap=(pl.col("price") * pl.col("volume")).sum() / pl.col("volume").sum(),
                   vol=pl.col("volume").sum() * 0.25, n=pl.len().cast(pl.Int32)))
        if minutes == 60:  # spread each hour onto its 4 quarters
            g = (g.with_columns(ts=pl.concat_list([pl.col("ts") + pl.duration(minutes=15 * q) for q in range(4)]))
                  .explode("ts"))
        tag = "pi15" if minutes == 15 else "pi60"
        return g.rename({"vwap": f"{tag}_vwap_lei_mwh", "vol": f"{tag}_volume_mwh", "n": f"{tag}_n"})

    df = grid.join(per("PT15M", 15), on="ts", how="left").join(per("PT60M", 60), on="ts", how="left")
    use15 = pl.col("pi15_vwap_lei_mwh").is_not_null()
    use60 = pl.col("pi60_vwap_lei_mwh").is_not_null()
    df = df.with_columns(
        pi_vwap_lei_mwh=pl.when(use15).then("pi15_vwap_lei_mwh").otherwise("pi60_vwap_lei_mwh"),
        pi_volume_mwh=pl.when(use15).then("pi15_volume_mwh").when(use60).then("pi60_volume_mwh").otherwise(0.0),
        pi_trades=pl.when(use15).then("pi15_n").when(use60).then("pi60_n").otherwise(0).cast(pl.Int32),
        pi_product=pl.when(use15).then(pl.lit("PT15M")).when(use60).then(pl.lit("PT60M"))
                     .otherwise(pl.lit(None, pl.Utf8)))
    return df.select(SCHEMA_COLS).sort("ts")


# ---------------------------------------------------------------- download (polite, cached, resumable)

class Client:
    """One cookie session + CSRF token; >= MIN_GAP_S between requests; refreshes the token on 419/expiry."""

    def __init__(self, cache: Path, offline: bool = False, gap: float = MIN_GAP_S):
        self.cache, self.offline, self.gap = cache, offline, max(gap, 1.0)
        self.opener = urllib.request.build_opener(urllib.request.HTTPCookieProcessor(http.cookiejar.CookieJar()))
        self.opener.addheaders = [("User-Agent", UA)]
        self.token: str | None = None
        self._last = 0.0
        self.requests = 0

    def _open(self, data: bytes | None = None) -> str:
        wait = self._last + self.gap - time.monotonic()
        if wait > 0:
            time.sleep(wait)
        try:
            with self.opener.open(URL, data=data, timeout=90) as r:
                return r.read().decode("utf-8", errors="replace")
        finally:
            self._last = time.monotonic()
            self.requests += 1

    def _refresh(self) -> None:
        m = re.search(r'name="_token"\s+value="([A-Za-z0-9]+)"', self._open())
        if not m:
            raise RuntimeError("no _token on the PI page (layout changed?)")
        self.token = m[1]

    def path(self, day: dt.date, tip: str) -> Path:
        return self.cache / tip / f"{day.isoformat()}.html.gz"

    def page(self, day: dt.date, tip: str) -> str | None:
        p = self.path(day, tip)
        if p.exists():
            return gzip.decompress(p.read_bytes()).decode("utf-8")
        if self.offline:
            return None
        for attempt in range(5):
            try:
                if self.token is None:
                    self._refresh()
                body = urllib.parse.urlencode(dict(_token=self.token, ziua=day.strftime("%d/%m/%Y"), buton="Refresh",
                                                   action="trading_for", limba="ro", tip=tip)).encode()
                html = self._open(body)
                caption, trades = parse_page(html)
                if page_date(caption) != day:  # stale token / session: the site re-renders today's page
                    raise PermissionError(f"caption {caption!r}")
                p.parent.mkdir(parents=True, exist_ok=True)
                p.write_bytes(gzip.compress(html.encode("utf-8"), 9))
                return html
            except (PermissionError, urllib.error.HTTPError) as e:
                if isinstance(e, urllib.error.HTTPError) and e.code not in (419, 429, 500, 502, 503, 504):
                    raise
                print(f"  {day} tip={tip}: {e}; new session, retry", file=sys.stderr)
                self.token = None
                time.sleep(10 * (attempt + 1))
            except (urllib.error.URLError, TimeoutError, ConnectionError) as e:
                print(f"  {day} tip={tip}: {e!r}; retry", file=sys.stderr)
                time.sleep(10 * (attempt + 1))
        raise RuntimeError(f"giving up on {day} tip={tip}")


def build(year: int = TEST_YEAR, cache: Path = CACHE, offline: bool = False) -> pl.DataFrame:
    """Download (or read cached) every trading-date page for both products and aggregate the delivery year."""
    client = Client(cache, offline=offline)
    days = [dt.date(year - 1, 12, 31) + dt.timedelta(d) for d in range((dt.date(year, 12, 31)
                                                                      - dt.date(year - 1, 12, 31)).days + 1)]
    frames, unknown, missing = [], {}, []
    t0 = time.time()
    for k, day in enumerate(days):
        for tip, product in (("15", "PT15M"), ("60", "PT60M")):
            html = client.page(day, tip)
            if html is None:
                missing.append((day, tip))
                continue
            df, unk = trades_frame(html, day, product)
            frames.append(df)
            for u in unk:
                unknown[u] = unknown.get(u, 0) + 1
        if client.requests and (k + 1) % 30 == 0:
            print(f"  {day}: {client.requests} requests, {time.time() - t0:.0f}s", file=sys.stderr)
    trades = pl.concat(frames)
    a, b = quarter_grid(year)["ts"][[0, -1]]
    trades = trades.filter(pl.col("ts").is_between(a, b))
    if unknown:
        print(f"  unknown labels (skipped): {dict(sorted(unknown.items())[:20])}", file=sys.stderr)
    if missing:
        print(f"  {len(missing)} pages not cached (offline): {missing[:5]}...", file=sys.stderr)
    print(f"  {trades.height:,} trades in {year} ({client.requests} HTTP requests)")
    return aggregate(trades, year)


def summary(df: pl.DataFrame) -> str:
    n = df.height
    by = dict(df.group_by("pi_product").len().iter_rows())
    v = df["pi_vwap_lei_mwh"].drop_nulls()
    return (f"{n:,} intervals: PT15M {by.get('PT15M', 0):,}, PT60M fallback {by.get('PT60M', 0):,}, "
            f"no trade {by.get(None, 0):,} (pi = pzu); pi_vwap mean {v.mean():.1f}, p5 {v.quantile(0.05):.1f}, "
            f"p95 {v.quantile(0.95):.1f} lei/MWh")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--year", type=int, default=TEST_YEAR)
    ap.add_argument("--cache", type=Path, default=CACHE, help="raw response cache (gzipped HTML per page)")
    ap.add_argument("--offline", action="store_true", help="use the cache only")
    ap.add_argument("--out", type=Path, default=OUT)
    ap.add_argument("--no-manifest", action="store_true")
    args = ap.parse_args()
    df = build(args.year, args.cache, args.offline)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    df.write_parquet(args.out, compression="zstd", compression_level=19)
    print(f"  wrote {args.out}: {summary(df)}")
    if not args.no_manifest and args.out.resolve().parent == RAW.resolve():
        from sentinel.data.download import write_manifest
        write_manifest()


if __name__ == "__main__":
    main()
