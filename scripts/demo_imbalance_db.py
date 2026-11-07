"""Build demo/imbalance.sqlite from the CSVs printed by `scripts/demo_data_dump.py imbalance` (card #193).

    python scripts/demo_data_load.py run.log --out /tmp/imb && python scripts/demo_imbalance_db.py /tmp/imb

Tables and index names mirror sentinel/eval/imbalance_replay.py TABLES (= packages/db/src/schema/imbalance.ts).
positions + imbalance_kpis: the whole replay (the UI's "whole replay" waterfall sums positions). portfolio_forecast:
day-ahead rows of the demo month only (the UI draws the band for one day of the demo week), to keep the file small.
Prices (pzu_lei_mwh / pi_lei_mwh) are left NULL: OPCOM data, internal use only, and the UI does not read them.
"""

from __future__ import annotations

import argparse
import csv
import sqlite3
from pathlib import Path

DDL = """
CREATE TABLE portfolio_forecast (issue_time text NOT NULL, ts text NOT NULL, policy text NOT NULL, horizon text,
  p10 real, p50 real, p90 real, run_id text);
CREATE INDEX portfolio_forecast_idx ON portfolio_forecast (ts, policy);
CREATE TABLE positions (ts text NOT NULL, policy text NOT NULL, bought_da real, traded_id real, actual real,
  dev_mwh real, intraday_lei real, imbalance_lei real, avoidable_lei real, pzu_lei_mwh real, pi_lei_mwh real,
  import_mwh real, export_mwh real, run_id text);
CREATE INDEX positions_idx ON positions (ts, policy);
CREATE TABLE imbalance_kpis (run_id text, policy text NOT NULL, name text NOT NULL, value real, unit text,
  label text, date_from text, date_to text);
CREATE INDEX imbalance_kpis_idx ON imbalance_kpis (policy, name);
"""
REAL = {"bought_da", "traded_id", "actual", "dev_mwh", "intraday_lei", "imbalance_lei", "avoidable_lei", "import_mwh",
        "export_mwh", "p10", "p50", "p90", "value", "pzu_lei_mwh", "pi_lei_mwh"}


def rows(path: Path, keep=lambda r: True):
    with path.open(newline="") as f:
        for r in csv.DictReader(f):
            if keep(r):
                yield {k: (None if v == "" else float(v) if k in REAL else v) for k, v in r.items()}


def insert(con: sqlite3.Connection, table: str, it) -> int:
    n = 0
    first = next(it, None)
    if first is None:
        return 0
    cols = list(first)
    sql = f"INSERT INTO {table} ({', '.join(cols)}) VALUES ({', '.join('?' * len(cols))})"
    con.execute(sql, [first[c] for c in cols])
    n = 1
    for r in it:
        con.execute(sql, [r[c] for c in cols])
        n += 1
    return n


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("src", type=Path, help="directory with positions.csv, portfolio_forecast_da.csv, imbalance_kpis.csv")
    ap.add_argument("--out", type=Path, default=Path("demo/imbalance.sqlite"))
    ap.add_argument("--pf-from", default="2025-05-31T21:00:00Z", help="first portfolio_forecast ts kept (UTC)")
    ap.add_argument("--pf-to", default="2025-07-01T21:00:00Z", help="end of the portfolio_forecast ts kept (UTC)")
    a = ap.parse_args()
    a.out.unlink(missing_ok=True)
    con = sqlite3.connect(a.out)
    con.executescript(DDL)
    n_pos = insert(con, "positions", rows(a.src / "positions.csv"))
    n_pf = insert(con, "portfolio_forecast",
                  rows(a.src / "portfolio_forecast_da.csv", lambda r: a.pf_from <= r["ts"] < a.pf_to))
    n_k = insert(con, "imbalance_kpis", rows(a.src / "imbalance_kpis.csv"))
    con.commit()
    con.execute("VACUUM")
    con.close()
    print(f"{a.out}: positions {n_pos}, portfolio_forecast {n_pf}, imbalance_kpis {n_k}, {a.out.stat().st_size / 1e6:.1f} MB")


if __name__ == "__main__":
    main()
