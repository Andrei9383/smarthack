"""The live feed's SQLite file (team/contracts/live.md). WAL, so the app can read while the feed writes."""

from __future__ import annotations

import os
import sqlite3
from pathlib import Path
from typing import Any, Iterable, Mapping, Optional

REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_DB = REPO_ROOT / "demo" / "live.sqlite"

SCHEMA = """
CREATE TABLE IF NOT EXISTS live_runs (
  run_id TEXT PRIMARY KEY,
  source TEXT NOT NULL,
  label TEXT NOT NULL,
  world TEXT NOT NULL,
  tick_s INTEGER NOT NULL,
  clock TEXT NOT NULL,
  feeders_json TEXT NOT NULL,
  model_versions_json TEXT,
  notes TEXT,
  created_at TEXT NOT NULL
);
CREATE TABLE IF NOT EXISTS live_ticks (
  run_id TEXT NOT NULL,
  ts TEXT NOT NULL,
  wall_at TEXT NOT NULL,
  status TEXT NOT NULL,
  n_readings INTEGER NOT NULL,
  PRIMARY KEY (run_id, ts)
);
CREATE TABLE IF NOT EXISTS live_readings (
  run_id TEXT NOT NULL,
  ts TEXT NOT NULL,
  site_id TEXT NOT NULL,
  feeder_id TEXT NOT NULL,
  pv_kw REAL,
  load_kw REAL,
  batt_kw REAL,
  soc REAL,
  net_kw REAL,
  PRIMARY KEY (run_id, ts, site_id)
);
CREATE TABLE IF NOT EXISTS live_predictions (
  run_id TEXT NOT NULL,
  ts TEXT NOT NULL,
  t0 TEXT NOT NULL,
  level TEXT NOT NULL,
  entity_id TEXT NOT NULL,
  feeder_id TEXT NOT NULL,
  quantity TEXT NOT NULL,
  method TEXT NOT NULL,
  q10 REAL NOT NULL,
  q50 REAL NOT NULL,
  q90 REAL NOT NULL,
  limit_kw REAL,
  flag TEXT,
  actual REAL,
  err REAL,
  err_flag INTEGER,
  n_sites INTEGER NOT NULL DEFAULT 1,
  model_version TEXT,
  is_fallback INTEGER NOT NULL DEFAULT 0,
  PRIMARY KEY (run_id, ts, level, entity_id, quantity, method)
);
CREATE INDEX IF NOT EXISTS live_predictions_feeder ON live_predictions (run_id, level, feeder_id, ts);
"""

PRED_COLS = ("run_id", "ts", "t0", "level", "entity_id", "feeder_id", "quantity", "method", "q10", "q50", "q90",
             "limit_kw", "flag", "n_sites", "model_version", "is_fallback")
READ_COLS = ("run_id", "ts", "site_id", "feeder_id", "pv_kw", "load_kw", "batt_kw", "soc", "net_kw")


def db_path(path: Optional[str | Path] = None) -> Path:
    return Path(path or os.environ.get("SENTINEL_LIVE_DB") or DEFAULT_DB)


def connect(path: Optional[str | Path] = None) -> sqlite3.Connection:
    p = db_path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    con = sqlite3.connect(p, isolation_level=None)  # explicit BEGIN/COMMIT per tick
    con.row_factory = sqlite3.Row
    con.execute("PRAGMA journal_mode=WAL")
    con.execute("PRAGMA synchronous=NORMAL")
    con.executescript(SCHEMA)
    return con


def insert_many(con: sqlite3.Connection, table: str, cols: Iterable[str], rows: Iterable[Mapping[str, Any]]) -> int:
    cols = tuple(cols)
    sql = f"INSERT OR IGNORE INTO {table} ({', '.join(cols)}) VALUES ({', '.join('?' * len(cols))})"
    data = [tuple(r.get(c) for c in cols) for r in rows]
    con.executemany(sql, data)
    return len(data)
