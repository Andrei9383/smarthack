"""SQLite audit store for the replay (card #17): create the DB from the Drizzle SQL, insert rows, never alter schema.

Schema source, in order: `packages/db/drizzle/*/migration.sql` or `packages/db/drizzle/*.sql` (#38, source of
truth) if present, else `sentinel/agent/schema_fallback.sql` (interim copy of #38's migration).

Inserts are column-intersection-safe, so a schema change on #38 never breaks the replay:
- `PRAGMA table_info` per table; only known columns are inserted; unknown fields are dropped with one warning
  per (table, column) (see `AuditDB.dropped`).
- Column name aliases (`ALIASES`): a row may say `level`, the table may call it `zone_type`; first match wins.
- Time values: pass `Ts(unix_seconds)` or an ISO-8601 UTC string; INTEGER columns get unix seconds
  (`SENTINEL_TS_UNIT=ms` for milliseconds), TEXT columns get ISO-8601 `...Z`.
- Derived columns when the table has them and the row does not: `step` (0..95 in the Europe/Bucharest day),
  `day` (local midnight, INTEGER unix time or TEXT YYYY-MM-DD).
- NOT NULL columns without default that the row lacks get a neutral value (0 / '') and a warning.
- Value maps (`VALUE_MAP`): contract enums -> #38's enums (asset_kind site -> prosumer, severity warning -> warn).
"""

from __future__ import annotations

import json
import math
import os
import sqlite3
import sys
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from enum import Enum
from pathlib import Path
from typing import Any, Callable, Iterable, Mapping, Optional, Sequence

from sentinel.contracts import Action

REPO_ROOT = Path(__file__).resolve().parents[2]
DRIZZLE_DIR = REPO_ROOT / "packages" / "db" / "drizzle"
FALLBACK_SQL = Path(__file__).with_name("schema_fallback.sql")

try:
    from zoneinfo import ZoneInfo

    _TZ: Any = ZoneInfo("Europe/Bucharest")
except Exception:  # pragma: no cover - no tzdata
    _TZ = None


class Ts(int):
    """A UTC timestamp in unix seconds; converted per column type on insert."""


def local_dt(ts: int) -> datetime:
    utc = datetime.fromtimestamp(int(ts), tz=timezone.utc)
    if _TZ is not None:
        return utc.astimezone(_TZ)
    off = 3 if 4 <= utc.month <= 10 else 2  # approximation without tzdata
    return (utc + timedelta(hours=off)).replace(tzinfo=None)


def local_step(ts: int) -> int:
    d = local_dt(ts)
    return (d.hour * 60 + d.minute) // 15


def local_midnight(ts: int) -> int:
    d = local_dt(ts)
    mid = d.replace(hour=0, minute=0, second=0, microsecond=0)
    if mid.tzinfo is None:
        return int(ts) - (d - mid).seconds
    return int(mid.timestamp())


def iso(ts: int) -> str:
    return datetime.fromtimestamp(int(ts), tz=timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def parse_iso(s: str) -> int:
    return int(datetime.fromisoformat(s.replace("Z", "+00:00")).timestamp())


def format_time(secs: int, affinity: str, unit: int = 1, date_only: bool = False) -> Any:
    """THE time formatter for every time column, by declared type (PRAGMA table_info):
    INTEGER/REAL/NUMERIC -> unix seconds UTC (x unit), TEXT -> ISO-8601 'YYYY-MM-DDTHH:MM:SSZ'.
    date_only (day/date columns): INTEGER -> local (Europe/Bucharest) midnight in unix time, TEXT -> 'YYYY-MM-DD'."""
    if date_only:
        if affinity in ("INTEGER", "REAL", "NUMERIC"):
            return local_midnight(secs) * unit
        return local_dt(secs).strftime("%Y-%m-%d")
    if affinity in ("INTEGER", "REAL", "NUMERIC"):
        return int(secs) * unit
    return iso(secs)


# Equivalent column names (any member may be in the row; the first one the table has is used).
ALIASES: list[tuple[str, ...]] = [
    ("ts", "t", "timestamp"),
    ("action_id", "ref_id", "id"),
    ("n_turbines", "turbine_count"),
    ("level", "zone_type"),
    ("q10_mw", "p10_mw"), ("q50_mw", "p50_mw"), ("q90_mw", "p90_mw"),
    ("q10_kw", "q10"), ("q50_kw", "q50"), ("q90_kw", "q90"),
    ("code", "kind"),
    ("rank", "rank_k"),
    ("fault", "fault_class"),
    ("lost_lei_day", "lei_at_risk"),
    ("energy_mwh", "production_mwh"),
    ("issues_json", "body_json"),
    ("hit", "hit_p10"),
    ("git_sha", "git_commit"),
    ("created_at", "started_at"),
    ("pv_ac_kw", "pv_kw"), ("act_pwt_kw", "act_pwt"),
    ("ambient_temp_c", "turbine_t"), ("gearbox_temp_c", "gearbox_t"), ("conv_temp_c", "conv_t"),
    ("gen_temp_c", "gen1_t"), ("transformer_temp_c", "transformer_t"),
    ("quality", "data_quality"),
    ("asset_kind", "asset_type"),
]
_ALIAS_OF: dict[str, tuple[str, ...]] = {n: g for g in ALIASES for n in g}
# Table renames between replay.md drafts and #38 (first existing wins).
TABLE_ALIASES: dict[str, tuple[str, ...]] = {
    "feeder_curves": ("feeder_curves", "feeder_steps"),
    "statuses": ("statuses", "asset_status"),
    "schedules": ("schedules", "battery_steps"),
    "reports": ("reports", "park_reports"),
    "audit_log": ("audit_log", "audit"),
}
# Contract enum values -> #38 schema enum values (by column name).
VALUE_MAP: dict[str, dict[str, str]] = {
    "asset_kind": {"site": "prosumer"},
    "severity": {"warning": "warn", "safety": "critical"},
}
TIME_COLS = {"ts", "t", "created_at", "started_at", "finished_at", "t0", "onset", "updated_at"}
DATE_COLS = {"day", "date"}
TIME_SOURCES = ("ts", "created_at", "t")


def find_schema_files(schema_dir: Optional[Path] = None) -> list[Path]:
    """Drizzle SQL files in apply order, or [FALLBACK_SQL]."""
    d = Path(schema_dir) if schema_dir is not None else DRIZZLE_DIR
    if d.is_dir():
        files = sorted(d.glob("*/migration.sql")) + sorted(d.glob("*.sql"))
        files = sorted(set(files), key=lambda p: (p.parent.name if p.name == "migration.sql" else p.stem))
        if files:
            return files
    return [FALLBACK_SQL]


def _affinity(decl: str) -> str:
    d = (decl or "").upper()
    if "INT" in d:
        return "INTEGER"
    if any(x in d for x in ("CHAR", "CLOB", "TEXT")):
        return "TEXT"
    if any(x in d for x in ("REAL", "FLOA", "DOUB")):
        return "REAL"
    if d == "" or "BLOB" in d:
        return "BLOB"
    return "NUMERIC"


@dataclass
class Col:
    name: str
    affinity: str
    notnull: bool
    has_default: bool
    pk: bool


class AuditDB:
    """One SQLite file for a replay: schema applied once, rows buffered and inserted in batches."""

    def __init__(self, path: str | os.PathLike, schema_dir: Optional[Path] = None, fresh: bool = True,
                 batch: int = 20000, warn: Optional[Callable[[str], None]] = None):
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        if fresh and self.path.exists():
            self.path.unlink()
        self.warn = warn or (lambda m: print(f"[audit] WARNING {m}", file=sys.stderr))
        self.conn = sqlite3.connect(str(self.path))
        self.conn.execute("PRAGMA journal_mode=OFF")
        self.conn.execute("PRAGMA synchronous=OFF")
        self.schema_files = find_schema_files(schema_dir)
        existing = {r[0] for r in self.conn.execute("SELECT name FROM sqlite_master WHERE type='table'")}
        if not existing:
            for f in self.schema_files:
                self.conn.executescript(f.read_text())
        self.schema_source = ",".join(str(f.relative_to(REPO_ROOT)) if f.is_relative_to(REPO_ROOT) else str(f)
                                      for f in self.schema_files)
        self.ts_unit = 1000 if os.environ.get("SENTINEL_TS_UNIT", "s").lower() == "ms" else 1
        self.tables: dict[str, dict[str, Col]] = {}
        for (name,) in self.conn.execute("SELECT name FROM sqlite_master WHERE type='table'").fetchall():
            cols = {}
            for cid, cname, ctype, notnull, dflt, pk in self.conn.execute(f"PRAGMA table_info(`{name}`)"):
                cols[cname] = Col(cname, _affinity(ctype), bool(notnull), dflt is not None, bool(pk))
            self.tables[name] = cols
        self._buf: dict[str, list[dict[str, Any]]] = {}
        self._plans: dict[tuple[str, tuple[str, ...]], tuple[str, list[tuple[str, str, Callable]]]] = {}
        self.dropped: dict[str, set[str]] = {}
        self.missing_tables: set[str] = set()
        self.counts: dict[str, int] = {}
        self.batch = batch

    # -- schema introspection ------------------------------------------------------------------------
    def resolve(self, table: str) -> Optional[str]:
        for t in TABLE_ALIASES.get(table, (table,)):
            if t in self.tables:
                return t
        return None

    def has_table(self, table: str) -> bool:
        return self.resolve(table) is not None

    def has_column(self, table: str, col: str) -> bool:
        t = self.resolve(table)
        return t is not None and (col in self.tables[t] or any(a in self.tables[t] for a in _ALIAS_OF.get(col, ())))

    # -- inserts -------------------------------------------------------------------------------------
    def add(self, table: str, row: Mapping[str, Any]) -> None:
        self._buf.setdefault(table, []).append(dict(row))
        if len(self._buf[table]) >= self.batch:
            self._flush_table(table)

    def add_many(self, table: str, rows: Iterable[Mapping[str, Any]]) -> None:
        for r in rows:
            self.add(table, r)

    def _converter(self, col: Col) -> Callable[[Any], Any]:
        vmap = VALUE_MAP.get(col.name)
        unit = self.ts_unit
        is_time = col.name in TIME_COLS
        aff = col.affinity

        def conv(v: Any) -> Any:
            if v is None:
                return None
            if isinstance(v, Enum):
                v = v.value
            if isinstance(v, Ts) or (is_time and isinstance(v, str) and len(v) >= 19 and v[4] == "-"):
                return format_time(int(v) if isinstance(v, Ts) else parse_iso(v), aff, unit)
            if isinstance(v, bool):
                return int(v)
            if isinstance(v, (dict, list, tuple)):
                return json.dumps(v, separators=(",", ":"), default=str)
            if isinstance(v, float) and (math.isnan(v) or math.isinf(v)):
                return None
            if vmap is not None and isinstance(v, str):
                return vmap.get(v, v)
            if hasattr(v, "item") and not isinstance(v, (str, bytes)):  # numpy scalar
                return v.item()
            return v

        return conv

    def _plan(self, table: str, keys: tuple[str, ...]):
        cache_key = (table, keys)
        if cache_key in self._plans:
            return self._plans[cache_key]
        real = self.resolve(table)
        if real is None:
            if table not in self.missing_tables:
                self.missing_tables.add(table)
                self.warn(f"table {table!r} not in schema ({self.schema_source}); rows dropped")
            self._plans[cache_key] = None  # type: ignore[assignment]
            return None
        cols = self.tables[real]
        mapping: list[tuple[str, str, Callable]] = []  # (row key, column, converter)
        used: set[str] = set()
        src = next((k for k in TIME_SOURCES if k in keys), None)  # row time used for step/day/date
        for k in keys:  # direct names first
            if k in cols and k not in used:
                mapping.append((k, k, self._converter(cols[k])))
                used.add(k)
        for k in keys:  # then aliases
            if k in cols:
                continue
            target = next((a for a in _ALIAS_OF.get(k, ()) if a in cols and a not in used), None)
            if target is not None:
                mapping.append((k, target, self._converter(cols[target])))
                used.add(target)
            elif not any(a in used for a in _ALIAS_OF.get(k, ())):
                if k == src and ("step" in cols or DATE_COLS & set(cols)):
                    continue  # used only to derive step / day
                if k not in self.dropped.setdefault(real, set()):
                    self.dropped[real].add(k)
                    self.warn(f"{real}.{k} not in schema; field dropped")
        # derived / required columns the row does not give
        for c in cols.values():
            if c.name in used:
                continue
            if c.name == "step" and src:
                mapping.append(("__step", "step", lambda v: v))
            elif c.name in DATE_COLS and src:
                mapping.append(("__date", c.name, lambda v, a=c.affinity: a))
            elif c.notnull and not c.has_default and not (c.pk and c.affinity == "INTEGER"):
                fill = 0 if c.affinity in ("INTEGER", "REAL", "NUMERIC") else ""
                self.warn(f"{real}.{c.name} is NOT NULL and not provided; filled with {fill!r}")
                mapping.append(("__fill", c.name, lambda v, f=fill: f))
            else:
                continue
            used.add(c.name)
        sql = f"INSERT OR REPLACE INTO `{real}` ({','.join('`%s`' % m[1] for m in mapping)}) " \
              f"VALUES ({','.join('?' * len(mapping))})"
        plan = (sql, mapping, real, cols)
        self._plans[cache_key] = plan  # type: ignore[assignment]
        return plan

    def _flush_table(self, table: str) -> None:
        rows = self._buf.pop(table, [])
        groups: dict[tuple[str, ...], list[dict[str, Any]]] = {}
        for r in rows:
            groups.setdefault(tuple(r.keys()), []).append(r)
        for keys, rs in groups.items():
            plan = self._plan(table, keys)
            if plan is None:
                continue
            sql, mapping, real, cols = plan
            src = next((k for k in TIME_SOURCES if k in keys), None)
            vals = []
            for r in rs:
                out = []
                ts = r.get(src) if src else None
                tsec = None
                if ts is not None:
                    tsec = int(ts) if isinstance(ts, (int, Ts)) else parse_iso(str(ts))
                for key, col, conv in mapping:
                    if key == "__step":
                        out.append(local_step(tsec) if tsec is not None else 0)
                    elif key == "__date":
                        out.append(None if tsec is None else format_time(tsec, conv(None), self.ts_unit, True))
                    elif key == "__fill":
                        out.append(conv(None))
                    else:
                        out.append(conv(r.get(key)))
                vals.append(out)
            self.conn.executemany(sql, vals)
            self.counts[real] = self.counts.get(real, 0) + len(vals)

    def flush(self) -> None:
        for t in list(self._buf):
            self._flush_table(t)
        self.conn.commit()

    def close(self) -> None:
        self.flush()
        self.conn.close()

    def table_count(self, table: str) -> int:
        self.flush()
        real = self.resolve(table)
        return 0 if real is None else self.conn.execute(f"SELECT COUNT(*) FROM `{real}`").fetchone()[0]

    # -- actions ------------------------------------------------------------------------------------
    def write_action(self, action: Action, rows: Optional[Mapping[str, Sequence[Mapping[str, Any]]]] = None,
                     *, ts: Optional[int] = None, extra: Optional[Mapping[str, Any]] = None) -> None:
        """One audit_log row for the action (+ its payload rows in their tables)."""
        a = action.audit
        pred = dict(a.prediction or {})
        if a.schedule is not None and not self.has_column("audit_log", "schedule_json"):
            pred["schedule"] = a.schedule  # #38 audit_log has no schedule column: keep it in prediction_json
        row = {
            "id": action.action_id, "action_id": action.action_id, "run_id": action.run_id,
            "ts": Ts(ts) if ts is not None else action.t, "actor": "agent", "action_type": action.type,
            "policy": action.policy, "model_version": a.model_version, "input_hash": a.input_hash,
            "prediction_json": pred, "threshold": a.threshold, "status": a.status, "schedule_json": a.schedule,
            "note": a.note,
        }
        p = action.payload
        for k in ("asset_id", "feeder_id", "park_id"):
            if p.get(k) is not None:
                row[k] = p[k]
        if p.get("site_id") and "asset_id" not in row:
            row["asset_id"] = p["site_id"]
        if extra:
            row.update(extra)
        self.add("audit_log", row)
        for table, rs in (rows or {}).items():
            self.add_many(table, rs)

    def summary(self) -> dict[str, Any]:
        self.flush()
        return {"schema": self.schema_source, "rows": dict(sorted(self.counts.items())),
                "dropped_fields": {t: sorted(c) for t, c in self.dropped.items() if c},
                "missing_tables": sorted(self.missing_tables)}
