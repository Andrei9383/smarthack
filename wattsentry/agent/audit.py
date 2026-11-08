"""SQLite audit store for the replay (card #17): create the DB from the Drizzle SQL, insert rows, never alter schema.

Schema source, in order: `packages/db/drizzle/*/migration.sql` or `packages/db/drizzle/*.sql` (#38, source of
truth) if present, else `wattsentry/agent/schema_fallback.sql` (interim copy of #38's migration).

Inserts are column-intersection-safe, so a schema change on #38 never breaks the replay:
- `PRAGMA table_info` per table; only known columns are inserted; unknown fields are dropped with one warning
  per (table, column) (see `AuditDB.dropped`).
- Column name aliases (`ALIASES`): a row may say `level`, the table may call it `zone_type`; first match wins.
- Time values: pass `Ts(unix_seconds)` or an ISO-8601 UTC string; INTEGER columns get unix seconds
  (`WATTSENTRY_TS_UNIT=ms` for milliseconds), TEXT columns get ISO-8601 `...Z`.
- Derived columns when the table has them and the row does not: `step` (0..95 in the Europe/Bucharest day),
  `day` (local midnight, INTEGER unix time or TEXT YYYY-MM-DD).
- NOT NULL columns without default that the row lacks get a neutral value (0 / '') and a warning.
- Value maps (`VALUE_MAP`): contract enums -> #38's enums (asset_kind site -> prosumer, severity warning -> warn).

Strict mode (`AuditDB(strict=True)`, the default of `replay.py` and the tests): a produced field with no column
(after aliases), a NOT NULL column we leave to the filler, or a missing table RAISES `SchemaMismatch`, except
for the fields in `ALLOWED_DROPS` -- the explicit list of things we knowingly don't store. Non-strict mode only
warns once per (table, field), so a replay against an older or newer schema still completes.
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
from operator import itemgetter
from pathlib import Path
from typing import Any, Callable, Iterable, Mapping, Optional, Sequence

from wattsentry.contracts import Action

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


_MEMO_MAX = 200_000
_FLOAT_ONLY = {float}
_step_memo: dict[int, int] = {}
_date_memo: dict[tuple[int, str, int], Any] = {}
_iso_memo: dict[str, int] = {}


def _memo_step(tsec: int) -> int:
    """local_step, memoised (a replay has few distinct timestamps)."""
    r = _step_memo.get(tsec)
    if r is None:
        if len(_step_memo) > _MEMO_MAX:
            _step_memo.clear()
        r = _step_memo[tsec] = local_step(tsec)
    return r


def _memo_date(tsec: int, affinity: str, unit: int) -> Any:
    """format_time(..., date_only=True), memoised."""
    k = (tsec, affinity, unit)
    r = _date_memo.get(k)
    if r is None:
        if len(_date_memo) > _MEMO_MAX:
            _date_memo.clear()
        r = _date_memo[k] = format_time(tsec, affinity, unit, True)
    return r


def _memo_parse_iso(s: str) -> int:
    r = _iso_memo.get(s)
    if r is None:
        if len(_iso_memo) > _MEMO_MAX:
            _iso_memo.clear()
        r = _iso_memo[s] = parse_iso(s)
    return r


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
# Fields the agent produces that #38 (d362a39) knowingly has no column for. Strict mode raises on any other drop.
# Every entry is review feedback for #38: add the column, or confirm the drop. Keys: table ("*" = any table).
ALLOWED_DROPS: dict[str, dict[str, str]] = {
    "*": {
        # app.md: "No step column: bucket by ts converted to Europe/Bucharest". #44's rows still carry it.
        "step": "no step column by design; UI derives it from ts (Europe/Bucharest)",
    },
    "feeders": {
        # roll-ups of assets; handy for the Feeders panel without a GROUP BY, but derivable
        "n_sites": "derivable: COUNT(assets) WHERE feeder_id",
        "installed_pv_kw": "derivable: SUM(assets.recorded_kwp) WHERE feeder_id",
        "installed_batt_kwh": "derivable: SUM(assets.battery_kwh) WHERE feeder_id",
    },
    "gis_flags": {
        # grid-twin suggestion confidence (separate from the feeder's trust score tau); kept in audit_log
        "confidence": "suggestion confidence; also in the gis_flag action's audit_log.prediction_json",
    },
    "zone_demand": {
        # #44 zone_demand_rows: per-step q90/capacity that drives the 24 h stress slider; #38 keeps peak_stress only
        "stress": "per-step stress (q90 / capacity); #38 stores only the zone-level peak_stress",
        # one model version per zone_demand action; audit_log.model_version has it
        "model_version": "in audit_log.model_version of the zone_demand action",
    },
}


class SchemaMismatch(RuntimeError):
    """Strict mode: the agent produced data the schema cannot hold (and it is not in ALLOWED_DROPS)."""


def allowed_drop(table: str, field: str) -> bool:
    return field in ALLOWED_DROPS.get(table, {}) or field in ALLOWED_DROPS["*"]


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
                 batch: int = 20000, warn: Optional[Callable[[str], None]] = None, strict: bool = False,
                 vacuum: bool = False):
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
        self.ts_unit = 1000 if os.environ.get("WATTSENTRY_TS_UNIT", "s").lower() == "ms" else 1
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
        self.strict = strict
        self.vacuum = vacuum
        self.filled: dict[str, set[str]] = {}

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

        def slow(v: Any) -> Any:
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

        # Fast path by exact type (same results as `slow`, which handles every other type): rows are mostly
        # plain float / int / str / None / Ts, and a replay has few distinct timestamps, so Ts are memoised.
        isfinite = math.isfinite
        ts_memo: dict[int, Any] = {}

        def conv(v: Any) -> Any:
            t = type(v)
            if t is float:
                return v if isfinite(v) else None
            if t is int:
                return v
            if t is str:
                if is_time and len(v) >= 19 and v[4] == "-":
                    return slow(v)
                return v if vmap is None else vmap.get(v, v)
            if v is None:
                return None
            if t is Ts:
                r = ts_memo.get(v)
                if r is None:
                    if len(ts_memo) > 200_000:
                        ts_memo.clear()
                    r = ts_memo[v] = format_time(int(v), aff, unit)
                return r
            return slow(v)

        # types conv returns unchanged (a float only when finite): lets _flush_table skip whole columns
        conv.passthrough = frozenset({int, type(None)} | (set() if is_time or vmap is not None else {str}))  # type: ignore[attr-defined]
        return conv

    def _plan(self, table: str, keys: tuple[str, ...]):
        cache_key = (table, keys)
        if cache_key in self._plans:
            return self._plans[cache_key]
        real = self.resolve(table)
        if real is None:
            if self.strict:
                raise SchemaMismatch(f"table {table!r} not in schema ({self.schema_source})")
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
                if self.strict and not allowed_drop(real, k):
                    raise SchemaMismatch(f"{real}.{k}: produced but no column in {self.schema_source} "
                                         f"(map it to #38's name, or add it to audit.ALLOWED_DROPS with a reason)")
                if k not in self.dropped.setdefault(real, set()):
                    self.dropped[real].add(k)
                    if not allowed_drop(real, k):
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
                if self.strict:
                    raise SchemaMismatch(f"{real}.{c.name} is NOT NULL and the row ({', '.join(keys)}) has no value")
                self.filled.setdefault(real, set()).add(c.name)
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
            # Column-wise conversion. Every non-derived mapping key is one of `keys`, so r[key] == r.get(key).
            columns: list[Any] = []
            tsecs = self._tsecs(rs, src)  # always (as before): an unparseable row time raises even if unused
            for key, col, conv in mapping:
                if key in ("__step", "__date", "__fill"):
                    if key == "__fill":
                        columns.append([conv(None)] * len(rs))
                    elif key == "__step":
                        columns.append([_memo_step(x) if x is not None else 0 for x in tsecs])
                    else:
                        aff, unit = conv(None), self.ts_unit
                        columns.append([None if x is None else _memo_date(x, aff, unit) for x in tsecs])
                else:
                    raw = list(map(itemgetter(key), rs))
                    types = set(map(type, raw))
                    keep = getattr(conv, "passthrough", frozenset())
                    if types <= keep or (types == _FLOAT_ONLY and all(map(math.isfinite, raw))):
                        columns.append(raw)  # conv would return every value unchanged
                    else:
                        columns.append(list(map(conv, raw)))
            vals = list(zip(*columns)) if columns else [() for _ in rs]
            self.conn.executemany(sql, vals)
            self.counts[real] = self.counts.get(real, 0) + len(vals)

    @staticmethod
    def _tsecs(rs: list[dict[str, Any]], src: Optional[str]) -> list[Optional[int]]:
        """Row time in unix seconds (None without a time field), as the derived step / day / date use it."""
        if not src:
            return [None] * len(rs)
        out: list[Optional[int]] = []
        for r in rs:
            ts = r.get(src)
            if ts is None:
                out.append(None)
            elif isinstance(ts, (int, Ts)):
                out.append(int(ts))
            else:
                out.append(_memo_parse_iso(str(ts)))
        return out

    def flush(self) -> None:
        for t in list(self._buf):
            self._flush_table(t)
        self.conn.commit()

    def close(self) -> None:
        self.flush()
        if self.vacuum:
            self.conn.execute("VACUUM")  # rebuilds indexes compactly (~6% smaller)
        self.conn.close()

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
                "unexpected_drops": {t: sorted(x for x in c if not allowed_drop(t, x))
                                     for t, c in self.dropped.items() if any(not allowed_drop(t, x) for x in c)},
                "filled_not_null": {t: sorted(c) for t, c in self.filled.items() if c},
                "missing_tables": sorted(self.missing_tables)}
