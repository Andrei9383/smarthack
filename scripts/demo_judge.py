"""Judge tables for the demo DB (card #192): replay#506 again on the lab, ship only judge_verdicts / trust / outcomes.

demo/wattsentry.sqlite is replay#506's exact output, made before the Judge (#149) existed. The Judge only records
verdicts (tests/test_judge_verdict.py::test_judge_on_and_off_give_identical_decisions, replay --judge-check), so a
re-run of #506's command (same world, run id 506) gives the same agent rows plus the Judge tables. The lab has no
artifact download, so `dump` prints the tables xz + base64 between markers and `load` reads them back from the log.

    lab submit 192 --run replay --needs gpu --cmd "sh rust/build.sh status && .venv/bin/python scripts/demo_judge.py dump"
    lab logs <run> > run.log && python scripts/demo_judge.py load run.log      # into demo/wattsentry.sqlite
    python scripts/demo_judge.py digest demo/wattsentry.sqlite                     # per-table row count + sha256

`dump` also prints the digest of every table of the re-run, to compare with `digest demo/wattsentry.sqlite`.
`load` creates the tables with packages/db's judge migration, inserts the rows and fails unless every table that was
there before has the same digest afterwards (no VACUUM: existing pages are not rewritten).
"""

from __future__ import annotations

import argparse
import base64
import hashlib
import json
import lzma
import os
import re
import sqlite3
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
JUDGE_TABLES = ("judge_verdicts", "trust", "outcomes")
MIGRATION = next((ROOT / "packages" / "db" / "drizzle").glob("*_judge_tables/migration.sql"))
BEGIN = re.compile(r"^=====BEGIN (\S+) ([0-9a-f]{64}) (\d+)=====$")
# replay#506's command (demo/README.md), run id pinned so the rows carry the demo's run ids (506:sentinel)
CACHES = [["-m", "wattsentry.data.windows"], ["-m", "wattsentry.sim.wind", "--no-plots"], ["-m", "wattsentry.gridtwin.run"]]
REPLAY = ["replay.py", "--world", "sim", "--policy", "baseline,sentinel", "--run-id", "506", "--judge-check"]


def tables(con: sqlite3.Connection) -> list[str]:
    return [r[0] for r in con.execute("SELECT name FROM sqlite_master WHERE type = 'table' AND name NOT LIKE "
                                      "'sqlite_%' ORDER BY name")]


def columns(con: sqlite3.Connection, table: str) -> list[str]:
    return [r[1] for r in con.execute(f'PRAGMA table_info("{table}")')]


def rows(con: sqlite3.Connection, table: str):
    try:
        return con.execute(f'SELECT * FROM "{table}" ORDER BY rowid')
    except sqlite3.OperationalError:  # WITHOUT ROWID
        return con.execute(f'SELECT * FROM "{table}"')


def _cell(v):
    return v.hex() if isinstance(v, bytes) else v


def digest(con: sqlite3.Connection, table: str, skip: tuple[str, ...] = ()) -> tuple[int, str]:
    """Row count and sha256 over every row in rowid order (floats as repr, so exact); `skip` columns left out."""
    cols = columns(con, table)
    keep = [i for i, c in enumerate(cols) if c not in skip]
    h = hashlib.sha256(json.dumps([cols[i] for i in keep]).encode())
    n = 0
    for r in rows(con, table):
        h.update(json.dumps([_cell(r[i]) for i in keep]).encode() + b"\n")
        n += 1
    return n, h.hexdigest()


def digests(path: Path) -> dict[str, tuple[int, str]]:
    con = sqlite3.connect(f"file:{path}?mode=ro", uri=True)
    out = {t: digest(con, t) for t in tables(con)}
    # runs: created_at / finished_at are wall clock, git_sha is the commit; the rest must match
    out["runs[-clock,sha]"] = digest(con, "runs", ("created_at", "finished_at", "git_sha"))
    out["runs[-clock,sha,notes]"] = digest(con, "runs", ("created_at", "finished_at", "git_sha", "notes"))
    for rid, notes in con.execute("SELECT id, notes FROM runs ORDER BY id"):
        for k, v in sorted(json.loads(notes or "{}").items()):
            out[f"runs.notes[{rid}].{k}"] = (1, hashlib.sha256(json.dumps(v, sort_keys=True).encode()).hexdigest())
    con.close()
    return out


def print_digests(path: Path) -> None:
    for name, (n, h) in digests(path).items():
        print(f"DIGEST {name} {n} {h}")


def emit(name: str, data: bytes) -> None:
    blob = base64.b64encode(lzma.compress(data, preset=9 | lzma.PRESET_EXTREME)).decode()
    print(f"=====BEGIN {name} {hashlib.sha256(data).hexdigest()} {len(data)}=====", flush=True)
    for i in range(0, len(blob), 4000):
        print(blob[i : i + 4000])
    print(f"=====END {name}=====", flush=True)


def table_json(con: sqlite3.Connection, table: str) -> bytes:
    """Column-oriented JSON (compresses ~3x better than rows): {"columns": [...], "data": [[col0...], [col1...]]}."""
    cols = columns(con, table)
    data: list[list] = [[] for _ in cols]
    for r in rows(con, table):
        for i, v in enumerate(r):
            data[i].append(_cell(v))
    return json.dumps({"table": table, "columns": cols, "data": data}, separators=(",", ":")).encode()


def run(cmd: list[str], log: Path) -> int:
    print("+ " + " ".join(cmd), flush=True)
    with log.open("ab") as f:
        rc = subprocess.call(cmd, stdout=f, stderr=subprocess.STDOUT, cwd=ROOT)
    if rc:
        print(f"# exit {rc}; last lines of {log}:\n" + "\n".join(log.read_text(errors="replace").splitlines()[-40:]))
    return rc


def dump(a: argparse.Namespace) -> int:
    out = Path(os.environ.get("LAB_OUTPUT_DIR", ROOT / ".lab-out" / "demo-judge"))
    out.mkdir(parents=True, exist_ok=True)
    db, log = out / "wattsentry.sqlite", out / "replay.log"
    db.unlink(missing_ok=True)
    py = sys.executable
    if a.overlay:  # e.g. card/145's Judge weights + slot fix before it is on main (worker checkout only, not committed)
        ok = subprocess.call(["git", "fetch", "-q", "origin", a.overlay], cwd=ROOT) == 0 and subprocess.call(
            ["git", "checkout", "FETCH_HEAD", "--", *a.overlay_paths.split(",")], cwd=ROOT) == 0
        print(f"# overlay {a.overlay} ({a.overlay_paths}): {'applied' if ok else 'NOT applied'}", flush=True)
    for c in CACHES:
        if run([py, *c], log):
            return 1
    rc = run([py, *REPLAY, "--out", str(db)], log)
    text = log.read_text(errors="replace").splitlines()
    for line in text:  # KPIs + judge lines only: the lab keeps ~1 MB of log per run, the data needs most of it
        if line.startswith(("LAB_METRIC", "# judge", "# replay", "[judge]", "# not stored", "# SCHEMA", "# FALLBACK")):
            print(line[:2000], flush=True)
    if rc:
        return rc
    print_digests(db)
    con = sqlite3.connect(db)
    for t in JUDGE_TABLES:
        if t in a.tables:
            emit(f"{t}.json", table_json(con, t))
    con.close()
    return 0


def extract(log: str) -> dict[str, bytes]:
    files: dict[str, bytes] = {}
    name, chunks, sha, size = None, [], "", 0
    for line in log.splitlines():
        m = BEGIN.match(line.strip())
        if m:
            name, sha, size, chunks = m.group(1), m.group(2), int(m.group(3)), []
        elif name and line.strip() == f"=====END {name}=====":
            data = lzma.decompress(base64.b64decode("".join(chunks)))
            if hashlib.sha256(data).hexdigest() != sha or len(data) != size:
                raise ValueError(f"{name}: checksum mismatch")
            files[name] = data
            name = None
        elif name:
            chunks.append(line.strip())
    return files


def load(a: argparse.Namespace) -> int:
    files = extract(Path(a.log).read_text(errors="replace"))
    missing = [t for t in JUDGE_TABLES if f"{t}.json" not in files]
    if missing:
        print(f"missing in log: {missing}", file=sys.stderr)
        return 1
    before = digests(a.db)
    con = sqlite3.connect(a.db)
    present = [t for t in JUDGE_TABLES if t in tables(con)]
    if present:
        print(f"{a.db} already has {present}", file=sys.stderr)
        return 1
    run_ids = {r[0] for r in con.execute("SELECT id FROM runs")}
    with con:
        for stmt in MIGRATION.read_text().split("--> statement-breakpoint"):
            if stmt.strip():
                con.execute(stmt)
        for t in JUDGE_TABLES:
            d = json.loads(files[f"{t}.json"])
            if columns(con, t) != d["columns"]:
                raise ValueError(f"{t}: columns {d['columns']} != schema {columns(con, t)}")
            bad = set(d["data"][d["columns"].index("run_id")]) - run_ids
            if bad:
                raise ValueError(f"{t}: run ids not in runs: {sorted(bad)}")
            cols = ", ".join(f'"{c}"' for c in d["columns"])
            con.executemany(f'INSERT INTO "{t}" ({cols}) VALUES ({", ".join("?" * len(d["columns"]))})',
                            zip(*d["data"]))
            print(f"{t}: {len(d['data'][0])} rows")
    ok = con.execute("PRAGMA integrity_check").fetchone()[0]
    con.close()
    after = digests(a.db)
    changed = [k for k in before if before[k] != after.get(k)]
    print(f"integrity_check {ok}; {len(before)} pre-existing digests, changed: {changed or 'none'}; "
          f"{Path(a.db).stat().st_size / 1e6:.2f} MB")
    return 0 if ok == "ok" and not changed else 1


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    sub = ap.add_subparsers(dest="cmd", required=True)
    d = sub.add_parser("dump", help="lab: re-run replay#506, print digests + Judge tables")
    d.add_argument("--tables", default=",".join(JUDGE_TABLES))
    d.add_argument("--overlay", default="", help="branch whose --overlay-paths are checked out before the replay")
    d.add_argument("--overlay-paths", default="models/judge.*,wattsentry/judge/slot.py")
    lo = sub.add_parser("load", help="local: import the Judge tables from a saved `lab logs` into the demo DB")
    lo.add_argument("log")
    lo.add_argument("--db", default=str(ROOT / "demo" / "wattsentry.sqlite"))
    g = sub.add_parser("digest", help="row count + sha256 per table")
    g.add_argument("db")
    a = ap.parse_args()
    if a.cmd == "dump":
        return dump(a)
    if a.cmd == "load":
        return load(a)
    print_digests(Path(a.db))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
