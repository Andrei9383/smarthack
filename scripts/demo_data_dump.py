"""Print small lab outputs into the run log so they can be committed under demo/.

The lab has no artifact download, only `lab logs <run>`. This script (re)generates the outputs on a GPU worker and
prints each file xz-compressed + base64 between markers; `scripts/demo_data_load.py` turns a saved log back into files.

    lab submit <card> --run replay --needs gpu --cmd "sh rust/build.sh status && .venv/bin/python scripts/demo_data_dump.py grid"
    lab submit <card> --run imbalance --cmd "sh rust/build.sh status && .venv/bin/python scripts/demo_data_dump.py imbalance --from-run 1071"

grid:      dso_value with --run-g1 (hosting_capacity.json + dso_value.json in one run) and voltage.json.
imbalance: the worker's stored imbalance.sqlite of --from-run if it is on this machine, else a fresh
           `replay.py --mode imbalance`; exported as compact CSV: imbalance_kpis (whole replay) and positions +
           day-ahead portfolio_forecast for ts in [--from, --to) only (the lab keeps ~1 MB of log per run).
"""

from __future__ import annotations

import argparse
import base64
import csv
import hashlib
import io
import lzma
import os
import sqlite3
import subprocess
import sys
from pathlib import Path

OUT = Path(os.environ.get("LAB_OUTPUT_DIR", ".lab-out/demo-dump"))
WORK = Path("/var/lib/lab-gpu/work")


def dump(name: str, data: bytes) -> None:
    blob = base64.b64encode(lzma.compress(data, preset=9 | lzma.PRESET_EXTREME)).decode()
    print(f"=====BEGIN {name} {hashlib.sha256(data).hexdigest()} {len(data)}=====", flush=True)
    for i in range(0, len(blob), 4000):
        print(blob[i : i + 4000])
    print(f"=====END {name}=====", flush=True)


def run(cmd: list[str]) -> int:
    print("+ " + " ".join(cmd), flush=True)
    return subprocess.call(cmd)


def grid() -> int:
    OUT.mkdir(parents=True, exist_ok=True)
    py = sys.executable
    rc = run([py, "-m", "wattsentry.eval.dso_value", "--world", "sim", "--build-caches", "--run-g1", "--out", str(OUT)])
    rc_v = run([py, "-m", "wattsentry.gridtwin.voltage", "--world", "sim", "--out", str(OUT / "voltage")])
    for name, path in [("hosting_capacity.json", OUT / "g1" / "hosting_capacity.json"),
                       ("dso_value.json", OUT / "dso_value.json"), ("voltage.json", OUT / "voltage" / "voltage.json")]:
        if path.exists():
            dump(name, path.read_bytes())
        else:
            print(f"# missing {path}")
    print(f"# exit codes: dso_value={rc} voltage={rc_v}")
    return rc


def table_csv(con: sqlite3.Connection, sql: str) -> bytes:
    cur = con.execute(sql)
    buf = io.StringIO()
    w = csv.writer(buf, lineterminator="\n")
    w.writerow([d[0] for d in cur.description])
    for row in cur:
        w.writerow(["" if v is None else (f"{v:.9g}" if isinstance(v, float) else v) for v in row])
    return buf.getvalue().encode()


def imbalance(from_run: str, ts_from: str, ts_to: str, regen: bool = True) -> int:
    cands = [WORK / "outputs" / from_run / "imbalance.sqlite", WORK / "runs" / from_run / ".lab-out" / "imbalance.sqlite"]
    src = next((p for p in cands if os.access(p, os.R_OK)), None)
    if src is None and not regen:
        print(f"# run {from_run} output not on this worker (host {os.uname().nodename}, user {os.getuid()})")
        return 3
    if src is None:
        OUT.mkdir(parents=True, exist_ok=True)
        src = OUT / "imbalance.sqlite"
        print(f"# run {from_run} output not on this worker: regenerating", flush=True)
        rc = run([sys.executable, "replay.py", "--mode", "imbalance", "--out", str(src)])
        if rc:
            return rc
        source = f"run {os.environ.get('LAB_RUN_ID', 'local')}"
    else:
        source = f"run {from_run}"
    print(f"# source: {src} ({source})", flush=True)
    con = sqlite3.connect(src)
    dump("imbalance_kpis.csv", table_csv(con, "SELECT * FROM imbalance_kpis"))
    win = f"ts >= '{ts_from}' AND ts < '{ts_to}'"
    dump("positions.csv", table_csv(con, "SELECT ts, policy, bought_da, traded_id, actual, dev_mwh, intraday_lei, "
                                         "imbalance_lei, avoidable_lei, import_mwh, export_mwh, run_id FROM positions "
                                         f"WHERE {win} ORDER BY ts, policy"))
    dump("portfolio_forecast_da.csv", table_csv(con, "SELECT issue_time, ts, policy, horizon, p10, p50, p90, run_id "
                                                     f"FROM portfolio_forecast WHERE horizon = 'da' AND {win} "
                                                     "ORDER BY ts, policy"))
    return 0


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("what", choices=["grid", "imbalance"])
    ap.add_argument("--from-run", default="1071")
    ap.add_argument("--from", dest="ts_from", default="2025-05-28T21:00:00Z", help="imbalance: first ts (UTC) kept")
    ap.add_argument("--to", dest="ts_to", default="2025-06-11T21:00:00Z", help="imbalance: end ts (UTC), exclusive")
    ap.add_argument("--no-regen", action="store_true", help="imbalance: fail instead of re-running the replay")
    a = ap.parse_args()
    return grid() if a.what == "grid" else imbalance(a.from_run, a.ts_from, a.ts_to, not a.no_regen)


if __name__ == "__main__":
    raise SystemExit(main())
