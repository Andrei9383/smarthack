"""Extract the files printed by scripts/demo_data_dump.py from a saved `lab logs <run>` (card #193).

    lab logs 1100 > /tmp/run.log && python scripts/demo_data_load.py /tmp/run.log --out demo/grid

Checks each file's sha256 and size against the BEGIN marker.
"""

from __future__ import annotations

import argparse
import base64
import hashlib
import lzma
import re
from pathlib import Path

BEGIN = re.compile(r"^=====BEGIN (\S+) ([0-9a-f]{64}) (\d+)=====$")


def extract(log: str) -> dict[str, bytes]:
    files: dict[str, bytes] = {}
    name, chunks = None, []
    for line in log.splitlines():
        m = BEGIN.match(line)
        if m:
            name, sha, size, chunks = m.group(1), m.group(2), int(m.group(3)), []
        elif name and line == f"=====END {name}=====":
            data = lzma.decompress(base64.b64decode("".join(chunks)))
            if hashlib.sha256(data).hexdigest() != sha or len(data) != size:
                raise ValueError(f"{name}: checksum mismatch")
            files[name] = data
            name = None
        elif name:
            chunks.append(line.strip())
    return files


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("log", type=Path)
    ap.add_argument("--out", type=Path, required=True)
    a = ap.parse_args()
    a.out.mkdir(parents=True, exist_ok=True)
    for name, data in extract(a.log.read_text(errors="replace")).items():
        (a.out / name).write_bytes(data)
        print(f"{a.out / name}: {len(data)} bytes")


if __name__ == "__main__":
    main()
