"""Decode the LABDUMP blocks printed by scripts/lab_fetch_outputs.sh (#202) from a saved `lab logs <run>` file.

    lab logs 1234 > /tmp/r1234.log
    python scripts/lab_decode_dump.py /tmp/r1234.log /tmp/recovered     # -> /tmp/recovered/<label>/<files>

Every file is checked against its LABDUMP-SHA line; exits 1 on any mismatch or a truncated block.
"""

from __future__ import annotations

import base64
import hashlib
import io
import re
import sys
import tarfile
from pathlib import Path


def decode(log: str, dest: Path) -> dict[str, list[str]]:
    shas: dict[tuple[str, str], str] = {}
    for m in re.finditer(r"^LABDUMP-SHA (\S+) ([0-9a-f]{64})\s+\./(\S+)$", log, re.M):
        shas[(m[1], m[3])] = m[2]
    out: dict[str, list[str]] = {}
    bad = []
    for m in re.finditer(r"^=====BEGIN LABDUMP (\S+)=====\n(.*?)^=====END LABDUMP \1=====$", log, re.M | re.S):
        label, body = m[1], "".join(m[2].split())
        raw = base64.b64decode(body, validate=True)
        target = dest / label
        target.mkdir(parents=True, exist_ok=True)
        with tarfile.open(fileobj=io.BytesIO(raw), mode="r:gz") as tf:
            tf.extractall(target, filter="data")
            names = [n.removeprefix("./") for n in tf.getnames()]
        for n in names:
            want = shas.get((label, n))
            got = hashlib.sha256((target / n).read_bytes()).hexdigest()
            if want != got:
                bad.append(f"{label}/{n}: sha256 {got} != {want}")
        out[label] = names
    begun = set(re.findall(r"^=====BEGIN LABDUMP (\S+)=====$", log, re.M))
    for label in sorted(begun - set(out)):
        bad.append(f"{label}: block not closed (log truncated?)")
    if bad:
        raise SystemExit("\n".join(bad))
    return out


def main(argv: list[str]) -> None:
    log, dest = Path(argv[0]).read_text(errors="replace"), Path(argv[1])
    for label, names in decode(log, dest).items():
        print(f"{label}: {len(names)} files ok -> {dest / label}")


if __name__ == "__main__":
    main(sys.argv[1:])
