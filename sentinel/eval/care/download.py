"""CARE-to-Compare downloader (Zenodo, Gück et al. 2024, doi:10.3390/data9120138). Never commit the data.

The archive is ~5.5 GB (record 15846963; 6.1 GB in the older 14958989), above our 2 GB in-run limit, so we never
fetch it whole. A ZIP keeps its directory at the end, so with HTTP range requests we read the directory and pull
only the members of the requested farm(s). Wind Farm A (EDP, onshore Portugal, the farm with named gearbox /
generator-bearing / transformer temperatures and failure events) is ~0.16 GB compressed.

Cache: ~/.cache/sentinel/care/<record>/ (or $SENTINEL_CACHE/care). Offline, or when the requested members exceed
`max_gb`, `fetch()` returns None and prints why: the eval reports `care_status` instead of failing.
"""

from __future__ import annotations

import io
import json
import os
import shutil
import urllib.request
import zipfile
from pathlib import Path

RECORDS = ("15846963", "14958989")  # newest first
API = "https://zenodo.org/api/records/{}"
UA = {"User-Agent": "sentinel-care/1.0 (smarthack; +card 43)"}


def cache_dir() -> Path:
    return Path(os.environ.get("SENTINEL_CACHE", Path.home() / ".cache" / "sentinel")) / "care"


class HTTPRangeFile(io.RawIOBase):
    """Read-only, seekable file over HTTP range requests (enough for zipfile)."""

    def __init__(self, url: str, timeout: float = 60.0):
        self.timeout = timeout
        self.pos = 0
        req = urllib.request.Request(url, method="HEAD", headers=UA)
        with urllib.request.urlopen(req, timeout=timeout) as r:
            self.size = int(r.headers["Content-Length"])
            self.url = r.url

    def seekable(self) -> bool:
        return True

    def readable(self) -> bool:
        return True

    def tell(self) -> int:
        return self.pos

    def seek(self, off: int, whence: int = 0) -> int:
        self.pos = off if whence == 0 else (self.pos + off if whence == 1 else self.size + off)
        return self.pos

    def read(self, n: int = -1) -> bytes:
        if n is None or n < 0:
            n = self.size - self.pos
        if n == 0 or self.pos >= self.size:
            return b""
        end = min(self.size, self.pos + n) - 1
        req = urllib.request.Request(self.url, headers={**UA, "Range": f"bytes={self.pos}-{end}"})
        with urllib.request.urlopen(req, timeout=self.timeout) as r:
            data = r.read()
        self.pos += len(data)
        return data


def archive_url(log=print) -> tuple[str, str, int] | None:
    """(record, url, bytes) of the CARE zip on Zenodo, or None when offline."""
    for rec in RECORDS:
        try:
            with urllib.request.urlopen(urllib.request.Request(API.format(rec), headers=UA), timeout=30) as r:
                meta = json.load(r)
        except Exception as e:  # offline, DNS, rate limit
            log(f"[care] Zenodo record {rec} unavailable: {e!r}")
            continue
        for f in meta.get("files", []):
            if f["key"].lower().endswith(".zip"):
                return rec, f["links"]["self"], int(f["size"])
    return None


def fetch(farms=("Wind Farm A",), max_gb: float = 2.0, log=print) -> Path | None:
    """Download (or reuse) the selected farms. → the extracted `CARE_To_Compare` directory, or None (skipped)."""
    found = archive_url(log)
    if found is None:
        log("[care] SKIPPED: Zenodo not reachable (offline?); no CARE score this run")
        return None
    rec, url, size = found
    root = cache_dir() / rec
    done = root / ".complete.json"
    if done.exists() and set(json.loads(done.read_text()).get("farms", [])) >= set(farms):
        log(f"[care] cached: {root}")
        return root / "CARE_To_Compare"
    log(f"[care] archive {size / 1e9:.2f} GB (record {rec}); fetching only {list(farms)} by HTTP range")
    try:
        zf = zipfile.ZipFile(HTTPRangeFile(url))
        members = [i for i in zf.infolist() if any(f"/{farm}/" in i.filename for farm in farms)
                   or i.filename.endswith("README.md")]
        need = sum(i.compress_size for i in members)
        if need > max_gb * 1e9:
            log(f"[care] SKIPPED: selected members are {need / 1e9:.2f} GB > {max_gb} GB limit")
            return None
        log(f"[care] downloading {len(members)} members, {need / 1e9:.3f} GB compressed")
        root.mkdir(parents=True, exist_ok=True)
        for i in members:
            if i.is_dir():
                continue
            dest = root / i.filename
            if dest.exists() and dest.stat().st_size == i.file_size:
                continue
            dest.parent.mkdir(parents=True, exist_ok=True)
            tmp = dest.with_suffix(dest.suffix + ".part")
            with zf.open(i) as src, open(tmp, "wb") as dst:
                shutil.copyfileobj(src, dst, 8 << 20)
            tmp.replace(dest)
    except Exception as e:
        log(f"[care] SKIPPED: download failed: {e!r}")
        return None
    done.write_text(json.dumps({"record": rec, "farms": list(farms), "archive_bytes": size}))
    return root / "CARE_To_Compare"
