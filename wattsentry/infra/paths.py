"""Where generated data is cached between runs."""
from __future__ import annotations

import os
from pathlib import Path


def cache_root() -> Path:
    """$WATTSENTRY_CACHE, else ~/.cache/wattsentry."""
    return Path(os.environ.get("WATTSENTRY_CACHE", Path.home() / ".cache" / "wattsentry"))
