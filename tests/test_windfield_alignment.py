"""The wind field is indexed by hours since the replay clock's start, so the two must agree."""

import json
import re
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent


def test_windfield_starts_at_the_replay_clock():
    replay = (ROOT / "apps/web/src/demo/replay.ts").read_text()
    match = re.search(r"WEEK_START_UTC = Date\.UTC\((\d+), (\d+), (\d+), (\d+), (\d+)\)", replay)
    assert match, "WEEK_START_UTC not found in replay.ts"
    year, month, day, hour, minute = map(int, match.groups())
    clock_start = datetime(year, month + 1, day, hour, minute, tzinfo=timezone.utc)

    meta = json.loads((ROOT / "apps/web/public/windfield/meta.json").read_text())
    field_start = datetime.fromisoformat(meta["hours"][0]["ts"].replace("Z", "+00:00"))

    assert field_start == clock_start, "regenerate with scripts/windfield.py --start"
    assert len(meta["hours"]) * 4 >= int(re.search(r"DAYS = (\d+)", replay).group(1)) * 96
