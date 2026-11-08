"""Context buckets of the trust ledger. A key is the six conditions joined by "|":

    kind | seen | battery | season | quality | ood        e.g. "site|unseen|batt|warm|clean|ood_lo"

Parents drop conditions from last to first ("site|unseen|batt|warm|clean" ... "site"), and the root is "*".
Values: kind site|turbine|zone|feeder · seen seen|unseen · battery batt|nobatt · season warm (Apr-Sep)|cold ·
quality clean|gappy (> 10 % missing steps) · ood ood_lo|ood_hi (OOD score below / above the model's threshold).
"""

from __future__ import annotations

import datetime as _dt

CONDITIONS = ("kind", "seen", "battery", "season", "quality", "ood")
ROOT = "*"


def season_of(month: int) -> str:
    return "warm" if 4 <= int(month) <= 9 else "cold"


def _flag(v, yes: str, no: str) -> str:
    if isinstance(v, str):
        s = v.strip().lower()
        if s in (yes, no):
            return s
        return yes if s in ("1", "true", "yes", "y") else no
    return yes if bool(v) else no


def bucket_key(kind: str = "site", seen=True, battery=False, season: str | int | _dt.date = "warm",
               quality="clean", ood_high=False) -> str:
    """Canonical full bucket key. `seen`/`battery`/`ood_high` take bools or their labels; `season` takes
    'warm'/'cold', a month number or a date; `quality` takes 'clean'/'gappy' or a bool meaning gappy."""
    if isinstance(season, (_dt.date, _dt.datetime)):
        season = season_of(season.month)
    elif isinstance(season, int):
        season = season_of(season)
    if isinstance(quality, str):
        quality = quality.strip().lower()
    else:
        quality = "gappy" if bool(quality) else "clean"
    return "|".join((str(kind), _flag(seen, "seen", "unseen"), _flag(battery, "batt", "nobatt"), str(season),
                     quality, _flag(ood_high, "ood_hi", "ood_lo")))


def truncate(key: str, level: int) -> str:
    """The ancestor of `key` that keeps its first `level` conditions (0 = root)."""
    if level <= 0 or key == ROOT:
        return ROOT
    return "|".join(key.split("|")[:level])


def ancestors(key: str) -> list[str]:
    """The key itself, then its parents from the most to the least specific, ending at the root."""
    n = 0 if key == ROOT else len(key.split("|"))
    return [truncate(key, lv) for lv in range(n, -1, -1)]
