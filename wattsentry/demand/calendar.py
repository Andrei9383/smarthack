"""Romanian calendar for demand: public holidays (incl. Orthodox Easter) and Europe/Bucharest local time.

No tz database needed: Romania follows the EU DST rule since 1997 (UTC+2, UTC+3 from the last Sunday of March
01:00 UTC to the last Sunday of October 01:00 UTC), so the sandbox does not need tzdata.
"""

from __future__ import annotations

from datetime import date, datetime, timedelta, timezone
from functools import lru_cache

import numpy as np

HDD_BASE_C = 18.0  # heating degree base (RO / EU convention)
CDD_BASE_C = 22.0  # cooling degree base


def orthodox_easter(year: int) -> date:
    """Orthodox Easter Sunday (Gregorian date). Meeus Julian algorithm + 13 days (valid 1900-2099)."""
    a, b, c = year % 4, year % 7, year % 19
    d = (19 * c + 15) % 30
    e = (2 * a + 4 * b - d + 34) % 7
    month, day = divmod(d + e + 114, 31)
    return date(year, month, day + 1) + timedelta(days=13)


@lru_cache(maxsize=None)
def ro_public_holidays(year: int) -> dict[date, str]:
    """Legal public holidays in Romania (Codul muncii art. 139) for `year`, date -> name."""
    easter = orthodox_easter(year)
    h: dict[date, str] = {
        date(year, 1, 1): "Anul Nou",
        date(year, 1, 2): "Anul Nou (ziua 2)",
        date(year, 1, 24): "Unirea Principatelor",
        easter - timedelta(days=2): "Vinerea Mare",
        easter: "Paștele",
        easter + timedelta(days=1): "Paștele (luni)",
        date(year, 5, 1): "Ziua Muncii",
        date(year, 6, 1): "Ziua Copilului",
        easter + timedelta(days=49): "Rusaliile",
        easter + timedelta(days=50): "Rusaliile (luni)",
        date(year, 8, 15): "Adormirea Maicii Domnului",
        date(year, 11, 30): "Sfântul Andrei",
        date(year, 12, 1): "Ziua Națională",
        date(year, 12, 25): "Crăciunul",
        date(year, 12, 26): "Crăciunul (ziua 2)",
    }
    if year >= 2024:  # Legea 52/2023
        h.setdefault(date(year, 1, 6), "Boboteaza")
        h.setdefault(date(year, 1, 7), "Sfântul Ioan")
    return h


def is_ro_holiday(d: date) -> bool:
    return d in ro_public_holidays(d.year)


def _last_sunday_utc(year: int, month: int) -> np.datetime64:
    """01:00 UTC on the last Sunday of `month` (EU DST switch instant)."""
    d = date(year, month, 31)
    d -= timedelta(days=(d.weekday() + 1) % 7)  # weekday(): Mon=0 .. Sun=6
    return np.datetime64(datetime(d.year, d.month, d.day, 1), "s")


def bucharest_offset_hours(ts_utc) -> np.ndarray:
    """UTC offset (2 or 3) of Europe/Bucharest for each UTC instant (numpy datetime64 array-like)."""
    ts = np.asarray(ts_utc, dtype="datetime64[s]")
    years = ts.astype("datetime64[Y]").astype(int) + 1970
    out = np.full(ts.shape, 2, dtype=np.int64)
    for y in np.unique(years):
        start, end = _last_sunday_utc(int(y), 3), _last_sunday_utc(int(y), 10)
        m = (years == y) & (ts >= start) & (ts < end)
        out[m] = 3
    return out


def to_bucharest(ts_utc) -> np.ndarray:
    """Naive local Europe/Bucharest wall-clock times (datetime64[s]) for UTC instants."""
    ts = np.asarray(ts_utc, dtype="datetime64[s]")
    return ts + bucharest_offset_hours(ts).astype("timedelta64[h]")


def local_hour(ts_utc) -> np.ndarray:
    loc = to_bucharest(ts_utc)
    return ((loc - loc.astype("datetime64[D]")).astype("timedelta64[m]").astype(int) // 60).astype(np.int64)


def utc_iso(ts: np.datetime64) -> str:
    return str(np.datetime64(ts, "s")) + "Z"


def parse_ts(ts: str | datetime | np.datetime64) -> np.datetime64:
    """UTC ISO string / aware datetime / datetime64 (assumed UTC) -> datetime64[s]."""
    if isinstance(ts, np.datetime64):
        return ts.astype("datetime64[s]")
    if isinstance(ts, str):
        ts = datetime.fromisoformat(ts.replace("Z", "+00:00"))
    if ts.tzinfo is not None:
        ts = ts.astimezone(timezone.utc).replace(tzinfo=None)
    return np.datetime64(ts, "s")


def calendar_features(ts_utc) -> dict[str, np.ndarray]:
    """Local calendar features per UTC instant: hour, dow (Mon=0), month, doy sin/cos, weekend, holiday flags.

    `daytype`: 0 = working day, 1 = Saturday, 2 = Sunday or public holiday (used by the national shape).
    """
    loc = to_bucharest(ts_utc)
    days = loc.astype("datetime64[D]")
    hour = ((loc - days).astype("timedelta64[m]").astype(int) // 60).astype(np.int64)
    dow = ((days.astype(np.int64) + 3) % 7).astype(np.int64)  # 1970-01-01 was a Thursday
    month = (days.astype("datetime64[M]").astype(int) % 12 + 1).astype(np.int64)
    doy = (days - days.astype("datetime64[Y]")).astype(int)

    uniq, inv = np.unique(days, return_inverse=True)
    pydays = uniq.astype(object)
    hol_u = np.array([is_ro_holiday(d) for d in pydays], dtype=bool)
    prev_u = np.array([is_ro_holiday(d + timedelta(days=1)) for d in pydays], dtype=bool)  # day before a holiday
    next_u = np.array([is_ro_holiday(d - timedelta(days=1)) for d in pydays], dtype=bool)  # day after a holiday
    holiday = hol_u[inv]
    weekend = dow >= 5
    # "punte": a working day squeezed between a holiday and a weekend is often taken off in RO
    bridge = ~weekend & ~holiday & ((next_u[inv] & (dow == 4)) | (prev_u[inv] & (dow == 0)))
    daytype = np.where(holiday | (dow == 6), 2, np.where(dow == 5, 1, 0))
    return {
        "hour": hour,
        "dow": dow,
        "month": month,
        "doy_sin": np.sin(2 * np.pi * doy / 365.25),
        "doy_cos": np.cos(2 * np.pi * doy / 365.25),
        "weekend": weekend.astype(np.int64),
        "holiday": holiday.astype(np.int64),
        "pre_holiday": prev_u[inv].astype(np.int64),
        "post_holiday": next_u[inv].astype(np.int64),
        "bridge": bridge.astype(np.int64),
        "daytype": daytype.astype(np.int64),
        "local_day": days,
    }


def degree_hours(temp_c) -> tuple[np.ndarray, np.ndarray]:
    t = np.asarray(temp_c, dtype=float)
    return np.maximum(0.0, HDD_BASE_C - t), np.maximum(0.0, t - CDD_BASE_C)
