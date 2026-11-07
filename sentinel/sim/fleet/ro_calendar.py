"""Romanian calendar: public holidays (incl. Orthodox Easter and Rusalii) and local-time features."""

from __future__ import annotations

import datetime as dt

import numpy as np
import pandas as pd

TZ = "Europe/Bucharest"


def orthodox_easter(year: int) -> dt.date:
    """Meeus' Julian algorithm, shifted to the Gregorian calendar (+13 days, valid 1900-2099)."""
    a, b, c = year % 4, year % 7, year % 19
    d = (19 * c + 15) % 30
    e = (2 * a + 4 * b - d + 34) % 7
    month = (d + e + 114) // 31
    day = (d + e + 114) % 31 + 1
    return dt.date(year, month, day) + dt.timedelta(days=13)


def ro_holidays(year: int) -> set[dt.date]:
    easter = orthodox_easter(year)
    fixed = [(1, 1), (1, 2), (1, 24), (5, 1), (6, 1), (8, 15), (11, 30), (12, 1), (12, 25), (12, 26)]
    if year >= 2024:  # Bobotează and Sf. Ioan became public holidays in 2024
        fixed += [(1, 6), (1, 7)]
    days = {dt.date(year, m, d) for m, d in fixed}
    for off in (-2, 0, 1, 49, 50):  # Good Friday, Easter Sunday + Monday, Rusalii Sunday + Monday
        days.add(easter + dt.timedelta(days=off))
    return days


def utc_index(year: int) -> pd.DatetimeIndex:
    """15-min UTC timestamps covering one local (Europe/Bucharest) calendar year."""
    start = pd.Timestamp(f"{year}-01-01", tz=TZ).tz_convert("UTC")
    end = pd.Timestamp(f"{year + 1}-01-01", tz=TZ).tz_convert("UTC")
    return pd.date_range(start, end, freq="15min", inclusive="left")


def local_features(ts_utc: pd.DatetimeIndex) -> dict[str, np.ndarray]:
    loc = ts_utc.tz_convert(TZ)
    dates = loc.date
    years = sorted({d.year for d in dates})
    hol = set().union(*(ro_holidays(y) for y in years))
    is_hol = np.fromiter((d in hol for d in dates), bool, len(dates))
    dow = loc.dayofweek.to_numpy()
    return {
        "hour": (loc.hour + loc.minute / 60).to_numpy(np.float32),   # local clock hour, 0..24
        "qh": (loc.hour * 4 + loc.minute // 15).to_numpy(),          # local quarter-hour index 0..95
        "doy": loc.dayofyear.to_numpy(),
        "month": loc.month.to_numpy(),
        "day_index": (loc.normalize() - loc[0].normalize()).days.to_numpy(),  # local day counter
        "off_day": (dow >= 5) | is_hol,                               # weekend or public holiday
        "holiday": is_hol,
    }
