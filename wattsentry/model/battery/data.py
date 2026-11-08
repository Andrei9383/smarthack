"""data-v2 access for the battery model (card #174): dense per-chunk arrays from I2's parquet files (#172).

Only TRAINING files are read for features (homes_public, meter_observed, meter_monthly, prices, weather,
standard_profile). The hidden files are read in two places only, both clearly named: ``labels`` (strategy and
battery parameters = classifier / regressor TARGETS) and ``truth_chunk`` (per-home import/export to SCORE
predictions, never a feature). tests/test_battery_model.py checks that no feature name is a hidden column.

Time axis: every array is (homes, T) on the dataset's full UTC 15-min axis (``World.ts``, local years
2022-2025 by default); ``World.cal`` holds local calendar fields and ``World.year`` the local year of each step.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Sequence

import numpy as np
import pandas as pd

TZ = "Europe/Bucharest"
DT_H = 0.25
STEPS_PER_DAY = 96
LABEL_COLUMNS = ("strategy", "has_battery", "cap_kwh", "p_kw", "reserve_frac", "k_top", "eta")


def resolve_data(out: str | Path | None = None, quick: bool = False, **cfg) -> Path:
    """Path of a generated data-v2 directory: ``out`` if given, else I2's cached generate() (regenerates where
    missing, ~3 min for 5000 homes on the GPU-host worker)."""
    if out:
        return Path(out)
    from wattsentry.sim.prosumers_v2 import QUICK, ProsumerConfig, generate

    kw = {**(QUICK if quick else {}), **cfg}
    return Path(generate(ProsumerConfig(**kw)))


@dataclass
class World:
    """Everything per step that is common to all homes (public)."""

    out: Path
    ts: pd.DatetimeIndex                 # UTC interval starts (T,)
    cal: dict[str, np.ndarray]           # qh, hour, dow, month, doy, day_index, off_day (local)
    year: np.ndarray                     # local year per step (T,)
    pzu: np.ndarray                      # PZU lei/MWh (T,)
    price: np.ndarray                    # MaxControl customer price lei/MWh (T,)
    rank: np.ndarray                     # rank of `price` in its local day, 0 = most expensive (T,)
    med: np.ndarray                      # the local day's median of `price` (T,)
    load_coef: np.ndarray                # standard (settlement) load profile shape (T,)
    homes: pd.DataFrame                  # homes_public, in home_id order
    weather: pd.DataFrame = field(repr=False)  # locality x 15 min
    n_days: int = 0
    day_start: np.ndarray = field(default=None, repr=False)  # first step of each local day (n_days,)
    day_month: np.ndarray = field(default=None, repr=False)  # month index (0-based from the first month) per day
    month_keys: list = field(default_factory=list)            # 'YYYY-MM' per month index

    @property
    def T(self) -> int:
        return len(self.ts)

    def steps_of_years(self, years: Sequence[int]) -> np.ndarray:
        return np.flatnonzero(np.isin(self.year, list(years)))

    def days_of_years(self, years: Sequence[int]) -> np.ndarray:
        return np.flatnonzero(np.isin(self.year[self.day_start], list(years)))


def load_world(out: str | Path) -> World:
    from wattsentry.sim.fleet.ro_calendar import local_features

    try:
        from wattsentry.kernels.strategies import day_rank_median
    except ImportError:  # pragma: no cover - I2 not on this branch yet
        raise
    out = Path(out)
    pr = pd.read_parquet(out / "prices.parquet").sort_values("ts").reset_index(drop=True)
    ts = pd.DatetimeIndex(pr["ts"]).tz_convert("UTC")
    cal = local_features(ts)
    loc = ts.tz_convert(TZ)
    cal["dow"] = loc.day_of_week.to_numpy()
    day = cal["day_index"].astype(np.int64)
    day = day - day[0]
    cal["day_index"] = day
    price = pr["customer_price_lei_mwh"].to_numpy(np.float64)
    rank, med = day_rank_median(price, day)
    sp = pd.read_parquet(out / "standard_profile.parquet").sort_values("ts")
    sp_ts = pd.DatetimeIndex(sp["ts"]).tz_convert("UTC").as_unit("ns")
    load_coef = pd.Series(sp["load_coef"].to_numpy(np.float64), index=sp_ts).reindex(ts.as_unit("ns")).ffill().bfill()
    homes = pd.read_parquet(out / "homes_public.parquet").sort_values("home_id").reset_index(drop=True)
    for c in ("home_id", "locality_id", "county", "contract", "meter", "split"):
        homes[c] = homes[c].astype(str)
    wx = pd.read_parquet(out / "weather.parquet")
    wx["point_id"] = wx["point_id"].astype(str)
    n_days = int(day[-1]) + 1
    day_start = np.searchsorted(day, np.arange(n_days))
    ym = loc.year.to_numpy() * 12 + loc.month.to_numpy() - 1
    ym0 = int(ym[0])
    day_month = (ym[day_start] - ym0).astype(np.int64)
    month_keys = [f"{(ym0 + k) // 12:04d}-{(ym0 + k) % 12 + 1:02d}" for k in range(int(day_month[-1]) + 1)]
    return World(out=out, ts=ts, cal=cal, year=loc.year.to_numpy(), pzu=pr["pzu_lei_mwh"].to_numpy(np.float64),
                 price=price, rank=rank.astype(np.int32), med=med, load_coef=load_coef.to_numpy(np.float64),
                 homes=homes, weather=wx, n_days=n_days, day_start=day_start, day_month=day_month,
                 month_keys=month_keys)


def _step_index(world: World, ts) -> np.ndarray:
    t0 = int(world.ts[0].value)
    return ((pd.DatetimeIndex(ts).as_unit("ns").asi8 - t0) // (900 * 10**9)).astype(np.int64)


def _dense(world: World, df: pd.DataFrame, home_ids: Sequence[str], cols: Sequence[str]) -> list[np.ndarray]:
    pos = {h: i for i, h in enumerate(home_ids)}
    rows = df["home_id"].astype(str).map(pos).to_numpy()
    t = _step_index(world, df["ts"])
    ok = (t >= 0) & (t < world.T) & ~pd.isna(rows)
    rows, t = rows[ok].astype(np.int64), t[ok]
    outs = []
    for c in cols:
        a = np.full((len(home_ids), world.T), np.nan, np.float32)
        a[rows, t] = df[c].to_numpy(np.float32)[ok]
        outs.append(a)
    return outs


def _read_parts(out: Path, sub: str, home_ids: Sequence[str], columns: Sequence[str]) -> pd.DataFrame:
    files = sorted((out / sub).glob("year=*/part-*.parquet"))
    filt = [("home_id", "in", list(home_ids))]
    dfs = [pd.read_parquet(p, filters=filt, columns=list(columns)) for p in files]
    dfs = [d for d in dfs if len(d)]
    return pd.concat(dfs, ignore_index=True) if dfs else pd.DataFrame(columns=list(columns))


def meter_chunk(world: World, home_ids: Sequence[str]) -> tuple[np.ndarray, np.ndarray]:
    """Smart-meter 15-min import / export kWh (n, T) float32; NaN = profile meter or reading never arrived.
    PPC receives day D at local D+1 00:00 (``available_at``): callers lag by >= 2 days for a day-ahead forecast."""
    df = _read_parts(world.out, "meter_observed", home_ids, ["ts", "home_id", "import_kwh", "export_kwh"])
    return tuple(_dense(world, df, home_ids, ["import_kwh", "export_kwh"]))  # type: ignore[return-value]


def monthly_chunk(world: World, home_ids: Sequence[str]) -> tuple[np.ndarray, np.ndarray]:
    """Monthly import / export kWh (n, n_months) for every home (smart + profile), NaN where missing.
    Month m is known from the start of month m+1."""
    mm = pd.read_parquet(world.out / "meter_monthly.parquet")
    mm = mm[mm["home_id"].astype(str).isin(set(home_ids))]
    pos = {h: i for i, h in enumerate(home_ids)}
    mpos = {k: i for i, k in enumerate(world.month_keys)}
    r = mm["home_id"].astype(str).map(pos).to_numpy()
    c = mm["month"].astype(str).map(mpos)
    ok = c.notna().to_numpy()
    imp = np.full((len(home_ids), len(world.month_keys)), np.nan)
    exp = imp.copy()
    imp[r[ok].astype(int), c[ok].astype(int)] = mm["import_kwh"].to_numpy(float)[ok]
    exp[r[ok].astype(int), c[ok].astype(int)] = mm["export_kwh"].to_numpy(float)[ok]
    return imp, exp


# ---------------------------------------------------------------------------------------- hidden: targets / scoring
def labels(out: str | Path, home_ids: Sequence[str] | None = None) -> pd.DataFrame:
    """HIDDEN. Strategy + battery parameters per home: classifier / regressor TARGETS only, never features."""
    h = pd.read_parquet(Path(out) / "homes.parquet", columns=["home_id", *LABEL_COLUMNS])
    h["home_id"] = h["home_id"].astype(str)
    h["strategy"] = h["strategy"].astype(str)
    h = h.set_index("home_id")
    return h.loc[list(home_ids)].reset_index() if home_ids is not None else h.reset_index()


def truth_chunk(world: World, home_ids: Sequence[str], years: Sequence[int]) -> tuple[np.ndarray, np.ndarray]:
    """HIDDEN. True per-home import / export kWh (n, T) for ``years`` (NaN elsewhere): scoring only (it is also
    what profile-metered homes physically did, which PPC only sees as monthly totals)."""
    files = [p for y in years for p in sorted((world.out / "truth/home_15min" / f"year={y}").glob("part-*.parquet"))]
    filt = [("home_id", "in", list(home_ids))]
    df = pd.concat([pd.read_parquet(p, filters=filt, columns=["ts", "home_id", "import_kwh", "export_kwh"])
                    for p in files], ignore_index=True)
    return tuple(_dense(world, df, home_ids, ["import_kwh", "export_kwh"]))  # type: ignore[return-value]


def chunks(home_ids: Sequence[str], size: int) -> list[list[str]]:
    ids = list(home_ids)
    return [ids[i:i + size] for i in range(0, len(ids), size)]
