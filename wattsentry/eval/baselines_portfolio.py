"""Portfolio baselines for the imbalance replay: what a supplier buys day-ahead today.

- ``profile``: DEER standard load profile x expected monthly consumption, minus a typical monthly solar curve x kWp
  (no weather, no interval meters).
- ``persistence``: the smart meter one week earlier (D-14, then the profile, when missing).
- ``forecast_only``: validated PV and load q50 per home, load - pv, no battery model (persistence where unknown).

``positions`` returns one row per 15-min step of the local test year with ``POSITION_COLUMNS``: ts (UTC),
policy, bought_da_mwh, traded_id_mwh (0 here), actual_mwh = import_mwh - export_mwh. MWh per step, + = net purchase;
import/export are summed per home first, so export_mwh is the gross injection. ``actual`` is the fault-injected
meter flow with gaps filled from the clean fleet (a logger fault does not change the physical flow).
"""

from __future__ import annotations

import json
import os
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable, Optional, Sequence

import numpy as np
import pandas as pd
import polars as pl

REPO_ROOT = Path(__file__).resolve().parents[2]
PROFILE_PATH = REPO_ROOT / "data" / "raw" / "standard_profile.parquet"
TZ = "Europe/Bucharest"
STEP_H = 0.25
H = 96
WEEK = 7 * H
POLICIES = ("profile", "persistence", "forecast_only")
POSITION_COLUMNS = {
    "ts": pl.Datetime("ns", "UTC"), "policy": pl.String, "bought_da_mwh": pl.Float64, "traded_id_mwh": pl.Float64,
    "actual_mwh": pl.Float64, "import_mwh": pl.Float64, "export_mwh": pl.Float64,
}
# Localities with town status (oraș); every other fleet locality is a commune -> DEER rural profile.
URBAN_LOCALITIES = frozenset({"voluntari", "otopeni", "bragadiru", "popesti_leordeni", "ghimbav", "ovidiu", "cisnadie"})
# Standard orientation for the typical solar curve: PPC knows kWp from the connection file, not tilt/azimuth.
STD_TILT, STD_AZIMUTH = 30.0, 180.0
LABEL = "synthetic RO homes, real RO weather (ERA5) + real RO standard profile (DEER PSC 2024)"


# ============================================================================================ time axis
def year_index(year: int) -> pd.DatetimeIndex:
    """15-min UTC step starts of one Europe/Bucharest calendar year (35040 or 35136 steps)."""
    start = pd.Timestamp(f"{year}-01-01", tz=TZ).tz_convert("UTC")
    end = pd.Timestamp(f"{year + 1}-01-01", tz=TZ).tz_convert("UTC")
    return pd.date_range(start, end, freq="15min", inclusive="left")


def _local(ts: pd.DatetimeIndex) -> dict[str, np.ndarray]:
    from wattsentry.sim.fleet.ro_calendar import ro_holidays

    loc = ts.tz_convert(TZ)
    dates = loc.date
    hol = set().union(*(ro_holidays(y) for y in sorted({d.year for d in dates})))
    is_hol = np.fromiter((d in hol for d in dates), bool, len(dates))
    workday = (loc.day_of_week.to_numpy() < 5) & ~is_hol
    return {"month": loc.month.to_numpy(), "slot": (loc.hour * 4 + loc.minute // 15).to_numpy(),
            "hour": loc.hour.to_numpy(), "workday": workday,
            "ym": (loc.year.to_numpy() * 12 + loc.month.to_numpy() - 1), "date": np.asarray(dates)}


def _utc_slot(ts: pd.DatetimeIndex) -> np.ndarray:
    return (ts.hour * 4 + ts.minute // 15).to_numpy()


# ============================================================================================ standard profile
def standard_profile(path: Path | str = PROFILE_PATH) -> pl.DataFrame:
    """DEER PSC weights: zone (urban|rural), season (warm|cold), day_type (workday|nonworkday), interval 1..96
    (local settlement interval, 1 = 00:00-00:15), weight (each 96-interval day sums to 1), r_workday."""
    return pl.read_parquet(path)


def psc_step_weights(ts: pd.DatetimeIndex, zone: str, prof: Optional[pl.DataFrame] = None) -> np.ndarray:
    """Un-normalised weight of every step under the DEER formulas (2)-(3): p_day_type(interval) x (r on workdays,
    1 on non-workdays). Dividing by the sum over a month gives the share of that month's energy in each step,
    which equals W_zl = r W / (r N_zl + N_znl) per workday and W_znl = W / (r N_zl + N_znl) per non-workday."""
    prof = standard_profile() if prof is None else prof
    p = prof.filter(pl.col("zone") == zone)
    if p.height != 4 * H:
        raise ValueError(f"standard profile has no complete '{zone}' zone")
    tab = {}
    for (season, day), g in p.group_by(["season", "day_type"]):
        g = g.sort("interval")
        tab[(season, day)] = (g["weight"].to_numpy(), float(g["r_workday"][0]))
    L = _local(ts)
    warm = (L["month"] >= 4) & (L["month"] <= 9)
    w = np.empty(len(ts))
    for season, smask in (("warm", warm), ("cold", ~warm)):
        for day, dmask in (("workday", L["workday"]), ("nonworkday", ~L["workday"])):
            m = smask & dmask
            pw, r = tab[(season, day)]
            w[m] = pw[L["slot"][m]] * (r if day == "workday" else 1.0)
    return w


def monthly_shares(ts: pd.DatetimeIndex, w: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """(month index 0..11 per step, share of its month's energy per step). Shares sum to 1 within each local month
    present in ts (DST days included: the 25-h day just gets 100 steps)."""
    L = _local(ts)
    ym = L["ym"]
    tot = np.zeros(ym.max() - ym.min() + 1)
    np.add.at(tot, ym - ym.min(), w)
    return L["month"] - 1, w / tot[ym - ym.min()]


# ============================================================================================ solar curve
def solar_climatology(weather: pl.DataFrame, years: Sequence[int], tilt: float = STD_TILT,
                      azimuth: float = STD_AZIMUTH) -> dict[str, np.ndarray]:
    """Typical monthly solar curve per locality: mean AC kW per kWp by (month 1..12, UTC 15-min slot) -> [12, 96],
    over the given past years of real weather (ERA5 at the locality), standard orientation, inverter = kWp.
    A climatology, not a forecast: the day's weather never enters the profile policy."""
    from wattsentry.model import physics as ph
    from wattsentry.sim.fleet.localities import BY_ID

    out = {}
    wx = weather.filter(pl.col("ts").dt.convert_time_zone(TZ).dt.year().is_in(list(years)))
    for (pid,), g in wx.group_by(["point_id"]):
        g = g.sort("ts")
        ts = pd.DatetimeIndex(g["ts"].to_pandas())
        loc = BY_ID.get(pid)
        lat, lon = (loc.lat, loc.lon) if loc else (45.9, 24.9)
        geom = ph.solar_geometry(ts, lat, lon)
        poa, _ = ph.poa(tilt, azimuth, geom[:, 0], geom[:, 1], g["dni_wm2"].to_numpy().astype(np.float64),
                        g["ghi_wm2"].to_numpy().astype(np.float64), g["dhi_wm2"].to_numpy().astype(np.float64))
        kw = ph.ac_power_kw(poa, g["temp_c"].to_numpy().astype(np.float64), 1.0, 1.0)
        month = ts.tz_convert(TZ).month.to_numpy() - 1
        slot = _utc_slot(ts)
        s = np.zeros((12, H))
        n = np.zeros((12, H))
        np.add.at(s, (month, slot), kw)
        np.add.at(n, (month, slot), 1)
        out[pid] = np.where(n > 0, s / np.maximum(n, 1), 0.0)
    if not out:
        raise ValueError(f"no weather rows in years {list(years)}")
    return out


def solar_kw_per_kwp(ts: pd.DatetimeIndex, clim: np.ndarray) -> np.ndarray:
    """Typical-curve kW per kWp on the ts axis (local month, UTC slot)."""
    return clim[ts.tz_convert(TZ).month.to_numpy() - 1, _utc_slot(ts)]


def typical_monthly_yield(ts: pd.DatetimeIndex, clim: np.ndarray) -> np.ndarray:
    """[12] kWh per kWp in each local month of ts under the typical curve."""
    m = ts.tz_convert(TZ).month.to_numpy() - 1
    y = np.zeros(12)
    np.add.at(y, m, solar_kw_per_kwp(ts, clim) * STEP_H)
    return y


# ============================================================================================ portfolio data
@dataclass
class Portfolio:
    """Everything the baselines may look at, on one test year.

    sites: site_id, locality_id, kwp_recorded, zone (urban|rural), smart (bool).
    ts: UTC step starts of the test year (T). meter_kw [S, W + T]: smart-meter connection flow in kW, + = net import,
      NaN = no D+1 reading (or a profile-metered home); the first W steps are the WEEKS weeks before the year.
    prev_ts / prev_meter_kw [S, T0]: the previous local year (expected consumption for the profile).
    actual_kw [S, T]: physical connection flow (+ import), no NaN. clim: locality -> [12, 96] kW/kWp curve."""

    sites: pl.DataFrame
    year: int
    ts: pd.DatetimeIndex
    meter_kw: np.ndarray
    prev_ts: pd.DatetimeIndex
    prev_meter_kw: np.ndarray
    actual_kw: np.ndarray
    clim: dict[str, np.ndarray]
    label: str = LABEL
    source: dict = field(default_factory=dict)
    lead: int = 2 * WEEK

    @property
    def n_sites(self) -> int:
        return self.sites.height

    def meter_lag(self, steps: int) -> np.ndarray:
        """[S, T] meter value `steps` before each test step (NaN where unknown)."""
        if steps > self.lead:
            raise ValueError(f"lag {steps} > {self.lead} steps of lead-in")
        return self.meter_kw[:, self.lead - steps:self.lead - steps + len(self.ts)]


def _pivot(df: pl.DataFrame, site_ids: list[str], ts: pd.DatetimeIndex, col: str) -> np.ndarray:
    """Long (ts, site_id, col) -> [S, T] float64 with NaN where missing."""
    out = np.full((len(site_ids), len(ts)), np.nan)
    si = {s: i for i, s in enumerate(site_ids)}
    t0 = ts[0].value
    sidx = np.array([si.get(s, -1) for s in df["site_id"].to_list()])
    tidx = (df["ts"].dt.epoch("ns").to_numpy() - t0) // (900 * 10**9)
    v = df[col].to_numpy().astype(np.float64)
    ok = (sidx >= 0) & (tidx >= 0) & (tidx < len(ts))
    out[sidx[ok], tidx[ok]] = v[ok]
    return out


def _read_years(tdir: Path, years: Sequence[int], cols: Sequence[str]) -> pl.DataFrame:
    parts = [pl.read_parquet(tdir / f"year={y}.parquet", columns=list(cols)) for y in years
             if (tdir / f"year={y}.parquet").exists()]
    if not parts:
        raise FileNotFoundError(f"no telemetry for {list(years)} under {tdir}")
    return pl.concat(parts)


def find_fleet_dirs(root: Optional[Path] = None) -> tuple[Path, Optional[Path]]:
    """(clean fleet dir, fault-injected dir or None). $WATTSENTRY_FLEET / $WATTSENTRY_FAULTS, else the dirs named in
    the window cache index (the data the models were trained on; built if missing), else the newest fleet-*."""
    if root is not None:
        root = Path(root)
        mf = root / "manifest.json"
        if mf.exists() and "fleet_dir" in json.loads(mf.read_text()):  # a faults dir
            return Path(json.loads(mf.read_text())["fleet_dir"]), root
        return root, None
    if os.environ.get("WATTSENTRY_FLEET"):
        f = os.environ.get("WATTSENTRY_FAULTS")
        return Path(os.environ["WATTSENTRY_FLEET"]), Path(f) if f else None
    from wattsentry.model.data import build_lab_windows, find_lab_windows

    w = find_lab_windows() or build_lab_windows()  # fresh worker: fleet + faults + windows, same as training
    if w is not None:
        key = json.loads((w / "index.json").read_text()).get("source_key", {})
        fleet, faults = w.parent / key.get("fleet", "-"), w.parent / key.get("faults", "-")
        if (fleet / "sites.parquet").exists():
            return fleet, faults if (faults / "telemetry").exists() else None
    base = Path(os.environ.get("WATTSENTRY_CACHE", Path.home() / ".cache" / "sentinel"))
    cands = sorted(base.glob("data-v*/fleet-*/manifest.json"), key=lambda p: p.stat().st_mtime)
    if cands:
        return cands[-1].parent, None
    from wattsentry.sim.fleet.config import FleetConfig
    from wattsentry.sim.fleet.generate import generate

    return Path(generate(FleetConfig(), plots=False)), None


def load_fleet(year: int = 2025, root: Optional[Path] = None, site_ids: Optional[Sequence[str]] = None) -> Portfolio:
    """The current fleet as a Portfolio (all homes smart-metered)."""
    fleet, faults = find_fleet_dirs(root)
    sites = pl.read_parquet(fleet / "sites.parquet").sort("site_id")
    if site_ids is not None:
        sites = sites.filter(pl.col("site_id").is_in(list(site_ids)))
    ids = sites["site_id"].to_list()
    sites = sites.select("site_id", "locality_id", "kwp_recorded").with_columns(
        zone=pl.when(pl.col("locality_id").is_in(list(URBAN_LOCALITIES))).then(pl.lit("urban")).otherwise(pl.lit("rural")),
        smart=pl.lit(True))
    ts = year_index(year)
    prev_ts = year_index(year - 1)
    lead = 2 * WEEK
    full = pd.DatetimeIndex(prev_ts[-lead:].append(ts))
    cols = ("ts", "site_id", "grid_w")
    tdir = (faults or fleet) / "telemetry"
    raw = _read_years(tdir, (year - 1, year), cols).filter(pl.col("site_id").is_in(ids))
    meter = -_pivot(raw, ids, full, "grid_w") / 1000.0           # + import, kW; NaN = gap
    prev = -_pivot(raw, ids, prev_ts, "grid_w") / 1000.0
    actual = meter[:, lead:].copy()
    if faults is not None and np.isnan(actual).any():
        clean = _read_years(fleet / "telemetry", (year,), cols).filter(pl.col("site_id").is_in(ids))
        actual = np.where(np.isnan(actual), -_pivot(clean, ids, ts, "grid_w") / 1000.0, actual)
    actual = np.nan_to_num(actual)
    weather = pl.read_parquet(fleet / "weather.parquet", columns=["ts", "point_id", "ghi_wm2", "dni_wm2", "dhi_wm2",
                                                                    "temp_c"])
    clim = solar_climatology(weather, list(range(year - 3, year)))
    return Portfolio(sites=sites, year=year, ts=ts, meter_kw=meter, prev_ts=prev_ts, prev_meter_kw=prev,
                     actual_kw=actual, clim=clim, lead=lead,
                     source={"fleet": str(fleet), "faults": str(faults) if faults else None,
                             "climatology_years": [year - 3, year - 2, year - 1]})


# ============================================================================================ policies (per home)
def expected_monthly_kwh(pf: Portfolio) -> np.ndarray:
    """[S, 12] expected gross consumption per home and month: last year's metered net import in that month plus
    the typical PV yield x kWp (PPC sees the meter, not the panels). Gaps are scaled up from the month's readings;
    a home with no readings at all gets the portfolio median per month."""
    m = pf.prev_ts.tz_convert(TZ).month.to_numpy() - 1
    S = pf.n_sites
    tot = np.zeros((S, 12))
    cnt = np.zeros((S, 12))
    n_all = np.bincount(m, minlength=12).astype(float)
    x = pf.prev_meter_kw
    ok = ~np.isnan(x)
    for k in range(12):
        sel = m == k
        tot[:, k] = np.nansum(x[:, sel], axis=1) * STEP_H
        cnt[:, k] = ok[:, sel].sum(1)
    net = np.where(cnt > 0, tot * n_all / np.maximum(cnt, 1), np.nan)
    kwp = pf.sites["kwp_recorded"].to_numpy().astype(float)
    locs = pf.sites["locality_id"].to_list()
    yld = np.stack([typical_monthly_yield(pf.prev_ts, pf.clim[l]) if l in pf.clim else np.zeros(12) for l in locs])
    e = net + yld * kwp[:, None]
    med = np.nanmedian(np.where(np.isfinite(e), e, np.nan), axis=0)
    e = np.where(np.isfinite(e), e, np.nan_to_num(med))
    return np.clip(e, 0.0, None)


def profile_site_kw(pf: Portfolio, prof: Optional[pl.DataFrame] = None) -> np.ndarray:
    """[S, T] kW, + = import: PSC share x expected monthly consumption - typical solar curve x kWp."""
    prof = standard_profile() if prof is None else prof
    e = expected_monthly_kwh(pf)
    out = np.zeros((pf.n_sites, len(pf.ts)))
    zones = pf.sites["zone"].to_numpy()
    for z in np.unique(zones):
        month, share = monthly_shares(pf.ts, psc_step_weights(pf.ts, str(z), prof))
        sel = zones == z
        out[sel] = e[sel][:, month] * share[None] / STEP_H
    kwp = pf.sites["kwp_recorded"].to_numpy().astype(float)
    locs = np.array(pf.sites["locality_id"].to_list())
    for l in np.unique(locs):
        if l in pf.clim:
            sel = locs == l
            out[sel] -= kwp[sel, None] * solar_kw_per_kwp(pf.ts, pf.clim[l])[None]
    return out


def persistence_site_kw(pf: Portfolio, fallback: Optional[np.ndarray] = None) -> np.ndarray:
    """[S, T] kW: meter one week earlier, else two weeks, else `fallback` (default: the home's profile)."""
    x = pf.meter_lag(WEEK).copy()
    smart = pf.sites["smart"].to_numpy()
    x[~smart] = np.nan
    x = np.where(np.isnan(x), pf.meter_lag(2 * WEEK), x)
    x[~smart] = np.nan
    if np.isnan(x).any():
        fb = profile_site_kw(pf) if fallback is None else fallback
        x = np.where(np.isnan(x), fb, x)
    return x


# --- forecast_only: PV + load forecasts through the Judge adapter ---------------------------------------------------------
def validated_q50(model: str, site_ids: Sequence[str], days: Sequence[str], q: np.ndarray) -> np.ndarray:
    """THE single seam to the Judge: raw (N, 96, 3) kW -> validated q50 (N, 96), NaN = not known."""
    from wattsentry.judge.adapter import RawBatch, validated_many

    vb = validated_many(model, list(site_ids), list(days), raw=RawBatch(q=np.asarray(q, np.float64)))
    return np.asarray(vb.q50, np.float64)


def model_forecasts_kw(pf: Portfolio, windows: Optional[Path] = None, model_dir: Optional[Path] = None,
                       batch: int = 512, log: Callable = print) -> dict[str, np.ndarray]:
    """Raw day-ahead PV (pvnet_fc, forecast weather) and load (loadnet seeds) quantiles [S, T, 3] kW on pf.ts,
    computed over whole UTC days of the window cache exactly like the replay's ModelBank. NaN = no output."""
    from types import SimpleNamespace

    from wattsentry.agent.predictors import FleetSlice, ModelBank, _members, models_dir
    from wattsentry.agent.world import find_windows

    root = Path(windows) if windows else find_windows()
    if root is None:  # fresh worker: build the window cache (fleet + faults + windows, several minutes)
        from wattsentry.model.data import build_lab_windows

        root = build_lab_windows()
    if root is None:
        raise FileNotFoundError("no window cache ($WATTSENTRY_WINDOWS or ~/.cache/wattsentry/data-v{N}/windows)")
    idx = json.loads((root / "index.json").read_text())
    cache_sites = [s["site_id"] for s in json.loads((root / "sites.json").read_text())]
    pos = {s: i for i, s in enumerate(cache_sites)}
    want = pf.sites["site_id"].to_list()
    site_idx = [pos[s] for s in want if s in pos]
    rows = np.array([i for i, s in enumerate(want) if s in pos])
    t_start = int(pd.Timestamp(idx["t_start"]).timestamp())
    t_end = t_start + int(idx["n_steps"]) * 900
    day = 86400
    t_from = max(t_start, (int(pf.ts[0].timestamp()) // day - 16) * day)
    t_to = min(t_end, -(-int(pf.ts[-1].timestamp() + 900) // day) * day)
    src = FleetSlice(root, site_idx, t_from, t_to)
    md = Path(model_dir or models_dir())
    out = {}
    for key, model, sub, kind in (("pv", "pvnet", "pvnet_fc", "pv"), ("load", "loadnet", "", "load")):
        t0 = time.perf_counter()
        members = _members(model, md / sub if sub else md)
        ns = SimpleNamespace(src=src, batch=batch, _valid_days=np.arange(15, src.n_days))
        q, _, _ = ModelBank._forecast(ns, members, kind, "forecast")
        full = np.full((pf.n_sites, len(pf.ts), 3), np.nan)
        off = (int(pf.ts[0].timestamp()) - t_from) // 900
        a, b = max(0, off), min(src.n_steps, off + len(pf.ts))
        full[rows, a - off:b - off] = q[:, a:b]
        out[key] = full
        log(f"[forecast_only] {model}: {len(members)} member(s), {len(site_idx)} sites x {src.n_days} days "
            f"in {time.perf_counter() - t0:.1f}s")
    return out


def _local_days(ts: pd.DatetimeIndex) -> tuple[list[str], list[np.ndarray]]:
    loc = ts.tz_convert(TZ)
    keys = loc.strftime("%Y-%m-%d")
    days, idx = np.unique(keys, return_inverse=True)
    return list(days), [np.flatnonzero(idx == k) for k in range(len(days))]


def forecast_only_site_kw(pf: Portfolio, raw: Optional[dict[str, np.ndarray]] = None,
                          fallback: Optional[np.ndarray] = None, **kw) -> tuple[np.ndarray, dict]:
    """[S, T] kW = validated load q50 - validated PV q50 per home (no battery model); fallback where unknown."""
    raw = model_forecasts_kw(pf, **kw) if raw is None else raw
    ids = pf.sites["site_id"].to_list()
    days, steps = _local_days(pf.ts)
    S = pf.n_sites
    q50 = {}
    for key, model in (("pv", "pvnet"), ("load", "loadnet")):
        qa = raw[key]
        out = np.full((S, len(pf.ts)), np.nan)
        # one adapter call per model: rows = (site, local day), 96 steps from local midnight
        N = S * len(days)
        qq = np.full((N, H, 3), np.nan)
        for d, st in enumerate(steps):
            n = min(H, len(st))
            qq[d::len(days), :n] = qa[:, st[:n]]
        v = validated_q50(model, [s for s in ids for _ in days], days * S, qq)
        for d, st in enumerate(steps):
            n = min(H, len(st))
            out[:, st[:n]] = v[d::len(days), :n]
            if len(st) > H:  # 25-h DST day: the extra hour keeps the raw q50
                out[:, st[H:]] = qa[:, st[H:], 1]
        q50[key] = out
    net = q50["load"] - q50["pv"]
    smart_or_any = np.isfinite(net)
    fb = persistence_site_kw(pf) if fallback is None else fallback
    info = {"model_coverage": float(smart_or_any.mean())}
    print(f"LAB_METRIC forecast_only_model_coverage={info['model_coverage']:.4f}", flush=True)
    return np.where(smart_or_any, net, fb), info


# ============================================================================================ portfolio positions
def _frame(pf: Portfolio, policy: str, site_kw: np.ndarray) -> pl.DataFrame:
    a = pf.actual_kw
    imp = np.clip(a, 0, None).sum(0) * STEP_H / 1000.0
    exp = np.clip(-a, 0, None).sum(0) * STEP_H / 1000.0
    return pl.DataFrame({
        "ts": pl.Series(pf.ts.tz_convert("UTC").as_unit("ns").tz_localize(None)).dt.replace_time_zone("UTC"),
        "policy": [policy] * len(pf.ts),
        "bought_da_mwh": site_kw.sum(0) * STEP_H / 1000.0,
        "traded_id_mwh": np.zeros(len(pf.ts)),
        "actual_mwh": imp - exp, "import_mwh": imp, "export_mwh": exp,
    }, schema=POSITION_COLUMNS)


def profile_position(pf: Portfolio) -> pl.DataFrame:
    return _frame(pf, "profile", profile_site_kw(pf))


def persistence_position(pf: Portfolio) -> pl.DataFrame:
    return _frame(pf, "persistence", persistence_site_kw(pf))


def forecast_only_position(pf: Portfolio, raw: Optional[dict[str, np.ndarray]] = None, **kw) -> pl.DataFrame:
    return _frame(pf, "forecast_only", forecast_only_site_kw(pf, raw, **kw)[0])


def positions(policy: str, year: int = 2025, *, fleet: Optional[Portfolio] = None, **kw) -> pl.DataFrame:
    """Position frame (see module doc) for one policy on the test year. `fleet` defaults to load_fleet(year)."""
    pf = fleet if fleet is not None else load_fleet(year)
    fn = {"profile": profile_position, "persistence": persistence_position,
          "forecast_only": forecast_only_position}.get(policy)
    if fn is None:
        raise ValueError(f"unknown policy {policy!r}; expected one of {POLICIES}")
    return fn(pf, **kw) if policy == "forecast_only" else fn(pf)


# ============================================================================================ metrics + CLI
