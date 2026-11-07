"""Features for the battery model (card #174, spec I step 5): expected PV / load, per-home strategy features over the
last 28 days of meter data, and per-step rows for the quantile LightGBM.

Day-ahead information rule: the forecast for local day D is issued on D-1 (before the PZU gate). Smart-meter data
of day X arrives at local X+1 00:00, so the newest complete meter day is D-2; monthly totals of month M arrive at
the start of M+1. Every feature for day D uses only meter days <= D-2 and months that ended before D-1.

M1 / M2 on data-v2: PPC sees no PV or household-load telemetry for these prosumers (only the meter), so the fleet's
PVNet / LoadNet (which need the site's own past PV / load) cannot run. ``expected_pv_load`` is THE seam that gives
the battery model its "M1 P50" and "M2 P50": PV = M1's physics prior (``sentinel.model.physics``: isotropic POA +
the PVWatts-style inverter on the RECORDED kWp / tilt / azimuth with the locality's weather), load = the standard
(settlement) load profile shape scaled per home so it reproduces the home's previous-month metered import.
Labels: proxy (physics / profile stand-ins for M1 / M2), on synthetic RO homes with real RO ERA5 weather.
Swap the body for C1-adapter output when a data-v2 M1 / M2 exists.
"""

from __future__ import annotations

from typing import Sequence

import numpy as np
import pandas as pd

from .data import DT_H, World

LOOKBACK_DAYS = 28
LAG_DAYS = 2              # newest meter day available for target day D is D-2
TOP_PRICE_RANK = 12       # "expensive" quarter-hours of the day (3 h)
NOSUN_FRAC = 0.02         # expected PV below 2 % of kWp counts as "no sun"

# Per-home strategy features (classifier input). Names never collide with hidden columns (tested).
SPEC_FEATURES = ["corr_exp_price", "share_exp_18_21", "mid_exp_vs_surplus", "eve_imp_vs_load", "peak_gap_h"]
EXTRA_FEATURES = ["share_exp_top_price", "exp_nosun_per_kwp", "exp_ratio", "imp_ratio", "night_imp_vs_load",
                  "meter_coverage", "m_exp_ratio", "m_imp_ratio"]
PUBLIC_FEATURES = ["kwp", "contract_maxcontrol", "battery_declared", "enrolled", "smart"]
CLS_FEATURES = SPEC_FEATURES + EXTRA_FEATURES + PUBLIC_FEATURES

# Per-step row features (quantile LightGBM, spec 5a) + a few per-home behaviour features from past meters.
ROW_FEATURES = ["pv_exp", "load_exp", "net_fo", "pzu", "price", "day_rank", "qh", "dow", "month", "off_day",
                "kwp", "contract_maxcontrol", "battery_declared", "enrolled",
                "lag2_net", "lag7_net", "lag2_exp", "lag7_exp",
                "share_exp_top_price", "exp_nosun_per_kwp", "share_exp_18_21", "exp_ratio", "m_exp_ratio"]
METER_ROW_FEATURES = ["lag2_net", "lag7_net", "lag2_exp", "lag7_exp", "share_exp_top_price", "exp_nosun_per_kwp",
                      "share_exp_18_21", "exp_ratio"]


# =============================================================================================== M1 / M2 stand-ins
def _locality_arrays(world: World) -> dict[str, dict[str, np.ndarray]]:
    """Per locality: weather aligned to world.ts and solar geometry (zenith, azimuth) at the step midpoint."""
    cache = getattr(world, "_loc_cache", None)
    if cache is not None:
        return cache
    from sentinel.model.physics import solar_geometry

    cache = {}
    wx = world.weather
    ts = world.ts.as_unit("ns")
    for loc, g in world.homes.groupby("locality_id"):
        w = wx[wx["point_id"] == loc].copy()
        w.index = pd.DatetimeIndex(w["ts"]).tz_convert("UTC").as_unit("ns")
        w = w[~w.index.duplicated()].reindex(ts).ffill().bfill()
        geom = solar_geometry(world.ts, float(g["lat"].mean()), float(g["lon"].mean()))
        cache[loc] = {"ghi": w["ghi_wm2"].to_numpy(np.float32), "dni": w["dni_wm2"].to_numpy(np.float32),
                      "dhi": w["dhi_wm2"].to_numpy(np.float32), "temp": w["temp_c"].to_numpy(np.float32),
                      "wind": w["wind10_ms"].to_numpy(np.float32),
                      "zen": geom[:, 0].astype(np.float32), "saz": geom[:, 1].astype(np.float32)}
    world._loc_cache = cache  # type: ignore[attr-defined]
    return cache


def expected_pv_kw(world: World, homes: pd.DataFrame, batch: int = 64) -> np.ndarray:
    """(n, T) kW: physics PV on recorded metadata (isotropic POA, same formulas as pvlib's 'isotropic' model and
    sentinel.model.physics.ac_power_kw)."""
    from sentinel.model.physics import ac_power_kw

    locs = _locality_arrays(world)
    out = np.zeros((len(homes), world.T), np.float32)
    for loc, idx in homes.groupby("locality_id").indices.items():
        a = locs[loc]
        zr = np.deg2rad(a["zen"])[None]
        up = (a["zen"] < 90)[None]
        for s in range(0, len(idx), batch):
            ii = idx[s:s + batch]
            h = homes.iloc[ii]
            tilt = np.deg2rad(h["tilt_recorded"].to_numpy(np.float32))[:, None]
            az = h["azimuth_recorded"].to_numpy(np.float32)[:, None]
            cos_aoi = np.cos(zr) * np.cos(tilt) + np.sin(zr) * np.sin(tilt) * np.cos(np.deg2rad(a["saz"][None] - az))
            poa = (a["dni"][None] * np.clip(cos_aoi, 0, None) + a["dhi"][None] * (1 + np.cos(tilt)) / 2
                   + a["ghi"][None] * 0.2 * (1 - np.cos(tilt)) / 2)
            poa = np.where(up, poa, 0.0).astype(np.float32)
            out[ii] = ac_power_kw(poa, a["temp"][None], h["kwp"].to_numpy(np.float32)[:, None],
                                  h["inverter_kw"].to_numpy(np.float32)[:, None], wind=a["wind"][None])
    return out


def _month_steps(world: World) -> list[np.ndarray]:
    sm = world.day_month[world.cal["day_index"]]
    bounds = np.flatnonzero(np.diff(sm)) + 1
    return np.split(np.arange(world.T), bounds)


def load_scale(world: World, pv: np.ndarray, imp_month: np.ndarray, iters: int = 10) -> np.ndarray:
    """c (n, n_months) such that sum_t max(c w_t - pv_t, 0) dt = metered import of that month (Newton from above;
    f is convex and increasing in c, so it converges monotonically)."""
    w = world.load_coef
    steps = _month_steps(world)
    c = np.full(imp_month.shape, np.nan)
    for m, st in enumerate(steps):
        I = imp_month[:, m]
        ok = np.isfinite(I) & (I > 0)
        if not ok.any():
            continue
        wm = w[st].astype(np.float64)
        p = pv[:, st].astype(np.float64)
        cm = (I + p.sum(1) * DT_H) / max(wm.sum() * DT_H, 1e-9)
        for _ in range(iters):
            x = cm[:, None] * wm[None] - p
            f = np.clip(x, 0, None).sum(1) * DT_H - I
            fp = ((x > 0) * wm[None]).sum(1) * DT_H
            cm = cm - f / np.maximum(fp, 1e-9)
        c[:, m] = np.where(ok, np.clip(cm, 0, None), np.nan)
    return c


def expected_load_kw(world: World, pv: np.ndarray, imp_month: np.ndarray) -> np.ndarray:
    """(n, T) kW: profile shape x the scale fitted on the PREVIOUS month (known at the start of this month)."""
    c = load_scale(world, pv, imp_month)
    c = pd.DataFrame(c.T).ffill().bfill().to_numpy().T              # missing months: last known scale
    med = np.nanmedian(c) if np.isfinite(c).any() else 0.0
    c = np.where(np.isfinite(c), c, med)
    prev = np.concatenate([c[:, :1], c[:, :-1]], axis=1)          # month m uses m-1 (first month: its own)
    sm = world.day_month[world.cal["day_index"]]
    return (prev[:, sm] * world.load_coef[None]).astype(np.float32)


def expected_pv_load(world: World, homes: pd.DataFrame, imp_month: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """THE M1 / M2 seam: day-ahead P50 PV and household load per home, kW (n, T). See the module docstring."""
    pv = expected_pv_kw(world, homes)
    return pv, expected_load_kw(world, pv, imp_month)


# =============================================================================================== strategy features
def _reduce_days(world: World, x: np.ndarray) -> np.ndarray:
    return np.add.reduceat(x, world.day_start, axis=1)


def daily_stats(world: World, imp: np.ndarray, exp: np.ndarray, pv: np.ndarray, load: np.ndarray,
                kwp: np.ndarray, batch: int = 64) -> dict[str, np.ndarray]:
    """Per home and local day (n, D) sums over VALID meter steps (both import and export present)."""
    parts = [_daily_stats(world, imp[s:s + batch], exp[s:s + batch], pv[s:s + batch], load[s:s + batch],
                          kwp[s:s + batch]) for s in range(0, len(imp), batch)]
    return {k: np.concatenate([p[k] for p in parts], axis=0) for k in parts[0]}


def _daily_stats(world: World, imp: np.ndarray, exp: np.ndarray, pv: np.ndarray, load: np.ndarray,
                 kwp: np.ndarray) -> dict[str, np.ndarray]:
    hour = world.cal["hour"]
    valid = np.isfinite(imp) & np.isfinite(exp)
    i0 = np.where(valid, imp, 0).astype(np.float64)
    e0 = np.where(valid, exp, 0).astype(np.float64)
    v = valid.astype(np.float64)
    p = (world.price / 1000.0)[None]                              # lei/kWh
    eve = ((hour >= 18) & (hour < 21))[None]
    mid = ((hour >= 11) & (hour < 15))[None]
    eimp = ((hour >= 18) & (hour < 23))[None]
    night = (hour < 6)[None]
    top = (world.rank < TOP_PRICE_RANK)[None]
    nosun = pv < NOSUN_FRAC * kwp[:, None]
    surplus = np.clip(pv - load, 0, None) * DT_H * v
    ld = load * DT_H * v
    R = lambda a: _reduce_days(world, a)  # noqa: E731
    d = {"n": R(v), "e": R(e0), "i": R(i0), "e2": R(e0 * e0), "p": R(p * v), "p2": R(p * p * v), "ep": R(e0 * p),
         "e_eve": R(e0 * eve), "e_mid": R(e0 * mid), "sur_mid": R(surplus * mid), "i_eve": R(i0 * eimp),
         "l_eve": R(ld * eimp), "i_night": R(i0 * night), "l_night": R(ld * night), "e_top": R(e0 * top),
         "e_nosun": R(e0 * nosun), "pv": R(pv * DT_H * v), "ld": R(ld)}
    # hour of peak export vs hour of peak price, per day with export
    D = world.n_days
    gap = np.full((len(imp), D), np.nan)
    bounds = np.r_[world.day_start, world.T]
    for k in range(D):
        a, b = bounds[k], bounds[k + 1]
        ek = e0[:, a:b]
        has = ek.max(1) > 0.01
        if has.any():
            gap[has, k] = np.abs(ek[has].argmax(1) - int(np.argmax(world.price[a:b]))) / 4.0
    d["gap"] = np.nan_to_num(gap)
    d["gap_n"] = np.isfinite(gap).astype(np.float64)
    # monthly expected PV / load (for the monthly ratios, all homes)
    d["pv_all"] = R(pv * DT_H)
    d["ld_all"] = R(load * DT_H)
    return d


def _window(cs: np.ndarray, days: np.ndarray, lag: int = LAG_DAYS, n: int = LOOKBACK_DAYS) -> np.ndarray:
    hi = np.clip(days - lag + 1, 0, cs.shape[1] - 1)
    lo = np.clip(days - lag + 1 - n, 0, cs.shape[1] - 1)
    return cs[:, hi] - cs[:, lo]


def _ratio(a, b, eps=1e-6):
    with np.errstate(divide="ignore", invalid="ignore"):
        return np.where(b > eps, a / np.maximum(b, eps), np.nan)


def strategy_features(world: World, daily: dict[str, np.ndarray], homes: pd.DataFrame, days: np.ndarray,
                      imp_month: np.ndarray, exp_month: np.ndarray) -> dict[str, np.ndarray]:
    """Features (n, len(days)) for target days ``days`` from meter days [D-29, D-2] and the last complete month."""
    cs = {k: np.concatenate([np.zeros((v.shape[0], 1)), np.cumsum(v, axis=1)], axis=1) for k, v in daily.items()}
    W = {k: _window(c, days) for k, c in cs.items()}
    n = W["n"]
    cov_n = np.maximum(n, 1)
    me, mp = W["e"] / cov_n, W["p"] / cov_n
    var_e = W["e2"] / cov_n - me ** 2
    var_p = W["p2"] / cov_n - mp ** 2
    cov = W["ep"] / cov_n - me * mp
    with np.errstate(divide="ignore", invalid="ignore"):
        corr = np.where((var_e > 1e-10) & (var_p > 1e-10), cov / np.sqrt(np.maximum(var_e * var_p, 1e-20)), np.nan)
    kwp = homes["kwp"].to_numpy(float)[:, None]
    n_days_obs = n / 96.0
    f = {
        "corr_exp_price": corr,
        "share_exp_18_21": _ratio(W["e_eve"], W["e"]),
        "mid_exp_vs_surplus": _ratio(W["e_mid"], W["sur_mid"]),
        "eve_imp_vs_load": _ratio(W["i_eve"], W["l_eve"]),
        "peak_gap_h": _ratio(W["gap"], W["gap_n"]),
        "share_exp_top_price": _ratio(W["e_top"], W["e"]),
        "exp_nosun_per_kwp": _ratio(W["e_nosun"], n_days_obs * kwp),
        "exp_ratio": _ratio(W["e"], W["pv"]),
        "imp_ratio": _ratio(W["i"], W["ld"]),
        "night_imp_vs_load": _ratio(W["i_night"], W["l_night"]),
        "meter_coverage": n / (LOOKBACK_DAYS * 96.0),
    }
    no_meter = n < 96 * 3  # fewer than 3 days of readings: meter features unknown
    for k in SPEC_FEATURES + EXTRA_FEATURES[:-3]:
        f[k] = np.where(no_meter, np.nan, f[k])
    # last complete month before the issue day (D-1)
    mi = world.day_month[np.clip(days - 1, 0, None)] - 1
    mpv = np.stack([daily["pv_all"][:, world.day_month == m].sum(1) for m in range(len(world.month_keys))], 1)
    mld = np.stack([daily["ld_all"][:, world.day_month == m].sum(1) for m in range(len(world.month_keys))], 1)
    ok = mi >= 0
    mic = np.clip(mi, 0, None)
    f["m_exp_ratio"] = np.where(ok[None], _ratio(exp_month[:, mic], mpv[:, mic]), np.nan)
    f["m_imp_ratio"] = np.where(ok[None], _ratio(imp_month[:, mic], mld[:, mic]), np.nan)
    shape = (len(homes), len(days))
    f["kwp"] = np.broadcast_to(kwp, shape)
    f["contract_maxcontrol"] = np.broadcast_to((homes["contract"] == "maxcontrol").to_numpy(float)[:, None], shape)
    f["battery_declared"] = np.broadcast_to(homes["battery_declared"].to_numpy(float)[:, None], shape)
    f["enrolled"] = np.broadcast_to(homes["enrolled"].to_numpy(float)[:, None], shape)
    f["smart"] = np.broadcast_to((homes["meter"] == "smart").to_numpy(float)[:, None], shape)
    return {k: np.asarray(f[k], np.float32) for k in CLS_FEATURES}


def stack_features(f: dict[str, np.ndarray], names: Sequence[str] = CLS_FEATURES) -> np.ndarray:
    """dict of (n, d) -> (n * d, F) rows in home-major order."""
    return np.stack([np.asarray(f[k], np.float32).reshape(-1) for k in names], axis=1)


# =============================================================================================== quantile rows
def row_features(world: World, homes: pd.DataFrame, rows: np.ndarray, steps: np.ndarray, pv: np.ndarray,
                 load: np.ndarray, imp: np.ndarray, exp: np.ndarray, sfeat_daily: dict[str, np.ndarray],
                 mask_meter: np.ndarray | None = None) -> np.ndarray:
    """(len(rows), len(ROW_FEATURES)) for (home row, step) pairs. ``sfeat_daily`` = strategy_features for ALL days
    (n, D). ``mask_meter`` (len(rows),) bool blanks meter features (trains the profile-meter path)."""
    day = world.cal["day_index"][steps]
    net = imp - exp
    lag = lambda a, k: np.where(steps - k >= 0, a[rows, np.clip(steps - k, 0, None)], np.nan)  # noqa: E731
    cols = {
        "pv_exp": pv[rows, steps], "load_exp": load[rows, steps],
        "pzu": world.pzu[steps], "price": world.price[steps], "day_rank": world.rank[steps],
        "qh": world.cal["qh"][steps], "dow": world.cal["dow"][steps], "month": world.cal["month"][steps],
        "off_day": world.cal["off_day"][steps],
        "kwp": homes["kwp"].to_numpy(float)[rows],
        "contract_maxcontrol": (homes["contract"] == "maxcontrol").to_numpy(float)[rows],
        "battery_declared": homes["battery_declared"].to_numpy(float)[rows],
        "enrolled": homes["enrolled"].to_numpy(float)[rows],
        "lag2_net": lag(net, LAG_DAYS * 96), "lag7_net": lag(net, 7 * 96),
        "lag2_exp": lag(exp, LAG_DAYS * 96), "lag7_exp": lag(exp, 7 * 96),
    }
    cols["net_fo"] = cols["load_exp"] - cols["pv_exp"]
    for k in ("share_exp_top_price", "exp_nosun_per_kwp", "share_exp_18_21", "exp_ratio", "m_exp_ratio"):
        cols[k] = sfeat_daily[k][rows, day]
    X = np.stack([np.asarray(cols[k], np.float32) for k in ROW_FEATURES], axis=1)
    if mask_meter is not None and mask_meter.any():
        for k in METER_ROW_FEATURES:
            X[mask_meter, ROW_FEATURES.index(k)] = np.nan
    return X
