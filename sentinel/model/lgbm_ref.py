"""#15's LightGBM quantile baseline, trained and scored inside a #14 run on the SAME windows, split and mask.

Uses #15's own code (sentinel.eval.baselines.pv.lgbm: features, params, train, predict_bulk) on a duck-typed
FleetArrays built from the WindowSource arrays, so "M1 vs LGBM" differs only in the model:
same sites (#10 split column), same days (#10 splits), same weather (actual for nowcast, forecast for forecast
mode), same pvlib-with-recorded-metadata channel, same 7-day reference. Train: #15's defaults (healthy daylight
steps, 400k sampled rows, 300 rounds) on up to `max_train_sites` train sites over the train days.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

H = 96
SITE_COLS = {"archetype": "base", "heating": "gas", "has_ac": False, "has_ev": False, "has_electric_boiler": False,
             "works_from_home": False}


@dataclass
class _FA:
    sites: pd.DataFrame
    days: pd.DatetimeIndex
    pv_kw: np.ndarray
    load_kw: np.ndarray
    weather: dict
    geom: dict
    pvlib_kw: np.ndarray
    pvlib_cs_kw: np.ndarray
    fault: np.ndarray

    @property
    def n(self) -> int:
        return len(self.sites)


def build_fa(src, sites: np.ndarray, d0: int, d1: int, mode: str = "nowcast") -> _FA:
    """Day-aligned [n, d, 96] float32 arrays for sites x days [d0, d1)."""
    sites = np.asarray(sites)
    n, nd = len(sites), d1 - d0
    sl = slice(d0 * H, d1 * H)

    def site_arr(name):
        return np.asarray(src.array(name)[sites, sl], np.float32).reshape(n, nd, H)

    valid = (np.asarray(src.array("valid")[sites, sl]) > 0).reshape(n, nd, H) if src.has("valid") else None
    pv, load = site_arr("pv_ac"), site_arr("load")
    if valid is not None:
        pv[~valid] = np.nan
        load[~valid] = np.nan
    loc = src.loc_idx()[sites]
    wx = np.asarray(src.array("weather_fc" if mode == "forecast" else "weather_act")[:, sl], np.float32)[loc]
    wx = wx.reshape(n, nd, H, -1)
    g = np.asarray(src.array("geom")[:, sl], np.float32)[loc].reshape(n, nd, H, 3)
    days = src.day_dates()[d0:d1]
    doy = np.broadcast_to(days.day_of_year.to_numpy()[None, :, None], (n, nd, H)).astype(np.float32)
    rows = []
    for i in sites:
        r = src.sites[i]
        d = {k: r.get(k) for k in ("site_id", "kwp_recorded", "tilt_recorded", "azimuth_recorded", "lat", "lon")}
        for k, default in SITE_COLS.items():
            v = r.get(k)
            if v is None and k == "has_ac":
                v = r.get("archetype") == "ac"
            if v is None and k == "has_ev":
                v = r.get("archetype") == "ev"
            d[k] = default if v is None else v
        rows.append(d)
    return _FA(
        sites=pd.DataFrame(rows), days=days, pv_kw=pv, load_kw=load,
        weather={"ghi": wx[..., 0], "dni": wx[..., 1], "dhi": wx[..., 2], "temp_c": wx[..., 3],
                 "wind10_ms": wx[..., 5], "snow_cm": wx[..., 6]},
        geom={"zenith": g[..., 0], "azimuth": g[..., 1], "cs_ghi": g[..., 2], "doy": doy},
        pvlib_kw=site_arr("pvlib_rec_fc" if mode == "forecast" else "pvlib_rec"), pvlib_cs_kw=site_arr("pvclear_rec"),
        fault=np.asarray(src.array("fault")[sites, sl]).reshape(n, nd, H),
    )


def available() -> bool:
    try:
        import lightgbm  # noqa: F401

        from sentinel.eval.baselines.pv import lgbm  # noqa: F401
    except ImportError:
        return False
    return True


def fit(src, splits, kind: str, mode: str = "nowcast", max_train_sites: int = 150, max_rows: int = 400_000,
        rounds: int = 300, seed: int = 0, log=print):
    from sentinel.eval.baselines.pv import lgbm

    rng = np.random.default_rng(seed)
    tr = np.sort(rng.choice(splits.train_sites, min(max_train_sites, len(splits.train_sites)), replace=False))
    days = splits.days["train"]
    d0, d1 = max(0, int(days.min()) - 7), int(days.max()) + 1
    fa = build_fa(src, tr, d0, d1, mode)
    day_mask = np.isin(np.arange(d0, d1), days)
    return lgbm.train(kind, fa, day_mask, np.ones(fa.n, bool), max_rows=max_rows, rounds=rounds, seed=seed, log=log)


def predict(bundle, src, sites: np.ndarray, day_idx: np.ndarray, mode: str = "nowcast", chunk_sites: int = 200):
    """q [N, 96, 3] kW for windows (site, day) (day-aligned, t0 = 00:00 UTC).

    Same features / post-processing as #15's predict_bulk, but only the requested (site, day) rows (and for PV only
    daylight steps) go through the booster, so scoring a sampled eval set doesn't cost the full sites x days grid."""
    from sentinel.eval.baselines.pv import lgbm
    from sentinel.eval.baselines.pv.features import stack

    sites, day_idx = np.asarray(sites), np.asarray(day_idx)
    out = np.zeros((len(sites), H, 3), np.float32)
    d0, d1 = max(0, int(day_idx.min()) - 7), int(day_idx.max()) + 1
    uniq = np.unique(sites)
    for k in range(0, len(uniq), chunk_sites):
        us = uniq[k:k + chunk_sites]
        fa = build_fa(src, us, d0, d1, mode)
        pos = np.searchsorted(us, sites)
        in_chunk = np.isin(sites, us)
        need = np.unique(day_idx[in_chunk]) - d0
        for di in lgbm._chunks(need):
            f = lgbm.pv_bulk_features(fa, di) if bundle.kind == "pv" else lgbm.load_bulk_features(fa, di)
            X = stack(f, bundle.features).reshape(fa.n, len(di), H, -1)
            dpos = {d: j for j, d in enumerate(di)}
            w = np.flatnonzero(in_chunk & np.isin(day_idx - d0, di))
            if not len(w):
                continue
            si, dj = pos[w], np.array([dpos[d] for d in day_idx[w] - d0])
            Xw = X[si, dj]  # [n_w, 96, F]
            if bundle.kind == "pv":
                day = fa.geom["zenith"][si, di[dj]] < lgbm.NIGHT_ZENITH
                kwp = fa.sites.kwp_recorded.to_numpy()[si]
                qd = np.clip(bundle.predict(Xw[day]), 0, None)  # [3, rows]
                q = np.zeros((len(w), H, 3), np.float32)
                q[day] = (qd * np.broadcast_to(kwp[:, None], day.shape)[day][None]).T
            else:
                sc = np.broadcast_to(np.asarray(f["scale"], np.float32), (fa.n, len(di), H))[si, dj]
                qd = np.clip(bundle.predict(Xw.reshape(-1, Xw.shape[-1])), 0, None).reshape(3, len(w), H)
                q = np.transpose(qd * sc[None], (1, 2, 0))
            out[w] = q
    return np.nan_to_num(out)
