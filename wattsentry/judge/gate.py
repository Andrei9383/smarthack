"""Independent data-quality gate: which asset-days may grade anyone.

Physics and the asset's own history only (measured telemetry, recorded metadata, pvlib clear-sky, own past energy);
no function takes a prediction, so a broken witness can never have its misses thrown out as "bad data".
Vectorised over [assets, days, 96]; a day that fails any check is gated (ungraded). Codes are bit masks.
"""

from __future__ import annotations

from typing import Mapping, Optional, Sequence

import numpy as np

PV_RANGE, PV_NIGHT, LOAD_FUSE, FROZEN, UNIT_JUMP, GAP, TURB_RANGE = (1 << i for i in range(7))

PV_MAX_FRAC = 1.2  # x kWp (recorded)
NIGHT_TOL_FRAC = 0.05  # x kWp allowed at night (sensor offset)
NIGHT_MAX_STEPS = 2  # more violating night steps than this -> gated
FROZEN_STEPS = 8  # same non-zero PV value this long while clear-sky changes
JUMP_LO, JUMP_HI = 0.2, 5.0  # daily energy vs the asset's own recent median
JUMP_HISTORY_DAYS = 30
JUMP_MIN_HISTORY = 3
MIN_REPORTED = 0.25  # share of steps reported; a long outage is ungraded
TURB_MAX_FRAC = 1.1
DEFAULT_FUSE_KW = 3 * 230 * 32 / 1000  # 3-phase 32 A connection, 22 kW, when the row has no `fuse_kw`


# ------------------------------------------------------------------------------------------ pvlib clear-sky
def clearsky_power(site_rows: Sequence[Mapping], day0_ts: int, n_days: int, steps: int = 96,
                   step_minutes: int = 15, loc_round: float = 0.1) -> np.ndarray:
    """pvlib clear-sky AC power [S, n_days, steps] (kW) from each site's RECORDED kWp / tilt / azimuth.

    Sun position is computed once per location cell (`loc_round` degrees, ~10 km: the same sky shape) at interval
    middles, then broadcast over sites (wattsentry.eval.baselines.pv.physical, the fair pvlib baseline).
    """
    import pandas as pd

    from wattsentry.eval.baselines.pv import physical as ph

    S = len(site_rows)
    T = n_days * steps
    if S == 0:
        return np.zeros((0, n_days, steps))
    ts = pd.date_range(pd.Timestamp(int(day0_ts), unit="s", tz="UTC"), periods=T, freq=f"{step_minutes}min")
    lat = np.array([float(r["lat"]) for r in site_rows])
    lon = np.array([float(r["lon"]) for r in site_rows])
    cell = np.stack([np.round(lat / loc_round), np.round(lon / loc_round)], 1)
    uniq, inv = np.unique(cell, axis=0, return_inverse=True)
    inv = inv.reshape(-1)
    keys = ("zenith", "azimuth", "cs_ghi", "cs_dni", "cs_dhi")
    geo = {k: np.empty((len(uniq), T)) for k in keys}
    doy = None
    for g, (la, lo) in enumerate(uniq):
        gg = ph.solar_geometry(ts, float(la * loc_round), float(lo * loc_round))
        for k in keys:
            geo[k][g] = gg[k]
        doy = gg["doy"]
    geom = {k: geo[k][inv] for k in keys}
    geom["doy"] = np.broadcast_to(doy, (S, T))

    def col(key: str, fallback: Optional[str] = None) -> np.ndarray:
        out = []
        for r in site_rows:
            v = r.get(key)
            if (v is None or not np.isfinite(float(v)) or float(v) <= 0) and fallback:
                v = r.get(fallback)
            out.append(float(v) if v is not None else np.nan)
        return np.array(out)[:, None]

    kwp = col("kwp_recorded")
    meta = {"kwp": kwp, "tilt": col("tilt_recorded"), "azimuth": col("azimuth_recorded"),
            "inverter_kw": np.where(np.isfinite(col("inverter_kw")), col("inverter_kw"), kwp)}
    with np.errstate(invalid="ignore"):
        ac = ph.clearsky_power_kw(meta, geom)
    return np.nan_to_num(np.asarray(ac, np.float64), nan=0.0).clip(min=0.0).reshape(S, n_days, steps)


# ------------------------------------------------------------------------------------------ helpers
def _run_max(mask: np.ndarray) -> np.ndarray:
    """Longest run of True along the last axis, vectorised (no per-step Python loop)."""
    m = mask.astype(np.int32)
    n = m.shape[-1]
    idx = np.arange(1, n + 1, dtype=np.int32)
    # position of the last False before each step (cumulative max of the index where mask is False)
    last_false = np.maximum.accumulate(np.where(m == 0, idx, 0), axis=-1)
    return (idx - last_false).max(axis=-1) if n else np.zeros(mask.shape[:-1], np.int32)


def own_history_jump(energy: np.ndarray, hist_days: int = JUMP_HISTORY_DAYS,
                     min_hist: int = JUMP_MIN_HISTORY) -> np.ndarray:
    """[A, D] bool: daily energy outside 0.2-5 x the median of the asset's own previous <= `hist_days` days.

    Strictly causal: day d only looks at days < d. NaN energies (gated / unreported) are ignored; fewer than
    `min_hist` usable past days -> not flagged.
    """
    A, D = energy.shape
    out = np.zeros((A, D), bool)
    if D < 2:
        return out
    pad = np.concatenate([np.full((A, hist_days), np.nan), energy], axis=1)
    win = np.lib.stride_tricks.sliding_window_view(pad, hist_days, axis=1)[:, :D]  # [A, D, hist] = days d-30..d-1
    cnt = np.isfinite(win).sum(-1)
    from .scores import nanquantile

    med = nanquantile(win, 0.5)[..., 0]
    ok = (cnt >= min_hist) & np.isfinite(med) & (med > 0) & np.isfinite(energy)
    out[ok] = (energy[ok] < JUMP_LO * med[ok]) | (energy[ok] > JUMP_HI * med[ok])
    return out


# ------------------------------------------------------------------------------------------ gates
def gate_pv(pv: np.ndarray, cs: np.ndarray, kwp: np.ndarray, step_h: float = 0.25) -> np.ndarray:
    """PV gate codes [S, D] (0 = clean). pv [S, D, 96] kW with NaN = not reported; cs = pvlib clear-sky kW."""
    S, D, T = pv.shape
    kwp = np.asarray(kwp, np.float64).reshape(S, 1, 1)
    rep = np.isfinite(pv)
    code = np.zeros((S, D), np.int64)
    x = np.where(rep, pv, 0.0)
    code |= np.where(((x < -1e-6) | (x > PV_MAX_FRAC * kwp)).any(-1), PV_RANGE, 0)
    night = cs <= 0
    interior = night.copy()
    interior[..., 1:] &= night[..., :-1]
    interior[..., :-1] &= night[..., 1:]
    bad_night = (interior & rep & (x > NIGHT_TOL_FRAC * kwp)).sum(-1)
    code |= np.where(bad_night > NIGHT_MAX_STEPS, PV_NIGHT, 0)
    same = rep[..., 1:] & rep[..., :-1] & (pv[..., 1:] == pv[..., :-1]) & (x[..., 1:] > 0)
    moving = cs[..., 1:] != cs[..., :-1]
    # a value pinned at the day's own maximum is inverter clipping (physical), not a frozen sensor
    dmax = np.max(x, axis=-1, keepdims=True)
    not_clip = x[..., 1:] < 0.98 * dmax
    code |= np.where(_run_max(same & moving & not_clip) >= FROZEN_STEPS - 1, FROZEN, 0)
    code |= np.where(rep.mean(-1) < MIN_REPORTED, GAP, 0)
    energy = np.where(rep.any(-1), x.sum(-1) * step_h, np.nan)
    energy = np.where(code == 0, energy, np.nan)  # gated days never become anyone's history
    code |= np.where(own_history_jump(energy), UNIT_JUMP, 0)
    return code


def gate_load(load: np.ndarray, fuse_kw: np.ndarray, step_h: float = 0.25) -> np.ndarray:
    S, D, T = load.shape
    fuse = np.asarray(fuse_kw, np.float64).reshape(S, 1, 1)
    rep = np.isfinite(load)
    x = np.where(rep, load, 0.0)
    code = np.zeros((S, D), np.int64)
    code |= np.where(((x < -1e-6) | (x > fuse)).any(-1), LOAD_FUSE, 0)
    code |= np.where(rep.mean(-1) < MIN_REPORTED, GAP, 0)
    energy = np.where((code == 0) & rep.any(-1), x.sum(-1) * step_h, np.nan)
    code |= np.where(own_history_jump(energy), UNIT_JUMP, 0)
    return code


def gate_turbine(power: np.ndarray, rated_kw: np.ndarray) -> np.ndarray:
    N, D, T = power.shape
    rated = np.asarray(rated_kw, np.float64).reshape(N, 1, 1)
    rep = np.isfinite(power)
    x = np.where(rep, power, 0.0)
    code = np.zeros((N, D), np.int64)
    code |= np.where(((x > TURB_MAX_FRAC * rated) | (x < -0.05 * rated)).any(-1), TURB_RANGE, 0)
    code |= np.where(rep.mean(-1) < MIN_REPORTED, GAP, 0)
    return code
