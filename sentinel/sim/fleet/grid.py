"""Feeder physics: passive households, per-phase net export, feeder export vs limit, inverter voltage.

Voltage at a site = feeder base (MV side, slow daily wander) + rise from net export on its TRUE phase (and a bit
from the other phases), scaled by its position along the feeder, + noise. That is the signal the grid-twin (#13)
uses to recover phases and feeders that the GIS records wrongly.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from .load import _OCC_OFFDAY, _OCC_WEEKDAY, _qh, national_shape
from .weather import Weather

V_NOMINAL = 230.0
PHASES = "ABC"


def passive_load(feeders: pd.DataFrame, w: Weather, cal: dict[str, np.ndarray], rng) -> np.ndarray:
    """(n_feeders, T) aggregate load of the feeder's households without PV, in W."""
    qh, off = cal["qh"], cal["off_day"]
    occ = np.where(off, _qh(_OCC_OFFDAY)[qh], _qh(_OCC_WEEKDAY)[qh])
    loc_index = {lid: i for i, lid in enumerate(w.locality_ids)}
    li = np.array([loc_index[x] for x in feeders.locality_id])
    temp = w.data["temp_air"][li]
    per_house = (230 + 330 * occ[None]) * national_shape(cal["doy"])[None] + 18 * np.maximum(15 - temp, 0)
    eps = rng.standard_normal(per_house.shape).astype(np.float32)
    noise = np.empty_like(eps)
    noise[:, 0] = eps[:, 0]
    for t in range(1, eps.shape[1]):
        noise[:, t] = 0.9 * noise[:, t - 1] + 0.436 * eps[:, t]
    return (per_house * (1 + 0.04 * noise) * feeders.n_passive_households.to_numpy()[:, None]).astype(np.float32)


def feeder_flows(sites: pd.DataFrame, feeders: pd.DataFrame, grid_w: np.ndarray, passive_w: np.ndarray,
                 w: Weather, rng) -> tuple[pd.DataFrame, np.ndarray]:
    """Return (feeder time series as long DataFrame, inverter voltage (n_sites, T))."""
    nf, T = len(feeders), grid_w.shape[1]
    fidx = pd.Index(feeders.feeder_id).get_indexer(sites.true_feeder_id)
    pidx = np.searchsorted(np.array(list(PHASES)), sites.true_phase.to_numpy())
    group = fidx * 3 + pidx
    onehot = np.zeros((nf * 3, len(sites)), np.float32)
    onehot[group, np.arange(len(sites))] = 1
    phase_w = (onehot @ grid_w).reshape(nf, 3, T) - passive_w[:, None, :] / 3   # net export per phase
    feeder_w = phase_w.sum(1)
    prosumer_w = feeder_w + passive_w

    # MV-side voltage: per-feeder base with a slow daily swing and a little noise.
    hour = w.ts.tz_convert("Europe/Bucharest").hour.to_numpy()
    swing = 0.006 * np.cos(2 * np.pi * (hour - 4) / 24)
    v_mv = V_NOMINAL * (feeders.v_base_pu.to_numpy()[:, None] + swing[None]
                        + 0.002 * rng.standard_normal((nf, T)))

    own = phase_w[fidx, pidx]
    rise = feeders.v_sens_v_per_kw.to_numpy()[fidx][:, None] * sites.true_feeder_position.to_numpy()[:, None] \
        * (own + 0.25 * (feeder_w[fidx] - own)) / 1000
    volt = (v_mv[fidx] + rise + 0.4 * rng.standard_normal(rise.shape)).astype(np.float32)

    ts = np.tile(w.ts.to_numpy(), nf)
    df = pd.DataFrame({
        "ts": ts,
        "feeder_id": np.repeat(feeders.feeder_id.to_numpy(), T),
        "net_export_w": feeder_w.ravel().astype(np.float32),
        "prosumer_export_w": prosumer_w.ravel().astype(np.float32),
        "passive_load_w": passive_w.ravel().astype(np.float32),
        "v_mv_v": v_mv.ravel().astype(np.float32),
        **{f"net_export_{p}_w": phase_w[:, k].ravel().astype(np.float32) for k, p in enumerate(PHASES)},
    })
    return df, volt
