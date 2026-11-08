"""Household load for Romanian prosumers at 15 min.

Archetype = heating (gas | electric | heat pump) x AC x EV x electric boiler x works-from-home, on top of an
occupancy profile (weekday vs weekend / public holiday), lighting after dark, appliance spikes and noise.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from .weather import Weather

# Hourly occupancy (0..1), local time.
_OCC_WEEKDAY = [.25, .25, .25, .25, .25, .3, .7, .85, .6, .2, .2, .2, .2, .2, .2, .3, .5, .8, 1, 1, 1, .9, .7, .4]
_OCC_OFFDAY = [.3, .25, .25, .25, .25, .25, .3, .5, .7, .75, .7, .7, .75, .6, .55, .6, .7, .85, 1, 1, 1, .9, .7, .45]
_OCC_WFH = [.25, .25, .25, .25, .25, .3, .7, .85, .7, .6, .6, .6, .7, .6, .6, .6, .7, .85, 1, 1, 1, .9, .7, .4]


def _qh(profile: list[float]) -> np.ndarray:
    return np.repeat(np.asarray(profile, np.float32), 4)


def national_shape(doy: np.ndarray) -> np.ndarray:
    """Seasonal multiplier of household consumption (winter higher)."""
    return (1 + 0.10 * np.cos(2 * np.pi * (doy - 15) / 365)).astype(np.float32)


def _ema(x: np.ndarray, tau_steps: float) -> np.ndarray:
    a = 1 / tau_steps
    out = np.empty_like(x)
    out[..., 0] = x[..., 0]
    for t in range(1, x.shape[-1]):
        out[..., t] = out[..., t - 1] + a * (x[..., t] - out[..., t - 1])
    return out


def _daily_event(rng, n: int, n_days: int, qh: np.ndarray, day: np.ndarray, p_day: float,
                 start_lo: int, start_hi: int, dur_lo: int, dur_hi: int) -> np.ndarray:
    on = rng.random((n, n_days)) < p_day
    start = rng.integers(start_lo, start_hi, (n, n_days))
    dur = rng.integers(dur_lo, dur_hi + 1, (n, n_days))
    rel = qh[None] - start[:, day]
    return on[:, day] & (rel >= 0) & (rel < dur[:, day])


def simulate_load(sites: pd.DataFrame, w: Weather, cal: dict[str, np.ndarray], rng, batch: int = 128) -> np.ndarray:
    n, T = len(sites), len(w.ts)
    out = np.empty((n, T), np.float32)
    qh, day, off = cal["qh"], cal["day_index"], cal["off_day"]
    n_days = int(day[-1]) + 1
    loc_index = {lid: i for i, lid in enumerate(w.locality_ids)}
    # Building inertia: heating follows a ~6 h lagged temperature, cooling a ~3 h lagged one.
    t_heat = _ema(w.data["temp_air"], 24)
    t_cool = _ema(w.data["temp_air"], 12)
    dark = (w.data["zenith"] > 85).astype(np.float32)
    season = national_shape(cal["doy"])

    occ_week, occ_off, occ_wfh = _qh(_OCC_WEEKDAY), _qh(_OCC_OFFDAY), _qh(_OCC_WFH)
    for b0 in range(0, n, batch):
        s = sites.iloc[b0:b0 + batch]
        m = len(s)
        li = np.array([loc_index[x] for x in s.locality_id])
        wfh = s.works_from_home.to_numpy()[:, None]
        occ = np.where(off[None], occ_off[qh][None], np.where(wfh, occ_wfh[qh][None], occ_week[qh][None]))

        base = 100 + 320 * occ + 150 * occ * dark[li]
        # Heating: setpoint 20 C when people are up, 17 C otherwise; only in the heating season.
        th = t_heat[li]
        setp = np.where(occ > 0.5, 20.0, 17.0)
        q_heat = s.true_ua_kw_per_k.to_numpy()[:, None] * np.maximum(setp - th, 0) * (th < 16) * 1000
        heating = s.heating.to_numpy()[:, None]
        cop = np.clip(2.2 + 0.07 * (th + 5), 1.8, 4.5)
        heat = np.where(heating == "electric", np.minimum(q_heat, 6000),
                        np.where(heating == "heat_pump", np.minimum(q_heat / cop, 3000), 60.0 * (th < 15)))
        # Cooling: AC above 25 C (lagged), stronger when someone is home.
        cool = s.has_ac.to_numpy()[:, None] * np.minimum(
            s.true_ua_kw_per_k.to_numpy()[:, None] * 0.5 * np.maximum(t_cool[li] - 25, 0) * 1000, 2200) \
            * (0.35 + 0.65 * occ)
        ev = _daily_event(rng, m, n_days, qh, day, 0.45, 72, 88, 4, 12) * s.has_ev.to_numpy()[:, None] * 3700
        boiler = (_daily_event(rng, m, n_days, qh, day, 0.9, 24, 30, 3, 5)
                  | _daily_event(rng, m, n_days, qh, day, 0.7, 76, 84, 2, 4)) * \
            s.has_electric_boiler.to_numpy()[:, None] * 2000
        spikes = (rng.random((m, T)) < 0.06 * occ) * rng.exponential(700, (m, T))

        eps = rng.standard_normal((m, T)).astype(np.float32)
        noise = np.empty_like(eps)
        noise[:, 0] = eps[:, 0]
        for t in range(1, T):
            noise[:, t] = 0.7 * noise[:, t - 1] + 0.714 * eps[:, t]
        mult = np.exp(0.25 * noise - 0.03)
        load = (base * season[None] * mult + heat + cool + ev + boiler + spikes) * s.true_load_scale.to_numpy()[:, None]
        out[b0:b0 + m] = load
    return out
