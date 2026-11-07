"""PV production per site from the TRUE metadata, with effects pvlib won't model from the recorded metadata:
horizon shading, inverter efficiency curve + clipping, degradation, snow cover, temperature, system losses."""

from __future__ import annotations

import numpy as np
import pandas as pd

from . import solar
from .weather import Weather

INV_STANDBY, INV_PRODUCING, INV_LIMITING = 0, 1, 2


def simulate_pv(sites: pd.DataFrame, w: Weather) -> tuple[np.ndarray, np.ndarray]:
    """Return (pv_ac_w, inv_status), both (n_sites, T)."""
    n, T = len(sites), len(w.ts)
    pac = np.zeros((n, T), np.float32)
    status = np.zeros((n, T), np.int8)
    years = (w.ts.year + (w.ts.dayofyear - 1) / 365.25).to_numpy()
    loc_index = {lid: i for i, lid in enumerate(w.locality_ids)}

    for lid, rows in sites.groupby("locality_id").indices.items():
        s = sites.iloc[rows]
        j = loc_index[lid]
        zen, saz = w.data["zenith"][j], w.data["azimuth"][j]
        ghi, dni, dhi = w.data["ghi"][j], w.data["dni"][j], w.data["dhi"][j]
        tair, wind, snow = w.data["temp_air"][j], w.data["wind_speed"][j], w.data["snow_depth_cm"][j]

        tilt = s.true_tilt_deg.to_numpy()[:, None]
        cos_aoi = np.clip(solar.aoi_cos(tilt, s.true_azimuth_deg.to_numpy()[:, None], zen, saz), 0, 1)
        iam = np.clip(1 - 0.05 * (1 / np.maximum(cos_aoi, 0.05) - 1), 0, 1)
        beam = dni * cos_aoi * iam

        # Horizon shading: beam blocked while the sun is low inside the obstruction's azimuth sector.
        shaded = s.true_shaded.to_numpy()[:, None]
        elev = 90 - zen
        d_az = np.abs((saz - s.true_shade_az_deg.to_numpy()[:, None] + 180) % 360 - 180)
        blocked = shaded & (d_az < s.true_shade_width_deg.to_numpy()[:, None] / 2) & \
            (elev < s.true_shade_elev_deg.to_numpy()[:, None])
        beam = np.where(blocked, 0, beam)

        cos_t = np.cos(np.radians(tilt))
        albedo = np.where(snow > 1, 0.6, 0.2)
        poa = beam + dhi * (1 + cos_t) / 2 * np.where(shaded, 0.92, 1.0) + ghi * albedo * (1 - cos_t) / 2

        cover = np.clip((snow - 1) / 8, 0, 1) * np.where(tilt < 25, 1.0, 0.6)
        t_cell = tair + poa / (25 + 6.84 * wind)
        age = np.maximum(years[None] - s.install_year.to_numpy()[:, None] - 0.5, 0)
        pdc = (s.true_kwp.to_numpy()[:, None] * poa * (1 + s.true_temp_coeff.to_numpy()[:, None] * (t_cell - 25))
               * s.true_system_losses.to_numpy()[:, None]
               * (1 - s.true_degradation_per_year.to_numpy()[:, None]) ** age * (1 - cover))
        pdc = np.maximum(pdc, 0)

        ac_w = s.inverter_ac_kw.to_numpy()[:, None] * 1000
        p_norm = pdc / ac_w
        eta = np.clip(0.97 * p_norm / (p_norm + 0.008) - 0.01 * p_norm, 0, 0.975)
        p = pdc * eta
        p = np.where(pdc < 15, 0, p)  # inverter asleep
        clipped = p > ac_w
        p = np.minimum(p, ac_w)
        pac[rows] = p
        status[rows] = np.where(p > 0, np.where(clipped, INV_LIMITING, INV_PRODUCING), INV_STANDBY)
    return pac, status
