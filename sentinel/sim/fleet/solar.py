"""Solar geometry, clear sky, decomposition and transposition in plain numpy (no pvlib on purpose:
the pvlib baseline in #15 must not share the simulator's physics)."""

from __future__ import annotations

import numpy as np
import pandas as pd

SOLAR_CONST = 1361.0


def solar_position(ts_utc: pd.DatetimeIndex, lat: float, lon: float) -> tuple[np.ndarray, np.ndarray]:
    """Return (zenith_deg, azimuth_deg) using the NOAA approximation (accuracy ~0.5 deg, plenty here).
    Timestamps are interval starts; the position is taken at the middle of the 15-min interval."""
    t = ts_utc + pd.Timedelta(minutes=7.5)
    doy = t.dayofyear.to_numpy()
    hour = (t.hour + t.minute / 60 + t.second / 3600).to_numpy()
    g = 2 * np.pi / 365 * (doy - 1 + (hour - 12) / 24)
    eqtime = 229.18 * (0.000075 + 0.001868 * np.cos(g) - 0.032077 * np.sin(g)
                       - 0.014615 * np.cos(2 * g) - 0.040849 * np.sin(2 * g))
    decl = (0.006918 - 0.399912 * np.cos(g) + 0.070257 * np.sin(g) - 0.006758 * np.cos(2 * g)
            + 0.000907 * np.sin(2 * g) - 0.002697 * np.cos(3 * g) + 0.00148 * np.sin(3 * g))
    tst = hour * 60 + eqtime + 4 * lon
    ha = np.radians(tst / 4 - 180)
    phi = np.radians(lat)
    cosz = np.sin(phi) * np.sin(decl) + np.cos(phi) * np.cos(decl) * np.cos(ha)
    cosz = np.clip(cosz, -1, 1)
    zen = np.arccos(cosz)
    sinz = np.maximum(np.sin(zen), 1e-6)
    cos_az = (np.sin(decl) - np.sin(phi) * cosz) / (np.cos(phi) * sinz)
    az = np.degrees(np.arccos(np.clip(cos_az, -1, 1)))
    az = np.where(ha > 0, 360 - az, az)  # afternoon -> west of south
    return np.degrees(zen), az


def extraterrestrial(doy: np.ndarray) -> np.ndarray:
    return SOLAR_CONST * (1 + 0.033 * np.cos(2 * np.pi * doy / 365))


def clearsky_ghi(zenith_deg: np.ndarray) -> np.ndarray:
    """Haurwitz clear-sky GHI (W/m2)."""
    cz = np.cos(np.radians(zenith_deg))
    out = np.zeros_like(cz)
    up = cz > 0.01
    out[up] = 1098.0 * cz[up] * np.exp(-0.057 / cz[up])
    return out


def erbs(ghi: np.ndarray, zenith_deg: np.ndarray, doy: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Split GHI into (DNI, DHI) with the Erbs diffuse-fraction model."""
    cz = np.cos(np.radians(zenith_deg))
    i0h = extraterrestrial(doy) * np.maximum(cz, 0.065)
    kt = np.clip(ghi / i0h, 0, 1)
    df = np.where(kt <= 0.22, 1 - 0.09 * kt,
                  np.where(kt <= 0.8, 0.9511 - 0.1604 * kt + 4.388 * kt**2 - 16.638 * kt**3 + 12.336 * kt**4, 0.165))
    dhi = ghi * df
    dni = np.where(cz > 0.065, (ghi - dhi) / np.maximum(cz, 0.065), 0.0)
    return np.clip(dni, 0, 1100), dhi


def aoi_cos(tilt_deg: np.ndarray, surf_az_deg: np.ndarray, zen_deg: np.ndarray, sun_az_deg: np.ndarray) -> np.ndarray:
    """cos(angle of incidence); tilt/surf_az shaped (n, 1), sun arrays shaped (T,) -> (n, T)."""
    t, sa = np.radians(tilt_deg), np.radians(surf_az_deg)
    z, a = np.radians(zen_deg), np.radians(sun_az_deg)
    return np.cos(z) * np.cos(t) + np.sin(z) * np.sin(t) * np.cos(a - sa)
