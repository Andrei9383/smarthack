"""Wind physics shared by the stand-in simulator, M6 features and the predictor (numpy only).

Plan §3.5 "Wind normalisation":
    v_hub  = v_100 · (h_hub / 100)^α          α ≈ 0.14
    ρ      = p / (R_d · T)                     R_d = 287.05 J/(kg·K), p in Pa, T in K
    v_norm = v_hub · (ρ / 1.225)^(1/3)         IEC 61400-12-1 density correction
Power curves are generic class curves (Betz-limited aero power with a soft knee), not vendor curves.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

R_D = 287.05  # J/(kg·K), dry air
RHO_REF = 1.225  # kg/m³, IEC reference density
ALPHA_DEFAULT = 0.14


def shear_to_hub(v_ref, h_hub, h_ref: float = 100.0, alpha: float = ALPHA_DEFAULT):
    """Power-law shear: wind at reference height (default 100 m, ERA5) → hub height."""
    return np.asarray(v_ref, dtype=np.float64) * (np.asarray(h_hub, dtype=np.float64) / h_ref) ** alpha


def air_density(pressure_hpa, temp_c):
    """ρ = p / (R_d · T) with p in hPa and T in °C (converted to Pa and K)."""
    p = np.asarray(pressure_hpa, dtype=np.float64) * 100.0
    t = np.asarray(temp_c, dtype=np.float64) + 273.15
    return p / (R_D * t)


def density_normalise(v_hub, rho):
    """v_norm = v_hub · (ρ / 1.225)^(1/3)."""
    return np.asarray(v_hub, dtype=np.float64) * (np.asarray(rho, dtype=np.float64) / RHO_REF) ** (1.0 / 3.0)


@dataclass(frozen=True)
class TurbineClass:
    """Generic turbine class (not a vendor curve)."""

    name: str
    rated_kw: float
    rotor_d_m: float
    hub_height_m: float
    cut_in_ms: float = 3.0
    cut_out_ms: float = 25.0
    cp: float = 0.43
    rot_min_rpm: float = 5.0
    rot_max_rpm: float = 14.0

    @property
    def area_m2(self) -> float:
        return float(np.pi * (self.rotor_d_m / 2) ** 2)

    def power_frac(self, v_norm):
        """Fraction of rated power at density-normalised hub wind (m/s). Soft knee, zero below cut-in / above cut-out."""
        v = np.asarray(v_norm, dtype=np.float64)
        vc = np.clip(v, 0.0, None)
        aero = 0.5 * RHO_REF * self.area_m2 * self.cp * (vc**3 - self.cut_in_ms**3) / (self.rated_kw * 1000.0)
        aero = np.clip(aero, 1e-9, None)
        k = 8.0
        frac = (aero ** (-k) + 1.0) ** (-1.0 / k)  # smooth min(aero, 1)
        frac = np.where((v < self.cut_in_ms) | (v >= self.cut_out_ms), 0.0, frac)
        return frac

    def rated_wind_ms(self) -> float:
        """Wind speed at which the curve reaches 99% of rated."""
        vs = np.linspace(self.cut_in_ms, self.cut_out_ms, 2201)
        return float(vs[np.argmax(self.power_frac(vs) >= 0.99)])

# Generic classes used by the stand-in parks (plan §4: 2.5 MW and 6.1 MW class).
CLASS_2500 = TurbineClass("generic-2.5MW", 2500.0, 100.0, 85.0, rot_min_rpm=5.0, rot_max_rpm=14.0)
CLASS_6100 = TurbineClass("generic-6.1MW", 6100.0, 158.0, 120.0, cp=0.42, rot_min_rpm=4.0, rot_max_rpm=10.5)
CLASSES = {c.name: c for c in (CLASS_2500, CLASS_6100)}


def class_for(rated_kw: float, rotor_d_m: float | None = None, hub_height_m: float | None = None) -> TurbineClass:
    """Pick the generic class closest in rated power (used when external turbines have other models)."""
    best = min(CLASSES.values(), key=lambda c: abs(c.rated_kw - rated_kw))
    if abs(best.rated_kw - rated_kw) < 1e-6 and rotor_d_m is None:
        return best
    return TurbineClass(
        f"generic-{rated_kw / 1000:.1f}MW",
        float(rated_kw),
        float(rotor_d_m or best.rotor_d_m * np.sqrt(rated_kw / best.rated_kw)),
        float(hub_height_m or best.hub_height_m),
        cp=best.cp,
        rot_min_rpm=best.rot_min_rpm,
        rot_max_rpm=best.rot_max_rpm,
    )
