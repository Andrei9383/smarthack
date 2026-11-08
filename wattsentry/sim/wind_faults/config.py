"""Fault injector configuration (card #23). Rates are per turbine-year unless marked per park-year."""

from __future__ import annotations

import dataclasses
from dataclasses import dataclass

# Injector kinds -> contract WindAnomaly class (wattsentry.contracts.envelope.WindAnomaly).
KIND_CLASS = {
    "fault_stop": "fault_stop",
    "curtailment": "curtailment",
    "icing": "icing",
    "yaw": "underperformance",
    "power_curve": "underperformance",
    "overheating": "overheating",
    "anemometer": "anemometer_fault",
    "data_gap": "data_gap",
}
KINDS = tuple(KIND_CLASS)


@dataclass(frozen=True)
class FaultConfig:
    seed: int = 23
    kinds: tuple[str, ...] = KINDS  # enabled kinds (tests enable one at a time)
    # single-turbine trips (2 h - 3 days)
    rate_stop: float = 5.0
    stop_hours: tuple[float, float] = (2.0, 72.0)
    # park-wide curtailment caps, per PARK-year; onsets weighted towards high production (1-8 h)
    rate_curtail: float = 30.0
    curtail_hours: tuple[float, float] = (1.0, 8.0)
    curtail_caps: tuple[float, ...] = (0.0, 0.3, 0.5, 0.7)  # cap as a fraction of rated
    curtail_cap_p: tuple[float, ...] = (0.15, 0.3, 0.3, 0.25)
    curtail_participation: float = 0.95  # share of the park's turbines that receive the setpoint
    # icing: real ERA5 spells with T < icing_temp_c and (RH >= icing_rh or precipitation), >= icing_min_hours long
    icing_temp_c: float = 1.0
    icing_rh: float = 90.0
    icing_precip_mm: float = 0.05
    icing_min_hours: float = 3.0
    p_icing_spell: float = 0.4  # share of qualifying spells that actually ice the park
    icing_stop_frac: float = 0.25  # iced turbines that the ice detector shuts down (still labelled icing)
    # underperformance: yaw misalignment (sector-dependent) and pitch / power-curve degradation (7-45 days)
    rate_yaw: float = 0.5
    yaw_deg: tuple[float, float] = (10.0, 25.0)
    rate_power_curve: float = 0.3
    power_curve_loss: tuple[float, float] = (0.08, 0.20)
    under_days: tuple[float, float] = (7.0, 45.0)
    # gearbox / generator-bearing degradation: residual drifting 3-21 days, often ending in a failure stop
    rate_overheat: float = 0.7
    overheat_days: tuple[float, float] = (3.0, 21.0)
    overheat_amp_c: tuple[float, float] = (8.0, 20.0)
    p_failure_after_overheat: float = 0.6
    failure_days: tuple[float, float] = (1.0, 5.0)
    # nacelle anemometer faults (bias / stuck / erratic), 2-20 days; power keeps following the real wind
    rate_anem: float = 1.0
    anem_days: tuple[float, float] = (2.0, 20.0)
    # communication gaps (all sensors null), some carrying integral PPC error codes
    rate_gap_short: float = 10.0  # 15 min - 12 h
    rate_gap_long: float = 0.4  # 1 - 7 days
    p_gap_coded: float = 0.3

    def to_json(self) -> dict:
        return dataclasses.asdict(self)

    def only(self, *kinds: str, **overrides) -> "FaultConfig":
        return dataclasses.replace(self, kinds=tuple(kinds), **overrides)
