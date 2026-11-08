"""Qualification, battery, feeder plan and commitment types (plan §3.4-3.5). Spec: team/contracts/opt.md.

The battery model and self-consumption rule live here so the simulator (#11) and the LP (#18) share one
definition: sites that are not in the pool keep running `self_consumption_step` (R3).
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from enum import Enum
from typing import Any, Mapping, Optional, Sequence

from .envelope import DAY_STEPS, STEP_MINUTES

DT_H = STEP_MINUTES / 60.0  # 0.25 h


class Status(str, Enum):
    ELIGIBLE = "eligible"
    DERATED = "derated"
    EXCLUDED = "excluded"
    HUMAN_REVIEW = "human_review"


@dataclass
class AssetStatus:
    """`qualify(asset, prediction, telemetry_age) -> AssetStatus`, for sites and turbines.

    headroom_factor in [0, 1]: 1 = full headroom, 0 = out of the pool. eligible -> 1, excluded / human_review -> 0,
    derated -> (0, 1) = 1 - estimated loss. `battery_ok` lets a site with a PV fault keep its battery in the pool.
    """

    asset_id: str
    status: str  # Status value
    headroom_factor: float
    reason: str  # short machine-readable code, e.g. "stale_telemetry", "pv_fault:soiling", "ood"
    battery_ok: bool = True
    detail: str = ""  # human text for the UI

    def __post_init__(self) -> None:
        self.status = Status(self.status).value
        self.headroom_factor = float(self.headroom_factor)
        if not 0.0 <= self.headroom_factor <= 1.0:
            raise ValueError("headroom_factor must be in [0, 1]")
        if self.status in (Status.EXCLUDED.value, Status.HUMAN_REVIEW.value) and self.headroom_factor != 0.0:
            raise ValueError(f"{self.status} must have headroom_factor 0")
        if self.status == Status.ELIGIBLE.value and self.headroom_factor != 1.0:
            raise ValueError("eligible must have headroom_factor 1")

    @property
    def in_pool(self) -> bool:
        return self.status in (Status.ELIGIBLE.value, Status.DERATED.value)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class BatterySpec:
    """One home battery (agreed with #11). Powers in kW, energies in kWh, efficiencies one-way."""

    capacity_kwh: float  # usable E_i
    p_max_kw: float  # P_i, same for charge and discharge
    eta_charge: float = 0.95
    eta_discharge: float = 0.95  # round trip ~0.90
    soc_min: float = 0.05  # fraction of capacity
    soc_max: float = 1.0
    comfort_reserve_kwh: float = 0.0  # R_i: never planned below this (customer promise)

    def __post_init__(self) -> None:
        if self.capacity_kwh < 0 or self.p_max_kw < 0:
            raise ValueError("capacity and power must be >= 0")
        if not (0 < self.eta_charge <= 1 and 0 < self.eta_discharge <= 1):
            raise ValueError("efficiencies must be in (0, 1]")
        if not 0 <= self.soc_min <= self.soc_max <= 1:
            raise ValueError("need 0 <= soc_min <= soc_max <= 1")
        if self.comfort_reserve_kwh > self.e_max:
            raise ValueError("comfort reserve above max energy")

    @property
    def e_min(self) -> float:
        """Lowest energy the planner may use: max(soc_min floor, comfort reserve)."""
        return max(self.soc_min * self.capacity_kwh, self.comfort_reserve_kwh)

    @property
    def e_max(self) -> float:
        return self.soc_max * self.capacity_kwh

    def step(self, e_kwh: float, batt_kw: float, dt_h: float = DT_H) -> tuple[float, float]:
        """Apply a requested battery power (+ charge, - discharge), clipped to power and energy limits.

        Returns (applied batt_kw, next energy kWh). Uses the soc_min floor, not the comfort reserve: the
        reserve binds the planner; the physical battery may go below it to serve load in the rule.
        """
        lo = self.soc_min * self.capacity_kwh
        p = max(-self.p_max_kw, min(self.p_max_kw, batt_kw))
        if p >= 0:
            p = min(p, max(0.0, (self.e_max - e_kwh) / (self.eta_charge * dt_h)))
            e_next = e_kwh + self.eta_charge * p * dt_h
        else:
            p = max(p, -max(0.0, (e_kwh - lo) * self.eta_discharge / dt_h))
            e_next = e_kwh + p * dt_h / self.eta_discharge
        return p, min(self.e_max, max(lo, e_next))


def self_consumption_step(
    battery: Optional[BatterySpec], e_kwh: float, pv_kw: float, load_kw: float, dt_h: float = DT_H
) -> tuple[float, float, float]:
    """Default rule (baseline + non-pool sites): charge from PV surplus, discharge to cover load.

    Returns (batt_kw, next energy kWh, grid_kw) with grid_kw > 0 = export, < 0 = import.
    """
    surplus = pv_kw - load_kw
    if battery is None or battery.capacity_kwh == 0:
        return 0.0, e_kwh, surplus
    batt_kw, e_next = battery.step(e_kwh, surplus, dt_h)
    return batt_kw, e_next, surplus - batt_kw


@dataclass
class FeederSpec:
    """Feeder limits from the grid-twin (#13). G_f = tau*G_max + (1-tau)*G_conservative; low trust only tightens."""

    feeder_id: str
    g_max_kw: float
    g_conservative_kw: float
    tau: float = 1.0  # GIS trust in [0, 1]

    def __post_init__(self) -> None:
        if not 0 <= self.tau <= 1:
            raise ValueError("tau must be in [0, 1]")
        if self.g_conservative_kw > self.g_max_kw:
            raise ValueError("g_conservative_kw must be <= g_max_kw")

    @property
    def g_kw(self) -> float:
        return self.tau * self.g_max_kw + (1 - self.tau) * self.g_conservative_kw


def _check_len(name: str, xs: Sequence[float], n: int) -> list[float]:
    xs = [float(x) for x in xs]
    if len(xs) != n:
        raise ValueError(f"{name} has {len(xs)} values, expected {n}")
    return xs


@dataclass
class FeederPlan:
    """`plan_feeder(feeder, eligible_sites, rest_of_feeder, forecasts, G, prices) -> FeederPlan`.

    schedule[site_id] = battery kW per step (+ charge, - discharge). export_curve = planned feeder net export kW
    (P50); slack = kW above the limit the LP could not remove (s_t). policy: "sentinel" | "self_consumption" |
    "threshold".
    """

    feeder_id: str
    t0: str
    policy: str
    g_kw: float
    schedule: dict[str, list[float]]
    export_curve: list[float]
    slack: list[float]
    horizon: int = DAY_STEPS
    solver_status: str = "optimal"
    objective: Optional[float] = None

    def __post_init__(self) -> None:
        self.schedule = {k: _check_len(f"schedule[{k}]", v, self.horizon) for k, v in self.schedule.items()}
        self.export_curve = _check_len("export_curve", self.export_curve, self.horizon)
        self.slack = _check_len("slack", self.slack, self.horizon)
        if any(s < -1e-6 for s in self.slack):
            raise ValueError("slack must be >= 0")

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class Commitment:
    """`commit(feeder, schedule, scenarios) -> Commitment`. Four numbers, never conflated (plan §3.5):
    nominal >= available >= committed; realised is filled in the next tick (None until measured). kW per step.
    """

    feeder_id: str
    t0: str
    nominal: list[float]
    available: list[float]
    committed: list[float]
    realised: Optional[list[Optional[float]]] = None
    haircut: float = 0.0  # h, telemetry-loss haircut
    horizon: int = DAY_STEPS
    extra: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        self.nominal = _check_len("nominal", self.nominal, self.horizon)
        self.available = _check_len("available", self.available, self.horizon)
        self.committed = _check_len("committed", self.committed, self.horizon)
        eps = 1e-6
        for t, (n, a, c) in enumerate(zip(self.nominal, self.available, self.committed)):
            if not (n + eps >= a and a + eps >= c and c >= -eps):
                raise ValueError(f"need nominal >= available >= committed >= 0 at step {t}: {n} {a} {c}")
        if self.realised is not None and len(self.realised) != self.horizon:
            raise ValueError("realised length != horizon")

    def hit_rate(self) -> Optional[float]:
        """Share of measured steps with committed > 0 where realised >= committed."""
        if self.realised is None:
            return None
        pairs = [(r, c) for r, c in zip(self.realised, self.committed) if r is not None and c > 0]
        if not pairs:
            return None
        return sum(r + 1e-6 >= c for r, c in pairs) / len(pairs)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, d: Mapping[str, Any]) -> "Commitment":
        return cls(**dict(d))
