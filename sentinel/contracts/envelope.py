"""Prediction envelope: the one output shape every model and baseline returns (plan §3.2, §3.4).

Spec: team/contracts/model.md. Stdlib only, so it imports anywhere (sandbox, workers, replay).
"""

from __future__ import annotations

import math
from dataclasses import asdict, dataclass, field
from datetime import datetime
from enum import Enum
from typing import Any, Mapping, Optional, Protocol, Sequence, runtime_checkable

STEP_MINUTES = 15
DAY_STEPS = 96  # 24 h at 15 min


class ModelId(str, Enum):
    """Registry ids (plan §3.2). Baselines and the mock use the same envelope."""

    PVNET = "pvnet"  # M1
    LOADNET = "loadnet"  # M2
    DEMANDNET = "demandnet"  # M3
    SITEHEALTH = "sitehealth"  # M4
    WINDPOWER = "windpower"  # M5
    WINDNBM = "windnbm"  # M6
    WINDFAILURE = "windfailure"  # M7
    GRIDTWIN = "gridtwin"  # M8
    # baselines / fallbacks
    PVLIB = "pvlib"
    PERSISTENCE = "persistence"
    LGBM = "lgbm"
    THRESHOLD_CUSUM = "threshold_cusum"
    WINDMON_RULES = "windmon_rules"
    BINNED_CURVE = "binned_curve"
    MOCK = "mock"


class AssetKind(str, Enum):
    SITE = "site"  # prosumer site (PV + battery + load)
    TURBINE = "turbine"
    FEEDER = "feeder"
    ZONE = "zone"  # demand zone: county / locality / feeder
    PARK = "park"


class PvFault(str, Enum):
    """M4 SiteHealth classes (8, incl. healthy). Same labels as the PV fault injector (#12)."""

    HEALTHY = "healthy"
    SOILING = "soiling"
    SHADING = "shading"
    STRING_OUTAGE = "string_outage"
    CLIPPING = "clipping"
    OVERVOLTAGE_TRIP = "overvoltage_trip"
    BATTERY_FADE = "battery_fade"
    SENSOR_FAULT = "sensor_fault"


class WindAnomaly(str, Enum):
    """M6 WindNBM classes (plan §3.5). Same labels as the wind fault injector (#23)."""

    NORMAL = "normal"
    FAULT_STOP = "fault_stop"
    CURTAILMENT = "curtailment"
    ICING = "icing"
    UNDERPERFORMANCE = "underperformance"  # incl. yaw misalignment, pitch / power-curve degradation
    OVERHEATING = "overheating"  # component temperature residual trending up
    ANEMOMETER_FAULT = "anemometer_fault"
    DATA_GAP = "data_gap"  # gap or suspect data


def _floats(xs: Optional[Sequence[float]]) -> Optional[list[float]]:
    return None if xs is None else [float(x) for x in xs]


def parse_utc(ts: str) -> datetime:
    """ISO 8601 UTC timestamp, e.g. '2025-06-01T10:00:00Z'. Storage is always UTC (plan §3.4)."""
    dt = datetime.fromisoformat(ts.replace("Z", "+00:00"))
    if dt.tzinfo is None or dt.utcoffset().total_seconds() != 0:
        raise ValueError(f"timestamp must be UTC with an explicit offset: {ts!r}")
    return dt


@dataclass
class Prediction:
    """What every model returns for one asset and one window.

    q10/q50/q90 have length `horizon` (steps of `step_minutes`), in `unit` (kW for power, MW for zones).
    Detectors (M4, M6) still fill q10/q50/q90 with the expected power they measured the fault against.
    """

    asset_id: str
    model: str  # a ModelId value
    model_version: str  # e.g. "pvnet-2025.11.07-r12" (run id) or "mock-1"
    t0: str  # UTC ISO 8601, start of the first horizon step
    horizon: int
    q10: list[float]
    q50: list[float]
    q90: list[float]
    ood_score: float = 0.0  # 0 = in-distribution; thresholds tuned on validation (#27)
    member_std: float = 0.0  # ensemble spread of q50, same unit
    fault_probs: Optional[dict[str, float]] = None  # PvFault -> p, sums to 1 (M4)
    anomaly_class: Optional[str] = None  # WindAnomaly value (M6)
    anomaly_score: Optional[float] = None  # 0..1 (M6)
    unit: str = "kW"
    step_minutes: int = STEP_MINUTES
    extra: dict[str, Any] = field(default_factory=dict)  # model-specific, never required by consumers

    def __post_init__(self) -> None:
        self.q10 = _floats(self.q10)
        self.q50 = _floats(self.q50)
        self.q90 = _floats(self.q90)
        self.ood_score = float(self.ood_score)
        self.member_std = float(self.member_std)
        if self.fault_probs is not None:
            self.fault_probs = {str(k): float(v) for k, v in self.fault_probs.items()}
        self.validate()

    def validate(self) -> None:
        if not self.asset_id:
            raise ValueError("asset_id is empty")
        parse_utc(self.t0)
        if self.horizon <= 0:
            raise ValueError("horizon must be > 0")
        for name in ("q10", "q50", "q90"):
            q = getattr(self, name)
            if len(q) != self.horizon:
                raise ValueError(f"{name} has {len(q)} values, horizon is {self.horizon}")
            if any(math.isnan(v) or math.isinf(v) for v in q):
                raise ValueError(f"{name} has NaN/inf")
        eps = 1e-6
        for t, (a, b, c) in enumerate(zip(self.q10, self.q50, self.q90)):
            if not (a <= b + eps and b <= c + eps):
                raise ValueError(f"quantiles cross at step {t}: {a} {b} {c}")
        if self.ood_score < 0 or self.member_std < 0:
            raise ValueError("ood_score and member_std must be >= 0")
        if self.fault_probs is not None:
            unknown = set(self.fault_probs) - {f.value for f in PvFault}
            if unknown:
                raise ValueError(f"unknown fault classes {sorted(unknown)}")
            s = sum(self.fault_probs.values())
            if abs(s - 1.0) > 1e-3:
                raise ValueError(f"fault_probs sum to {s}, not 1")
        if self.anomaly_class is not None and self.anomaly_class not in {a.value for a in WindAnomaly}:
            raise ValueError(f"unknown anomaly_class {self.anomaly_class!r}")

    # -- helpers consumers use -------------------------------------------------
    @property
    def p_fault(self) -> float:
        """P(any fault) = 1 - P(healthy). 0 if the model has no fault head."""
        if not self.fault_probs:
            return 0.0
        return 1.0 - self.fault_probs.get(PvFault.HEALTHY.value, 0.0)

    @property
    def top_fault(self) -> Optional[str]:
        if not self.fault_probs:
            return None
        return max(self.fault_probs.items(), key=lambda kv: kv[1])[0]

    def interval_width(self) -> float:
        """Mean q90 - q10, the 'not sure' signal next to member_std and ood_score."""
        return sum(c - a for a, c in zip(self.q10, self.q90)) / self.horizon

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, d: Mapping[str, Any]) -> "Prediction":
        return cls(**dict(d))


@runtime_checkable
class Predictor(Protocol):
    """`predict(asset, window) -> Prediction` — what #14, #15, #41 and MockPredictor implement.

    `asset` is a metadata row (dict, see data.md / wind.md); `window` is a Window (see model.md).
    """

    model: str
    model_version: str

    def predict(self, asset: Mapping[str, Any], window: "Window") -> Prediction: ...


@dataclass
class Window:
    """Inputs for one prediction. Arrays are plain sequences or numpy arrays, shape noted per field.

    PV / site (M1, M4): weather [T_in, C_w] over the 24 h ahead (forecast weather in forecast mode, never ERA5
    actuals), solar geometry [T_in, 3] (zenith, azimuth, clearsky_ghi), reference power [T_ref] = the site's own
    past pv_ac_w for 7-14 days, residual_summary [14, C_r] daily residual stats, battery [T_in, 2] (soc, batt_w).
    Load (M2): weather + calendar, past load_w. Wind (M6): turbine telemetry [T_in, C_wind] + park context.
    """

    t0: str
    horizon: int = DAY_STEPS
    weather: Any = None
    solar_geometry: Any = None
    reference: Any = None
    residual_summary: Any = None
    battery: Any = None
    calendar: Any = None
    history: Any = None  # past target (load_w for M2, act_pwt for M6), never the target window itself
    park_context: Any = None
    meta: dict[str, Any] = field(default_factory=dict)
