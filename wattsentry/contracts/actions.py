"""Agent actions: JSON payload + audit record, one SQLite row each (table names from the Drizzle schema)."""

from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, dataclass, field
from enum import Enum
from typing import Any, Mapping, Optional

from .envelope import parse_utc


class ActionType(str, Enum):
    BATTERY_SCHEDULE = "battery_schedule"
    FEEDER_COMMITMENT = "feeder_commitment"
    WORK_ORDER = "work_order"
    ALERT = "alert"
    GIS_FLAG = "gis_flag"
    HUMAN_REVIEW = "human_review"
    PARK_REPORT = "park_report"
    ZONE_DEMAND = "zone_demand"


class CrewType(str, Enum):
    RESIDENTIAL = "residential_installer"
    WIND = "wind_technician"
    REMOTE = "remote_check"  # sensor faults: never a truck


class Severity(str, Enum):
    INFO = "info"
    WARNING = "warning"
    CRITICAL = "critical"
    SAFETY = "safety"  # e.g. icing


# Required payload keys per action type (extra keys are allowed).
PAYLOAD_KEYS: dict[str, tuple[str, ...]] = {
    "battery_schedule": ("site_id", "feeder_id", "batt_kw"),  # batt_kw[96], + charge
    "feeder_commitment": ("feeder_id", "nominal", "available", "committed"),  # kW[96]
    "work_order": ("asset_id", "asset_kind", "fault", "crew_type", "score", "lost_kwh_day", "lost_lei_day"),
    "alert": ("asset_id", "asset_kind", "severity", "code", "message"),
    "gis_flag": ("section_id", "recorded_feeder", "recorded_phase", "suggested_feeder", "suggested_phase", "tau"),
    "human_review": ("asset_id", "asset_kind", "reason"),
    "park_report": ("park_id", "date", "energy_mwh", "expected_mwh", "availability", "curtailed_mwh", "issues"),
    "zone_demand": ("zone_id", "level", "q50_mw", "q90_mw", "peak_t", "peak_mw"),
}


def input_hash(obj: Any) -> str:
    """Stable sha256 (first 16 hex) of any JSON-able input, so an audit row pins exactly what the model saw."""
    blob = json.dumps(obj, sort_keys=True, separators=(",", ":"), default=str).encode()
    return hashlib.sha256(blob).hexdigest()[:16]


@dataclass
class AuditRecord:
    """Why the agent did it: what model, on what input, what it predicted, what threshold, what status."""

    model_version: str
    input_hash: str
    prediction: dict[str, Any]  # compact: e.g. {"q50_sum_kwh":..., "p_fault":..., "top_fault":...}
    threshold: Optional[float]
    status: str  # AssetStatus.status at decision time
    schedule: Optional[list[float]] = None  # battery kW, when the action carries one
    note: str = ""

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class Action:
    action_id: str
    type: str  # ActionType value
    t: str  # UTC decision time (the tick)
    policy: str  # "sentinel" | "self_consumption" | "threshold" | ...
    run_id: str  # lab run id of the replay that produced it
    payload: dict[str, Any]
    audit: AuditRecord
    meta: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        self.type = ActionType(self.type).value
        if isinstance(self.audit, Mapping):
            self.audit = AuditRecord(**self.audit)
        parse_utc(self.t)
        missing = [k for k in PAYLOAD_KEYS[self.type] if k not in self.payload]
        if missing:
            raise ValueError(f"{self.type} payload missing {missing}")

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    def to_json(self) -> str:
        return json.dumps(self.to_dict(), sort_keys=True, separators=(",", ":"))

    @classmethod
    def from_json(cls, s: str) -> "Action":
        return cls(**json.loads(s))
