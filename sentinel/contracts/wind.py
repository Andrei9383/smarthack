"""Wind telemetry contract: PPC HackITall schema (plan §3.4, decisions 14:15). Spec: team/contracts/wind.md.

Tags look like `WP<park>.TID<turbine>.<sensor>`, e.g. `WP1.TID07.act_pwt`. Power in kW, wind in km/h, temps in °C.
Integral values on a numeric sensor are error codes, not measurements. Real PPC data must drop in unchanged, so the
sensor names below are the only place to rename if PPC's sample differs.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Any, Optional

TAG_RE = re.compile(r"^WP(?P<park>[A-Za-z0-9]+)\.TID(?P<turbine>[A-Za-z0-9]+)\.(?P<sensor>[A-Za-z0-9_]+)$")

# sensor -> (unit, description). Exactly the sensors in PPC's DataDictionary (WindMon repo, reviewed on #7).
# PPC data has no wind direction, yaw, rotor speed, pitch or status tags.
SENSORS: dict[str, tuple[str, str]] = {
    "act_pwt": ("kW", "active power at the turbine (unit varies per turbine: see power_scale in turbine metadata)"),
    "wd_spd": ("km/h", "nacelle anemometer wind speed"),
    "conv_t": ("degC", "converter temperature"),
    "gearbox_t": ("degC", "gearbox temperature"),
    "gen1_t": ("degC", "generator bearing 1 temperature"),
    "gen2_t": ("degC", "generator bearing 2 temperature"),
    "transformer_t": ("degC", "transformer temperature"),
    "turbine_t": ("degC", "ambient temperature at the turbine"),
}

# Sensors whose values are always integer codes (never flagged as error codes). None in PPC's schema.
CODE_SENSORS: set[str] = set()

KMH_TO_MS = 1 / 3.6


@dataclass(frozen=True)
class Tag:
    park: str
    turbine: str
    sensor: str

    @property
    def turbine_id(self) -> str:
        """Canonical asset id used everywhere else (envelope, SQLite): 'WP1.TID07'."""
        return f"WP{self.park}.TID{self.turbine}"

    def __str__(self) -> str:
        return f"{self.turbine_id}.{self.sensor}"


def parse_tag(tag: str, strict: bool = False) -> Tag:
    """Parse 'WP1.TID07.act_pwt'. strict=True also rejects sensors not in SENSORS."""
    m = TAG_RE.match(tag.strip())
    if not m:
        raise ValueError(f"not a PPC tag: {tag!r}")
    t = Tag(m["park"], m["turbine"], m["sensor"])
    if strict and t.sensor not in SENSORS:
        raise ValueError(f"unknown sensor {t.sensor!r} in {tag!r}")
    return t


def make_tag(park: str | int, turbine: str | int, sensor: str) -> str:
    """Build a tag. PPC ids are hex-like strings (WP98FA99B8.TID374A8025) and pass through unchanged; the
    zero-padding of int turbine numbers is only a convenience for the synthetic parks."""
    turbine = f"{turbine:02d}" if isinstance(turbine, int) else turbine
    return f"WP{park}.TID{turbine}.{sensor}"


def is_error_code(sensor: str, value: Any) -> bool:
    """PPC convention: an integral value on a continuous sensor is an error code, not a reading.

    PPC's raw `current_value` column is float64, so codes arrive as floats with no decimals (5021.0). Rule:
    - temps and wd_spd: any integral value is a code;
    - act_pwt: integral and non-zero is a code (0.0 kW is a real reading: a stopped turbine);
    - NaN / None are gaps, not codes; bools are never codes.
    """
    if sensor in CODE_SENSORS or value is None or isinstance(value, bool):
        return False
    try:
        v = float(value)
    except (TypeError, ValueError):
        return False
    if v != v or v in (float("inf"), float("-inf")):  # NaN / inf
        return False
    if v != round(v):
        return False
    if sensor == "act_pwt":
        return v != 0.0
    return True


def kmh_to_ms(v_kmh: Optional[float]) -> Optional[float]:
    return None if v_kmh is None else v_kmh * KMH_TO_MS


# Long-format telemetry table (one row per tag reading). See wind.md for the wide per-turbine view.
TELEMETRY_LONG_COLUMNS = {
    "ts": "timestamp[UTC]",
    "tag": "string",
    "value": "float64",
    "is_code": "bool",  # True if the source value was an integer error code (value holds the code)
}

# Raw PPC exports, exactly as delivered (reader maps them to the tables above/below).
RAW_TELEMETRY_COLUMNS = {
    "datetime_current_value": "int64",  # epoch milliseconds, UTC
    "current_value": "float64",  # reading, or an error code (integral value, see is_error_code)
    "tag_name": "string",  # WP<park>.TID<turbine>.<sensor>
}
RAW_METEO_COLUMNS = {
    "data_timestamp": "timestamp",  # UTC
    "fk_wtg_id": "string",  # turbine id (meteo is per TURBINE, not per park)
    "wind_speed_mean_5min": "string",  # m/s, arrives as a string -> cast to float64
}
RAW_TURBINE_COLUMNS = {
    "pk_wtg_id": "string",
    "fk_wtg_model_id": "string",
    "fk_park_id": "string",
    "fk_turbine_id": "string",
    "lat_y": "float64",
    "long_x": "float64",
}

# Normalised 5-min meteo wind, one series per turbine. wdir_deg is optional (ERA5 only, not in PPC data).
METEO_COLUMNS = {
    "ts": "timestamp[UTC]",
    "turbine_id": "string",
    "wind_ms": "float64",
    "wdir_deg": "float64?",
}

TURBINE_META_COLUMNS = {
    "turbine_id": "string",  # WP<park>.TID<turbine>
    "park_id": "string",  # WP<park>
    "model": "string",  # e.g. "generic-2.5MW"
    "rated_kw": "float64",
    "hub_height_m": "float64",
    "rotor_d_m": "float64",
    "lat": "float64",
    "lon": "float64",
    "county": "string",
    "power_unit": "string",  # unit of act_pwt for this turbine (mostly "kW")
    "power_scale": "float64",  # multiply act_pwt by this to get kW
}
