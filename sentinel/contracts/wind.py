"""Wind telemetry contract: PPC HackITall schema (plan §3.4, decisions 14:15). Spec: team/contracts/wind.md.

Tags look like `WP<park>.TID<turbine>.<sensor>`, e.g. `WP1.TID07.act_pwt`. Power in kW, wind in km/h, temps in °C.
Integer values on a numeric sensor are error codes, not measurements. Real PPC data must drop in unchanged, so the
sensor names below are the only place to rename if PPC's sample differs.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Any, Optional

TAG_RE = re.compile(r"^WP(?P<park>[A-Za-z0-9]+)\.TID(?P<turbine>[A-Za-z0-9]+)\.(?P<sensor>[A-Za-z0-9_]+)$")

# sensor -> (unit, description). Names beyond act_pwt / wd_spd follow WindMon; confirm against PPC's sample.
SENSORS: dict[str, tuple[str, str]] = {
    "act_pwt": ("kW", "active power at the turbine"),
    "wd_spd": ("km/h", "nacelle anemometer wind speed"),
    "wd_dir": ("deg", "wind direction"),
    "nac_dir": ("deg", "nacelle position (yaw)"),
    "rot_spd": ("rpm", "rotor speed"),
    "pitch": ("deg", "blade pitch angle"),
    "conv_tmp": ("degC", "converter temperature"),
    "gbx_tmp": ("degC", "gearbox oil / bearing temperature"),
    "gen_brg_tmp": ("degC", "generator bearing temperature"),
    "trf_tmp": ("degC", "transformer temperature"),
    "amb_tmp": ("degC", "ambient temperature"),
    "status": ("code", "turbine state / error code (integer)"),
}

# Sensors whose values are always integer codes (never flagged as error codes).
CODE_SENSORS = {"status"}

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
    turbine = f"{turbine:02d}" if isinstance(turbine, int) else turbine
    return f"WP{park}.TID{turbine}.{sensor}"


def is_error_code(sensor: str, value: Any) -> bool:
    """PPC convention: an integer on a continuous sensor is an error code, not a reading.

    Ints (and floats with no fractional part that arrive typed as int) count; NaN / None are gaps, not codes.
    """
    if sensor in CODE_SENSORS or value is None or isinstance(value, bool):
        return False
    return isinstance(value, int)


def kmh_to_ms(v_kmh: Optional[float]) -> Optional[float]:
    return None if v_kmh is None else v_kmh * KMH_TO_MS


# Long-format telemetry table (one row per tag reading). See wind.md for the wide per-turbine view.
TELEMETRY_LONG_COLUMNS = {
    "ts": "timestamp[UTC]",
    "tag": "string",
    "value": "float64",
    "is_code": "bool",  # True if the source value was an integer error code (value holds the code)
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
}
