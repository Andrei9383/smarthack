"""M3 DemandNet (card #44): zone demand P10/P50/P90, peak hour and stress, feeder -> locality -> county.

v1 `aggregate_scenarios` (scenario sums of household load, no new model) -> `zone_demand_rows` / `zone_demand_actions`.
v2 `DemandNet` (LightGBM quantile per county) with RO calendar features; `python -m wattsentry.demand` trains it.
"""

from .aggregate import (
    LEVELS,
    ROW_FIELDS,
    DemandAggregate,
    ZoneDemand,
    aggregate_scenarios,
    build_hierarchy,
    mock_load_scenarios,
    scenarios_from_quantiles,
    zone_demand_actions,
    zone_demand_rows,
)
from .calendar import bucharest_offset_hours, calendar_features, orthodox_easter, ro_public_holidays
from .data import LoadSeries, load_national
from .model import DemandNet, NationalShape, evaluate_zone

__all__ = [
    "LEVELS",
    "ROW_FIELDS",
    "DemandAggregate",
    "ZoneDemand",
    "aggregate_scenarios",
    "build_hierarchy",
    "mock_load_scenarios",
    "scenarios_from_quantiles",
    "zone_demand_actions",
    "zone_demand_rows",
    "bucharest_offset_hours",
    "calendar_features",
    "orthodox_easter",
    "ro_public_holidays",
    "LoadSeries",
    "load_national",
    "DemandNet",
    "NationalShape",
    "evaluate_zone",
]
