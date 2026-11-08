"""Wind simulator: ERA5-driven telemetry for three RO parks in the PPC SCADA schema, with labelled faults.

    python -m wattsentry.sim.wind [--out DIR] [--force] [--no-plots]
    # -> $WATTSENTRY_WIND_DIR or $WATTSENTRY_CACHE/data-v{N}/wind/

Deterministic and cached by a digest of both configs, the data/raw inputs and the simulator source; bump
WIND_DATA_VERSION when the output changes meaning or shape. Inputs (committed): data/raw/weather.parquet (ERA5 at
each park point), osm_wind_turbines.parquet (real positions), entsoe_ro + ro_installed_capacity (capacity-factor
calibration only). Parks in parks.py (WP2 is held out of M6 training). Physics in generate.py: ERA5 interpolated to
15 min, AR(1) turbulence, seasonal shear to hub height, Jensen wakes, air density, class power curve; one ERA5 speed
factor calibrates the clean 2024 capacity factor to the real RO national one. Faults come from
wattsentry.sim.wind_faults; curtailment and icing are labelled as whole episodes, and M6 trains on the `effective`
span where power was actually cut (WATTSENTRY_WIND_LABELS=episode scores whole episodes).
"""

from .config import WIND_DATA_VERSION, WindSimConfig, default_out_dir

__all__ = ["WIND_DATA_VERSION", "WindSimConfig", "default_out_dir"]
