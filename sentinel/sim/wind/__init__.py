"""Wind simulator (card #22): ERA5-driven telemetry for three RO parks in the PPC SCADA schema.

    python -m sentinel.sim.wind            # -> ~/.cache/sentinel/data-v{N}/wind/ (or $SENTINEL_WIND_DIR)

See README.md in this folder. Faults come from sentinel.sim.wind_faults (card #23).
"""

from .config import WIND_DATA_VERSION, WindSimConfig, default_out_dir

__all__ = ["WIND_DATA_VERSION", "WindSimConfig", "default_out_dir"]
