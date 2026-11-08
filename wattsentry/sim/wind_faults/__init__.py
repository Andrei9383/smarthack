"""Wind fault injector (card #23): exact-onset labelled faults on top of wattsentry.sim.wind's clean telemetry.

    from wattsentry.sim.wind.generate import generate_clean
    from wattsentry.sim.wind_faults import FaultConfig, inject
    res = inject(generate_clean(), FaultConfig())     # res.sensors, res.err, res.episodes (fault_labels rows)

`python -m wattsentry.sim.wind` runs both and writes the dataset; `python -m wattsentry.sim.wind_faults` prints the
label summary of the cached dataset. Classes = wattsentry.contracts.envelope.WindAnomaly (see config.KIND_CLASS).
"""

from .config import KIND_CLASS, KINDS, FaultConfig
from .inject import FaultResult, inject

__all__ = ["FaultConfig", "FaultResult", "KINDS", "KIND_CLASS", "inject"]
