"""Wind fault injector: exact-onset labelled faults on top of wattsentry.sim.wind's clean telemetry.

    from wattsentry.sim.wind.generate import generate_clean
    from wattsentry.sim.wind_faults import FaultConfig, inject
    res = inject(generate_clean(), FaultConfig())     # res.sensors, res.err, res.episodes (fault_labels rows)

Classes = wattsentry.contracts.envelope.WindAnomaly (see config.KIND_CLASS); rates per turbine-year in FaultConfig.
"""

from .config import KIND_CLASS, KINDS, FaultConfig
from .inject import FaultResult, inject

__all__ = ["FaultConfig", "FaultResult", "KINDS", "KIND_CLASS", "inject"]
