"""PV fault injector (card #12): labelled faults with exact onsets on top of the fleet sim, or on real PVDAQ series.

    python -m wattsentry.sim.faults            # default fleet -> ~/.cache/wattsentry/data-vN/faults-<digest>/
    from wattsentry.sim.faults import inject   # DataFrame API (PVDAQ: ts, site_id, pv_ac_w + sites lat/lon)

Classes = contracts PvFault (minus healthy); labels = data.md fault_labels. See kinds.py for each signature.
"""

from .dataset import inject_dataset
from .inject import FaultConfig, inject, inject_batch, labels_frame
from .kinds import APPLY, INV_OVERVOLTAGE

__all__ = ["APPLY", "FaultConfig", "INV_OVERVOLTAGE", "inject", "inject_batch", "inject_dataset", "labels_frame"]
