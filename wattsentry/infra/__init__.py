"""ML run helpers: `LAB_METRIC` lines, stage timing, ONNX export."""
from .metrics import emit
from .timing import stage

__all__ = ["emit", "stage"]
