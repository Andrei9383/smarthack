"""wattsentry.kernels.profile_replay.shares: inclusive time per function, recursion counted once (card #93)."""
from types import SimpleNamespace as F

from wattsentry.kernels.profile_replay import shares


def test_shares_counts_recursion_once():
    leaf = F(function="conv", time=1.0, children=[])
    inner = F(function="_flush_table", time=2.0, children=[leaf])
    outer = F(function="_flush_table", time=3.0, children=[inner, F(function="other", time=0.5, children=[])])
    root = F(function="<module>", time=4.0, children=[outer, F(function="conv", time=0.5, children=[])])
    assert shares(root) == {"_flush_table": 3.0, "conv": 1.5}
