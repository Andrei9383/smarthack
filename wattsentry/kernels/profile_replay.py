"""Profile replay.py with pyinstrument and print the share of wall time per hotspot (card #93).

    python -m wattsentry.kernels.profile_replay [--policy baseline,sentinel] [replay args ...]

Needs ``pip install pyinstrument`` (dev only, not in requirements.txt). Prints ``LAB_METRIC t_wall_s`` and
``LAB_METRIC pct_<name>`` for each function in HOTSPOTS (inclusive time, recursion counted once). Rule for a new
Rust kernel (team/infra.md): a Python loop that is >= 10% of wall and gets >= 3x from Rust.
"""

from __future__ import annotations

import sys

HOTSPOTS = ("solve_lp", "solve_via_data", "get_problem_data", "_flush_table", "conv", "format_time", "qualify_many",
            "_assess_site", "rel_width", "predict", "site_day", "commit_fn", "rollout", "default_components")


def shares(root, names=HOTSPOTS) -> dict[str, float]:
    tot: dict[str, float] = {}

    def walk(f, active: frozenset) -> None:
        hit = f.function in names and f.function not in active
        if hit:
            tot[f.function] = tot.get(f.function, 0.0) + f.time
        for c in f.children:
            walk(c, active | {f.function} if hit else active)

    walk(root, frozenset())
    return tot


def main(argv: list[str] | None = None) -> int:
    from pyinstrument import Profiler

    from wattsentry.agent.replay import main as replay

    args = list(sys.argv[1:] if argv is None else argv)
    with Profiler(interval=0.001) as prof:
        replay(args)
    root = prof.last_session.root_frame()
    wall = root.time
    print(f"LAB_METRIC t_wall_s={wall:.6g}")
    for k, v in sorted(shares(root).items(), key=lambda kv: -kv[1]):
        print(f"# {v:8.2f}s {100 * v / wall:5.1f}%  {k}")
        print(f"LAB_METRIC pct_{k}={100 * v / wall:.4g}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
