"""Replay a test week through the agent and write every row to SQLite (stub; #17 fills it in).

Usage: python replay.py --policy baseline,sentinel --out $LAB_OUTPUT_DIR/sentinel.sqlite
"""
import argparse
import os
import pathlib


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--policy", default="baseline,sentinel", help="comma-separated policies to replay")
    ap.add_argument("--out", default=os.path.join(os.environ.get("LAB_OUTPUT_DIR", "."), "sentinel.sqlite"))
    args = ap.parse_args()

    pathlib.Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    policies = [p for p in args.policy.split(",") if p]
    print(f"replay stub: policies={policies} out={args.out} (no rows written until #17)")
    for name in ("peak_reduction", "violations_avoided", "commitment_hit_rate"):
        print(f"LAB_METRIC {name}=0")


if __name__ == "__main__":
    main()
