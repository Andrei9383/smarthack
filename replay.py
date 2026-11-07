"""Replay a test week through the agent and write every row to SQLite (card #17).

Usage: python replay.py --policy baseline,sentinel --out $LAB_OUTPUT_DIR/sentinel.sqlite [--world mock|sim]
       [--days 7] [--seed 0] [--resolve-hourly] [--fallback all|planner,...] [--full] [--no-strict]
Default: demo-size DB (< 50 MB) and strict schema mode (raises on fields #38 cannot hold, see audit.ALLOWED_DROPS).
Implementation: sentinel/agent/replay.py. Prints LAB_METRIC peak_reduction, minutes_over_limit (per policy),
p10_hit_rate, comfort_breaches, lei_per_home, alerts_per_turbine_day.
"""

from sentinel.agent.replay import main

if __name__ == "__main__":
    main()
