"""Replay a test week through the agent and write every row to SQLite (implementation: wattsentry/agent/replay.py).

Usage: python replay.py --policy baseline,sentinel --out $LAB_OUTPUT_DIR/wattsentry.sqlite [--world mock|sim]
       [--days 7] [--seed 0] [--mode imbalance] ...   (python replay.py --help lists every flag)
"""

from wattsentry.agent.replay import main

if __name__ == "__main__":
    main()
