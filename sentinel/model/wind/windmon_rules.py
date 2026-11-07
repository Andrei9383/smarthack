"""Import shim for #41: WindMon's rules now live in sentinel.eval.baselines.wind.windmon_rules (card #24)."""

from sentinel.eval.baselines.wind.windmon_rules import (  # noqa: F401
    FROZEN_STEPS,
    RULE_CLASS,
    RULE_PRIORITY,
    RULES,
    TEMP_JUMP_C,
    TEMP_LIMITS,
    anomalies_table,
    raw_alert_count,
    windmon_classes,
    windmon_flags,
)
