"""Wind baselines: WindMon rules and the shim (stand-in synth data)."""

import pytest

from wattsentry.eval.baselines.wind import windmon_rules
from wattsentry.model.wind.metrics import event_metrics
from wattsentry.model.wind.synth import SynthConfig, generate


@pytest.fixture(scope="module")
def data():
    return generate(SynthConfig(days=60, scale=0.1))


def test_shim_reexports_rules():
    from wattsentry.model.wind import windmon_rules as shim

    assert shim.windmon_flags is windmon_rules.windmon_flags and shim.RULES == windmon_rules.RULES


def test_event_metrics_run_on_windmon(data):
    d = data
    wm = windmon_rules.windmon_classes(windmon_rules.windmon_flags(d.sensors, d.err, d.rated_kw))
    m = event_metrics(d.labels, wm, full=False)
    assert 0 <= m["wind_precision"] <= 1 and 0 <= m["wind_recall"] <= 1


def test_anomalies_table(data):
    d = data
    f = windmon_rules.windmon_flags(d.sensors, d.err, d.rated_kw)
    df = windmon_rules.anomalies_table(f, d.ts, d.turbine_ids)
    assert df.height == windmon_rules.raw_alert_count(f) and set(df["rule"].unique()) <= set(windmon_rules.RULES)
