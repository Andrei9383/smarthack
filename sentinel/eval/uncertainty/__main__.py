"""`python -m sentinel.eval.uncertainty` — end-to-end uncertainty demo on MockPredictor forecasts (CPU).

Prints LAB_METRIC coverage_10_90 / p10_hit_rate / p10_miss_kw (+ extras). Optional outputs: calibration JSONs,
tuned thresholds (never overwrites models/thresholds.json unless asked explicitly).
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from sentinel.eval.uncertainty.demo import run
from sentinel.eval.uncertainty.thresholds import save_thresholds


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(prog="python -m sentinel.eval.uncertainty", description=__doc__)
    ap.add_argument("--feeders", type=int, default=8)
    ap.add_argument("--sites", type=int, default=10)
    ap.add_argument("--cal-days", type=int, default=30)
    ap.add_argument("--test-days", type=int, default=10)
    ap.add_argument("--scenarios", type=int, default=200)
    ap.add_argument("--actual-draws", type=int, default=30, help="truth draws scored per feeder-day")
    ap.add_argument("--haircut", type=float, default=0.05)
    ap.add_argument("--inflate", type=float, default=1.5, help="truth spread / mock-claimed spread")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--calibration-out", type=Path, default=None, help="dir for pv/load calibration JSON")
    ap.add_argument("--thresholds-out", type=Path, default=None, help="write tuned thresholds JSON here")
    ap.add_argument("--report", type=Path, default=None, help="write a JSON report here")
    a = ap.parse_args(argv)

    res = run(n_feeders=a.feeders, n_sites=a.sites, cal_days=a.cal_days, test_days=a.test_days,
              n_scenarios=a.scenarios, actual_draws=a.actual_draws, haircut=a.haircut, inflate=a.inflate,
              seed=a.seed)
    m = res["metrics"]
    for k in ("coverage_10_90", "coverage_10_90_raw", "coverage_10_90_load", "pinball_pv", "pinball_pv_raw",
              "p10_hit_rate", "p10_miss_kw", "p10_miss_kw_max", "p10_shortfall_share", "n_steps",
              "committed_kwh_per_feeder_day", "nominal_kwh_per_feeder_day", "runtime_s"):
        v = m[k]
        print(f"LAB_METRIC {k}={v:.4f}" if isinstance(v, float) else f"LAB_METRIC {k}={v}")
    if a.calibration_out:
        for kind, cal in res["calibrations"].items():
            cal.save(a.calibration_out / f"calibration_{kind}.json")
    if a.thresholds_out:
        save_thresholds(res["thresholds"], a.thresholds_out)
    if a.report:
        a.report.parent.mkdir(parents=True, exist_ok=True)
        a.report.write_text(json.dumps({"metrics": m, "reports": res["reports"], "thresholds": res["thresholds"]},
                                       indent=2) + "\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
