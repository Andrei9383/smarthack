"""Year-replay benchmark of the trust ledger (#147 acceptance: < 10 s for a full year, all buckets).

    python -m sentinel.judge.bench_ledger [--sites 1000 --turbines 120 --days 365 --err-rows 8]

Synthetic outcome rows shaped like the grader's (§8 + bucket columns): 6 forecast witnesses/substitutes with
calibration + sharpness + `norm_err` rows per home-day, 2 home detectors and 2 turbine detectors with sparse
alarm-precision / miss-score rows, crew outcomes observed 1-5 days late. Prints LAB_METRIC t_ledger_* lines.
"""

from __future__ import annotations

import argparse
import time

import numpy as np
import polars as pl

from sentinel.infra.metrics import emit
from sentinel.judge.ledger import Ledger, Priors

FORECASTS = ("pvnet", "loadnet", "lgbm_pv", "lgbm_load", "pvlib", "persistence_load")
HOME_DET = ("sitehealth", "threshold_cusum")
WIND_DET = ("windnbm", "windmon_rules")


def synth(sites: int, turbines: int, days: int, err_rows: int, seed: int = 0) -> pl.DataFrame:
    rng = np.random.default_rng(seed)
    d0 = np.datetime64("2025-01-01").astype("datetime64[D]").astype(np.int64)
    seen, batt = rng.random(sites) < 0.7, rng.random(sites) < 0.3
    frames = []
    day = np.repeat(np.arange(days), sites) + d0
    aid = np.tile(np.arange(sites), days)
    n = len(day)
    for m in FORECASTS:
        ood = rng.random(n) < 0.05
        gappy = rng.random(n) < 0.03
        for s in ("calibration", "sharpness"):
            frames.append(pl.DataFrame({"model": m, "model_version": f"{m}-1", "asset_id": aid, "day": day,
                                        "score_name": s, "score": rng.beta(8, 2, n), "weight": 1.0,
                                        "observed_at": day + 1, "gated": (rng.random(n) < 0.02).astype(np.int8),
                                        "kind": "site", "seen": seen[aid], "battery": batt[aid],
                                        "gappy": gappy, "ood_high": ood}))
        k = err_rows
        frames.append(pl.DataFrame({"model": m, "model_version": f"{m}-1", "asset_id": np.repeat(aid, k),
                                    "day": np.repeat(day, k), "score_name": "norm_err",
                                    "score": rng.exponential(1.0, n * k), "weight": 1.0,
                                    "observed_at": np.repeat(day, k) + 1, "gated": np.int8(0), "kind": "site",
                                    "seen": np.repeat(seen[aid], k), "battery": np.repeat(batt[aid], k),
                                    "gappy": np.repeat(gappy, k), "ood_high": np.repeat(ood, k)}))
    for kind, models, count in (("site", HOME_DET, sites), ("turbine", WIND_DET, turbines)):
        tday = np.repeat(np.arange(days), count) + d0
        taid = np.tile(np.arange(count), days)
        for m in models:
            sel = rng.random(len(tday)) < 0.15  # graded detector days (alarms, audits, residual checks)
            nn = int(sel.sum())
            late = rng.integers(1, 6, nn)
            frames.append(pl.DataFrame({"model": m, "model_version": f"{m}-1", "asset_id": taid[sel],
                                        "day": tday[sel], "score_name": np.where(rng.random(nn) < 0.3,
                                                                                 "alarm_precision", "miss_score"),
                                        "score": (rng.random(nn) < 0.85).astype(float),
                                        "weight": np.where(rng.random(nn) < 0.2, 0.5, rng.uniform(1, 5, nn)),
                                        "observed_at": tday[sel] + late, "gated": np.int8(0), "kind": kind,
                                        "seen": rng.random(nn) < 0.7, "battery": rng.random(nn) < 0.3,
                                        "gappy": rng.random(nn) < 0.03, "ood_high": rng.random(nn) < 0.05}))
    df = pl.concat(frames, how="vertical_relaxed")
    return df.with_columns(pl.col("day").cast(pl.Int32).cast(pl.Date),
                           pl.col("observed_at").cast(pl.Int32).cast(pl.Date),
                           pl.col("asset_id").cast(pl.Utf8))


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--sites", type=int, default=1000)
    ap.add_argument("--turbines", type=int, default=120)
    ap.add_argument("--days", type=int, default=365)
    ap.add_argument("--err-rows", type=int, default=8)
    a = ap.parse_args()
    df = synth(a.sites, a.turbines, a.days, a.err_rows)
    n_score = df.filter(pl.col("score_name") != "norm_err").height
    print(f"rows: {df.height:,} ({n_score:,} score rows, {df.height - n_score:,} norm_err rows)")
    t = time.perf_counter()
    led = Ledger(df, Priors())
    t1 = time.perf_counter()
    rows = led.trust_rows()
    t2 = time.perf_counter()
    print(f"trust rows: {rows.height:,}; alerts: {led.alerts().height}")
    emit("t_ledger_build_s", t1 - t)
    emit("t_ledger_rows_s", t2 - t1)
    emit("t_ledger_year_s", t2 - t)


if __name__ == "__main__":
    main()
