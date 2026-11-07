"""Train + evaluate M3 DemandNet (card #44). `train.py --model demandnet` (lab run type train_cpu) calls `main()`.

  python -m sentinel.demand [--data-dir data/raw] [--out $LAB_OUTPUT_DIR] [--counties B:0.11,CT:0.035] [--quick]

1. v2 on national RO load: ENTSO-E RO total load if #9's parquet is in data/raw ("real RO"), else a synthetic
   RO-shaped series ("synthetic RO"). DemandNet vs persistence (day, week) and national-shape scaling on the test
   split. Headline LAB_METRICs come from this zone.
2. v2 per county on synthetic county series (always "synthetic RO" — county load is not published).
3. v1 demo: MockPredictor load scenarios aggregated feeder -> locality -> county -> zone_demand rows + actions.
Writes model files, demand_report.json, zone_demand_v1_rows.json and zone_demand_v1_actions.jsonl to the output dir.
"""

from __future__ import annotations

import argparse
import json
import os
import time
from pathlib import Path
from typing import Any

import numpy as np

from .aggregate import LABEL_SYNTHETIC, aggregate_scenarios, mock_load_scenarios, zone_demand_actions, zone_demand_rows
from .data import load_national, synthetic_county_split
from .model import DemandNet, NationalShape, default_split, evaluate_zone

DEFAULT_COUNTIES = "B:0.11,CT:0.035,IS:0.03,VS:0.012"  # rough shares of RO consumption (synthetic split)


def _fmt(v: Any) -> str:
    return f"{v:.6g}" if isinstance(v, float) else str(v)


def _metric(name: str, value: Any) -> None:
    print(f"LAB_METRIC {name}={_fmt(value)}", flush=True)


def demo_hierarchy() -> tuple[list[dict], list[dict]]:
    """Small RO hierarchy for the v1 demo: 2 counties, 5 localities, 10 feeders, 20 homes each."""
    layout = {"CT": ["Constanța", "Medgidia", "Cernavodă"], "VS": ["Vaslui", "Bârlad"]}
    archetypes = ["base", "heat_pump", "electric_heating", "ac", "base"]
    sites, feeders = [], []
    n = 0
    for county, locs in layout.items():
        for li, loc in enumerate(locs):
            for k in range(2):
                fid = f"F{county}{li}{k}"
                g = 55.0 if (county, li, k) == ("CT", 1, 0) else 80.0 + 15.0 * k  # one congested feeder
                feeders.append({"feeder_id": fid, "locality": loc, "county": county, "g_max_kw": g,
                                "g_conservative_kw": 0.8 * g, "tau": 1.0})
                for _ in range(20):
                    sites.append({"site_id": f"S{n:04d}", "feeder_id": fid, "locality": loc, "county": county,
                                  "archetype": archetypes[n % len(archetypes)]})
                    n += 1
    return sites, feeders


def run_v1_demo(out: Path, t0: str, run_id: str, n_scenarios: int = 200) -> dict[str, Any]:
    sites, feeders = demo_hierarchy()
    scen = mock_load_scenarios(sites, t0, n_scenarios, seed=0, common=0.1)
    agg = aggregate_scenarios(scen, sites, feeders, t0=t0, label=LABEL_SYNTHETIC,
                              model_version="demandnet-v1-agg-mock")
    rows = zone_demand_rows(agg, run_id)
    actions = zone_demand_actions(agg, run_id=run_id)
    (out / "zone_demand_v1_rows.json").write_text(json.dumps(rows))
    (out / "zone_demand_v1_actions.jsonl").write_text("\n".join(a.to_json() for a in actions) + "\n")
    # how much would summing site quantiles overstate the zone P90? (the reason for scenario sums)
    site_q90 = np.quantile(scen, 0.9, axis=0)  # [N, T] kW
    overstate = {}
    for z in agg.level("county"):
        idx = [i for i, s in enumerate(sites) if s["county"] == z.zone_id]
        naive = site_q90[idx].sum(axis=0) / 1000.0
        overstate[z.zone_id] = round(float(naive.max() / max(z.q90_mw)) - 1, 4)
    return {
        "label": "synthetic RO (MockPredictor load scenarios)",
        "t0": t0,
        "n_sites": len(sites),
        "n_scenarios": n_scenarios,
        "zones": {z.zone_id: {"level": z.level, "peak_hour_local": z.peak_hour_local, "peak_mw": round(z.peak_mw, 4),
                              "p90_peak_mw": round(z.p90_peak_mw, 4), "capacity_mw": round(z.capacity_mw, 4),
                              "stress": round(z.stress, 4)} for z in agg.zones},
        "sum_of_site_q90_overstates_zone_q90_peak_by": overstate,
        "n_rows": len(rows),
        "n_actions": len(actions),
    }


def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser(prog="python -m sentinel.demand", description="Train + evaluate M3 DemandNet")
    ap.add_argument("--model", default="demandnet", help="ignored (train.py dispatch compatibility)")
    ap.add_argument("--data-dir", default=None, help="where #9's raw parquet lives (default <repo>/data/raw)")
    ap.add_argument("--out", default=os.environ.get("LAB_OUTPUT_DIR", "./.lab-out"))
    ap.add_argument("--counties", default=DEFAULT_COUNTIES, help="CODE:share,... synthetic county split ('' = none)")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--rounds", type=int, default=600, help="max boosting rounds (early stopping on validation)")
    ap.add_argument("--quick", action="store_true", help="fewer rounds, no counties (smoke run)")
    ap.add_argument("--v1-t0", default="2025-01-14T22:00:00Z", help="v1 demo day start (UTC; local midnight)")
    ap.add_argument("--no-synthetic", action="store_true", help="fail instead of falling back to synthetic RO")
    args, _unknown = ap.parse_known_args(argv)

    t_start = time.time()
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    run_id = os.environ.get("LAB_RUN_ID", "local")
    rounds = 150 if args.quick else args.rounds
    version = f"demandnet-v2-{run_id}"

    national = load_national(args.data_dir, allow_synthetic=not args.no_synthetic, seed=args.seed)
    print(f"data: {national.source} [{national.label}; temperature {national.temp_label}] "
          f"{national.ts[0]} .. {national.ts[-1]} ({np.isfinite(national.load_mw).sum()} hourly values)", flush=True)
    split = default_split(national.ts)
    shape = NationalShape.fit(national, split["train"])
    model = DemandNet(num_boost_round=rounds, seed=args.seed, model_version=version, shape=shape)
    res_nat = evaluate_zone(national, model, shape=shape, split=split)

    res_counties: dict[str, Any] = {}
    if args.counties and not args.quick:
        shares = {c.split(":")[0]: float(c.split(":")[1]) for c in args.counties.split(",") if ":" in c}
        for cs in synthetic_county_split(national, shares, seed=args.seed + 7):
            res_counties[cs.zone_id] = evaluate_zone(cs, model, shape=shape, split=split)

    v1 = run_v1_demo(out, args.v1_t0, run_id)
    files = model.save(out)

    label = national.label
    print(f"# headline metrics: zone RO, {label} (test split: {res_nat['split']})", flush=True)
    for k in ("demand_nmae", "demand_peak_hour_err", "demand_pinball_q50", "demand_coverage_10_90",
              "baseline_persistence_day_nmae", "baseline_persistence_day_peak_hour_err",
              "baseline_persistence_week_nmae", "baseline_persistence_week_peak_hour_err",
              "baseline_shape_scaling_nmae", "baseline_shape_scaling_peak_hour_err", "skill_vs_best_baseline"):
        _metric(k, res_nat[k])
    _metric("demand_real_ro", 1 if label == "real RO" else 0)
    if res_counties:
        _metric("demand_county_nmae_mean", float(np.mean([r["demand_nmae"] for r in res_counties.values()])))
        _metric("baseline_county_best_nmae_mean",
                float(np.mean([r[f"baseline_{r['best_baseline']}_nmae"] for r in res_counties.values()])))

    report = {
        "card": 44,
        "model": "demandnet",
        "model_version": version,
        "run_id": run_id,
        "label": label,
        "note": ("metrics on ENTSO-E RO total load" if label == "real RO" else
                 "NO ENTSO-E file found: synthetic RO-shaped series, exercises the code path only"),
        "national": res_nat,
        "counties": res_counties,
        "counties_label": "synthetic RO (county split of the national series)",
        "v1_demo": v1,
        "files": files + ["zone_demand_v1_rows.json", "zone_demand_v1_actions.jsonl"],
        "seconds": round(time.time() - t_start, 1),
    }
    (out / "demand_report.json").write_text(json.dumps(report, indent=2, default=str))
    print(f"wrote {out / 'demand_report.json'} ({report['seconds']} s)", flush=True)


if __name__ == "__main__":
    main()
