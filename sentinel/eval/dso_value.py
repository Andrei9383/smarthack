"""DSO value in lei + GridTwin (M8) in the loop (card G3 #161, team/features/grid.md steps 3-4).

    python -m sentinel.eval.dso_value [--world sim|mock] [--days 7] [--out DIR] [--skip-replay]

Lab (needs the fleet / windows / grid-twin caches, same chain as `make replay-demo`):
    lab submit 161 --run replay --cmd "sh rust/build.sh status && .venv/bin/python -m sentinel.data.windows && \
        .venv/bin/python -m sentinel.gridtwin.run && .venv/bin/python -m sentinel.eval.dso_value --world sim"

Deferred capex (step 3), per feeder that Sentinel lets take more prosumers:
    value = cost x (1 - 1 / (1 + r)^years)
    years = ln((N + HC_sentinel) / (N + HC_baseline)) / ln(1 + growth)
i.e. how many years of prosumer growth (compound, `growth` per year on the N homes the feeder has today) the extra
hosting capacity absorbs before the feeder needs the same reinforcement. Assumptions, all stated in the output:
- cost: REINFORCEMENT_COST_LEI, a PLACEHOLDER ([COST POST / CABLU], to be replaced by the GIS mentor's figure);
- r = 7 %, growth = GROWTH_RATE (assumed);
- HC_*: G1 #159's hosting capacity, read in `hosting_capacity()` only. Until G1 merges that function returns a STUB,
  and every lei number built on it is labelled so.
DEO's public estimate (other operator, not PPC / Retele Electrice) gives an order-of-magnitude cross-check per
prosumer. Every printed number carries a label (synthetic RO / proxy / other-operator) and the run id.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import time
from pathlib import Path
from typing import Any, Mapping, Optional

DISCOUNT_RATE = 0.07  # r, assumption (grid.md step 3)
GROWTH_RATE = 0.15  # prosumers per feeder per year, compound; assumption (RO grew far faster in 2023-24, slowing)
# PLACEHOLDER [COST POST / CABLU]: one LV reinforcement (new 20/0.4 kV substation or parallel LV cable). Not a PPC
# figure; ask the GIS mentor. Context only: Retele Electrice invested ~1.4 bn lei in 2025 incl. ~1,000 new and 440
# modernised substations plus 2 M smart meters (Financial Intelligence, Apr 2026).
REINFORCEMENT_COST_LEI = 500_000.0
REINFORCEMENT_COST_IS_PLACEHOLDER = True
# Other operator: Distributie Energie Oltenia, Feb 2024 (Economica): ~400 M EUR to prepare the grid for 100k-150k
# prosumers -> 2,667-4,000 EUR per new prosumer. EUR/RON assumed 4.97.
DEO_EUR = 400e6
DEO_PROSUMERS = (100_000, 150_000)
EUR_RON = 4.97

LABEL_SYN = "synthetic RO"
LABEL_PROXY = "proxy"
LABEL_DEO = "other-operator (DEO), not PPC"

# STUB until G1 #159 lands: hosting capacity (homes that can be added before minutes over the limit exceed the
# tolerance) per feeder, baseline rule vs Sentinel. Shaped like G1's planned output; numbers are NOT measured.
_HC_STUB = {"baseline": 0, "sentinel": 8}


def deferred_value(cost: float, years: float, r: float = DISCOUNT_RATE) -> float:
    """Present value of pushing a `cost` back by `years`: cost x (1 - 1/(1+r)^years)."""
    if years <= 0 or cost <= 0:
        return 0.0
    return cost * (1.0 - math.exp(-years * math.log1p(r)))  # = 1/(1+r)^years, no overflow for huge years


def years_deferred(n_homes: int, hc_baseline: float, hc_sentinel: float, growth: float = GROWTH_RATE) -> float:
    """Years of compound prosumer growth the extra hosting capacity absorbs (0 if Sentinel adds none)."""
    if hc_sentinel <= hc_baseline or growth <= 0:
        return 0.0
    n = max(1.0, float(n_homes))
    return math.log((n + hc_sentinel) / (n + hc_baseline)) / math.log(1.0 + growth)


def hosting_capacity(world=None) -> dict[str, dict[str, Any]]:
    """THE one read of G1 #159 (hosting capacity per feeder). STUB until G1 merges: congested feeders get the
    constant _HC_STUB. Replace the body with G1's output on `lab sync`; callers do not change."""
    feeders = {}
    if world is not None:
        for fid, f in world.feeders.items():
            if f.congested:
                feeders[fid] = len(f.site_ids)
    else:
        feeders = {"F7": 24}
    return {fid: {"n_homes": n, "hc_baseline": _HC_STUB["baseline"], "hc_sentinel": _HC_STUB["sentinel"],
                  "source": "STUB pending G1 #159", "run_id": None} for fid, n in feeders.items()}


def deo_cost_per_prosumer_lei() -> tuple[float, float]:
    lo, hi = (DEO_EUR / n * EUR_RON for n in reversed(DEO_PROSUMERS))
    return lo, hi


def deferred_capex(hc: Mapping[str, Mapping[str, Any]], cost: float = REINFORCEMENT_COST_LEI,
                   r: float = DISCOUNT_RATE, growth: float = GROWTH_RATE) -> dict[str, Any]:
    rows = []
    for fid, h in sorted(hc.items()):
        y = years_deferred(h["n_homes"], h["hc_baseline"], h["hc_sentinel"], growth)
        gain = max(0.0, float(h["hc_sentinel"]) - float(h["hc_baseline"]))
        lo, hi = deo_cost_per_prosumer_lei()
        rows.append({"feeder_id": fid, "n_homes": h["n_homes"], "hc_baseline": h["hc_baseline"],
                     "hc_sentinel": h["hc_sentinel"], "hosting_capacity_gain": gain, "years_deferred": round(y, 3),
                     "deferred_capex_lei": round(deferred_value(cost, y, r), 0),
                     "deo_basis_lei": [round(deferred_value(gain * c, y, r), 0) for c in (lo, hi)],
                     "hc_source": h.get("source"), "hc_run_id": h.get("run_id")})
    stub = any("STUB" in str(x["hc_source"]) for x in rows)
    return {
        "deferred_capex_lei": round(sum(x["deferred_capex_lei"] for x in rows), 0),
        "deferred_capex_lei_deo_basis": [round(sum(x["deo_basis_lei"][i] for x in rows), 0) for i in (0, 1)],
        "hosting_capacity_gain": sum(x["hosting_capacity_gain"] for x in rows),
        "feeders": rows,
        "assumptions": {"reinforcement_cost_lei": cost, "cost_is_placeholder": REINFORCEMENT_COST_IS_PLACEHOLDER,
                        "cost_note": "PLACEHOLDER [COST POST / CABLU]: ask the GIS mentor",
                        "discount_rate": r, "growth_rate": growth, "eur_ron": EUR_RON,
                        "deo_cost_per_prosumer_lei": [round(c) for c in deo_cost_per_prosumer_lei()],
                        "hosting_capacity": "STUB pending G1 #159" if stub else "G1 #159"},
        "label": f"{LABEL_PROXY} (placeholder cost{', STUB hosting capacity' if stub else ''})",
        "label_deo": LABEL_DEO,
    }


# ============================================================================================ M8 inputs
def twin_outputs(world=None):
    """(TwinResult.sites, fleet truth sites, source) for the recovery rate: the grid-twin cache that
    `python -m sentinel.gridtwin.run` writes (SENTINEL_GRIDTWIN or the default dir), else the twin in-process on
    the world's fleet. None in the mock world (its M8 is a stand-in, not a result)."""
    import pandas as pd

    from sentinel.gridtwin.run import default_out

    d = Path(os.environ.get("SENTINEL_GRIDTWIN") or default_out())
    if (d / "sites.parquet").exists() and (d / "metrics.json").exists():
        fleet = Path(json.loads((d / "metrics.json").read_text())["fleet_dir"])
        return pd.read_parquet(d / "sites.parquet"), pd.read_parquet(fleet / "sites.parquet"), f"cache {d}"
    fleet = getattr(world, "windows", None)
    if fleet is None:
        return None
    idx = json.loads((Path(fleet) / "index.json").read_text())
    fleet_dir = Path(fleet).parent / idx.get("source_key", {}).get("fleet", "")
    from sentinel.gridtwin.twin import load_voltages, run_twin

    s, v = load_voltages(fleet_dir)
    res = run_twin(s, v, pd.read_parquet(fleet_dir / "feeders.parquet"))
    return res.sites, s, f"in-process on {fleet_dir}"


def case_world(world, twin_sites, truth, days: int):
    """SimWorld holding the hero feeder plus the other feeder of its best correct M8 feeder move (the demo subset
    may not contain it). Returns (world, note); the input world if it already holds such a case."""
    from sentinel.agent.world import HERO_FEEDER_ID, SimWorld
    from sentinel.gridtwin.inloop import pick_case, validated_moves

    truth_l = {s: w.feeder_id for s, w in world.sites.items()}
    if pick_case(validated_moves(world.gis_flags(), world.week_start), truth_l, HERO_FEEDER_ID):
        return world, "demo feeder subset"
    hero = world.hero_source
    s = twin_sites.merge(truth[["site_id", "feeder_id"]], on="site_id")
    c = s[s.feeder_flag & (s.likely_feeder_id == s.feeder_id) & ((s.gis_feeder_id == hero) | (s.feeder_id == hero))]
    if c.empty:
        return world, f"no correct M8 feeder move touches the hero feeder ({hero})"
    c = c.assign(on_hero=c.gis_feeder_id == hero).sort_values(["on_hero", "feeder_score"], ascending=False)
    r = c.iloc[0]
    other = r.feeder_id if r.gis_feeder_id == hero else r.gis_feeder_id
    w = SimWorld(days=days, feeders=[hero, other], hero=hero, wind=False)
    return w, f"hero {hero} + {other} (fleet ids)"


# ============================================================================================ main
def _emit(name: str, value: float, label: str, run_id: str) -> None:
    from sentinel.infra.metrics import emit

    emit(name, value)
    print(f"#   {name}: {label}, run {run_id}")


def main(argv: Optional[list[str]] = None) -> dict[str, Any]:
    ap = argparse.ArgumentParser(prog="python -m sentinel.eval.dso_value", description=__doc__.split("\n\n")[0])
    ap.add_argument("--world", default="sim", choices=["sim", "mock"])
    ap.add_argument("--days", type=int, default=7)
    ap.add_argument("--out", type=Path, default=Path(os.environ.get("LAB_OUTPUT_DIR", ".")))
    ap.add_argument("--skip-replay", action="store_true", help="only the recovery rate and the lei (no replays)")
    a = ap.parse_args(argv)
    t0 = time.perf_counter()
    run_id = os.environ.get("LAB_RUN_ID") or "local"
    rid = f"{a.world}#{run_id}"
    from sentinel.agent.loop import default_components
    from sentinel.agent.world import HERO_DAY, make_world
    from sentinel.gridtwin import inloop

    world = make_world(a.world, days=a.days, **({"wind": False} if a.world == "sim" else {}))
    label = LABEL_SYN if a.world == "sim" else "mock world (dev only, not a result)"
    out: dict[str, Any] = {"run_id": run_id, "world": a.world, "label": label, "adapter": inloop.ADAPTER}

    # (a) recovery of the corrupted GIS records
    tw = twin_outputs(world) if a.world == "sim" else None
    if tw is not None:
        out["recovery"] = {**inloop.recovery(tw[0], tw[1]), "source": tw[2]}
        _emit("gis_recovery_rate", out["recovery"]["gis_recovery_rate"], label, rid)
        print(f"# recovery: {json.dumps(out['recovery'])}")

    comps = default_components(world)
    if not a.skip_replay:
        # (b) minutes over the limit with vs without the tau limit
        te = inloop.tau_effect(world, comps, run_id)
        out["tau_effect"] = te
        for k in ("minutes_over_g_max_tau", "minutes_over_g_max_no_tau", "minutes_over_g_tau_tau",
                  "minutes_over_g_tau_no_tau", "kwh_over_g_max_tau", "kwh_over_g_max_no_tau", "lei_cost_delta_tau"):
            if k in te:
                _emit(k, te[k], label, rid)
        print(f"# tau effect: tau {te['tau']}  G_max {te['g_max_kw']}  G_tau {te['g_tau_kw']}")
        for arm, v in te["arms"].items():
            print(f"#   {arm}: vs G_max {v['vs_g_max']}  vs G_tau {v['vs_g_tau']}  cost {v['lei_cost']} lei")
        # (c) one home M8 moves, and the hero feeder's plan diff
        cw, note = (case_world(world, tw[0], tw[1], a.days) if tw is not None else (world, "mock"))
        cc = comps if cw is world else default_components(cw)
        case = inloop.moved_home_case(cw, cc, d=HERO_DAY)
        case["world_note"] = note
        out["moved_home"] = case
        if case.get("found"):
            print(f"# moved home: {case['site_id']} GIS {case['recorded_feeder']} -> M8 {case['suggested_feeder']} "
                  f"(truth {case['true_feeder']}, score {case['score']}, {case['message']}), {case['local_date']}")
            print(f"#   {case['hero']} plan on GIS membership: physical {case['physical_gis']}")
            print(f"#   {case['hero']} plan on M8 membership:  physical {case['physical_m8']}")
            for r in case["plan_diff"]:
                print(f"#   diff {r}")
            _emit("moved_home_minutes_over_gis", case["physical_gis"]["minutes_over"], label, rid)
            _emit("moved_home_minutes_over_m8", case["physical_m8"]["minutes_over"], label, rid)
        else:
            print(f"# moved home: none ({case.get('note')}; {note})")

    # deferred capex (lei), on G1's hosting capacity (STUB until #159 merges)
    hc = hosting_capacity(world)
    cap = deferred_capex(hc)
    out["dso_value"] = cap
    _emit("deferred_capex_lei", cap["deferred_capex_lei"], cap["label"], rid)
    print(f"#   deferred_capex_lei DEO basis {cap['deferred_capex_lei_deo_basis']} lei: {LABEL_DEO}")
    for r in cap["feeders"]:
        print(f"#   {r}")
    print(f"LAB_METRIC dso_value_seconds={time.perf_counter() - t0:.1f}")
    a.out.mkdir(parents=True, exist_ok=True)
    (a.out / "dso_value.json").write_text(json.dumps(out, default=str, ensure_ascii=False, indent=1))
    print(f"wrote {a.out / 'dso_value.json'}")
    return out


if __name__ == "__main__":
    main()
