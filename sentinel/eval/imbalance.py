"""Avoidable imbalance cost in lei: one number per policy, shared by Imbalance (I6, I8) and Wind (W3, W5) (card #158).

Spec I §7 (feature-imbalance-prosumatori.md, "Pasul 7"), sign convention for wind from spec W §4.

All inputs are per 15-min step: quantities in MWh, prices in lei/MWh. Arrays broadcast (e.g. scenarios x steps).
Sign convention = **consumption** (the supplier buys):

    actual    = what the portfolio really took from the grid (import - export), MWh
    bought_da = what was bought day-ahead on PZU, MWh
    traded_id = what was bought (+) or sold (-) on the intraday market (PI), MWh
    dev       = actual - bought_da - traded_id      > 0: short (bought too little), < 0: long (too much)

What the supplier pays: PZU on ``bought_da``, PI on ``traded_id``; the deficit is bought at the imbalance price
``pzu + a*|pzu|``, the surplus sold at ``pzu - b*|pzu|``. Minus the perfect-foresight cost (everything bought on PZU,
``actual * pzu``) this leaves

    intraday  = traded_id * (pi - pzu)
    imbalance = a*|pzu|*short + b*|pzu|*long
    avoidable = intraday + imbalance

Imbalance price = PZU x (1 +- a|b) is a **proxy** (spec I §6), a and b are calibrated once (``calibrate`` below) so
that the ``profile`` policy costs 200-375 lei per MWh injected by prosumers (AFEER), then frozen in
``models/imbalance_ab.json``. Stated deviation from the spec: the spec writes ``a*pzu*short + b*pzu*long``; we use
``|pzu|`` so that being short or long never *earns* money in negative-price quarter-hours (1.4% of 2025, mostly at
solar noon, exactly when prosumers are long). Identical whenever pzu >= 0.

**Wind / any producer: production enters with a minus sign**, as negative demand (spec W §4):

    avoidable_cost(actual=-prod_mwh, bought_da=-sold_da_mwh, traded_id=-sold_id_mwh, pzu=pzu, pi=pi)
    # sold_id_mwh: sold more (+) or bought back (-) on PI
    # dev = sold_da + sold_id - prod  > 0: sold more than produced (short, pays a), < 0: produced more (long, b)

``wind_avoidable_cost`` does that flip for you.

With a = b the cost of every policy scales by the same factor, so a % reduction does not depend on the value; with
a != b it depends only on a/b. ``q_star(a, b) = a/(a+b)`` is the purchase quantile that minimises the expected
imbalance term (newsvendor), used by I5 and W3.
"""

from __future__ import annotations

import json
from functools import lru_cache
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
AB_PATH = ROOT / "models" / "imbalance_ab.json"
KEYS = ("avoidable_lei", "intraday_lei", "imbalance_lei", "abs_dev_mwh", "short_mwh", "long_mwh")


@lru_cache(maxsize=4)
def _ab_file(path: str) -> tuple[float, float]:
    d = json.loads(Path(path).read_text())
    return float(d["a"]), float(d["b"])


def load_ab(path: Path | str = AB_PATH) -> tuple[float, float]:
    """Frozen ``(a, b)`` from models/imbalance_ab.json."""
    return _ab_file(str(path))


def q_star(a: float | None = None, b: float | None = None) -> float:
    """Day-ahead purchase quantile ``a / (a + b)`` (P50 when a = b). Frozen values when not given."""
    if a is None or b is None:
        a0, b0 = load_ab()
        a, b = (a0 if a is None else a), (b0 if b is None else b)
    return float(a) / (float(a) + float(b))


def _resolve(a, b) -> tuple[float, float]:
    if a is None or b is None:
        a0, b0 = load_ab()
        a, b = (a0 if a is None else a), (b0 if b is None else b)
    return float(a), float(b)


def cost_steps(actual, bought_da, traded_id, pzu, pi, a: float | None = None,
               b: float | None = None) -> dict[str, np.ndarray]:
    """Per-step terms (broadcast shape of the inputs): ``dev, short, long`` (MWh) and ``intraday, imbalance,
    avoidable`` (lei). ``traded_id`` / ``pi`` may be None (no intraday trading: 0 MWh, pi = pzu)."""
    a, b = _resolve(a, b)
    actual = np.asarray(actual, dtype=np.float64)
    bought_da = np.asarray(bought_da, dtype=np.float64)
    pzu = np.asarray(pzu, dtype=np.float64)
    traded = np.zeros_like(actual) if traded_id is None else np.asarray(traded_id, dtype=np.float64)
    pi = pzu if pi is None else np.asarray(pi, dtype=np.float64)
    dev = actual - bought_da - traded
    short = np.clip(dev, 0.0, None)
    long = np.clip(-dev, 0.0, None)
    intraday = traded * (pi - pzu)
    apzu = np.abs(pzu)
    imbalance = a * apzu * short + b * apzu * long
    intraday, imbalance = np.broadcast_arrays(intraday, imbalance)
    return {"dev": dev, "short": short, "long": long,
            "intraday": intraday, "imbalance": imbalance, "avoidable": intraday + imbalance}


def avoidable_cost(actual, bought_da, traded_id, pzu, pi, a: float | None = None, b: float | None = None,
                   axis: int | tuple[int, ...] | None = None) -> dict:
    """Spec I §7. Returns ``avoidable_lei, intraday_lei, imbalance_lei, abs_dev_mwh, short_mwh, long_mwh``.

    ``axis=None`` sums everything and returns floats; otherwise sums over ``axis`` and returns arrays (e.g. ``axis=-1``
    on scenarios x steps gives one cost per scenario). ``a, b = None`` -> frozen values from models/imbalance_ab.json.
    Consumption sign; for producers use ``wind_avoidable_cost`` or pass every quantity with a minus sign.
    """
    s = cost_steps(actual, bought_da, traded_id, pzu, pi, a, b)
    terms = {"avoidable_lei": s["avoidable"], "intraday_lei": s["intraday"], "imbalance_lei": s["imbalance"],
             "abs_dev_mwh": np.abs(s["dev"]), "short_mwh": s["short"], "long_mwh": s["long"]}
    if axis is None:
        return {k: float(v.sum()) for k, v in terms.items()}
    return {k: v.sum(axis=axis) for k, v in terms.items()}


def wind_avoidable_cost(prod_mwh, sold_da_mwh, sold_id_mwh, pzu, pi, a: float | None = None,
                        b: float | None = None, axis=None) -> dict:
    """Producer side (spec W §4): ``prod_mwh`` produced, ``sold_da_mwh`` sold on PZU, ``sold_id_mwh`` sold more (+)
    or bought back (-) on PI. short_mwh = sold but not produced (pays a), long_mwh = produced but not sold (b)."""
    neg = lambda x: None if x is None else -np.asarray(x, dtype=np.float64)  # noqa: E731
    return avoidable_cost(neg(prod_mwh), neg(sold_da_mwh), neg(sold_id_mwh), pzu, pi, a, b, axis=axis)


def lei_per_mwh(res: dict, mwh: float) -> float:
    """``avoidable_lei / mwh`` (lei per MWh injected by prosumers, or per MWh produced for wind)."""
    return float(res["avoidable_lei"]) / float(mwh) if mwh else float("nan")


# --------------------------------------------------------------------------------------------------------------------
# Calibration of a, b (run once, result frozen in models/imbalance_ab.json):
#   python -m sentinel.eval.imbalance --calibrate [--fleet DIR] [--year 2024] [--write]
# --------------------------------------------------------------------------------------------------------------------
AFEER_LO, AFEER_HI = 200.0, 375.0  # lei per MWh injected by prosumers, publicly reported by suppliers (AFEER)
AB_RATIO = 2.0                     # a = 2 b: a deficit costs twice what a surplus does (spec: choose a >= b)


def _fleet_dir(fleet: str | None) -> Path:
    if fleet:
        return Path(fleet)
    from sentinel.sim.fleet import FleetConfig, generate
    return Path(generate(FleetConfig()))


def _portfolio(fleet: Path, year: int):
    """Per local year: ts, portfolio net import (MWh/step), per-home export summed (MWh/step), and the
    fleet-average load per home + PV per kWp, per step (kW). Synthetic RO."""
    import polars as pl
    sites = pl.read_parquet(fleet / "sites.parquet").select("site_id", "kwp_true")
    t = (pl.scan_parquet(fleet / "telemetry" / f"year={year}.parquet")
         .select("ts", "site_id", "pv_ac_w", "load_w", "grid_w")
         .with_columns(pl.col("grid_w").fill_null(pl.col("pv_ac_w").fill_null(0) - pl.col("load_w").fill_null(0))))
    g = (t.group_by("ts").agg(
            (-pl.col("grid_w")).sum().alias("net_w"),
            pl.col("grid_w").clip(lower_bound=0).sum().alias("exp_w"),
            pl.col("load_w").mean().alias("load_w"),
            pl.col("pv_ac_w").sum().alias("pv_w"),
            pl.len().alias("n"))
         .sort("ts").collect())
    kwp = float(sites["kwp_true"].sum())
    to_mwh = 0.25 / 1e6
    return {"ts": g["ts"].to_numpy(), "net_mwh": g["net_w"].to_numpy() * to_mwh,
            "exp_mwh": g["exp_w"].to_numpy() * to_mwh,
            "load_kw_home": g["load_w"].to_numpy() / 1e3, "pv_kw_per_kwp": g["pv_w"].to_numpy() / 1e3 / kwp,
            "n_homes": int(g["n"].max()), "kwp": kwp}


def _local_keys(ts: np.ndarray):
    import pandas as pd
    loc = pd.DatetimeIndex(ts).tz_localize("UTC").tz_convert("Europe/Bucharest") if pd.DatetimeIndex(ts).tz is None \
        else pd.DatetimeIndex(ts).tz_convert("Europe/Bucharest")
    qh = (loc.hour * 4 + loc.minute // 15).to_numpy()
    month = loc.month.to_numpy()
    weekend = (loc.day_of_week >= 5).astype(int)
    return month, weekend, qh


def _shape(values: np.ndarray, keys: tuple[np.ndarray, ...]) -> dict:
    out: dict = {}
    for i, k in enumerate(zip(*keys)):
        out.setdefault(k, []).append(values[i])
    return {k: float(np.mean(v)) for k, v in out.items()}


def profile_position(prev: dict, cur: dict) -> np.ndarray:
    """Minimal stand-in for I3's ``profile`` policy (#173, not merged yet): what a supplier buys day-ahead without
    weather or meter data. Load = class-average load shape (month x weekday/weekend x quarter-hour) from the
    previous year x homes (the role of the distributors' standard load profile x expected consumption); minus a
    typical monthly solar curve (previous year's average kW per kWp by month x quarter-hour) x portfolio kWp."""
    m0, w0, q0 = _local_keys(prev["ts"])
    load_shape = _shape(prev["load_kw_home"], (m0, w0, q0))
    pv_shape = _shape(prev["pv_kw_per_kwp"], (m0, q0))
    m, w, q = _local_keys(cur["ts"])
    load = np.array([load_shape.get((a, b_, c), np.nan) for a, b_, c in zip(m, w, q)])
    pv = np.array([pv_shape.get((a, c), 0.0) for a, c in zip(m, q)])
    load = np.where(np.isnan(load), np.nanmean(load), load)
    return (load * cur["n_homes"] - pv * cur["kwp"]) * 0.25 / 1e3  # kW -> MWh per step


def calibrate(fleet: str | None = None, year: int = 2024, ratio: float = AB_RATIO,
              target: float | None = None) -> dict:
    """Solve a (with b = a / ratio) so the profile policy costs ``target`` lei/MWh injected on ``year``.
    The profile policy does not trade intraday and its position does not depend on a, b, so the cost is linear:
    cost = a*S + b*L with S, L the |pzu|-weighted short / long MWh -> closed form."""
    from sentinel.data.prices import prices_at
    target = (AFEER_LO + AFEER_HI) / 2 if target is None else target
    fdir = _fleet_dir(fleet)
    prev, cur = _portfolio(fdir, year - 1), _portfolio(fdir, year)
    pos = profile_position(prev, cur)
    pzu, _ = prices_at(cur["ts"])
    inj = float(cur["exp_mwh"].sum())
    unit_a = avoidable_cost(cur["net_mwh"], pos, None, pzu, None, a=1.0, b=0.0)["imbalance_lei"]
    unit_b = avoidable_cost(cur["net_mwh"], pos, None, pzu, None, a=0.0, b=1.0)["imbalance_lei"]
    a = target * inj / (unit_a + unit_b / ratio)
    b = a / ratio
    res = avoidable_cost(cur["net_mwh"], pos, None, pzu, None, a=a, b=b)
    return {
        "a": round(a, 4), "b": round(b, 4), "ratio_a_over_b": ratio, "q_star": round(a / (a + b), 4),
        "target_lei_per_mwh_injected": target, "band_lei_per_mwh_injected": [AFEER_LO, AFEER_HI],
        "calibration": {
            "year": year, "policy": "profile (minimal stand-in, see sentinel.eval.imbalance.profile_position; "
                                    "I3 #173 not merged)",
            "fleet": fdir.name, "n_homes": cur["n_homes"], "kwp": round(cur["kwp"], 1),
            "injected_mwh": round(inj, 2), "net_import_mwh": round(float(cur["net_mwh"].sum()), 2),
            "abs_dev_mwh": round(res["abs_dev_mwh"], 2), "short_mwh": round(res["short_mwh"], 2),
            "long_mwh": round(res["long_mwh"], 2), "avoidable_lei": round(res["avoidable_lei"], 0),
            "lei_per_mwh_injected": round(lei_per_mwh(res, inj), 1),
        },
        "labels": {"prices": "real RO (OPCOM PZU, hourly x4 before 2025-10-01)", "portfolio": "synthetic RO",
                   "imbalance_price": "proxy: PZU +- a|b x |PZU|", "target": "real RO (AFEER, public)"},
        "formula": "imbalance = a*|pzu|*short + b*|pzu|*long; intraday = traded_id*(pi-pzu)",
    }


def _check_year(fleet: str | None, year: int, a: float, b: float) -> dict:
    from sentinel.data.prices import prices_at
    fdir = _fleet_dir(fleet)
    prev, cur = _portfolio(fdir, year - 1), _portfolio(fdir, year)
    pos = profile_position(prev, cur)
    pzu, _ = prices_at(cur["ts"])
    res = avoidable_cost(cur["net_mwh"], pos, None, pzu, None, a=a, b=b)
    inj = float(cur["exp_mwh"].sum())
    return {"year": year, "injected_mwh": round(inj, 2), "abs_dev_mwh": round(res["abs_dev_mwh"], 2),
            "avoidable_lei": round(res["avoidable_lei"], 0), "lei_per_mwh_injected": round(lei_per_mwh(res, inj), 1)}


def main(argv: list[str] | None = None) -> int:
    import argparse
    import os
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--calibrate", action="store_true")
    ap.add_argument("--fleet", default=None, help="fleet cache dir (default: generate(FleetConfig()))")
    ap.add_argument("--year", type=int, default=2024, help="calibration year (validation split)")
    ap.add_argument("--check-year", type=int, default=2025, help="year to report with the frozen a, b")
    ap.add_argument("--ratio", type=float, default=AB_RATIO)
    ap.add_argument("--write", action="store_true", help="write models/imbalance_ab.json")
    args = ap.parse_args(argv)
    if not args.calibrate:
        ap.print_help()
        return 0
    out = calibrate(args.fleet, args.year, args.ratio)
    chk = _check_year(args.fleet, args.check_year, out["a"], out["b"])
    out["check"] = {**chk, "note": "same frozen a, b on the test year; not used to fit"}
    run = os.environ.get("LAB_RUN_ID") or os.environ.get("LAB_RUN")
    if run:
        out["lab_run"] = run
    print(json.dumps(out, indent=1))
    c = out["calibration"]
    print(f"LAB_METRIC imbalance_a={out['a']}")
    print(f"LAB_METRIC imbalance_b={out['b']}")
    print(f"LAB_METRIC profile_lei_per_mwh_injected_{args.year}={c['lei_per_mwh_injected']}")
    print(f"LAB_METRIC profile_abs_dev_mwh_{args.year}={c['abs_dev_mwh']}")
    print(f"LAB_METRIC profile_lei_per_mwh_injected_{chk['year']}={chk['lei_per_mwh_injected']}")
    if args.write:
        AB_PATH.write_text(json.dumps(out, indent=1) + "\n")
        print(f"wrote {AB_PATH}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
