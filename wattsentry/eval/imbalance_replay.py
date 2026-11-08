"""Imbalance replay: the supplier policies settled on the same year, meters and prices.

    python replay.py --mode imbalance --policy profile,persistence,forecast_only,sentinel \
        --out $LAB_OUTPUT_DIR/imbalance.sqlite

Baselines (``profile``, ``persistence``, ``forecast_only``) come from ``baselines_portfolio``; ``sentinel`` buys the
quantile q* of the portfolio forecast (``portfolio``) on PZU only; ``sentinel_id`` adds the 15:00 re-forecast traded on
PI. Opt-in level-3 policies (``wattsentry.opt.level3.L3_POLICIES``) add enrolled batteries to the 15:00 LP; their
dispatch changes the metered flow, so ``positions.bought_da`` = purchase - dispatch change.

Every policy is settled on the same metered portfolio flow with ``wattsentry.eval.imbalance.cost_steps`` (frozen a, b)
at the prices of ``wattsentry.data.prices.prices_at``. LAB_METRICs unsuffixed = ``sentinel``, ``_<policy>`` for each.
Outputs: one SQLite file with the ``TABLES`` below (replaced if present), the same tables as parquet next to it, and
imbalance_report.json.
"""

from __future__ import annotations

import argparse
import json
import os
import sqlite3
import time
from pathlib import Path
from typing import Any, Mapping, Optional, Sequence

import numpy as np
import pandas as pd

from wattsentry.opt.level3 import L3_POLICIES

TZ = "Europe/Bucharest"
POLICIES = ("profile", "persistence", "forecast_only", "sentinel")
EXTRA_POLICIES = ("sentinel_id",)
BASELINES = ("profile", "persistence", "forecast_only")
HEADLINE = "sentinel"
LABEL_QTY = "synthetic RO"
UNITS = {"avoidable_cost_lei": "lei", "intraday_lei": "lei", "imbalance_lei": "lei", "abs_dev_mwh": "MWh",
         "short_mwh": "MWh", "long_mwh": "MWh", "double_cover_mwh_18_21": "MWh", "abs_dev_mwh_17_21": "MWh",
         "lei_per_mwh_injected": "lei/MWh", "position_nmae": "ratio", "bias_mwh": "MWh", "export_mwh": "MWh",
         "actual_mwh": "MWh", "steps": "count", "avoidable_cost_reduction_vs_profile": "ratio",
         "portfolio_nmae": "ratio", "portfolio_coverage_p10_p90": "ratio", "avoidable_cost_lei_per_day": "lei"}

# Output tables (SQLite + parquet). ts / issue_time = ISO-8601 UTC text ("2025-01-01T00:00:00Z") in SQLite,
# datetime[UTC] in parquet. Quantities MWh per 15-min step, + = the portfolio takes from the grid (net purchase).
TABLES = {
    "portfolio_forecast": {  # one row per (issue_time, ts, policy, horizon)
        "issue_time": "text", "ts": "text", "policy": "text",  # profile|persistence|forecast_only|sentinel
        "horizon": "text",                                      # da (day-ahead, D-1 11:00 UTC) | id (15:00 re-forecast)
        "p10": "real", "p50": "real", "p90": "real",           # MWh/step; p10/p90 NULL for point policies
        "run_id": "text"},
    "positions": {  # one row per (ts, policy)
        "ts": "text", "policy": "text",                        # + sentinel_id (level 2)
        "bought_da": "real", "traded_id": "real", "actual": "real",  # MWh/step
        "dev_mwh": "real",                                      # actual - bought_da - traded_id (> 0 short)
        "intraday_lei": "real", "imbalance_lei": "real", "avoidable_lei": "real",  # lei per step (cost_steps)
        "pzu_lei_mwh": "real", "pi_lei_mwh": "real",           # prices used
        "import_mwh": "real", "export_mwh": "real",            # metered portfolio gross flows (>= 0)
        "run_id": "text"},
    "imbalance_kpis": {  # one row per (policy, name): totals over the replayed days
        "run_id": "text", "policy": "text", "name": "text", "value": "real", "unit": "text", "label": "text",
        "date_from": "text", "date_to": "text"},
}


# ============================================================================================ pure evaluation
def local_hour(ts: Any, tz: str = TZ) -> np.ndarray:
    """Local clock hour (float) of UTC step starts (naive = UTC)."""
    idx = pd.DatetimeIndex(np.asarray(ts).ravel())
    idx = idx.tz_localize("UTC") if idx.tz is None else idx
    loc = idx.tz_convert(tz)
    return (loc.hour + loc.minute / 60.0).to_numpy().reshape(np.shape(ts))


def evaluate(ts: Any, actual: np.ndarray, export_mwh: np.ndarray,
             positions: Mapping[str, tuple[np.ndarray, Optional[np.ndarray]]], pzu: np.ndarray,
             pi: Optional[np.ndarray] = None, *, a: Optional[float] = None, b: Optional[float] = None
             ) -> tuple[dict[str, dict[str, float]], dict[str, dict[str, np.ndarray]]]:
    """Settle every policy on the same steps. positions: {policy: (bought_da, traded_id or None)} MWh/step.

    Returns (summary {policy: {name: total}}, steps {policy: cost_steps arrays}). double_cover_mwh_18_21 =
    MWh bought beyond need (bought + traded - actual > 0) between 18:00 and 21:00 local (the 18:00 story);
    lei_per_mwh_injected = avoidable lei / MWh exported by the prosumers (gross, per home) on the same steps."""
    from wattsentry.eval.imbalance import cost_steps

    actual = np.asarray(actual, dtype=np.float64)
    hrs = local_hour(ts)
    eve = (hrs >= 18) & (hrs < 21)
    e17 = (hrs >= 17) & (hrs < 21)
    exp_tot = float(np.sum(export_mwh))
    den = float(np.abs(actual).sum())
    summary, steps = {}, {}
    for pol, (bought, traded) in positions.items():
        s = cost_steps(actual, bought, traded, pzu, pi, a, b)
        dev = s["dev"]
        avoid = float(s["avoidable"].sum())
        summary[pol] = {
            "avoidable_cost_lei": avoid, "intraday_lei": float(s["intraday"].sum()),
            "imbalance_lei": float(s["imbalance"].sum()), "abs_dev_mwh": float(np.abs(dev).sum()),
            "short_mwh": float(s["short"].sum()), "long_mwh": float(s["long"].sum()),
            "double_cover_mwh_18_21": float(s["long"][eve].sum()), "abs_dev_mwh_17_21": float(np.abs(dev[e17]).sum()),
            "lei_per_mwh_injected": avoid / exp_tot if exp_tot > 0 else float("nan"),
            "position_nmae": float(np.abs(dev).sum() / den) if den > 0 else float("nan"),
            "bias_mwh": float(-dev.sum()), "export_mwh": exp_tot, "actual_mwh": float(actual.sum()),
            "steps": float(actual.size)}
        steps[pol] = s
    return summary, steps


def reduction_vs(summary: Mapping[str, Mapping[str, float]], pol: str, ref: str = "profile",
                 key: str = "avoidable_cost_lei") -> float:
    """1 - cost(pol) / cost(ref): the headline percentage (independent of a = b's level)."""
    if pol not in summary or ref not in summary or not summary[ref][key]:
        return float("nan")
    return 1.0 - summary[pol][key] / summary[ref][key]


def align_days(ts_days: np.ndarray, year_ts: pd.DatetimeIndex) -> tuple[np.ndarray, np.ndarray]:
    """ts_days (D, T) naive-UTC step starts of the scenario days; year_ts = the baselines' UTC axis.
    Returns (keep (D,) bool: days whose every step is on the axis, idx (D, T) positions into year_ts, -1 = absent)."""
    flat = pd.DatetimeIndex(np.asarray(ts_days).ravel())
    flat = flat.tz_localize("UTC") if flat.tz is None else flat.tz_convert("UTC")
    axis = year_ts.tz_convert("UTC") if year_ts.tz is not None else year_ts.tz_localize("UTC")
    idx = axis.get_indexer(flat).reshape(np.shape(ts_days))
    return (idx >= 0).all(axis=1), idx


# ============================================================================================ policies
def sentinel_positions(src: Any, sites: np.ndarray, val_days: np.ndarray, test_days: np.ndarray, *,
                       models: Mapping[str, Sequence[Any]], k: int, seed: int, chunk_days: int,
                       region: str = "county", log=print) -> dict[str, Any]:
    """The portfolio forecast on the test days: calibrated day-ahead band + purchase (q*), 15:00 band + PI trade.

    Same seeds and conformal fit as ``python -m wattsentry.eval.portfolio`` (val = conformal, test seed + 7919), so
    overlapping days match its eval run. Returns ts (D, T) naive UTC, bought, traded, da_cal, id_cal, fo_cal,
    mask, actual_i5 (the window cache's meter truth, for the diagnostic), conformal json, timings."""
    from wattsentry.eval import portfolio as P

    fleet = P.fleet_from_source(src, region)
    timings: dict[str, float] = {}
    val = P.run_split(src, fleet, sites, val_days, models=models, n_scen=k, seed=seed, chunk_days=chunk_days,
                      timings=timings, log=log)
    test = P.run_split(src, fleet, sites, test_days, models=models, n_scen=k, seed=seed + 7919,
                       chunk_days=chunk_days, timings=timings, log=log)
    conf_da = P.TotalConformal.fit(val["da_bands"], val["actual"])
    conf_id = P.TotalConformal.fit(np.where(val["mask"][..., None], val["id_bands"], np.nan),
                                   np.where(val["mask"], val["actual"], np.nan))
    conf_fo = P.TotalConformal.fit(val["fo_bands"], val["actual"])
    bought = conf_da.stretch(test["da_q"][..., 0], test["da_q"][..., 1])
    q_id = conf_id.stretch(test["id_q"][..., 0], test["id_q"][..., 1])
    traded = np.where(test["mask"], q_id - bought, 0.0)
    return {"ts": test["ts"], "bought": bought, "traded": traded, "mask": test["mask"],
            "da_cal": conf_da.apply(test["da_bands"]), "id_cal": conf_id.apply(test["id_bands"]),
            "fo_cal": conf_fo.apply(test["fo_bands"]), "actual_i5": test["actual"],
            "conformal": {"da": conf_da.to_json(), "id": conf_id.to_json(), "forecast_only": conf_fo.to_json()},
            "q_star": P.purchase_quantile(), "behaviour": P.BEHAVIOUR_NAME, "timings": timings,
            "val_days": val["day"], "adapter_counts": {"val": val["counts"], "test": test["counts"]}}


def baseline_positions(site_ids: Sequence[str], year: int, policies: Sequence[str], log=print
                       ) -> tuple[Any, dict[str, np.ndarray]]:
    """Baseline positions on the local test year: (Portfolio, {policy: bought_da (T,) MWh on pf.ts})."""
    from wattsentry.eval import baselines_portfolio as B

    t0 = time.perf_counter()
    pf = B.load_fleet(year, site_ids=list(site_ids))
    log(f"[imbalance] baselines: {pf.n_sites} homes, {len(pf.ts)} steps of {year} in {time.perf_counter() - t0:.1f}s")
    out = {}
    for pol in policies:
        t0 = time.perf_counter()
        df = B.positions(pol, year, fleet=pf)
        if df.height != len(pf.ts):
            raise RuntimeError(f"{pol}: {df.height} rows, expected {len(pf.ts)}")
        out[pol] = df["bought_da_mwh"].to_numpy().astype(np.float64)
        log(f"[imbalance] {pol}: {time.perf_counter() - t0:.1f}s")
    return pf, out


# ============================================================================================ outputs
def _iso(ts: pd.DatetimeIndex) -> list[str]:
    return list(ts.tz_convert("UTC").strftime("%Y-%m-%dT%H:%M:%SZ"))


def build_tables(ts: pd.DatetimeIndex, actual: np.ndarray, imp: np.ndarray, exp: np.ndarray,
                 positions: Mapping[str, tuple[np.ndarray, Optional[np.ndarray]]],
                 steps: Mapping[str, Mapping[str, np.ndarray]], pzu: np.ndarray, pi: np.ndarray,
                 bands: Mapping[tuple[str, str], tuple[np.ndarray, Optional[np.ndarray]]],
                 summary: Mapping[str, Mapping[str, float]], extra_kpis: Mapping[str, Mapping[str, float]],
                 labels: Mapping[str, str], run_id: str) -> dict[str, pd.DataFrame]:
    """DataFrames in the ``TABLES`` layout (ts as UTC datetimes; SQLite writer turns them into ISO text).

    bands: {(policy, horizon): ((n, 3) p10/p50/p90 or (n,) point, step mask (T,) or None)}."""
    ts = pd.DatetimeIndex(ts)
    ts = ts.tz_localize("UTC") if ts.tz is None else ts.tz_convert("UTC")
    pos = []
    for pol, (bought, traded) in positions.items():
        s = steps[pol]
        pos.append(pd.DataFrame({
            "ts": ts, "policy": pol, "bought_da": np.asarray(bought, np.float64),
            "traded_id": np.zeros(len(ts)) if traded is None else np.asarray(traded, np.float64),
            "actual": actual, "dev_mwh": s["dev"], "intraday_lei": s["intraday"], "imbalance_lei": s["imbalance"],
            "avoidable_lei": s["avoidable"], "pzu_lei_mwh": pzu, "pi_lei_mwh": pi, "import_mwh": imp,
            "export_mwh": exp, "run_id": run_id}))
    day0 = ts.floor("D")
    fc = []
    for (pol, hz), (v, mask) in bands.items():
        v = np.asarray(v, np.float64)
        m = np.ones(len(ts), bool) if mask is None else np.asarray(mask, bool)
        point = v.ndim == 1
        issue = day0 - pd.Timedelta(hours=13) if hz == "da" else day0 + pd.Timedelta(hours=13)
        fc.append(pd.DataFrame({
            "issue_time": issue[m], "ts": ts[m], "policy": pol, "horizon": hz,
            "p10": np.full(m.sum(), np.nan) if point else v[m, 0], "p50": v[m] if point else v[m, 1],
            "p90": np.full(m.sum(), np.nan) if point else v[m, 2], "run_id": run_id}))
    d_from, d_to = str(ts[0].date()), str(ts[-1].date())
    kp = []
    for pol, vals in list(summary.items()) + [(p, v) for p, v in extra_kpis.items()]:
        for name, val in vals.items():
            kp.append({"run_id": run_id, "policy": pol, "name": name, "value": float(val),
                       "unit": UNITS.get(name, ""), "label": labels.get(name, labels["default"]),
                       "date_from": d_from, "date_to": d_to})
    return {"positions": pd.concat(pos, ignore_index=True), "portfolio_forecast": pd.concat(fc, ignore_index=True),
            "imbalance_kpis": pd.DataFrame(kp, columns=list(TABLES["imbalance_kpis"]))}


def write_sqlite(path: Path | str, tables: Mapping[str, pd.DataFrame]) -> dict[str, int]:
    """Replace the ``TABLES`` tables in ``path`` (other tables are left alone) -> rows per table."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    con = sqlite3.connect(path)
    rows = {}
    try:
        for name, df in tables.items():
            cols = TABLES[name]
            df = df.copy()
            for c in ("ts", "issue_time"):
                if c in df:
                    df[c] = _iso(pd.DatetimeIndex(df[c]))
            con.execute(f'DROP TABLE IF EXISTS "{name}"')
            con.execute(f'CREATE TABLE "{name}" (' + ", ".join(f'"{c}" {t.upper()}' for c, t in cols.items()) + ")")
            vals = [tuple(None if (isinstance(x, float) and np.isnan(x)) else x for x in r)
                    for r in df[list(cols)].itertuples(index=False, name=None)]
            con.executemany(f'INSERT INTO "{name}" VALUES ({", ".join("?" * len(cols))})', vals)
            key = "ts, policy" if "ts" in cols else "policy, name"
            con.execute(f'CREATE INDEX IF NOT EXISTS "{name}_idx" ON "{name}" ({key})')
            rows[name] = len(vals)
        con.commit()
    finally:
        con.close()
    return rows


# ============================================================================================ lab run
def _emit(name: str, value: float) -> None:
    from wattsentry.infra.metrics import emit

    if value is not None and np.isfinite(value):
        emit(name, value)


def _sample(days: np.ndarray, n: int, rng: np.random.Generator) -> np.ndarray:
    days = np.asarray(days)
    return days if n <= 0 or n >= len(days) else np.sort(rng.choice(days, n, replace=False))


def run(args: argparse.Namespace, log=print) -> dict[str, Any]:
    from wattsentry import kernels
    from wattsentry.data.prices import load_prices, prices_at
    from wattsentry.eval import portfolio as P
    from wattsentry.eval.imbalance import load_ab
    from wattsentry.model.data import make_splits, open_source

    t_all = time.perf_counter()
    policies = [p.strip() for p in args.policy.split(",") if p.strip()]
    bad = [p for p in policies if p not in POLICIES + EXTRA_POLICIES + L3_POLICIES]
    if bad:
        raise SystemExit(f"unknown policy {bad}; expected {POLICIES + EXTRA_POLICIES + L3_POLICIES}")
    run_id = args.run_id or os.environ.get("LAB_RUN_ID") or "local"
    src = open_source(args.data)
    n_all = len(src.sites)
    sites = np.arange(n_all) if args.max_sites <= 0 else np.arange(min(args.max_sites, n_all))
    site_ids = [str(src.sites[i]["site_id"]) for i in sites]
    sp = make_splits(src)
    rng = np.random.default_rng(args.seed)
    lag_ok = lambda d: d[d >= 16]  # noqa: E731  14 history days + 1 day of meter lag
    val_days = _sample(lag_ok(np.concatenate([sp.days["val"], sp.days["cal"]])), args.val_days, rng)
    test_days = _sample(lag_ok(sp.days["test"]), args.test_days, rng)
    md = Path(args.models)
    models = {"pv_da": P.find_models("pvnet", md / "pvnet_fc"), "pv_id": P.find_models("pvnet", md),
              "load": P.find_models("loadnet", md)}
    log(f"[imbalance] {len(sites)} homes, val {len(val_days)} days, test {len(test_days)} days, K={args.k}, "
        f"policies {policies}, kernels={kernels.backend()}", flush=True)

    t0 = time.perf_counter()
    sen = sentinel_positions(src, sites, val_days, test_days, models=models, k=args.k, seed=args.seed,
                             chunk_days=args.chunk_days, region=args.region, log=log)
    t_sentinel = time.perf_counter() - t0
    year = int(pd.Timestamp(sen["ts"][0, 0]).year)
    t0 = time.perf_counter()
    pf, base = baseline_positions(site_ids, year, [p for p in policies if p in BASELINES], log=log)
    t_baselines = time.perf_counter() - t0

    keep, idx = align_days(sen["ts"], pf.ts)
    if not keep.any():
        raise RuntimeError("no scenario day lies inside the baselines' test year")
    ix = idx[keep].ravel()
    ts = pf.ts[ix]
    actual_kw = pf.actual_kw[:, ix]
    imp = np.clip(actual_kw, 0, None).sum(0) * P.KW_TO_MWH
    exp = np.clip(-actual_kw, 0, None).sum(0) * P.KW_TO_MWH
    actual = imp - exp
    pick = lambda a: np.asarray(a)[keep].reshape((-1,) + np.shape(a)[2:])  # noqa: E731  (D, T, ...) -> (n, ...)
    positions: dict[str, tuple[np.ndarray, Optional[np.ndarray]]] = {}
    for pol in policies:
        if pol in BASELINES:
            positions[pol] = (base[pol][ix], None)
        elif pol == "sentinel":
            positions[pol] = (pick(sen["bought"]), None)  # level 1: PZU only
        elif pol == "sentinel_id":
            positions[pol] = (pick(sen["bought"]), pick(sen["traded"]))
    pzu, pi = prices_at(ts.tz_convert("UTC").tz_localize(None).to_numpy())
    n_days = int(keep.sum())
    l3 = None
    l3_pols = [p for p in policies if p in L3_POLICIES]
    if l3_pols:  # level 3: enrolled batteries in the 15:00 LP
        from wattsentry.opt.level3 import replay_level3

        a3, b3 = load_ab()
        l3 = replay_level3(src, sites, np.asarray(test_days)[keep], ts=np.asarray(sen["ts"])[keep],
                           bought=np.asarray(sen["bought"])[keep], id_cal=np.asarray(sen["id_cal"])[keep],
                           mask=np.asarray(sen["mask"])[keep], pzu=pzu.reshape(n_days, -1), pi=pi.reshape(n_days, -1),
                           policies=l3_pols, a=a3, b=b3, share=args.l3_share, p_noncompliance=args.l3_noncompliance,
                           n_nodes=args.l3_nodes, seed=args.seed, log=log)
        for pol in l3_pols:
            positions[pol] = (pick(sen["bought"]) - l3[pol]["delta"].ravel(), l3[pol]["traded"].ravel())
    summary, steps = evaluate(ts, actual, exp, positions, pzu, pi)

    da_cal = pick(sen["da_cal"])
    head: dict[str, float] = {
        "portfolio_nmae": P.nmae(da_cal[:, 1], actual), "portfolio_coverage_p10_p90": P.coverage(da_cal, actual),
        "portfolio_coverage_p10_p90_forecast_only": P.coverage(pick(sen["fo_cal"]), actual)}
    extra = {}
    for pol in summary:
        extra[pol] = {"avoidable_cost_reduction_vs_profile": reduction_vs(summary, pol),
                      "avoidable_cost_lei_per_day": summary[pol]["avoidable_cost_lei"] / n_days}
    sen_name = HEADLINE if HEADLINE in summary else ("sentinel_id" if "sentinel_id" in summary else None)
    if sen_name:
        extra[sen_name].update(head)

    a, b = load_ab()
    prices = load_prices()
    pi_src = sorted(set(prices["pi_src"].to_list()))
    pi_label = "real RO (OPCOM PI)" if "opcom_pi" in pi_src else "PI = PZU (no PI data yet, I1 #171): intraday term 0"
    labels = {"default": f"{LABEL_QTY} quantities; real RO PZU prices; proxy a, b",
              "abs_dev_mwh": LABEL_QTY, "double_cover_mwh_18_21": LABEL_QTY, "abs_dev_mwh_17_21": LABEL_QTY,
              "position_nmae": LABEL_QTY, "portfolio_nmae": LABEL_QTY, "portfolio_coverage_p10_p90": LABEL_QTY,
              "portfolio_coverage_p10_p90_forecast_only": LABEL_QTY, "export_mwh": LABEL_QTY,
              "actual_mwh": LABEL_QTY, "bias_mwh": LABEL_QTY, "short_mwh": LABEL_QTY, "long_mwh": LABEL_QTY,
              "steps": LABEL_QTY}

    # ---- metrics: unsuffixed = sentinel (level 1), then every policy suffixed
    for name in ("avoidable_cost_lei", "lei_per_mwh_injected", "abs_dev_mwh", "double_cover_mwh_18_21"):
        if sen_name:
            _emit(name, summary[sen_name][name])
    for name, v in head.items():
        _emit(name, v)
    if sen_name:
        _emit("avoidable_cost_reduction_vs_profile", extra[sen_name]["avoidable_cost_reduction_vs_profile"])
    for pol in summary:
        for name in ("avoidable_cost_lei", "lei_per_mwh_injected", "abs_dev_mwh", "double_cover_mwh_18_21",
                     "position_nmae", "imbalance_lei", "intraday_lei", "bias_mwh"):
            _emit(f"{name}_{pol}", summary[pol][name])
        _emit(f"avoidable_cost_reduction_vs_profile_{pol}", extra[pol]["avoidable_cost_reduction_vs_profile"])
    if l3 is not None:
        cost = {p: summary[p]["avoidable_cost_lei"] for p in summary}
        if "sentinel_id" in cost and "sentinel_l3" in cost:
            _emit("l3_extra_saving_vs_l2_lei", cost["sentinel_id"] - cost["sentinel_l3"])
            _emit("l3_extra_saving_vs_l2_share", 1.0 - cost["sentinel_l3"] / cost["sentinel_id"])
        if "sentinel_id_lp" in cost:
            for p in ("sentinel_l3", "sentinel_l3_oracle", "sentinel_l3_all"):
                if p in cost:
                    _emit(f"l3_battery_saving_lei_{p}", cost["sentinel_id_lp"] - cost[p])
        for p, acc in l3["diag"]["per_policy"].items():
            for k_, v in acc.items():
                _emit(f"l3_{k_}_{p}", float(v))
        _emit("l3_battery_homes", l3["diag"]["battery_homes"])
        _emit("l3_feeder_limit", float(l3["diag"]["feeder_limit"]))
        _emit("t_level3_s", l3["diag"]["t_level3_s"])
    _emit("export_mwh", float(exp.sum()))
    _emit("days", n_days)
    _emit("homes", len(sites))
    _emit("q_star", sen["q_star"])
    _emit("imbalance_a", a)
    _emit("imbalance_b", b)
    _emit("pi_real", float("opcom_pi" in pi_src))
    act_diff = float(np.abs(pick(sen["actual_i5"]) - actual).sum())
    _emit("actual_meter_vs_window_abs_mwh", act_diff)
    _emit("rust_kernels", float(kernels.backend() == "rust"))
    timings = {**sen["timings"], "t_sentinel_s": t_sentinel, "t_baselines_s": t_baselines}
    print("# strategy_f1: not available on the current fleet (no hidden battery strategies; needs data-v2 I2 #172 "
          "and the I4 #174 classifier)", flush=True)

    # ---- tables
    bands: dict[tuple[str, str], tuple[np.ndarray, Optional[np.ndarray]]] = {}
    for pol in policies:
        if pol in BASELINES:
            bands[(pol, "da")] = (positions[pol][0], None)
        elif pol == "sentinel":
            bands[(pol, "da")] = (da_cal, None)
        elif pol == "sentinel_id":
            bands[("sentinel", "id")] = (pick(sen["id_cal"]), pick(sen["mask"]))
    tables = build_tables(ts, actual, imp, exp, positions, steps, pzu, pi, bands, summary, extra, labels, run_id)
    out = Path(args.out)
    rows = write_sqlite(out, tables)
    pq_dir = out.parent
    for name, df in tables.items():
        df.to_parquet(pq_dir / f"{name}.parquet", index=False)
    timings["t_total_s"] = time.perf_counter() - t_all
    for k_, v in timings.items():
        _emit(k_, v)
    report = {
        "card": 176, "run_id": run_id, "policies": policies, "headline": sen_name, "days": n_days,
        "date_from": str(ts[0].date()), "date_to": str(ts[-1].date()), "homes": len(sites),
        "summary": summary, "extra": extra, "a": a, "b": b, "q_star": sen["q_star"], "pi_src": pi_src,
        "labels": {"quantities": f"{LABEL_QTY} (fleet {src.index.get('source_key', {}).get('fleet', '?')}, "
                                 f"meters = faults {pf.source.get('faults')})",
                   "pzu": "real RO (OPCOM; hourly x4 before 2025-10-01, 15-min after)", "pi": pi_label,
                   "imbalance_ab": "proxy (X1 #158, AFEER 200-375 lei/MWh injected)",
                   "standard_profile": "real RO (DEER PSC, I3 #173)",
                   "intraday_weather": "proxy: ERA5 analysis as the 15:00 weather update (I5 #175)"},
        "strategy_f1": None, "strategy_f1_reason": "current fleet has no hidden strategies; needs data-v2 + I4",
        "share_maxcontrol": None, "conformal": sen["conformal"], "behaviour": sen["behaviour"],
        "actual_meter_vs_window_abs_mwh": act_diff, "adapter_counts": sen["adapter_counts"],
        "kernels": kernels.backend(), "timings": timings, "rows": rows, "sqlite": str(out),
        "tables": TABLES, "test_days_dropped": int((~keep).sum()),
        "level3": None if l3 is None else l3["diag"]}
    (pq_dir / "imbalance_report.json").write_text(json.dumps(report, indent=1, default=float))
    log(f"[imbalance] {n_days} days {report['date_from']}..{report['date_to']}, rows {rows} -> {out} "
        f"in {timings['t_total_s']:.0f}s", flush=True)
    for pol, s in summary.items():
        log(f"[imbalance] {pol:14s} avoidable {s['avoidable_cost_lei']:12.0f} lei  "
            f"{s['lei_per_mwh_injected']:7.1f} lei/MWh inj  abs_dev {s['abs_dev_mwh']:8.1f} MWh  "
            f"double_cover_18_21 {s['double_cover_mwh_18_21']:7.1f} MWh  "
            f"vs profile {100 * extra[pol]['avoidable_cost_reduction_vs_profile']:+.1f}%", flush=True)
    return report


def parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(prog="replay.py --mode imbalance", description=__doc__.split("\n")[0])
    ap.add_argument("--policy", default=",".join(POLICIES + EXTRA_POLICIES),
                    help=f"comma list of {', '.join(POLICIES + EXTRA_POLICIES)}")
    ap.add_argument("--out", default=os.path.join(os.environ.get("LAB_OUTPUT_DIR", "."), "imbalance.sqlite"),
                    help="SQLite file (its imbalance tables are replaced); parquet + report go next to it")
    ap.add_argument("--data", default="lab", help="lab | auto | <windows dir> (wattsentry.model.data)")
    ap.add_argument("--models", default=str(Path(__file__).resolve().parents[2] / "models"))
    ap.add_argument("--val-days", type=int, default=120, help="sampled 2024 days for the conformal fit (0 = all)")
    ap.add_argument("--test-days", type=int, default=0, help="sampled 2025 days (0 = the whole test year)")
    ap.add_argument("--k", type=int, default=200, help="portfolio scenarios")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--region", default="county")
    ap.add_argument("--chunk-days", type=int, default=8)
    ap.add_argument("--max-sites", type=int, default=0, help="first N homes (smoke runs)")
    ap.add_argument("--run-id", default=None)
    ap.add_argument("--l3-share", type=float, default=0.10,
                    help="level 3: share of battery homes enrolled (sentinel_l3*)")
    ap.add_argument("--l3-noncompliance", type=float, default=0.10, help="level 3: P(a home opts out on a day)")
    ap.add_argument("--l3-nodes", type=int, default=20, help="level 3: quantile nodes of the 15:00 band per step")
    return ap


def main(argv: Optional[Sequence[str]] = None) -> dict[str, Any]:
    args, _ = parser().parse_known_args(argv)
    if args.data == "v2":  # data-v2 MaxControl sensitivity
        from wattsentry.eval.imbalance_v2 import main as v2_main

        return v2_main(argv)
    return run(parser().parse_args(argv))


if __name__ == "__main__":
    main()
