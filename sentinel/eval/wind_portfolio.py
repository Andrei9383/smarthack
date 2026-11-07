"""W3 Wind D1 (card #166, spec W step 4): how much each park sells tomorrow, and the avoidable imbalance cost.

    python -m sentinel.eval.wind_portfolio [--m5 train|<dir / .onnx / .json>] [--n-scen 100] [--max-days N]

For every UTC day D the park sells on PZU at D-1 10:00 UTC (before the PZU gate, same origin as W2's avail_eval),
one position per 15-min step. Four policies (spec W §4):

| policy               | day-ahead position                                                                    |
|----------------------|---------------------------------------------------------------------------------------|
| `manufacturer_curve` | generic class curve x forecast hub wind, all turbines up (+ empirical residual offset) |
| `persistence`        | park power 24 h earlier (48 h for steps after the origin: not observed yet) + offset  |
| `windpower_only`     | M5 park scenarios, all turbines available                                              |
| `sentinel`           | M5 x availability (W2: M6/M7 through the Judge adapter); at D-1 15:00 local          |
|                      | (13:00 UTC) the forecast is redone with the newer M6 reads and M5's origin-residual   |
|                      | correction, and the difference is sold (+) or bought back (-) on PI                   |

Extra rows, not policies: `sentinel_da` (sentinel without the PI correction: what the correction adds) and
`perfect` (sells exactly what is produced: the "done when" check, 0 lei).

**How much to sell.** X1's `q_star(a, b) = a / (a + b)` is the newsvendor level for *demand* (consumption sign: a
deficit, actual > bought, pays a). Production is negative demand (spec W §4), so the level a/(a+b) of -production is
the level **b/(a+b) of production**: a producer that sold more than it made is short and pays a, so with a = 2b it
sells its P33, not its P67. `sell_level(a, b)` returns that level (P50 when a = b). Scenario policies read it from the
W2 park scenarios after split-conformal (fitted on validation days of the training parks, as W2); the two point
baselines get the same level from the empirical quantile of their park residuals on the same rows.

**Cost.** `sentinel.eval.imbalance.cost_steps` (X1) with production entered with a minus sign, as negative demand;
dev = sold_da + sold_id - produced, > 0 = short (pays a|pzu|), < 0 = long (pays b|pzu|); the intraday term is
traded_id x (pi - pzu) with traded_id = -sold_id. pi = pzu until I1 (#171) ships data/raw/prices_pi.parquet, so the
PI correction is free for now (stated). Steps where the park was curtailed or had a data gap (injector label, as
W2's scoring mask) and steps without a price are left out of every policy alike.

**promised_but_not_delivered_mwh.** Energy sold on turbines that were already stopped when it was sold: per step,
the position x the share of the policy's own expected production that comes from turbines with a fault_stop label
both at the sale origin and at the delivery step (labels: scoring only). Share weights: manufacturer = curve power,
persistence = the lagged power, windpower_only = M5 P50, sentinel = M5 P50 x availability.

Metrics (LAB_METRIC, test year): `{metric}_{policy}` on all test park-days, `stopped_{metric}_{policy}` on park-days
with a stopped turbine, `holdout_{metric}_{policy}` on the park M5 never saw; unsuffixed `wind_avoidable_cost_lei`,
`lei_per_mwh_produced`, `wind_abs_dev_mwh`, `promised_but_not_delivered_mwh` = sentinel on all test park-days.
Prices real RO (OPCOM PZU), imbalance price proxy (PZU +- a|b x |PZU|, X1's frozen a, b), parks synthetic RO.

Positions for W5 (#168, park battery): `$LAB_OUTPUT_DIR/wind_positions.parquet`, one row per (policy, park, step):
see `positions_frame` / `load_positions`.
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path
from typing import Mapping, Optional, Sequence

import numpy as np

from sentinel.eval.imbalance import cost_steps, load_ab, q_star

LABEL = "synthetic RO parks (ERA5-driven #22/#23) x real RO prices (OPCOM PZU) x proxy imbalance price"
POLICIES = ("manufacturer_curve", "persistence", "windpower_only", "sentinel")
ROWS = POLICIES + ("sentinel_da", "perfect")
METRICS = ("wind_avoidable_cost_lei", "lei_per_mwh_produced", "abs_dev_mwh", "promised_but_not_delivered_mwh")
H = 96
DA_LEAD = 14 * 4  # sale for day D at D-1 10:00 UTC (W2's origin; PZU gate is 11:00 CET)
ID_LEAD = 11 * 4  # PI correction at D-1 13:00 UTC = 15:00 EET (16:00 EEST); PI for D opens 15:00 CET
KW15_TO_MWH = 0.25 / 1000.0


# ================================================================================================ core (pure numpy)
def sell_level(a: Optional[float] = None, b: Optional[float] = None) -> float:
    """Production quantile to sell day-ahead: b / (a + b) = 1 - q_star(a, b). Frozen X1 values when not given."""
    return 1.0 - q_star(a, b)


def wind_cost_steps(prod_mwh, sold_da_mwh, sold_id_mwh, pzu, pi, a: Optional[float] = None,
                    b: Optional[float] = None) -> dict[str, np.ndarray]:
    """Per-step producer cost via X1's `cost_steps` with every quantity entered with a minus sign (spec W §4).
    Returns dev (= sold_da + sold_id - prod; > 0 short), short, long (MWh), intraday, imbalance, avoidable (lei)."""
    neg = lambda x: None if x is None else -np.asarray(x, np.float64)  # noqa: E731
    return cost_steps(neg(prod_mwh), neg(sold_da_mwh), neg(sold_id_mwh), pzu, pi, a, b)


def stopped_share(w: np.ndarray, stopped: np.ndarray) -> np.ndarray:
    """Share of the expected production w [..., n] (>= 0) that comes from turbines flagged in stopped [..., n]."""
    w = np.clip(np.nan_to_num(np.asarray(w, np.float64)), 0, None)
    tot = w.sum(-1)
    return np.where(tot > 0, (w * stopped).sum(-1) / np.maximum(tot, 1e-12), 0.0)


def promised_not_delivered(sold_da_mwh, share_da, sold_id_mwh=None, share_id=None) -> np.ndarray:
    """Per-step MWh sold on already-stopped turbines: sold_da x share_da (+ sold_id x share_id), floored at 0."""
    x = np.asarray(sold_da_mwh, np.float64) * share_da
    if sold_id_mwh is not None:
        x = x + np.asarray(sold_id_mwh, np.float64) * share_id
    return np.clip(x, 0.0, None)


def portfolio_metrics(prod_mwh, sold_da_mwh, sold_id_mwh, pzu, pi, mask=None, pbnd_mwh=None,
                      a: Optional[float] = None, b: Optional[float] = None) -> dict[str, float]:
    """Spec W §4 metrics over the steps in mask (all broadcast together)."""
    s = wind_cost_steps(prod_mwh, sold_da_mwh, sold_id_mwh, pzu, pi, a, b)
    prod = np.broadcast_to(np.asarray(prod_mwh, np.float64), s["dev"].shape)
    sold = np.broadcast_to(np.asarray(sold_da_mwh, np.float64) + (0 if sold_id_mwh is None else sold_id_mwh),
                           s["dev"].shape)
    m = np.ones(s["dev"].shape, bool) if mask is None else np.broadcast_to(mask, s["dev"].shape)
    produced = float(prod[m].sum())
    return {"wind_avoidable_cost_lei": float(s["avoidable"][m].sum()),
            "lei_per_mwh_produced": float(s["avoidable"][m].sum()) / produced if produced > 0 else float("nan"),
            "abs_dev_mwh": float(np.abs(s["dev"])[m].sum()),
            "promised_but_not_delivered_mwh": float("nan") if pbnd_mwh is None else
            float(np.broadcast_to(pbnd_mwh, m.shape)[m].sum()),
            "short_mwh": float(s["short"][m].sum()), "long_mwh": float(s["long"][m].sum()),
            "imbalance_lei": float(s["imbalance"][m].sum()), "intraday_lei": float(s["intraday"][m].sum()),
            "produced_mwh": produced, "sold_mwh": float(sold[m].sum()), "n_steps": int(m.sum())}


def residual_offsets(p: np.ndarray, y: np.ndarray, mask: np.ndarray, levels: Sequence[float]) -> np.ndarray:
    """Empirical quantiles of y - p on the mask rows, one per level (point forecast -> distribution)."""
    r = (y - p)[mask & np.isfinite(y) & np.isfinite(p)]
    return np.quantile(r, list(levels)) if len(r) else np.zeros(len(levels))


def conformal_level(cal, raw: np.ndarray, park: str) -> tuple[np.ndarray, np.ndarray]:
    """W2's ParkConformal applied to scenario quantiles raw [..., 4] = (P10, P50, P90, sell level), fractions of
    capacity. The sell level goes through the same monotone map as W2's calibrated scenarios, so it equals the
    sell-level quantile of the calibrated scenarios. Returns (calibrated P10/P50/P90 [..., 3], sell level [...])."""
    q, sc = cal.apply(raw[..., :3], 1.0, park, scen=raw[..., 3][None])
    return q, sc[0]


# ================================================================================================ data helpers
def manufacturer_frac(data, chunk_days: int = 30) -> np.ndarray:
    """[T, N] generic class curve (power / rated) at the density-normalised forecast hub wind (M5's `cf` feature:
    the simulator's own curve, so a strong baseline). Built in time chunks."""
    from sentinel.model.windpower.features import build, weather_per_turbine

    T = data.n_steps
    w = weather_per_turbine(data.weather, data.park_of)
    out = np.empty((T, len(data.turbines)), np.float32)
    for s in range(0, T, chunk_days * H):
        e = min(T, s + chunk_days * H)
        out[s:e] = build(data.turbines, data.ts[s:e], {k: v[s:e] for k, v in data.meteo.items()},
                         {k: v[s:e] for k, v in w.items()})["cf"]
    return out


def persistence_index(d0: int, i0: int) -> np.ndarray:
    """Lagged step index for each step of the day starting at d0, sold at origin i0: 24 h earlier when that is
    before the origin, else 48 h earlier (the last same quarter-hour already measured)."""
    t = d0 + np.arange(H)
    return np.where(t - H < i0, t - H, t - 2 * H)


def nowcast(m5, q_kw: np.ndarray, resid0_kw: np.ndarray, horizon_h: np.ndarray) -> np.ndarray:
    """M5's nowcast_adjust per step, each step with its own horizon (nearest fitted beta_h)."""
    out = np.empty_like(q_kw)
    keys = sorted({int(k) for k in m5.meta.get("beta_h", {})} or {24})
    near = np.array([min(keys, key=lambda k: abs(k - h)) for h in horizon_h])
    for k in np.unique(near):
        m = near == k
        out[m] = m5.nowcast_adjust(q_kw[m], resid0_kw, int(k))
    return out


# ================================================================================================ positions table
def positions_frame(ts: np.ndarray, park_ids: Sequence[str], cap_mw: np.ndarray, rows: Mapping[str, dict],
                    actual_mwh: np.ndarray, pzu: np.ndarray, pi: np.ndarray, extra: Mapping[str, np.ndarray],
                    a: float, b: float):
    """Long table, one row per (policy, park, step). ts [D, H]; per-policy dict with sold_da, sold_id, fc
    ([D, H, P, 3] P10/P50/P90 MWh of the day-ahead forecast), pbnd ([D, H, P] MWh); actual [D, H, P]; pzu/pi [D, H];
    extra: [D, H, P] columns (bool / float) repeated for every policy."""
    import polars as pl

    D, _, P = actual_mwh.shape
    tsf = np.broadcast_to(np.asarray(ts).astype("datetime64[us]")[:, :, None], (D, H, P)).ravel()
    park = np.broadcast_to(np.array(park_ids, dtype=object)[None, None], (D, H, P)).ravel()
    pz = np.broadcast_to(pzu[:, :, None], (D, H, P))
    pp = np.broadcast_to(pi[:, :, None], (D, H, P))
    frames = []
    for name, r in rows.items():
        s = wind_cost_steps(actual_mwh, r["sold_da"], r["sold_id"], pz, pp, a, b)
        cols = {"policy": np.full(D * H * P, name, dtype=object), "ts": tsf, "park_id": park,
                "cap_mw": np.broadcast_to(cap_mw[None, None], (D, H, P)).ravel(),
                "actual_mwh": actual_mwh.ravel(), "sold_da_mwh": r["sold_da"].ravel(),
                "sold_id_mwh": np.broadcast_to(r["sold_id"], (D, H, P)).ravel(),
                "fc_p10_mwh": r["fc"][..., 0].ravel(), "fc_p50_mwh": r["fc"][..., 1].ravel(),
                "fc_p90_mwh": r["fc"][..., 2].ravel(), "pzu_lei_mwh": pz.ravel(), "pi_lei_mwh": pp.ravel(),
                "dev_mwh": s["dev"].ravel(), "short_mwh": s["short"].ravel(), "long_mwh": s["long"].ravel(),
                "imbalance_lei": s["imbalance"].ravel(), "intraday_lei": s["intraday"].ravel(),
                "avoidable_lei": s["avoidable"].ravel(), "pbnd_mwh": r["pbnd"].ravel()}
        cols.update({k: np.broadcast_to(v, (D, H, P)).ravel() for k, v in extra.items()})
        frames.append(pl.DataFrame({k: (list(v) if v.dtype == object else v) for k, v in cols.items()}))
    df = pl.concat(frames)
    return df.with_columns(pl.col("ts").dt.replace_time_zone("UTC"))


def load_positions(path) -> "object":
    """wind_positions.parquet (a W3 run's $LAB_OUTPUT_DIR) -> polars DataFrame. Filter `score_ok` before summing
    costs (curtailed / data-gap / unpriced steps are excluded for every policy)."""
    import polars as pl

    return pl.read_parquet(path)


# ================================================================================================ evaluation
def parse_args(argv):
    ap = argparse.ArgumentParser(prog="python -m sentinel.eval.wind_portfolio", description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--m5", default="train", help="train | binned | <dir / .onnx / .json> (as W2's avail_eval)")
    ap.add_argument("--m6-dir", default=None)
    ap.add_argument("--m7-dir", default=None)
    ap.add_argument("--n-scen", type=int, default=100)
    ap.add_argument("--rho", type=float, default=0.97)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--a", type=float, default=None, help="imbalance a (default: X1's frozen models/imbalance_ab.json)")
    ap.add_argument("--b", type=float, default=None)
    ap.add_argument("--max-days", type=int, default=0, help="debug: only the first N val and N test days")
    ap.add_argument("--synth", action="store_true", help="stand-in generator instead of #22/#23 output (debug)")
    ap.add_argument("--synth-days", type=int, default=731)
    ap.add_argument("--no-positions", action="store_true", help="skip writing wind_positions.parquet")
    ap.add_argument("--out", default=None)
    return ap.parse_args(argv)


def main(argv=None) -> dict:
    from sentinel.data.prices import prices_at
    from sentinel.eval.report import Report, emit
    from sentinel.infra.timing import stage
    from sentinel.model.wind.data import POWER, load_wind, make_splits
    from sentinel.model.windpower import avail_eval as w2
    from sentinel.model.windpower.availability import (AvailConfig, M7Batch, StopSurvival, availability,
                                                       fit_icing_factor, icing_risk, judge_reads, stop_runs_h)
    from sentinel.model.windpower.features import weather_per_turbine
    from sentinel.model.windpower.scenarios import ParkConformal, common_levels, park_scenarios, wake_ratio

    args = parse_args(argv if argv is not None else sys.argv[1:])
    t0 = time.time()
    a, b = load_ab() if args.a is None or args.b is None else (args.a, args.b)
    tau = sell_level(a, b)
    levels = (0.1, 0.5, 0.9, tau)
    rep = Report(vars(args), args.out)
    out_dir = rep.out_dir
    out_dir.mkdir(parents=True, exist_ok=True)
    m6_dir = Path(args.m6_dir) if args.m6_dir else w2.REPO / "models"
    m7_dir = Path(args.m7_dir) if args.m7_dir else w2.REPO / "models"

    with stage("data"):
        if not args.synth:
            from sentinel.model.wind.data import external_wind_dir

            if external_wind_dir() is None:
                from sentinel.sim.wind.build import build as build_sim

                build_sim()
        from sentinel.model.wind.synth import SynthConfig

        data = load_wind(prefer_external=not args.synth, synth_config=SynthConfig(days=args.synth_days))
        sp = make_splits(data)
    T, N, P = data.n_steps, len(data.turbines), len(data.park_ids)
    rated = data.rated_kw.astype(np.float64)
    power = data.sensors[POWER].astype(np.float64)
    lab = data.labels if data.labels is not None else np.zeros((T, N), np.uint8)
    po = data.park_of
    holdout = data.park_ids.index(sp.holdout_park)
    label = LABEL if not data.source.startswith("synth") else LABEL.replace("ERA5-driven #22/#23", "stand-in synth")
    print(f"[w3] {data.source}: {N} turbines, {T} steps, parks {data.park_ids} (holdout {sp.holdout_park}); "
          f"a={a:.4f} b={b:.4f} -> sell production P{100 * tau:.1f}", flush=True)

    with stage("m5"):
        m5 = w2.get_m5(args.m5, out_dir, args.seed)
        q = w2.predict_all(m5, data)  # [T, N, 3] kW
    print(f"[w3] M5 {m5.model_version} ({'MLP' if m5.sessions else 'binned fallback'}), beta_h "
          f"{m5.meta.get('beta_h')}", flush=True)
    with stage("m6"):
        m6 = w2.score_m6(data, m6_dir, log=print)
    with stage("m7"):
        m7 = w2.score_m7(data, m7_dir, log=print)
    with stage("curve"):
        cf = manufacturer_frac(data)

    # ---- W2's label-free fits on training days (stop survival, icing factor)
    w_t = weather_per_turbine(data.weather, po)
    surv = StopSurvival.fit(stop_runs_h(np.where(sp.train_t[:, None], m6["cls"], 0), w2.FS))
    risk_all = icing_risk(w_t, T, N, AvailConfig(survival=surv))
    ice_f, _ = fit_icing_factor(np.where(sp.train_t[:, None], power, np.nan), q[..., 1].astype(np.float64),
                                risk_all & sp.train_t[:, None])
    cfg = AvailConfig(survival=surv, icing_factor=ice_f)

    # ---- days (as W2): forecast for UTC day D sold at D-1 10:00
    day0 = np.flatnonzero((data.ts.astype("datetime64[m]") - data.ts.astype("datetime64[D]")).astype(int) == 0)
    day0 = day0[(day0 >= 36 * H) & (day0 + H <= T)]
    val_d = day0[sp.val_t[day0] & sp.val_t[np.minimum(day0 + H - 1, T - 1)]]
    test_d = day0[sp.test_t[day0]]
    if args.max_days:
        val_d, test_d = val_d[: args.max_days], test_d[: args.max_days]
    days = np.concatenate([val_d, test_d])
    D = len(days)
    stop_rl = w2.run_length(m6["cls"] == w2.FS)
    ids = data.turbine_ids
    parks, ratio_all = wake_ratio(m5, data.turbines, data.meteo, w_t, T)
    pk_idx = [parks.index(p) for p in data.park_ids]
    caps = np.array([rated[po == p].sum() for p in range(P)])  # kW
    cols_of = [np.flatnonzero(po == p) for p in range(P)]
    m7_day0 = data.ts[0].astype("datetime64[D]")
    is_normal = m6["cls"] == 0

    def avail_at(i: int, sl: slice):
        oday = str(data.ts[i].astype("datetime64[D]"))
        b6 = w2.origin_batch(m6, i, rated, power, stop_rl)
        b7 = None
        if m7 is not None:
            dk = int((data.ts[i].astype("datetime64[D]") - m7_day0).astype(int)) - 1
            if 0 <= dk < len(m7["p7"]):
                b7 = M7Batch(m7["p7"][dk], m7["alarm"][dk])
        reads = judge_reads(ids, oday, b6, b7)
        return availability(reads, data.ts[sl], weather={k: v[sl] for k, v in w_t.items()}, origin=data.ts[i],
                            cfg=cfg)

    SCEN = ("windpower_only", "sentinel_da", "sentinel_id")
    raw = {k: np.full((D, H, P, 4), np.nan) for k in SCEN}  # P10, P50, P90, sell level (frac of park capacity)
    point = {k: np.full((D, H, P), np.nan) for k in ("manufacturer_curve", "persistence")}
    share = {k: np.zeros((D, H, P)) for k in ("manufacturer_curve", "persistence", "windpower_only", "sentinel_da",
                                              "sentinel_id", "perfect")}
    with stage("forecast"):
        for k, d0 in enumerate(days):
            i0, i1 = d0 - DA_LEAD, d0 - ID_LEAD
            sl = slice(d0, d0 + H)
            av0, av1 = avail_at(i0, sl), avail_at(i1, sl)
            qd = q[sl].astype(np.float64)
            # residual at the 15:00 origin (actual - P50) on turbines M6 calls normal: M5's nowcast input
            r0 = np.where(is_normal[i1 - 1] & np.isfinite(power[i1 - 1]), power[i1 - 1] - q[i1 - 1, :, 1], np.nan)
            qd1 = nowcast(m5, qd, r0, (np.arange(H) + ID_LEAD) / 4.0)
            lag = power[persistence_index(d0, i0)]  # [H, N] kW
            stop0 = (lab[sl] == w2.FS) & (lab[i0 - 1] == w2.FS)[None]  # stopped at the sale and at delivery
            stop1 = (lab[sl] == w2.FS) & (lab[i1 - 1] == w2.FS)[None]
            u = common_levels(args.n_scen, H, P, args.rho, args.seed + k)  # common random numbers across policies
            wts = {"manufacturer_curve": (cf[sl] * rated[None], stop0), "persistence": (lag, stop0),
                   "windpower_only": (qd[..., 1], stop0), "sentinel_da": (qd[..., 1] * av0.a, stop0),
                   "sentinel_id": (qd1[..., 1] * av1.a, stop1), "perfect": (power[sl], stop0)}
            for p in range(P):
                c = cols_of[p]
                r_t = ratio_all[sl, pk_idx[p]]
                for name, qq, av in (("windpower_only", qd, None), ("sentinel_da", qd, av0), ("sentinel_id", qd1, av1)):
                    sc = park_scenarios(qq, rated, c, r_t, u[..., p], None if av is None else av.p_up,
                                        None if av is None else av.derate)
                    raw[name][k, :, p] = np.moveaxis(np.quantile(sc, levels, axis=0), 0, -1) / caps[p]
                point["manufacturer_curve"][k, :, p] = (cf[sl][:, c] * rated[c]).sum(1) / caps[p]
                fin = np.isfinite(lag[:, c])
                point["persistence"][k, :, p] = np.where(fin.any(1), np.nansum(lag[:, c], 1) / np.maximum(
                    (fin * rated[c]).sum(1), 1e-9), np.nan)
                for name, (w, st) in wts.items():
                    share[name][k, :, p] = stopped_share(w[:, c], st[:, c])
            if k % 50 == 0:
                print(f"[w3] day {k}/{D} {str(data.ts[d0])[:10]}", flush=True)

    # ---- truth, masks (as W2) and prices per (day, step, park)
    idx = days[:, None] + np.arange(H)[None]  # [D, H]
    yk = power[idx]
    y = np.stack([np.nansum(yk[..., c], -1) / caps[p] for p, c in enumerate(cols_of)], -1)  # frac of capacity
    lab_d, m6_d = lab[idx], m6["cls"][idx]
    nan_any = np.stack([(~np.isfinite(yk[..., c])).any(-1) for c in cols_of], -1)
    bad = (w2.CUR, w2.GAP)
    score_ok = ~nan_any & np.stack([~np.isin(lab_d[..., c], bad).any(-1) for c in cols_of], -1)
    cal_ok = ~nan_any & np.stack([~np.isin(m6_d[..., c], bad).any(-1) for c in cols_of], -1)
    pzu, pi = prices_at(data.ts[idx].ravel(), strict=False)
    pzu, pi = pzu.reshape(D, H), pi.reshape(D, H)
    priced = np.isfinite(pzu) & np.isfinite(pi)
    score_ok &= priced[:, :, None]
    pzu, pi = np.nan_to_num(pzu), np.nan_to_num(pi)
    is_val = np.isin(days, val_d)[:, None, None]
    is_test = np.isin(days, test_d)[:, None, None]
    train_park = (np.arange(P) != holdout)[None, None]
    stopped_day = np.stack([(lab_d[..., c] == w2.FS).any((1, 2)) for c in cols_of], -1)[:, None, :]
    cal_rows = np.broadcast_to(is_val & train_park & cal_ok, y.shape)

    # ---- distributions -> positions (fractions of capacity), calibrated on validation days x training parks
    parkname = np.broadcast_to(np.array(data.park_ids)[None, None], y.shape)
    fc, sell, conf = {}, {}, {}
    for name in SCEN:
        c = ParkConformal.fit(raw[name][..., :3][cal_rows], y[cal_rows], parkname[cal_rows])
        conf[name] = c.to_json()
        fc[name] = np.empty((D, H, P, 3))
        sell[name] = np.empty((D, H, P))
        for p, pid in enumerate(data.park_ids):
            fc[name][:, :, p], sell[name][:, :, p] = conformal_level(c, raw[name][:, :, p], pid)
    offsets = {}
    for name, x in point.items():
        off = residual_offsets(x, y, cal_rows, levels)
        offsets[name] = off.tolist()
        fc[name] = np.clip(x[..., None] + off[None, None, None, :3], 0, 1)
        sell[name] = np.clip(np.nan_to_num(x) + off[3], 0, 1)

    to_mwh = caps[None, None] * KW15_TO_MWH  # frac -> MWh per step
    act = np.nan_to_num(y) * to_mwh
    zero = np.zeros_like(act)
    rows = {}
    for name in ("manufacturer_curve", "persistence", "windpower_only", "sentinel_da"):
        s_da = sell[name] * to_mwh
        rows[name] = {"sold_da": s_da, "sold_id": zero, "fc": fc[name] * to_mwh[..., None],
                      "pbnd": promised_not_delivered(s_da, share[name])}
    s_da = sell["sentinel_da"] * to_mwh
    s_id = (sell["sentinel_id"] - sell["sentinel_da"]) * to_mwh
    rows["sentinel"] = {"sold_da": s_da, "sold_id": s_id, "fc": fc["sentinel_da"] * to_mwh[..., None],
                        "pbnd": promised_not_delivered(s_da, share["sentinel_da"], s_id, share["sentinel_id"])}
    rows["perfect"] = {"sold_da": act, "sold_id": zero, "fc": np.repeat(act[..., None], 3, -1),
                       "pbnd": promised_not_delivered(act, share["perfect"])}
    rows = {k: rows[k] for k in ROWS}

    # ---- metrics per group
    base = score_ok & is_test
    groups = {"test": base, "stopped": base & stopped_day, "holdout": base & ~train_park,
              "stopped_holdout": base & stopped_day & ~train_park, "val": score_ok & is_val}
    pz3, pi3 = pzu[:, :, None], pi[:, :, None]
    res = {g: {name: portfolio_metrics(act, r["sold_da"], r["sold_id"], pz3, pi3, np.broadcast_to(m, act.shape),
                                       r["pbnd"], a, b) for name, r in rows.items()} for g, m in groups.items()}
    for g in ("test", "stopped", "holdout"):
        pre = "" if g == "test" else f"{g}_"
        for name in ROWS:
            for mtr in METRICS:
                emit(f"{pre}{mtr}_{name}", res[g][name][mtr])
    for g in ("test", "stopped", "holdout"):
        pre = "" if g == "test" else f"{g}_"
        s_ = res[g]["sentinel"]["wind_avoidable_cost_lei"]
        for other in ("manufacturer_curve", "persistence", "windpower_only"):
            o = res[g][other]["wind_avoidable_cost_lei"]
            emit(f"{pre}sentinel_cost_cut_vs_{other}_pct", 100 * (1 - s_ / o) if o else float("nan"))
        emit(f"{pre}pi_correction_saving_lei", res[g]["sentinel_da"]["wind_avoidable_cost_lei"] - s_)
    t = res["test"]["sentinel"]
    emit("wind_avoidable_cost_lei", t["wind_avoidable_cost_lei"])
    emit("lei_per_mwh_produced", t["lei_per_mwh_produced"])
    emit("wind_abs_dev_mwh", t["abs_dev_mwh"])
    emit("promised_but_not_delivered_mwh", t["promised_but_not_delivered_mwh"])
    emit("perfect_forecast_cost_lei", res["test"]["perfect"]["wind_avoidable_cost_lei"])
    emit("sell_level", tau)
    emit("imbalance_a", a)
    emit("imbalance_b", b)
    emit("n_test_days", len(test_d))
    emit("n_stopped_park_days_test", int((stopped_day[:, 0] & is_test[:, 0]).sum()))
    emit("test_produced_mwh", res["test"]["perfect"]["produced_mwh"])
    best = min(POLICIES, key=lambda n: res["test"][n]["wind_avoidable_cost_lei"])
    emit("sentinel_cheapest_test", float(best == "sentinel"))

    if not args.no_positions:
        with stage("positions"):
            extra = {"score_ok": score_ok,
                     "split": np.where(np.broadcast_to(is_test, y.shape), "test", "val").astype(object),
                     "holdout": np.broadcast_to(~train_park, y.shape), "stopped_day": np.broadcast_to(stopped_day, y.shape)}
            df = positions_frame(data.ts[idx], data.park_ids, caps / 1000.0, rows, act, pzu, pi, extra, a, b)
            df.write_parquet(out_dir / "wind_positions.parquet")
            print(f"[w3] wrote {out_dir / 'wind_positions.parquet'} ({len(df)} rows)", flush=True)

    cfg_out = {"a": a, "b": b, "sell_level": tau, "da_origin": "D-1 10:00 UTC",
               "id_origin": "D-1 13:00 UTC (15:00 EET)", "conformal": conf, "baseline_offsets": offsets,
               "m5": m5.model_version, "m6": m6["version"], "m7": None if m7 is None else m7["version"],
               "beta_h": m5.meta.get("beta_h"), "n_scen": args.n_scen, "rho": args.rho, "label": label,
               "pi_source": "pzu (I1 #171 not merged: intraday term 0)" if not np.any(pi != pzu) else "opcom_pi"}
    for name, x in {"label": label, "data_source": data.source, "splits": sp.describe(), "results": res,
                    "config": cfg_out}.items():
        rep.table(name, x)
    rep.write()
    emit("t_total_s", time.time() - t0)
    print(f"[w3] {label}, test year, {len(test_d)} days x {P} parks:", flush=True)
    for g in ("test", "stopped", "holdout"):
        print(f"  [{g}] " + " | ".join(f"{n} {res[g][n]['wind_avoidable_cost_lei'] / 1e3:,.0f}k lei "
                                        f"{res[g][n]['lei_per_mwh_produced']:.1f} lei/MWh "
                                        f"{res[g][n]['abs_dev_mwh']:,.0f} MWh dev "
                                        f"{res[g][n]['promised_but_not_delivered_mwh']:,.1f} MWh pbnd"
                                        for n in ROWS), flush=True)
    return {"results": res, "config": cfg_out}


if __name__ == "__main__":
    main()
