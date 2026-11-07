"""python train.py --model battery [flags]: train + evaluate the battery behaviour model on data-v2 (card #174).

Splits (spec I step 5, no leakage): train = local years 2022-2023, validation = 2024, test = 2025; homes with
``split == unseen`` (30 %, drawn by I2) never appear in training and are the test population.

Pass 1 (per chunk of homes): meter + expected PV / load -> strategy features for every target day -> classifier
samples (train homes, every --issue-every days of the train years; validation = the same homes in 2024) and sampled
per-step rows for the quantile LightGBM (smart train homes; meter features blanked on --mask-frac of the homes so
the model also serves profile-metered homes). Fit. Pass 2: forecast the validation subset (train homes, 2024) to
choose soft vs hard mixing, then the unseen homes in 2025 -> metrics.

Metrics (LAB_METRIC, label: synthetic RO homes / real RO ERA5 weather and OPCOM PZU / proxy M1-M2 stand-ins):
  strategy_f1 (macro, unseen homes x test days), strategy_f1_home (one guess per home: the most frequent),
  net_nmae / net_nmae_17_21 (the two-stage model, home level: sum|pred - y| / sum|y| over homes x steps),
  *_portfolio (on the sum over homes), and the same for lgbm (5a P50), fo (forecast_only = load - pv, no battery)
  and persist (same step a week earlier from the meter; forecast_only for profile homes). t_<stage>_s per stage.
Outputs ($LAB_OUTPUT_DIR or --out): battery/ (models + battery_report.json), battery_guesses_test.parquet.
"""

from __future__ import annotations

import argparse
import json
import os
import time
from pathlib import Path

import numpy as np
import pandas as pd

from . import data as D
from . import features as F
from .classifier import PARAMS, PRIOR, STRATEGIES, StrategyClassifier, macro_f1
from .net_model import CAL_DAYS, NetQuantileModel, mix, residual_bias, simulate, simulate_guess

LABEL = "synthetic RO homes (data-v2) / real RO ERA5 weather + OPCOM PZU / proxy M1-M2 (physics + profile)"


def metric(name: str, value) -> None:
    try:
        v = float(value)
    except (TypeError, ValueError):
        return
    if np.isfinite(v):
        print(f"LAB_METRIC {name}={v:.6g}", flush=True)


def parse_args(argv=None):
    ap = argparse.ArgumentParser(prog="battery", description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--model", default="battery", help="ignored (train.py dispatch)")
    ap.add_argument("--data-dir", default=None, help="a generated data-v2 dir (default: I2's cached generate())")
    ap.add_argument("--quick", action="store_true", help="I2's quick world (smoke only)")
    ap.add_argument("--n-homes", type=int, default=None, help="data-v2 homes to GENERATE (default I2's 5000)")
    ap.add_argument("--max-homes", type=int, default=0, help="use at most this many homes (random, seeded); 0 = all")
    ap.add_argument("--train-years", default="2022,2023")
    ap.add_argument("--val-years", default="2024")
    ap.add_argument("--test-years", default="2025")
    ap.add_argument("--chunk", type=int, default=250, help="homes per chunk (memory ~ chunk x 4 years)")
    ap.add_argument("--issue-every", type=int, default=7, help="classifier samples every N target days")
    ap.add_argument("--rows-per-chunk", type=int, default=150_000, help="quantile-model rows sampled per chunk")
    ap.add_argument("--mask-frac", type=float, default=0.35, help="smart homes whose meter features are blanked")
    ap.add_argument("--val-homes", type=int, default=400, help="train homes forecast on the validation year")
    ap.add_argument("--rounds", type=int, default=400)
    ap.add_argument("--lr", type=float, default=0.05)
    ap.add_argument("--leaves", type=int, default=31)
    ap.add_argument("--net-leaves", type=int, default=63)
    ap.add_argument("--class-weight", choices=["sqrt", "balanced", "none"], default="sqrt")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", default=None)
    return ap.parse_args(argv)


def _years(s: str) -> list[int]:
    return [int(x) for x in str(s).split(",") if x.strip()]


# ============================================================================================ chunk features
class Chunk:
    """All per-home arrays of one chunk of homes (full time axis)."""

    def __init__(self, world: D.World, homes: pd.DataFrame):
        ids = homes["home_id"].tolist()
        self.homes = homes.reset_index(drop=True)
        self.imp, self.exp = D.meter_chunk(world, ids)
        self.imp_m, self.exp_m = D.monthly_chunk(world, ids)
        self.pv, self.load = F.expected_pv_load(world, self.homes, self.imp_m)
        self.daily = F.daily_stats(world, self.imp, self.exp, self.pv, self.load, self.homes["kwp"].to_numpy(float))
        self.sf = F.strategy_features(world, self.daily, self.homes, np.arange(world.n_days), self.imp_m, self.exp_m)
        n = len(self.homes)
        prior = {k: np.full(n, PRIOR[k]) for k in PARAMS}
        prior["k_top"] = prior["k_top"].astype(np.int32)
        self.sims = {name: simulate(self.pv, self.load, np.full(n, code, np.int8), prior, world.price, world.rank,
                                    world.med, world.cal["day_index"])[0]
                     for name, code in (("sim_self", 1), ("sim_price", 2))}

    def cls_rows(self, rows: np.ndarray, days: np.ndarray) -> np.ndarray:
        return F.stack_features({k: v[np.ix_(rows, days)] for k, v in self.sf.items()})


def sample_days(world: D.World, years, every: int, rng=None, start: int = 35) -> np.ndarray:
    days = world.days_of_years(years)
    days = days[days >= start]
    off = int(rng.integers(every)) if rng is not None else 0
    return days[off::every]


# ============================================================================================ forecast (pass 2)
def forecast_chunk(world: D.World, ch: Chunk, years, clf: StrategyClassifier, netq: NetQuantileModel | None,
                   batch: int = 50) -> dict:
    """Day-ahead forecasts for the chunk's homes over ``years`` (contiguous steps): per-strategy simulations, soft
    and hard mixes, forecast_only, persistence and the quantile LightGBM. Net import kWh per 15 min (n, Ty)."""
    steps = world.steps_of_years(years)
    days = world.days_of_years(years)
    n = len(ch.homes)
    # run from CAL_DAYS before the period so the first days already have a residual history
    d0 = max(0, int(days[0]) - CAL_DAYS)
    days_x = np.arange(d0, days[-1] + 1)
    steps_x = np.arange(world.day_start[d0], steps[-1] + 1)
    crop_d, crop_s = slice(len(days_x) - len(days), None), slice(len(steps_x) - len(steps), None)
    g = clf.guess(ch.cls_rows(np.arange(n), days_x))
    nd = len(days_x)
    probs = g["probs"].reshape(n, nd, len(STRATEGIES))
    step_day = world.cal["day_index"][steps_x] - d0
    month_params = []
    sm = world.day_month[world.cal["day_index"][steps_x]]
    for m in np.unique(sm):
        first = int(np.flatnonzero(world.day_month[days_x] == m)[0])  # params estimated on the month's first day
        prm = {k: g[k].reshape(n, nd)[:, first] for k in PARAMS}
        month_params.append((sm == m, prm))
    var = simulate_guess(world, steps_x, ch.pv[:, steps_x], ch.load[:, steps_x], month_params)
    soft = mix(var, probs, step_day) * D.DT_H
    net_x = (ch.imp - ch.exp)[:, steps_x]
    cal = soft + residual_bias(world, steps_x, net_x - soft)
    out = {"steps": steps, "days": days, "probs": probs[:, crop_d],
           "code": g["code"].reshape(n, nd)[:, crop_d], "conf": g["confidence"].reshape(n, nd)[:, crop_d],
           "params": {k: g[k].reshape(n, nd)[:, crop_d] for k in PARAMS},
           "sim_soft": soft[:, crop_s], "sim_hard": (mix(var, probs, step_day, hard=True) * D.DT_H)[:, crop_s],
           "sim_cal": cal[:, crop_s], "fo": var[0][:, crop_s] * D.DT_H}
    net = ch.imp - ch.exp
    lag = np.clip(steps - 7 * 96, 0, None)
    out["persist"] = np.where(np.isfinite(net[:, lag]), net[:, lag], out["fo"]).astype(np.float32)
    if netq is not None:
        q = np.empty((n, len(steps), 3), np.float32)
        for s in range(0, n, batch):
            rows = np.repeat(np.arange(s, min(s + batch, n)), len(steps))
            st = np.tile(steps, min(s + batch, n) - s)
            X = F.row_features(world, ch.homes, rows, st, ch.pv, ch.load, ch.imp, ch.exp, ch.sf, sims=ch.sims)
            q[s:s + batch] = netq.predict(X).reshape(-1, len(steps), 3)
        out["lgbm_q"] = q
        out["lgbm"] = q[..., 1]
    return out


class Scorer:
    """Accumulates home-level and portfolio nMAE over chunks (sum|e| / sum|y|), all steps and local 17-21."""

    def __init__(self, world: D.World, steps: np.ndarray, methods):
        h = world.cal["hour"][steps]
        self.eve = (h >= 17) & (h < 21)
        self.methods = list(methods)
        self.abs = {m: [0.0, 0.0] for m in self.methods}      # [all, 17-21]
        self.den = [0.0, 0.0]
        self.port = {m: np.zeros(len(steps)) for m in self.methods}
        self.port_y = np.zeros(len(steps))
        self.by_strat: dict = {}
        self.cover = [0.0, 0.0]

    def add(self, fc: dict, y: np.ndarray, strat: np.ndarray) -> None:
        ok = np.isfinite(y)
        y0 = np.where(ok, y, 0.0)
        self.den[0] += np.abs(y0).sum()
        self.den[1] += np.abs(y0[:, self.eve]).sum()
        self.port_y += y0.sum(0)
        for m in self.methods:
            e = np.where(ok, np.abs(fc[m] - y0), 0.0)
            self.abs[m][0] += e.sum()
            self.abs[m][1] += e[:, self.eve].sum()
            self.port[m] += np.where(ok, fc[m], 0.0).sum(0)
            for s in np.unique(strat):
                r = strat == s
                b = self.by_strat.setdefault((m, s), [0.0, 0.0])
                b[0] += e[r][:, self.eve].sum()
                b[1] += np.abs(y0[r][:, self.eve]).sum()
        if "lgbm_q" in fc:
            q = fc["lgbm_q"]
            ins = (y0 >= q[..., 0] - 1e-6) & (y0 <= q[..., 2] + 1e-6)
            self.cover[0] += (ins & ok).sum()
            self.cover[1] += ok.sum()

    def results(self, prefix: str = "") -> dict:
        r = {}
        for m in self.methods:
            r[f"{prefix}{m}_net_nmae"] = self.abs[m][0] / max(self.den[0], 1e-9)
            r[f"{prefix}{m}_net_nmae_17_21"] = self.abs[m][1] / max(self.den[1], 1e-9)
            e = np.abs(self.port[m] - self.port_y)
            r[f"{prefix}{m}_portfolio_nmae"] = e.sum() / max(np.abs(self.port_y).sum(), 1e-9)
            r[f"{prefix}{m}_portfolio_nmae_17_21"] = e[self.eve].sum() / max(np.abs(self.port_y[self.eve]).sum(), 1e-9)
        for (m, s), (a, b) in self.by_strat.items():
            r[f"{prefix}{m}_net_nmae_17_21_{s}"] = a / max(b, 1e-9)
        if self.cover[1]:
            r[f"{prefix}lgbm_coverage_10_90"] = self.cover[0] / self.cover[1]
        return r


# ============================================================================================ main
def run(argv=None) -> dict:
    a = parse_args(argv)
    t_all = time.perf_counter()
    rng = np.random.default_rng(a.seed)
    out_dir = Path(a.out or os.environ.get("LAB_OUTPUT_DIR", ".lab-out"))
    mdir = out_dir / "battery"
    mdir.mkdir(parents=True, exist_ok=True)
    timing = {}

    t = time.perf_counter()
    gen = {"n_homes": a.n_homes} if a.n_homes else {}
    data_dir = D.resolve_data(a.data_dir, quick=a.quick, **gen)
    world = D.load_world(data_dir)
    homes = world.homes
    if a.max_homes and len(homes) > a.max_homes:
        homes = homes.iloc[np.sort(rng.choice(len(homes), a.max_homes, replace=False))].reset_index(drop=True)
    lab = D.labels(data_dir, homes["home_id"]).set_index("home_id")
    code_of = {s: i for i, s in enumerate(STRATEGIES)}
    y_home = lab["strategy"].map(code_of).to_numpy(np.int64)
    homes = homes.assign(_y=y_home)
    tr_y, va_y, te_y = _years(a.train_years), _years(a.val_years), _years(a.test_years)
    have = set(np.unique(world.year))
    if not (set(tr_y) | set(va_y) | set(te_y)) <= have:
        raise SystemExit(f"battery: data has local years {sorted(have)}, splits need {tr_y}/{va_y}/{te_y}")
    timing["data"] = time.perf_counter() - t
    print(f"[battery] data-v2 {data_dir}: {len(homes)} homes "
          f"({(homes.split == 'unseen').sum()} unseen, {(homes.meter == 'smart').sum()} smart), T={world.T}", flush=True)

    # ---------------------------------------------------------------------------------------------- pass 1
    t = time.perf_counter()
    Xs, ys, Ps, Xv, yv, Pv = [], [], [], [], [], []
    R, Ry, Rv, Rvy = [], [], [], []
    tr_steps, va_steps = world.steps_of_years(tr_y), world.steps_of_years(va_y)
    train_homes = homes[homes.split != "unseen"]
    for ci, hc in enumerate(D.chunks(train_homes["home_id"], a.chunk)):
        hdf = train_homes[train_homes.home_id.isin(hc)].reset_index(drop=True)
        ch = Chunk(world, hdf)
        n = len(hdf)
        lab_c = lab.loc[hdf.home_id]
        for days, XX, YY, PP in ((sample_days(world, tr_y, a.issue_every, rng), Xs, ys, Ps),
                                 (sample_days(world, va_y, a.issue_every, rng, start=0), Xv, yv, Pv)):
            XX.append(ch.cls_rows(np.arange(n), days))
            YY.append(np.repeat(hdf["_y"].to_numpy(), len(days)))
            PP.append({k: np.repeat(np.where(lab_c["has_battery"], lab_c[k], np.nan), len(days)) for k in PARAMS})
        smart = np.flatnonzero((hdf.meter == "smart").to_numpy())
        masked = rng.random(n) < a.mask_frac
        for steps, nrows, XX, YY in ((tr_steps, a.rows_per_chunk, R, Ry), (va_steps, a.rows_per_chunk // 4, Rv, Rvy)):
            if not len(smart):
                continue
            rows = smart[rng.integers(len(smart), size=nrows * 2)]
            st = steps[rng.integers(len(steps), size=nrows * 2)]
            yy = (ch.imp - ch.exp)[rows, st]
            ok = np.flatnonzero(np.isfinite(yy))[:nrows]
            rows, st = rows[ok], st[ok]
            XX.append(F.row_features(world, hdf, rows, st, ch.pv, ch.load, ch.imp, ch.exp, ch.sf, masked[rows],
                                      sims=ch.sims))
            YY.append(yy[ok])
        print(f"[battery] pass 1 chunk {ci}: {n} homes, {time.perf_counter() - t:.0f}s", flush=True)
        del ch
    cat = lambda L: np.concatenate(L) if L else np.zeros((0, len(F.CLS_FEATURES)), np.float32)  # noqa: E731
    catP = lambda L: {k: np.concatenate([p[k] for p in L]) for k in PARAMS}  # noqa: E731
    Xs, ys, Xv, yv = cat(Xs), np.concatenate(ys), cat(Xv), np.concatenate(yv)
    timing["features"] = time.perf_counter() - t

    t = time.perf_counter()
    clf = StrategyClassifier.fit(Xs, ys, catP(Ps), X_val=Xv, y_val=yv, P_val=catP(Pv), rounds=a.rounds, lr=a.lr,
                                 leaves=a.leaves, seed=a.seed, class_weight=a.class_weight)
    clf.save(mdir)
    f1v, _ = macro_f1(yv, clf.predict_proba(Xv).argmax(1))
    timing["fit_cls"] = time.perf_counter() - t
    t = time.perf_counter()
    netq = NetQuantileModel.fit(np.concatenate(R), np.concatenate(Ry), np.concatenate(Rv), np.concatenate(Rvy),
                                rounds=a.rounds, lr=a.lr, leaves=a.net_leaves, seed=a.seed)
    netq.save(mdir)
    timing["fit_net"] = time.perf_counter() - t
    print(f"[battery] fit: classifier {len(Xs)} rows (val macro-F1 {f1v:.3f}), net quantile "
          f"{sum(map(len, Ry))} rows", flush=True)

    # ---------------------------------------------------------------------------------------------- pass 2
    methods = ["sim_soft", "sim_hard", "sim_cal", "lgbm", "fo", "persist"]
    res = {"strategy_f1_val_seen_homes": f1v}
    t = time.perf_counter()
    vh = train_homes.iloc[np.sort(rng.choice(len(train_homes), min(a.val_homes, len(train_homes)), replace=False))]
    for tag, hset, years in (("val_", vh, va_y), ("", homes[homes.split == "unseen"], te_y)):
        steps = world.steps_of_years(years)
        sc = Scorer(world, steps, methods)
        yt, yp, conf, hid = [], [], [], []
        for hc in D.chunks(hset["home_id"], a.chunk):
            hdf = hset[hset.home_id.isin(hc)].reset_index(drop=True)
            ch = Chunk(world, hdf)
            fc = forecast_chunk(world, ch, years, clf, netq)
            ti, te = D.truth_chunk(world, hdf.home_id.tolist(), years)
            sc.add(fc, (ti - te)[:, steps], lab.loc[hdf.home_id, "strategy"].to_numpy())
            every = np.arange(0, len(fc["days"]), a.issue_every)
            yt.append(np.repeat(hdf["_y"].to_numpy(), len(every)))
            yp.append(fc["code"][:, every].reshape(-1))
            conf.append(fc["conf"][:, every].reshape(-1))
            hid.append(np.repeat(hdf.home_id.to_numpy(), len(every)))
            del ch, fc
        r = sc.results(tag)
        yt, yp, conf, hid = map(np.concatenate, (yt, yp, conf, hid))
        f1, per = macro_f1(yt, yp)
        r[f"{tag}strategy_f1"] = f1
        r[f"{tag}strategy_acc"] = float((yt == yp).mean())
        r[f"{tag}battery_f1"] = macro_f1((yt > 0).astype(int), (yp > 0).astype(int), labels=[1])[0]
        r[f"{tag}strategy_conf_mean"] = float(conf.mean())
        r[f"{tag}strategy_low_conf_share"] = float((conf < 0.6).mean())
        for k, v in per.items():
            r[f"{tag}strategy_f1_{k}"] = v
        g = pd.DataFrame({"home_id": hid, "y": yt, "p": yp, "conf": conf})
        maj = g.groupby("home_id").agg(y=("y", "first"), p=("p", lambda s: np.bincount(s).argmax()),
                                       conf=("conf", "mean"))
        r[f"{tag}strategy_f1_home"] = macro_f1(maj["y"].to_numpy(), maj["p"].to_numpy())[0]
        if not tag:
            maj.assign(strategy_true=np.asarray(STRATEGIES)[maj.y], strategy_guess=np.asarray(STRATEGIES)[maj.p]) \
               .drop(columns=["y", "p"]).reset_index().to_parquet(out_dir / "battery_guesses_test.parquet")
            conf_mat = pd.crosstab(pd.Series(np.asarray(STRATEGIES)[yt], name="true"),
                                   pd.Series(np.asarray(STRATEGIES)[yp], name="guess"))
            res["confusion_test"] = conf_mat.to_dict(orient="index")  # {true: {guess: n}}
        res.update(r)
    timing["eval"] = time.perf_counter() - t

    # the two-stage model's mixing is chosen on validation (train homes, 2024), then reported on test
    best = min(("sim_soft", "sim_hard", "sim_cal"), key=lambda m: res[f"val_{m}_net_nmae"])
    res["sim_mode"] = best
    for k in ("net_nmae", "net_nmae_17_21", "portfolio_nmae", "portfolio_nmae_17_21"):
        res[f"{k}" if k.startswith("net") else f"net_{k}"] = res[f"{best}_{k}"]
    res["importance"] = clf.importance()
    timing["total"] = time.perf_counter() - t_all
    for k, v in timing.items():
        res[f"t_{k}_s"] = v
    res.update({"label": LABEL, "data_dir": str(data_dir), "n_homes": int(len(homes)), "args": vars(a)})
    (mdir / "battery_report.json").write_text(json.dumps(res, indent=1, default=str))
    for k, v in res.items():
        if not isinstance(v, (dict, str, list)):
            metric(k, v)
    print(f"[battery] label: {LABEL}")
    print(f"[battery] test (unseen homes, {te_y}): strategy_f1={res['strategy_f1']:.3f}  two-stage ({best}) "
          f"net_nmae={res['net_nmae']:.3f} 17-21={res['net_nmae_17_21']:.3f} | forecast_only "
          f"{res['fo_net_nmae']:.3f} / {res['fo_net_nmae_17_21']:.3f} | lgbm {res['lgbm_net_nmae']:.3f} / "
          f"{res['lgbm_net_nmae_17_21']:.3f} | portfolio 17-21: two-stage {res['net_portfolio_nmae_17_21']:.3f} "
          f"vs forecast_only {res['fo_portfolio_nmae_17_21']:.3f}", flush=True)
    return res


def main(argv: list[str] | None = None) -> None:
    run(argv)


if __name__ == "__main__":
    main()
