"""M5 WindPower training + evaluation: per-turbine P10/P50/P90 and park totals, 1-24 h.

    python train.py --model windpower [--seed 0] [--epochs 8] [--normal m6|labels] [--synth]

Turbine level: MLP (features.FEATS + turbine and model embeddings) -> P10/P50/P90 of power / rated, a residual on
the generic class curve with pinball loss, trained only on steps M6's label-free normal selection keeps (labels only
score). Held-out park turbines use the "unseen" embedding. Horizon h: meteo at the target time stands in for the NWP
(perfect prog) + the origin residual decayed by beta_h. Park: sum of turbine P50 x availability x per-sector wake ratio;
P10/P90 by spread multipliers fitted on validation park totals. Baselines on the same masks: manufacturer curve,
persistence, binned. Real-RO check: hourly fleet capacity factor vs Energy-Charts RO wind, shape only.

LAB_METRIC val_* = validation days x training parks, *_holdout = held-out park x test year. Artifacts in
$LAB_OUTPUT_DIR: windpower.s<seed>.onnx + .json (WindPowerModel), eval_report.json.
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
import sys
import threading
import time

import numpy as np

from wattsentry.eval.baselines.wind.curve import data_inputs, model_key
from wattsentry.eval.metrics import coverage, pinball
from wattsentry.eval.report import Report, emit
from wattsentry.infra.timing import stage
from wattsentry.model.wind.data import CLASS_INDEX, POWER, external_wind_dir, load_wind, make_splits
from wattsentry.model.wind.synth import SynthConfig

from .binned import fit_binned
from .features import FEATS, TURB_W, build_data, gather, park_sector
from .model import N_SEC, WindPowerModel, park_aggregate

TAUS = (0.1, 0.5, 0.9)
HORIZONS = (1, 6, 24)  # hours; 24 = day-ahead (D1)
UNSCORED = [CLASS_INDEX[c] for c in ("curtailment", "data_gap")]  # not forecastable from weather: park scoring skips
LABEL_SIM = "synthetic RO (ERA5-driven #22/#23 parks)"
LABEL_REAL = "real RO (Energy-Charts / ENTSO-E RO wind onshore; shape only)"


def scores(q, y, m) -> dict:
    m = m & np.isfinite(y) & np.isfinite(q).all(-1)
    return {"nmae": float(np.abs(y - q[..., 1])[m].mean()) if m.any() else float("nan"),
            "pinball": float(np.mean([pinball(q[..., i], y, t, m) for i, t in enumerate(TAUS)])),
            "coverage": coverage(q[..., 0], q[..., 2], y, m), "n": int(m.sum())}


def shift(x, s):
    """x[t - s] along time (NaN / False fill)."""
    out = np.full_like(x, False if x.dtype == bool else np.nan)
    out[s:] = x[: len(x) - s]
    return out


def offsets(r, m):
    """Empirical 10 / 50 / 90 % residual quantiles: turns a point forecast into a P10/P50/P90 baseline."""
    r = r[m & np.isfinite(r)]
    return np.array([np.quantile(r, t) if t != 0.5 and len(r) else 0.0 for t in TAUS])


def normal_mask(data, sp, mode: str = "m6", log=print) -> tuple[np.ndarray, dict]:
    """Steps M6 calls normal [T, N]. 'm6' = M6's label-free normal selection (wattsentry.model.wind.normal_select, the
    mask M6 itself trains on, curves fitted on training time); 'labels' = injector labels (optimistic, for comparison).
    Single access point: swapping in M6 verdicts from the Judge adapter changes only this function."""
    rated = data.rated_kw.astype(np.float64)
    y = data.sensors[POWER] / rated[None]
    ok = np.isfinite(y) & (data.err == 0)
    if mode == "labels":
        if data.labels is None:
            raise SystemExit("--normal labels needs fault labels")
        return ok & (data.labels == 0), {"mode": "labels"}
    from wattsentry.model.wind.normal_select import select_normal, selection_quality

    v = data_inputs(data)
    power_ok, _ = select_normal(data.sensors, data.err, rated, v["nacelle"], v["met"], sp.train_t,
                                data.meteo.get("wdir_deg"))
    info = {"mode": "m6_label_free"}
    if data.labels is not None:
        where = np.zeros_like(power_ok)
        where[np.ix_(sp.train_t, sp.train_turbines)] = True
        info |= selection_quality(power_ok, data.labels, where)
    return ok & power_ok, info


class GpuUtil:
    """Samples GPU utilisation (nvidia-smi, else torch.cuda.utilization) every second while active. The GPUs are shared
    with the training campaign, so the number is the device's, not only this process's."""

    def __init__(self, period: float = 1.0):
        self.period, self.samples, self._stop = period, [], threading.Event()
        dev = (os.environ.get("CUDA_VISIBLE_DEVICES") or "0").split(",")[0]
        self.cmd = (["nvidia-smi", "--query-gpu=utilization.gpu", "--format=csv,noheader,nounits", "-i", dev]
                    if shutil.which("nvidia-smi") else None)

    def _read(self):
        if self.cmd:
            try:
                return float(subprocess.check_output(self.cmd, text=True, timeout=5).strip().splitlines()[0])
            except Exception:
                self.cmd = None
        torch = sys.modules.get("torch")
        try:
            return float(torch.cuda.utilization()) if torch is not None and torch.cuda.is_available() else None
        except Exception:
            return None

    def _loop(self):
        while not self._stop.wait(self.period):
            u = self._read()
            if u is not None:
                self.samples.append(u)

    def __enter__(self):
        self._t = threading.Thread(target=self._loop, daemon=True)
        self._t.start()
        return self

    def __exit__(self, *exc):
        self._stop.set()
        self._t.join(timeout=10)
        if self.samples:
            emit("gpu_util", float(np.mean(self.samples)))
            emit("gpu_util_max", float(np.max(self.samples)))


# --------------------------------------------------------------------------------------------------------------
def make_net(n_turb: int, n_model: int, hidden: int):
    import torch
    import torch.nn as nn
    import torch.nn.functional as F

    ci = FEATS.index("cf")

    class Net(nn.Module):
        def __init__(self):
            super().__init__()
            self.te, self.me = nn.Embedding(n_turb + 1, 8), nn.Embedding(n_model, 4)
            h = hidden
            self.net = nn.Sequential(nn.Linear(len(FEATS) + 12, h), nn.SiLU(), nn.Linear(h, h), nn.SiLU(),
                                     nn.Linear(h, h), nn.SiLU(), nn.Linear(h, 3))

        def forward(self, x, t, m):
            o = self.net(torch.cat([x, self.te(t), self.me(m)], 1))
            q50 = x[:, ci] + o[:, 1]
            return torch.stack([q50 - F.softplus(o[:, 0]), q50, q50 + F.softplus(o[:, 2])], 1)

    return Net()


def train_net(X, tid, mid, y, Xv, tidv, midv, yv, n_turb, n_model, args, log=print):
    import torch

    dev = args.device or ("cuda" if torch.cuda.is_available() else "cpu")
    torch.manual_seed(args.seed)
    tau = torch.tensor(TAUS, device=dev)

    def loss_fn(q, yy):
        d = yy[:, None] - q
        return torch.maximum(tau * d, (tau - 1) * d).mean()

    T = lambda a, dt=torch.float32: torch.as_tensor(a, dtype=dt, device=dev)  # noqa: E731
    X, tid, mid, y = T(X), T(tid, torch.long), T(mid, torch.long), T(y)
    Xv, tidv, midv, yv = T(Xv), T(tidv, torch.long), T(midv, torch.long), T(yv)
    net = make_net(n_turb, n_model, args.hidden).to(dev)
    opt = torch.optim.AdamW(net.parameters(), lr=args.lr, weight_decay=1e-4)
    steps = args.epochs * (len(y) // args.batch + 1)
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, args.lr, total_steps=steps)
    best, state, hist = np.inf, None, []
    for ep in range(args.epochs):
        net.train()
        perm, tl = torch.randperm(len(y), device=dev), 0.0
        for i in range(0, len(y), args.batch):
            b = perm[i : i + args.batch]
            t = torch.where(torch.rand(len(b), device=dev) < args.unk, 0, tid[b])  # learn the unseen-turbine embedding
            loss = loss_fn(net(X[b], t, mid[b]), y[b])
            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()
            sched.step()
            tl += loss.item() * len(b)
        net.eval()
        with torch.no_grad():
            vl = float(np.mean([loss_fn(net(Xv[i : i + 65536], tidv[i : i + 65536], midv[i : i + 65536]),
                                        yv[i : i + 65536]).item() for i in range(0, len(yv), 65536)]))
        hist.append({"epoch": ep, "train_pinball": tl / len(y), "val_pinball": vl})
        log(f"[windpower] epoch {ep} train {tl / len(y):.5f} val {vl:.5f} ({dev})", flush=True)
        if vl < best:
            best, state = vl, {k: v.detach().clone() for k, v in net.state_dict().items()}
    net.load_state_dict(state)
    net.eval()

    def predict(Xa, ta, ma):
        with torch.no_grad():
            return np.concatenate([net(T(Xa[i : i + 262144]), T(ta[i : i + 262144], torch.long),
                                       T(ma[i : i + 262144], torch.long)).cpu().numpy()
                                   for i in range(0, len(Xa), 262144)]) if len(Xa) else np.zeros((0, 3), np.float32)

    return net, predict, hist, dev


# --------------------------------------------------------------------------------------------------------------
def real_ro(data, fleet: dict[str, np.ndarray]) -> dict:
    """Hourly fleet capacity factor vs RO wind generation / installed capacity, test year (shape only)."""
    import polars as pl

    from wattsentry.sim.wind.config import RAW

    e = pl.read_parquet(RAW / "entsoe_ro.parquet").filter(pl.col("series") == "gen_wind")
    cap = pl.read_parquet(RAW / "ro_installed_capacity.parquet").filter(pl.col("series") == "gen_wind")
    cap = dict(zip(cap["year"].to_list(), cap["gw"].to_list()))
    th = data.ts.astype("datetime64[h]")
    year = int(th.max().astype("datetime64[Y]").astype(int) + 1970)
    e = e.with_columns(pl.col("ts").dt.replace_time_zone(None).dt.truncate("1h").alias("h")).group_by("h").agg(
        pl.col("mw").mean()).sort("h").filter(pl.col("h").dt.year() == year)
    hrs = e["h"].to_numpy().astype("datetime64[h]")
    real = e["mw"].to_numpy() / (cap.get(year, max(cap.values())) * 1000)
    uh, inv = np.unique(th, return_inverse=True)
    idx = np.searchsorted(uh, hrs)
    ok = (idx < len(uh)) & (uh[np.minimum(idx, len(uh) - 1)] == hrs)
    out = {"label": LABEL_REAL, "year": year, "hours": int(ok.sum())}
    if ok.sum() < 48:
        return out
    for name, f in fleet.items():
        fh = np.bincount(inv, np.nan_to_num(f), len(uh)) / np.maximum(np.bincount(inv, np.isfinite(f), len(uh)), 1)
        a, b = fh[idx[ok]], real[ok]
        d = len(a) // 24 * 24
        out[name] = {"corr_hourly": float(np.corrcoef(a, b)[0, 1]),
                     "corr_daily": float(np.corrcoef(a[:d].reshape(-1, 24).mean(1), b[:d].reshape(-1, 24).mean(1))[0, 1]),
                     "cf_mean": float(a.mean()), "real_cf_mean": float(b.mean())}
    return out


def parse_args(argv):
    ap = argparse.ArgumentParser(prog="python train.py --model windpower", description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--synth", action="store_true", help="stand-in generator (model.wind.synth) instead of the wind simulator's output")
    ap.add_argument("--synth-days", type=int, default=731)
    ap.add_argument("--synth-scale", type=float, default=1.0)
    ap.add_argument("--holdout-park", default=None)
    ap.add_argument("--normal", choices=("m6", "labels"), default="m6", help="training steps: M6 normal or labels")
    ap.add_argument("--epochs", type=int, default=8)
    ap.add_argument("--lr", type=float, default=3e-3)
    ap.add_argument("--batch", type=int, default=8192)
    ap.add_argument("--hidden", type=int, default=128)
    ap.add_argument("--unk", type=float, default=0.15, help="share of training rows fed the unknown-turbine id")
    ap.add_argument("--max-rows", type=int, default=3_000_000)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--device", default=None)
    ap.add_argument("--out", default=None)
    ap.add_argument("--no-export", action="store_true")
    ap.add_argument("--model", default="windpower", help="ignored (train.py dispatch)")
    return ap.parse_args(argv)


def main(argv=None) -> dict:
    args = parse_args(argv if argv is not None else sys.argv[1:])
    t0, rng = time.time(), np.random.default_rng(args.seed)
    rep = Report(vars(args), args.out)
    out_dir = rep.out_dir
    run_id = os.environ.get("LAB_RUN_ID") or os.environ.get("LAB_RUN") or "local"
    version = f"windpower-r{run_id}-s{args.seed}"

    with stage("data"):
        if not args.synth and external_wind_dir() is None:
            from wattsentry.sim.wind.build import build as build_sim

            build_sim()
        data = load_wind(prefer_external=not args.synth,
                         synth_config=SynthConfig(days=args.synth_days, scale=args.synth_scale))
        sp = make_splits(data, args.holdout_park)
    label = "synthetic RO (stand-in synth.py)" if data.source.startswith("synth") else LABEL_SIM
    print(f"[windpower] {data.source}: {len(data.turbines)} turbines, {data.n_steps} steps; {sp.describe()}", flush=True)

    rated = data.rated_kw.astype(np.float64)
    y = data.sensors[POWER] / rated[None]
    lab = data.labels if data.labels is not None else np.zeros(y.shape, np.uint8)
    N, T = len(data.turbines), data.n_steps

    with stage("normal"):
        m6n, sel = normal_mask(data, sp, args.normal)
    for k, v in sel.items():
        if isinstance(v, (int, float)):
            emit(f"normal_sel_{k}", v)
    emit("normal_label_free", float(args.normal == "m6"))
    truth = (lab == 0) & np.isfinite(y) & (data.err == 0) if data.labels is not None else m6n  # scoring only
    tr_t = np.zeros((T, N), bool)
    tr_t[:, sp.train_turbines] = True
    tr_m = m6n & sp.train_t[:, None] & tr_t  # fit
    vsel = m6n & sp.val_t[:, None] & tr_t  # early stopping, beta_h (label-free)
    vm = truth & sp.val_t[:, None] & tr_t  # validation score
    hm = truth & sp.test_t[:, None] & ~tr_t  # held-out park x test year

    with stage("features"):
        base = build_data(data)
        v_met = np.clip(np.nan_to_num(data_inputs(data)["met"]), 0, 35)
    tid_of = np.zeros(N, np.int64)
    tid_of[sp.train_turbines] = np.arange(1, len(sp.train_turbines) + 1)  # 0 = unknown (held-out park)
    keys = [model_key(t) for t in data.turbines]
    mkeys = sorted(set(keys))
    mid_of = np.array([mkeys.index(k) for k in keys])

    def rows(m, cap):
        ti, tj = np.nonzero(m)
        if len(ti) > cap:
            k = rng.choice(len(ti), cap, replace=False)
            ti, tj = ti[k], tj[k]
        return gather(base, ti, tj), tid_of[tj], mid_of[tj], y[ti, tj]

    with stage("train"), GpuUtil():
        net, predict, hist, dev = train_net(*rows(tr_m, args.max_rows), *rows(vsel, 300_000), len(sp.train_turbines),
                                            len(mkeys), args)
    emit("n_params", sum(p.numel() for p in net.parameters()))
    with stage("infer"):
        q = np.empty((T, N, 3), np.float32)
        for s in range(0, T, 4096):  # full [T, N, 3] prediction in time chunks
            ti, tj = (a.ravel() for a in np.meshgrid(np.arange(s, min(s + 4096, T)), np.arange(N), indexing="ij"))
            q[s : s + 4096] = predict(gather(base, ti, tj), tid_of[tj], mid_of[tj]).reshape(-1, N, 3)
        q = np.sort(np.clip(q, 0.0, 1.05), -1)

    # --- turbine level vs baselines (curve-only: no origin information) --------------------------------------------
    with stage("eval"):
        bp = fit_binned(data.turbines, v_met, y, tr_m)
        binned = bp.predict_frac(data.turbines, v_met)
        cf = base["cf"].astype(np.float64)
        manu = cf[..., None] + offsets(y - cf, tr_m)[None, None]
        systems = {"model": q, "binned": binned, "manufacturer": manu}
        res = {s: {"val": scores(x, y, vm), "test_holdout": scores(x, y, hm)} for s, x in systems.items()}
        for s, x in res.items():
            p = "" if s == "model" else f"baseline_{s}_"
            for k in ("nmae", "pinball", "coverage"):
                emit(f"{p}val_{k}_wind", x["val"][k])
                emit(f"{p}test_{k}_wind_holdout", x["test_holdout"][k])
        per_class = {}
        for i, k in enumerate(mkeys):  # pinball per turbine class (spec)
            c = vm & (mid_of == i)[None]
            per_class[k] = {s: scores(x, y, c) for s, x in systems.items()}
            emit(f"val_pinball_wind_{k}", per_class[k]["model"]["pinball"])
            emit(f"baseline_manufacturer_val_pinball_wind_{k}", per_class[k]["manufacturer"]["pinball"])

        # horizons: origin residual decay (beta_h on validation, label-free mask) vs persistence
        steps_h = {h: 4 * h for h in HORIZONS}
        hz, qh, beta_h = {}, {}, {}
        for h, s in steps_h.items():
            r0 = np.where(shift(m6n, s), shift(y - q[..., 1], s), 0.0)
            beta = float(np.nansum((r0 * (y - q[..., 1]))[vsel]) / max(np.nansum((r0**2)[vsel]), 1e-9))
            beta_h[h] = beta
            qh[h] = q + (beta * r0)[..., None]
            yp = shift(y, s)
            pers = yp[..., None] + offsets(y - yp, tr_m)[None, None]
            hz[h] = {"beta": beta, "model": {"val": scores(qh[h], y, vm), "test_holdout": scores(qh[h], y, hm)},
                     "persistence": {"val": scores(pers, y, vm), "test_holdout": scores(pers, y, hm)}}
            emit(f"val_nmae_wind_h{h}", hz[h]["model"]["val"]["nmae"])
            emit(f"test_nmae_wind_holdout_h{h}", hz[h]["model"]["test_holdout"]["nmae"])
            emit(f"baseline_persist_val_nmae_wind_h{h}", hz[h]["persistence"]["val"]["nmae"])
            emit(f"baseline_persist_test_nmae_wind_holdout_h{h}", hz[h]["persistence"]["test_holdout"]["nmae"])

    # --- park: sum x availability x sector wake ratio; P10/P90 by spread multipliers fitted on validation ------------
    with stage("park"):
        po, P = data.park_of, len(data.park_ids)
        sec = park_sector(base, po, P, N_SEC)
        yk, qk = y * rated, q[..., 1] * rated
        num, den = np.zeros((P + 1, N_SEC)), np.zeros((P + 1, N_SEC))  # row P = pooled over training parks
        for k in range(N_SEC):
            m = tr_m & (sec[:, po] == k)
            for p in range(P):
                c = po == p
                num[[p, P], k] += np.nansum(yk[:, c][m[:, c]])
                den[[p, P], k] += np.nansum(qk[:, c][m[:, c]])
        pooled = np.where(den[P] > 0, num[P] / np.maximum(den[P], 1e-9), 1.0)
        ratio = np.clip(np.where(den[:P] > 1e3 * rated.mean(), num[:P] / np.maximum(den[:P], 1e-9), pooled), 0.8, 1.2)
        holdout_p = data.park_ids.index(sp.holdout_park)
        ratio[holdout_p] = np.clip(pooled, 0.8, 1.2)  # unseen park: pooled ratio only (no peeking at its training days)
        unscored = np.isin(lab, UNSCORED) | ~np.isfinite(y)
        caps = np.array([rated[po == p].sum() for p in range(P)])
        yy = np.stack([np.nansum(yk[:, po == p], 1) / caps[p] for p in range(P)], 1)  # park actual [T, P], frac of cap
        ok = np.stack([~unscored[:, po == p].any(1) for p in range(P)], 1)
        manu_k = cf * rated
        bin_k = binned[..., 1] * rated
        tr_rows = sp.train_t[:, None] & ok & (np.arange(P) != holdout_p)[None]
        park_res, mult, headline = {}, {}, {}
        for h, s in steps_h.items():
            avail = (~shift(lab == CLASS_INDEX["fault_stop"], s)).astype(np.float64)  # labels proxy for availability
            sysq = {}
            raw = np.stack([park_aggregate(qh[h][:, po == p] * rated[po == p][None, :, None], avail[:, po == p],
                                           ratio[p, sec[:, p]], (1.0, 1.0)) / caps[p] for p in range(P)], 1)  # [T, P, 3]
            # spread multipliers on validation park totals (training parks)
            vrows = sp.val_t[:, None] & ok & np.isfinite(shift(yy, s)) & (np.arange(P) != holdout_p)[None]
            lo, hi = raw[..., 1] - raw[..., 0], raw[..., 2] - raw[..., 1]
            a = -np.quantile(((yy - raw[..., 1]) / np.maximum(lo, 1e-4))[vrows], 0.1)
            b = np.quantile(((yy - raw[..., 1]) / np.maximum(hi, 1e-4))[vrows], 0.9)
            mult[h] = (float(max(a, 0.0)), float(max(b, 0.0)))
            sysq["model"] = np.stack([np.clip(raw[..., 1] - mult[h][0] * lo, 0, None), raw[..., 1],
                                      raw[..., 1] + mult[h][1] * hi], -1)
            for name, xk in (("manufacturer", manu_k), ("binned", bin_k)):
                p50 = np.stack([(avail[:, po == p] * xk[:, po == p]).sum(1) / caps[p] for p in range(P)], 1)
                sysq[name] = p50[..., None] + offsets(yy - p50, tr_rows)[None, None]
            yp = shift(yy, s)
            sysq["persist"] = yp[..., None] + offsets(yy - yp, tr_rows & np.isfinite(yp))[None, None]
            score_ok = ok & np.isfinite(yp)
            vmask = score_ok & sp.val_t[:, None] & (np.arange(P) != holdout_p)[None]
            hmask = score_ok & sp.test_t[:, None] & (np.arange(P) == holdout_p)[None]
            tmask = score_ok & sp.test_t[:, None]
            park_res[h] = {name: {"val": scores(x, yy, vmask), "test_holdout": scores(x, yy, hmask),
                                  "test_all_parks": scores(x, yy, tmask),
                                  "val_per_park": {pid: scores(x[:, [p]], yy[:, [p]], vmask[:, [p]])
                                                   for p, pid in enumerate(data.park_ids) if p != holdout_p}}
                           for name, x in sysq.items()}
            r = park_res[h]
            sfx = "" if h == 24 else f"_h{h}"
            emit(f"val_park_nmae{sfx}", r["model"]["val"]["nmae"])
            emit(f"val_park_pinball{sfx}", r["model"]["val"]["pinball"])
            emit(f"val_park_coverage{sfx}", r["model"]["val"]["coverage"])
            emit(f"park_nmae_h{h}_holdout", r["model"]["test_holdout"]["nmae"])
            emit(f"park_pinball_h{h}_holdout", r["model"]["test_holdout"]["pinball"])
            emit(f"park_coverage_h{h}_holdout", r["model"]["test_holdout"]["coverage"])
            for name in ("manufacturer", "persist", "binned"):
                emit(f"baseline_{name}_val_park_nmae{sfx}", r[name]["val"]["nmae"])
                emit(f"baseline_{name}_val_park_pinball{sfx}", r[name]["val"]["pinball"])
                emit(f"baseline_{name}_park_nmae_h{h}_holdout", r[name]["test_holdout"]["nmae"])
            beats = all(r["model"]["val"][k] < r[b]["val"][k] for b in ("manufacturer", "persist")
                        for k in ("nmae", "pinball"))
            headline[h] = beats
            emit(f"park_beats_baselines_val{sfx}", float(beats))

    with stage("realro"):
        fleet = {k: (x * rated).sum(1) / rated.sum() for k, x in
                 (("model_p50", q[..., 1]), ("binned_p50", binned[..., 1]), ("manufacturer_p50", cf))}
        fleet["sim_actual"] = np.nansum(y * rated, 1) / rated.sum()
        ro = real_ro(data, fleet)
        for k in fleet:
            if k in ro:
                emit(f"realro_corr_hourly_{k}", ro[k]["corr_hourly"])
                emit(f"realro_corr_daily_{k}", ro[k]["corr_daily"])

    meta = {"model": "windpower", "model_version": version, "label": label, "data_source": data.source,
            "feats": list(FEATS), "models": mkeys,
            "models_rated": {k: float(np.median(rated[mid_of == i])) for i, k in enumerate(mkeys)},
            "tid_of": {data.turbine_ids[j]: int(tid_of[j]) for j in range(N) if tid_of[j] > 0},
            "wake_ratio": {pid: ratio[p].round(4).tolist() for p, pid in enumerate(data.park_ids) if p != holdout_p},
            "wake_ratio_pooled": np.clip(pooled, 0.8, 1.2).round(4).tolist(),
            "park_mult": {str(h): list(v) for h, v in mult.items()}, "beta_h": {str(h): b for h, b in beta_h.items()},
            "normal_select": sel.get("mode"), "holdout_park": sp.holdout_park, "taus": list(TAUS),
            "binned": {"curves": bp.curves, "rated_kw": bp.rated_kw, "edges": bp.edges},
            "nwp": "perfect prog: ERA5 meteo wind at the target time (no NWP error model)"}
    if not args.no_export:
        with stage("export"):
            import torch

            from wattsentry.infra.export import export_onnx

            out_dir.mkdir(parents=True, exist_ok=True)
            onnx_path = out_dir / f"windpower.s{args.seed}.onnx"
            cpu = net.to("cpu").eval()
            n = 512
            ti, tj = np.arange(n) % T, np.arange(n) % N
            ex = (torch.from_numpy(gather(base, ti, tj)), torch.from_numpy(tid_of[tj]), torch.from_numpy(mid_of[tj]))
            parity = export_onnx(cpu, ex, onnx_path, ["x", "tid", "mid"], ["q"],
                                 dynamic_axes={k: {0: "n"} for k in ("x", "tid", "mid", "q")})
            onnx_path.with_suffix(".json").write_text(json.dumps(meta, default=float))
            # predict-API parity: WindPowerModel (onnxruntime, features rebuilt from raw arrays) vs the training path
            sl = slice(0, min(T, 96 * 2))
            sub = data.subset(t_slice=sl)
            api = WindPowerModel.load(onnx_path).predict_frac(sub.turbines, sub.ts, sub.meteo,
                                                              {k: v[:, data.park_of] for k, v in sub.weather.items()})
            emit("onnx_parity_max_abs", parity)
            inner = slice(0, sl.stop - TURB_W)  # the turbulence proxy's centred window sees the slice edge
            emit("api_parity_max_abs", float(np.abs(api[inner] - q[inner]).max()))
            torch.save({"state": net.state_dict(), "meta": meta}, out_dir / f"windpower.s{args.seed}.pt")

    emit("val_nmae_wind", res["model"]["val"]["nmae"])
    emit("val_pinball_wind", res["model"]["val"]["pinball"])
    emit("t_total_s", time.time() - t0)
    for name, x in {"label": label, "real_ro_label": LABEL_REAL, "data_source": data.source, "splits": sp.describe(),
                    "normal_select": sel, "train_history": hist, "device": dev, "nwp": meta["nwp"],
                    "turbine": res, "turbine_per_class": per_class, "horizons": hz, "park": park_res,
                    "park_beats_baselines_val": {str(h): v for h, v in headline.items()},
                    "park_spread_mult": mult, "wake_ratio": meta["wake_ratio"], "real_ro": ro}.items():
        rep.table(name, x)
    rep.write()
    r24 = park_res[24]
    print(f"[windpower] {label}: park val nMAE h24 model {r24['model']['val']['nmae']:.4f} vs manufacturer "
          f"{r24['manufacturer']['val']['nmae']:.4f} vs persistence {r24['persist']['val']['nmae']:.4f} "
          f"(beats both: {headline[24]}); turbine val nMAE {res['model']['val']['nmae']:.4f} | {time.time() - t0:.0f}s",
          flush=True)
    return {"turbine": res, "horizons": hz, "park": park_res, "real_ro": ro, "headline": headline, "meta": meta}


if __name__ == "__main__":
    main()
