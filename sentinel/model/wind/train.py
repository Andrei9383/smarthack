"""M6 WindNBM training + evaluation vs WindMon rules (card #41).

    python -m sentinel.model.wind.train [--epochs 20 --lr 2e-3 --max-minutes 25]
    train.py --model windnbm ...   →  sentinel.model.wind.train.main(argv)

Pipeline: load wind (#22/#23 output if present, else stand-in synth) → splits (time + held-out park) → NBM (GRU,
pinball, normal steps only) → binned meteo curve → LightGBM classifier on residual + park-context features →
per-class thresholds tuned on validation and frozen → held-out park / test year metrics for M6 and the WindMon
rules on the same data → ONNX (opset 17, onnxruntime parity) + classifier + calibration JSON + report + manifest
in $LAB_OUTPUT_DIR (default ./.lab-out).
"""

from __future__ import annotations

import argparse
import contextlib
import hashlib
import json
import os
import platform
import subprocess
import sys
import time
from dataclasses import asdict
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

from .data import CLASSES, CLASS_INDEX, TEMP_SENSORS, load_wind, make_splits
from .features import (
    DEFAULT_MERGE,
    DEFAULT_MIN_RUN,
    MIN_HISTORY_STEPS,
    NBM_IN,
    T_RISE_SCALE,
    V_SCALE,
    clf_features,
    curtail_ok,
    decide,
    fit_met_curve,
    input_ranges,
    nbm_inputs,
    neighbours_for,
    ood_steps,
    run_nbm,
)
from .metrics import event_metrics, step_confusion
from .normal_select import select_normal, selection_quality
from .synth import SynthConfig
from .windmon_rules import RULES, raw_alert_count, windmon_classes, windmon_flags

K = len(CLASSES)
LABEL = "synthetic RO"


# ----------------------------------------------------------------------------------------------------------------
class Run:
    def __init__(self, max_minutes: float):
        self.t0 = time.time()
        self.deadline = self.t0 + max_minutes * 60
        self.timings: dict[str, float] = {}
        self.metrics: dict[str, float] = {}

    def left(self) -> float:
        return self.deadline - time.time()

    @contextlib.contextmanager
    def stage(self, name: str):
        t = time.time()
        print(f"[windnbm] stage {name} ...", flush=True)
        yield
        dt = time.time() - t
        self.timings[f"t_{name}_s"] = round(dt, 2)
        self.emit(f"t_{name}_s", dt)

    def emit(self, name: str, value: float):
        value = float(value)
        self.metrics[name] = value
        print(f"LAB_METRIC {name}={value:.6g}", flush=True)


def _git_sha() -> str:
    try:
        return subprocess.check_output(["git", "rev-parse", "HEAD"], stderr=subprocess.DEVNULL, text=True).strip()
    except Exception:
        return os.environ.get("LAB_COMMIT", "unknown")


def _sha256(p: Path) -> str:
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for b in iter(lambda: f.read(1 << 20), b""):
            h.update(b)
    return h.hexdigest()


def parse_args(argv):
    ap = argparse.ArgumentParser(prog="windnbm", description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--epochs", type=int, default=20)
    ap.add_argument("--steps-per-epoch", type=int, default=300)
    ap.add_argument("--batch", type=int, default=256)
    ap.add_argument("--lr", type=float, default=2e-3)
    ap.add_argument("--hidden", type=int, default=96)
    ap.add_argument("--emb", type=int, default=8)
    ap.add_argument("--heads", type=int, default=3)
    ap.add_argument("--seq-len", type=int, default=128)
    ap.add_argument("--warm", type=int, default=32, help="steps at the start of each window excluded from the loss")
    ap.add_argument("--temp-weight", type=float, default=0.5)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--max-minutes", type=float, default=25.0, help="hard budget; training stops early to export")
    ap.add_argument("--reserve-minutes", type=float, default=None, help="time kept for post-training stages")
    ap.add_argument("--days", type=int, default=731, help="synthetic data only")
    ap.add_argument("--scale", type=float, default=1.0, help="synthetic data only: turbines per park multiplier")
    ap.add_argument("--data-seed", type=int, default=41)
    ap.add_argument("--synth", action="store_true", help="force the stand-in generator even if #22 output exists")
    ap.add_argument("--holdout-park", default=None)
    ap.add_argument("--clf-rounds", type=int, default=600)
    ap.add_argument("--clf-lr", type=float, default=0.05)
    ap.add_argument("--clf-normal", type=int, default=300_000, help="normal rows sampled for the classifier")
    ap.add_argument("--clf-per-class", type=int, default=80_000)
    ap.add_argument("--normal-select", choices=("labels", "label_free"), default="labels",
                    help="how NBM training picks 'normal' steps: injector labels (optimistic) or telemetry only (#70)")
    ap.add_argument("--no-eval-seen", action="store_true")
    ap.add_argument("--device", default=None)
    ap.add_argument("--out", default=None)
    ap.add_argument("--model", default="windnbm", help="ignored (train.py dispatch)")
    ap.add_argument("--profile", action=argparse.BooleanOptionalAction, default=False,
                    help="exact data-wait vs compute split (cuda sync per step)")
    ap.add_argument("--fused-adam", action=argparse.BooleanOptionalAction, default=True,
                    help="fused AdamW kernel on CUDA")
    ap.add_argument("--compile", default="off", choices=["off", "default", "reduce-overhead", "max-autotune-no-cudagraphs"],
                    help="torch.compile mode for the train forward (compile time logged as t_compile_s)")
    return ap.parse_args(argv)


# ----------------------------------------------------------------------------------------------------------------
def main(argv: list[str] | None = None) -> None:
    args = parse_args(argv if argv is not None else sys.argv[1:])
    if sys.platform == "darwin":  # torch + lightgbm each bring a libomp; on macOS they segfault unless single-threaded
        os.environ.setdefault("OMP_NUM_THREADS", "1")
    import lightgbm as lgb
    import torch

    from .nbm import WindNBM, n_params, pinball

    run = Run(args.max_minutes)
    reserve = args.reserve_minutes if args.reserve_minutes is not None else min(8.0, 0.35 * args.max_minutes)
    out = Path(args.out or os.environ.get("LAB_OUTPUT_DIR", ".lab-out"))
    out.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(args.seed)
    torch.manual_seed(args.seed)
    dev = torch.device(args.device or ("cuda" if torch.cuda.is_available() else "cpu"))
    use_amp = dev.type == "cuda" and torch.cuda.is_bf16_supported()
    run_id = os.environ.get("LAB_RUN_ID") or os.environ.get("LAB_RUN") or datetime.now(timezone.utc).strftime("%Y%m%d%H%M%S")
    version = f"windnbm-r{run_id}" + ("-lf" if args.normal_select == "label_free" else "")
    print(f"[windnbm] device={dev} bf16={use_amp} out={out} version={version}", flush=True)

    # --- data -------------------------------------------------------------------------------------------------
    with run.stage("data"):
        synth_cfg = SynthConfig(seed=args.data_seed, days=args.days, scale=args.scale)
        data = load_wind(prefer_external=not args.synth, synth_config=synth_cfg)
        sp = make_splits(data, args.holdout_park)
        park_of = data.park_of
        rated = data.rated_kw.astype(np.float64)
        is_large = (rated >= 4000).astype(np.float32)
        pres_turb = data.weather["pressure_hpa"][:, park_of]
    label = LABEL + (" (stand-in)" if data.source.startswith("synth") else "")  # until #22 lands
    if data.labels is None:
        raise SystemExit("no labels in the wind data (fault_labels missing): cannot train/evaluate M6")
    print(f"[windnbm] {len(data.turbines)} turbines, {data.n_steps} steps, source={data.source}; splits {sp.describe()}")
    N, T = len(data.turbines), data.n_steps

    with run.stage("windows"):
        label_free = args.normal_select == "label_free"
        X, Y, M = nbm_inputs(data.sensors, data.meteo, pres_turb, data.ts, rated, is_large, labels=data.labels,
                             train_mask_labels=not label_free)
        sel_quality = None
        if label_free:
            # #70: real SCADA has no labels, so pick normal steps from telemetry alone (labels only for diagnostics)
            v_nac = np.where(X[..., 1] > 0, X[..., 0] * V_SCALE, np.nan).T
            v_met = np.where(X[..., 3] > 0, X[..., 2] * V_SCALE, np.nan).T
            power_ok, temp_ok = select_normal(data.sensors, data.err, rated, v_nac, v_met, sp.train_t,
                                              data.meteo.get("wdir_deg"))
            M[..., 0] &= power_ok.T
            M[..., 1:] &= temp_ok.T[..., None]
            where = np.zeros_like(power_ok)
            where[np.ix_(sp.train_t, sp.train_turbines)] = True
            sel_quality = selection_quality(power_ok, data.labels, where)
            for k, v in sel_quality.items():
                run.emit(f"normal_sel_{k}", v)
            del v_nac, v_met
        run.emit("normal_label_free", float(label_free))
        emb_idx = np.zeros(N, dtype=np.int64)
        emb_idx[sp.train_turbines] = np.arange(1, len(sp.train_turbines) + 1)
        L, warm = args.seq_len, args.warm

        def starts_in(mask):
            c = np.concatenate([[0], np.cumsum(mask.astype(np.int64))])
            s = np.arange(0, T - L + 1)
            return s[(c[s + L] - c[s]) == L]

        tr_starts, va_starts = starts_in(sp.train_t), starts_in(sp.val_t)
        Xg = torch.from_numpy(X).to(dev)
        Yg = torch.from_numpy(Y).to(dev)
        Mg = torch.from_numpy(M).to(dev)
        ar = torch.arange(L, device=dev)
        tt = torch.from_numpy(sp.train_turbines).to(dev)
        embg = torch.from_numpy(emb_idx).to(dev)

        def batch(starts, n, gen):
            j = tt[torch.from_numpy(gen.integers(0, len(sp.train_turbines), n)).to(dev)]
            s = torch.from_numpy(starts[gen.integers(0, len(starts), n)]).to(dev)
            idx = s[:, None] + ar[None]
            return Xg[j[:, None], idx], Yg[j[:, None], idx], Mg[j[:, None], idx], embg[j]

        vgen = np.random.default_rng(1234)
        val_batches = [batch(va_starts, args.batch, vgen) for _ in range(8)]

    # --- NBM training -------------------------------------------------------------------------------------------
    model = WindNBM(len(sp.train_turbines), hidden=args.hidden, emb=args.emb, heads=args.heads).to(dev)
    run.emit("n_params", n_params(model))
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4,
                            **({"fused": True} if dev.type == "cuda" and args.fused_adam else {}))
    total_steps = args.epochs * args.steps_per_epoch
    warm_steps = max(1, total_steps // 20)

    def lr_lambda(i):  # linear warm-up, then cosine to 5 % of the peak
        if i < warm_steps:
            return (i + 1) / warm_steps
        x = min(1.0, (i - warm_steps) / max(1, total_steps - warm_steps))
        return 0.05 + 0.95 * 0.5 * (1 + np.cos(np.pi * x))

    sched = torch.optim.lr_scheduler.LambdaLR(opt, lr_lambda)
    if dev.type == "cuda":
        torch.cuda.reset_peak_memory_stats()

    from ..harness import compile_net

    fwd_train = compile_net(model.forward_all, args.compile)

    def loss_fn(xb, yb, mb, tb, train: bool):
        with torch.autocast("cuda", dtype=torch.bfloat16, enabled=use_amp):
            pq, tq = (fwd_train if train else model.forward_all)(xb, tb)
        pq, tq = pq.float()[:, warm:], tq.float()[:, warm:]
        yb, mb = yb.float()[:, warm:], mb[:, warm:]
        if train:
            boot = (torch.rand(xb.shape[0], model.heads, device=dev) < 0.8)
        else:
            boot = torch.ones(xb.shape[0], model.heads, dtype=torch.bool, device=dev)
        lp = sum(pinball(pq[:, :, h], yb[..., 0], mb[..., 0] & boot[:, h, None]) for h in range(model.heads)) / model.heads
        lt = pinball(tq, yb[..., 1:], mb[..., 1:])
        return lp + args.temp_weight * lt, lp, lt

    best, best_state, epochs_done, steps_done = float("inf"), None, 0, 0
    gen = np.random.default_rng(args.seed)
    stopped_early = False
    from ..harness import Throughput

    tp = Throughput(run.emit, dev.type, sync=args.profile)
    with run.stage("train"):
        for ep in range(args.epochs):
            model.train()
            tl = torch.zeros((), device=dev)  # summed on the device: no per-step host sync (#140)
            for _ in range(args.steps_per_epoch):
                with tp.wait():
                    xb, yb, mb, tb = batch(tr_starts, args.batch, gen)
                t_c = time.perf_counter()
                with tp.compute(xb.shape[0]):
                    loss, _, _ = loss_fn(xb, yb, mb, tb, True)
                    opt.zero_grad(set_to_none=True)
                    loss.backward()
                    torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                    opt.step()
                    sched.step()
                    tl += loss.detach()
                steps_done += 1
                if args.compile != "off" and steps_done == 1:
                    if dev.type == "cuda":
                        torch.cuda.synchronize()
                    run.emit("t_compile_s", time.perf_counter() - t_c)
                if run.left() < reserve * 60:
                    stopped_early = True
                    break
            tl = float(tl)
            model.eval()
            with tp.val(), torch.no_grad():
                vl = np.mean([[float(v) for v in loss_fn(*b, False)] for b in val_batches], axis=0)
            epochs_done = ep + 1
            print(f"[windnbm] epoch {ep + 1}/{args.epochs} train {tl / max(1, args.steps_per_epoch):.4f} "
                  f"val {vl[0]:.4f} (power {vl[1]:.4f} temp {vl[2]:.4f}) left {run.left() / 60:.1f} min", flush=True)
            if vl[0] < best:
                best = vl[0]
                best_state = {k: v.detach().clone() for k, v in model.state_dict().items()}
            if stopped_early:
                print("[windnbm] time budget reached: stopping training to export", flush=True)
                break
    tp.finish()
    model.load_state_dict(best_state)
    model.eval()
    run.emit("epochs_done", epochs_done)
    run.emit("val_loss", best)
    if dev.type == "cuda":
        run.emit("gpu_peak_mem_mb", torch.cuda.max_memory_allocated() / 2**20)
    del Xg, Yg, Mg, val_batches
    if dev.type == "cuda":
        torch.cuda.empty_cache()

    # --- NBM inference on everything ------------------------------------------------------------------------------
    def torch_fn(xb, tb):
        with torch.no_grad():
            a, b, c = model(torch.from_numpy(xb).to(dev), torch.from_numpy(tb).to(dev))
        return a.float().cpu().numpy(), b.float().cpu().numpy(), c.float().cpu().numpy()

    with run.stage("infer"):
        pq, pstd, tqr = run_nbm(torch_fn, X, emb_idx, batch=512)
        amb = data.sensors["turbine_t"]
        tq = tqr * T_RISE_SCALE + np.nan_to_num(amb, nan=10.0)[..., None, None]

    # NBM quality on validation (seen turbines, unseen time; normal steps) and on the held-out park (test year)
    y = data.sensors["act_pwt"] / rated[None]
    normal = (data.labels == 0) & np.isfinite(y)  # truth, for the reported metrics only
    vm = normal & sp.val_t[:, None]
    vm[:, sp.holdout_turbines] = False
    # what training-side calibration may use: label-free mode never sees the labels
    fit_normal = (power_ok & np.isfinite(y)) if label_free else normal
    vm_fit = fit_normal & sp.val_t[:, None]
    vm_fit[:, sp.holdout_turbines] = False
    err50 = np.abs(y - pq[..., 1])
    run.emit("val_nmae_wind", float(err50[vm].mean()))
    taus = np.array([0.1, 0.5, 0.9])
    d = y[..., None] - pq
    pin = np.maximum(taus * d, (taus - 1) * d).mean(-1)
    run.emit("val_pinball_wind", float(pin[vm].mean()))
    cover = (y >= pq[..., 0]) & (y <= pq[..., 2])
    run.emit("val_coverage_wind", float(cover[vm].mean()))
    hm = normal & sp.test_t[:, None]
    hm[:, sp.train_turbines] = False
    run.emit("test_nmae_wind_holdout", float(err50[hm].mean()))
    tmae = []
    for k, s in enumerate(TEMP_SENSORS):
        e = np.abs(data.sensors[s] - tq[:, :, k, 1])
        ok = vm & np.isfinite(e)
        tmae.append(float(e[ok].mean()))
    run.emit("val_temp_mae_c", float(np.mean(tmae)))
    # conformal-style additive offsets for the power interval (validation, normal steps)
    o10 = float(np.quantile((y - pq[..., 0])[vm_fit], 0.1))
    o90 = float(np.quantile((y - pq[..., 2])[vm_fit], 0.9))
    cal_cover = ((y >= pq[..., 0] + o10) & (y <= pq[..., 2] + o90))[hm].mean()
    run.emit("test_coverage_wind_holdout_cal", float(cal_cover))
    ranges = input_ranges(X[sp.train_turbines][:, sp.train_t])
    ood = ood_steps(X, ranges).T  # [T, N]
    vn_met = X[..., 2].T * V_SCALE
    met_curve = fit_met_curve(vn_met[:, sp.train_turbines][sp.train_t], y[:, sp.train_turbines][sp.train_t],
                              is_large[sp.train_turbines], fit_normal[:, sp.train_turbines][sp.train_t])
    del X, Y, M

    # --- classifier features ----------------------------------------------------------------------------------
    test_start = int(np.argmax(sp.test_t))
    lat = np.array([t["lat"] for t in data.turbines])
    lon = np.array([t["lon"] for t in data.turbines])

    def park_feats(p, t_lo, t_hi):
        cols = np.flatnonzero(park_of == p)
        sl = slice(t_lo, t_hi)
        wpark = {k: v[sl, p] for k, v in data.weather.items()}
        F, names = clf_features(
            {k: v[sl][:, cols] for k, v in data.sensors.items()}, data.err[sl][:, cols],
            {k: v[sl][:, cols] for k, v in data.meteo.items()}, wpark, rated[cols], is_large[cols],
            neighbours_for(lat[cols], lon[cols]), pq[sl][:, cols], pstd[sl][:, cols] / 1.0, tq[sl][:, cols], met_curve,
        )
        return cols, F, names

    train_parks = sorted({int(park_of[j]) for j in sp.train_turbines})
    hold_park = int(park_of[sp.holdout_turbines[0]])
    with run.stage("features"):
        Xtr, ytr, val_parts = [], [], []
        feat_names = None
        for p in train_parks:
            cols, F, feat_names = park_feats(p, 0, test_start)
            lab = data.labels[:test_start][:, cols]
            trm = sp.train_t[:test_start]
            Ft, lt = F[trm].reshape(-1, F.shape[-1]), lab[trm].ravel()
            Xtr.append(Ft)
            ytr.append(lt)
            vmask = sp.val_t[:test_start]
            val_parts.append((cols, F[vmask], lab[vmask]))
            del F
        Xtr, ytr = np.concatenate(Xtr), np.concatenate(ytr)
        pick = []
        for c in range(K):
            idx = np.flatnonzero(ytr == c)
            cap = args.clf_normal if c == 0 else args.clf_per_class
            if len(idx) > cap:
                idx = rng.choice(idx, cap, replace=False)
            pick.append(idx)
        pick = np.concatenate(pick)
        Xs, ys = Xtr[pick], ytr[pick]
        del Xtr, ytr
        counts = np.bincount(ys, minlength=K)
        print(f"[windnbm] classifier rows per class: {dict(zip(CLASSES, counts.tolist()))}", flush=True)

    with run.stage("classifier"):
        w_cls = np.where(counts > 0, (counts.max() / np.maximum(counts, 1)) ** 0.5, 0.0)
        Xv = np.concatenate([f.reshape(-1, f.shape[-1]) for _, f, _ in val_parts])
        yv = np.concatenate([l.ravel() for _, _, l in val_parts])
        vs = rng.choice(len(yv), min(len(yv), 300_000), replace=False)
        dtr = lgb.Dataset(Xs, ys, weight=w_cls[ys], feature_name=feat_names, free_raw_data=True)
        dva = lgb.Dataset(Xv[vs], yv[vs], weight=w_cls[yv[vs]], reference=dtr)
        params = dict(objective="multiclass", num_class=K, learning_rate=args.clf_lr, num_leaves=63, min_data_in_leaf=50,
                      feature_fraction=0.8, bagging_fraction=0.8, bagging_freq=1, lambda_l2=1.0, verbose=-1,
                      seed=args.seed, num_threads=max(1, (os.cpu_count() or 4)))
        clf_deadline = run.deadline - max(60.0, 0.5 * reserve * 60)

        def time_cb(env):
            if time.time() > clf_deadline:
                raise lgb.callback.EarlyStopException(env.iteration, env.evaluation_result_list)

        booster = lgb.train(params, dtr, num_boost_round=args.clf_rounds, valid_sets=[dva],
                            callbacks=[lgb.early_stopping(50, verbose=False), lgb.log_evaluation(50), time_cb])
        run.emit("clf_rounds", booster.best_iteration or booster.current_iteration())
        pv = booster.predict(Xv, num_iteration=booster.best_iteration).astype(np.float32)
        del Xv, Xs

    # --- tune thresholds on validation (frozen afterwards) -------------------------------------------------------
    with run.stage("tune"):
        val_probs, val_truth, val_cm = [], [], []
        o = 0
        for cols, f, l in val_parts:
            n = f.shape[0] * f.shape[1]
            val_probs.append(pv[o : o + n].reshape(f.shape[0], f.shape[1], K))
            val_truth.append(l)
            val_cm.append(curtail_ok(f, feat_names))
            o += n
        Pv = np.concatenate(val_probs, axis=1)
        Tv = np.concatenate(val_truth, axis=1)
        Cv = np.concatenate(val_cm, axis=1)
        del val_parts, pv

        def objective(th):
            m = event_metrics(Tv, decide(Pv, th, curtail_mask=Cv), full=False)
            pr, rc = m["wind_precision"], m["wind_recall"]
            f05 = 1.25 * pr * rc / (0.25 * pr + rc) if pr + rc else 0.0  # F0.5: alert fatigue weighs more
            return f05 + 0.5 * m["curtail_vs_fault_f1"] + 0.25 * m["class_correct_recall_macro"], m

        th = {c: 0.5 for c in CLASSES[1:]}
        best_obj, best_m = objective(th)
        grid = [0.15, 0.25, 0.35, 0.45, 0.55, 0.65, 0.75, 0.85, 0.93]
        for sweep in range(2):
            for c in CLASSES[1:]:
                for g in grid:
                    if run.left() < 90:
                        break
                    cand = dict(th, **{c: g})
                    ob, m = objective(cand)
                    if ob > best_obj + 1e-6:
                        best_obj, best_m, th = ob, m, cand
        print(f"[windnbm] tuned thresholds {th} → val {best_m}", flush=True)
        for k in ("wind_precision", "wind_recall", "alerts_per_turbine_day", "curtail_vs_fault_f1"):
            run.emit(f"val_{k}", best_m[k])
        del Pv, Tv, Cv

    # --- held-out park, test year: M6 vs WindMon rules on the same data ------------------------------------------
    with run.stage("eval"):
        lo = max(0, test_start - MIN_HISTORY_STEPS)
        cols, F, _ = park_feats(hold_park, lo, T)
        F = F[test_start - lo :]
        probs = booster.predict(F.reshape(-1, F.shape[-1]), num_iteration=booster.best_iteration)
        probs = probs.reshape(F.shape[0], F.shape[1], K).astype(np.float32)
        cmask = curtail_ok(F, feat_names)
        del F
        pred = decide(probs, th, curtail_mask=cmask)
        truth = data.labels[test_start:][:, cols]
        m6 = event_metrics(truth, pred)
        m6["step_confusion"] = step_confusion(truth, pred)
        sens = {k: v[test_start:][:, cols] for k, v in data.sensors.items()}
        flags = windmon_flags(sens, data.err[test_start:][:, cols], rated[cols])
        wpred = windmon_classes(flags)
        wm = event_metrics(truth, wpred)
        wm["step_confusion"] = step_confusion(truth, wpred)
        wm["raw_rule_alerts_per_turbine_day"] = raw_alert_count(flags) / (len(cols) * truth.shape[0] / 96)
        wm["rule_alerts_per_turbine_day"] = {
            r: raw_alert_count({r: flags[r]}) / (len(cols) * truth.shape[0] / 96) for r in RULES
        }
        # alert fatigue: share of curtailment episodes on which each system raised an O&M alert
        for name, mm, pr in (("m6", m6, pred), ("windmon", wm, wpred)):
            from .metrics import A_IDX, truth_episodes

            n_c = n_alert = 0
            for j in range(truth.shape[1]):
                for c, s, e in truth_episodes(truth[:, j]):
                    if c == CLASS_INDEX["curtailment"]:
                        n_c += 1
                        n_alert += int(np.isin(pr[s:e, j], A_IDX).any())
            mm["curtailment_episodes_alerted_frac"] = n_alert / n_c if n_c else 0.0
        for k in ("wind_precision", "wind_recall", "alerts_per_turbine_day", "curtail_vs_fault_f1"):
            run.emit(k, m6[k])
        for k in ("wind_precision", "wind_recall", "alerts_per_turbine_day", "curtail_vs_fault_f1"):
            run.emit(f"baseline_windmon_{k}", wm[k])
        run.emit("baseline_windmon_raw_alerts_per_turbine_day", wm["raw_rule_alerts_per_turbine_day"])
        run.emit("true_events_per_turbine_day", m6["true_actionable_per_turbine_day"])
        run.emit("wind_class_recall_macro", m6["class_correct_recall_macro"])
        for name, mm in (("M6", m6), ("WindMon", wm)):
            print(f"[windnbm] {name} per class (held-out park, test year): "
                  + json.dumps({c: [v["episodes"], v["detected_any_alert"], v["class_correct"]]
                                for c, v in mm["per_class"].items()}), flush=True)
            print(f"[windnbm] {name} episode confusion rows=truth cols=pred {CLASSES}: "
                  f"{mm['episode_confusion']['matrix']}", flush=True)
        print(f"[windnbm] curtailment episodes alerted: M6 {m6['curtailment_episodes_alerted_frac']:.3f} "
              f"WindMon {wm['curtailment_episodes_alerted_frac']:.3f}; WindMon per rule "
              f"{json.dumps(wm['rule_alerts_per_turbine_day'])}", flush=True)
        seen = None
        if not args.no_eval_seen and run.left() > 120:
            seen_t, seen_p, seen_w = [], [], []
            for p in train_parks:
                cols_s, Fs, _ = park_feats(p, lo, T)
                Fs = Fs[test_start - lo :]
                ps = booster.predict(Fs.reshape(-1, Fs.shape[-1]), num_iteration=booster.best_iteration)
                seen_p.append(decide(ps.reshape(Fs.shape[0], Fs.shape[1], K).astype(np.float32), th,
                                     curtail_mask=curtail_ok(Fs, feat_names)))
                seen_t.append(data.labels[test_start:][:, cols_s])
                fl = windmon_flags({k: v[test_start:][:, cols_s] for k, v in data.sensors.items()},
                                   data.err[test_start:][:, cols_s], rated[cols_s])
                seen_w.append(windmon_classes(fl))
                del Fs
            st = np.concatenate(seen_t, 1)
            seen = {"m6": event_metrics(st, np.concatenate(seen_p, 1)),
                    "windmon": event_metrics(st, np.concatenate(seen_w, 1))}
            run.emit("seen_wind_recall", seen["m6"]["wind_recall"])
            run.emit("seen_alerts_per_turbine_day", seen["m6"]["alerts_per_turbine_day"])

    # --- export ---------------------------------------------------------------------------------------------------
    with run.stage("export"):
        model_cpu = WindNBM(len(sp.train_turbines), hidden=args.hidden, emb=args.emb, heads=args.heads)
        model_cpu.load_state_dict({k: v.float().cpu() for k, v in model.state_dict().items()})
        model_cpu.eval()
        onnx_path = out / "windnbm.onnx"
        xd = torch.zeros(2, 128, len(NBM_IN))
        td = torch.tensor([0, 1], dtype=torch.int64)
        torch.onnx.export(
            model_cpu, (xd, td), str(onnx_path), input_names=["x", "tid"], output_names=["pq", "pstd", "tq"],
            dynamic_axes={"x": {0: "B", 1: "L"}, "tid": {0: "B"}, "pq": {0: "B", 1: "L"}, "pstd": {0: "B", 1: "L"},
                          "tq": {0: "B", 1: "L"}},
            opset_version=17, dynamo=False,
        )
        import onnxruntime as ort

        sess = ort.InferenceSession(str(onnx_path), providers=["CPUExecutionProvider"])
        pg = np.random.default_rng(7)
        xs = pg.normal(0, 0.5, (4, 200, len(NBM_IN))).astype(np.float32)
        tids = np.array([0, 1, 2, len(sp.train_turbines)], dtype=np.int64)
        with torch.no_grad():
            ref = [t.numpy() for t in model_cpu(torch.from_numpy(xs), torch.from_numpy(tids))]
        got = sess.run(None, {"x": xs, "tid": tids})
        parity = max(float(np.abs(a - b).max()) for a, b in zip(ref, got))
        run.emit("onnx_parity_max_abs", parity)
        if parity > 1e-3:
            raise SystemExit(f"ONNX parity check failed: {parity}")
        booster.save_model(str(out / "windnbm_clf.txt"), num_iteration=booster.best_iteration)
        torch.save({"state_dict": model_cpu.state_dict(), "args": vars(args)}, out / "windnbm.pt")

        vm_ = vm_fit  # validation thresholds for the qualify step (#26/#27)
        width = (pq[..., 2] - pq[..., 0])[vm_]
        calib = {
            "model": "windnbm",
            "model_version": version,
            "label": label,
            "quantile_offsets": {"all": {"q10": o10, "q90": o90, "unit": "fraction of rated_kw"}},
            "thresholds": {
                "ood": float(np.quantile(ood[sp.val_t][:, sp.train_turbines], 0.99)),
                "width": float(np.quantile(width, 0.95)),
                "member_std": float(np.quantile(pstd[vm_], 0.95)),
                "p_fault": float(min(th.values())),
            },
            "class_thresholds": th,
            "min_run": DEFAULT_MIN_RUN,
            "merge": DEFAULT_MERGE,
            "curtail_min_park_frac": 0.15,
            "classes": CLASSES,
            "turbine_index": {data.turbines[j]["turbine_id"]: int(emb_idx[j]) for j in range(N) if emb_idx[j] > 0},
            "met_curve": met_curve,
            "input_ranges": ranges,
            "feature_names": feat_names,
            "nbm_inputs": NBM_IN,
            "temp_sensors": TEMP_SENSORS,
            "v_scale": V_SCALE,
            "t_rise_scale": T_RISE_SCALE,
            "chunk": 672,
            "warm": 96,
            "min_history_steps": MIN_HISTORY_STEPS,
        }
        (out / "windnbm_thresholds.json").write_text(json.dumps(calib, indent=1))
        report = {
            "model": "windnbm",
            "model_version": version,
            "label": label,
            "data_source": data.source,
            "normal_select": {"mode": args.normal_select, "quality_on_train": sel_quality,
                              "note": "the LightGBM classifier and threshold tuning still use labels (supervised); "
                                      "label_free only changes what the NBM and its calibration treat as normal"},
            "splits": sp.describe(),
            "eval_scope": f"held-out park {sp.holdout_park}, test year {sp.test_year}",
            "definitions": __import__("sentinel.model.wind.metrics", fromlist=["x"]).__doc__,
            "m6": m6,
            "baseline_windmon": wm,
            "seen_parks_test": seen,
            "validation": best_m,
            "class_thresholds": th,
            "nbm": {k: run.metrics.get(k) for k in ("val_nmae_wind", "val_pinball_wind", "val_coverage_wind",
                                                    "test_nmae_wind_holdout", "test_coverage_wind_holdout_cal",
                                                    "val_temp_mae_c")},
            "feature_importance_gain": dict(sorted(
                zip(feat_names, booster.feature_importance("gain").tolist()), key=lambda kv: -kv[1])[:25]),
            "training": {"epochs_done": epochs_done, "steps": steps_done, "stopped_early": stopped_early},
        }
        (out / "windnbm_report.json").write_text(json.dumps(report, indent=1, default=float))

    run.emit("t_total_s", time.time() - run.t0)
    files = {p.name: _sha256(p) for p in sorted(out.glob("windnbm*"))}
    import lightgbm
    import onnx
    import onnxruntime

    manifest = {
        "model": "windnbm",
        "model_version": version,
        "git_sha": _git_sha(),
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "data": {"source": data.source, "fingerprint": data.fingerprint(), "synth_config": asdict(synth_cfg)
                 if data.source.startswith("synth") else None},
        "config": vars(args),
        "device": str(dev),
        "gpu": torch.cuda.get_device_name(0) if dev.type == "cuda" else None,
        "versions": {"python": platform.python_version(), "torch": torch.__version__, "onnx": onnx.__version__,
                     "onnxruntime": onnxruntime.__version__, "lightgbm": lightgbm.__version__,
                     "numpy": np.__version__},
        "files": files,
        "metrics": run.metrics,
        "timings": run.timings,
    }
    (out / "manifest.json").write_text(json.dumps(manifest, indent=1, default=str))
    print(f"[windnbm] done in {time.time() - run.t0:.0f}s → {out}", flush=True)


if __name__ == "__main__":
    main()
