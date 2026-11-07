"""M4 SiteHealth: 8-class fault head on M1/M2 residuals + telemetry + 14-day summary (1D-CNN, focal loss).

    python train.py --model sitehealth [--max-minutes 25 --m1-minutes 7 --m2-minutes 4 --epochs 40 ...]

Pipeline in one run (so any worker can reproduce it): train M1 PVNet and M2 LoadNet briefly (or load
--m1-ckpt / --m2-ckpt from earlier runs), predict expected PV / load for every site-day, then train M4 on 24 h windows
ending at any quarter-hour. Baselines on the same windows: a threshold rule (energy ratio vs scaled pvlib, frozen
meter, trip codes) and CUSUM on the scaled-pvlib residual. Thresholds frozen on validation at a 5% healthy-window false-alarm rate.

LAB_METRIC: val_prauc (macro PR-AUC over the 7 fault classes), val_prauc_binary, test_prauc, hours_to_detect
(median, test episodes, hourly sliding windows up to 72 h after onset), detect_rate_72h, false alarms per site-day,
baseline_threshold_* / baseline_cusum_*.
"""

from __future__ import annotations

import argparse
import math
import time
from concurrent.futures import ThreadPoolExecutor

import numpy as np

from . import features as F
from . import metrics as M
from .data import H, make_splits, open_source, window_labels
from .harness import RunContext, export_onnx
from .synth import FAULTS

N_CLS = len(FAULTS)
DETECT_HORIZON_H = 72


def parse_args(argv=None):
    ap = argparse.ArgumentParser(prog="train.py --model sitehealth")
    ap.add_argument("--epochs", type=int, default=40)
    ap.add_argument("--steps-per-epoch", type=int, default=250)
    ap.add_argument("--lr", type=float, default=2e-3)
    ap.add_argument("--weight-decay", type=float, default=1e-4)
    ap.add_argument("--batch", type=int, default=256)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--width", type=int, default=48)
    ap.add_argument("--dropout", type=float, default=0.15)
    ap.add_argument("--gamma", type=float, default=2.0, help="focal loss gamma")
    ap.add_argument("--stat-noise", type=float, default=0.3, help="noise on static features in training")
    ap.add_argument("--fault-frac", type=float, default=0.5, help="share of train windows drawn around fault episodes")
    ap.add_argument("--patience", type=int, default=8)
    ap.add_argument("--fa-budget", type=float, default=0.05, help="healthy-window false-alarm rate for thresholds")
    ap.add_argument("--max-minutes", type=float, default=25.0)
    ap.add_argument("--m1-minutes", type=float, default=None, help="M1 budget (default 28%% of --max-minutes)")
    ap.add_argument("--m2-minutes", type=float, default=None, help="M2 budget (default 16%% of --max-minutes)")
    ap.add_argument("--m1-ckpt", default=None, help="pvnet.pt from an earlier run (skips M1 training)")
    ap.add_argument("--m2-ckpt", default=None, help="loadnet.pt from an earlier run (skips M2 training)")
    ap.add_argument("--data", default="auto")
    ap.add_argument("--synth-sites", type=int, default=None)
    ap.add_argument("--device", default="auto")
    ap.add_argument("--eval-limit", type=int, default=0)
    ap.add_argument("--max-episodes", type=int, default=0, help="cap test episodes for hours_to_detect (0 = all)")
    ap.add_argument("--no-export", action="store_true")
    ap.add_argument("--tag", default="")
    return ap.parse_args(argv)


# --- upstream models ---------------------------------------------------------------------------------------
def _load_ckpt(path, model, device):
    import torch

    from .forecast import build_model

    ck = torch.load(path, map_location=device, weights_only=False)
    a = argparse.Namespace(**ck["args"])
    net = build_model(model, a, ck["dims"], ck["stats"]).to(device)
    net.load_state_dict(ck["state"])
    return net, ck["stats"], a.mode


def upstream_predictions(src, model, net, stats, mode, device, amp, batch=2048):
    """Day-aligned predictions for every site-day -> [S, T, 3] float16 kW (first 14 days: pvlib / history-free fill)."""
    from .forecast import predict

    S, T, D = len(src.sites), src.n_steps, src.n_days
    kind = "pv" if model == "pvnet" else "load"
    out = np.zeros((S, T, 3), np.float16)
    if kind == "pv":
        pl = np.asarray(src.array("pvlib_rec"), np.float32)
        out[..., 0], out[..., 1], out[..., 2] = 0.7 * pl, pl, 1.2 * pl
    else:
        ld = np.asarray(src.array("load")[:, : 14 * H], np.float32).mean(1, keepdims=True)
        out[..., 0], out[..., 1], out[..., 2] = 0.5 * ld, ld, 2 * ld
    s_, d_ = np.meshgrid(np.arange(S), np.arange(F.HIST_DAYS, D), indexing="ij")
    s_, d_ = s_.ravel(), d_.ravel()
    for k in range(0, len(s_), 32 * batch):
        ss, dd = s_[k:k + 32 * batch], d_[k:k + 32 * batch]
        q = predict(net, src, kind, ss, dd * H, mode, stats, device, batch=batch, amp=amp)
        idx = dd[:, None] * H + np.arange(H)[None]
        out[ss[:, None], idx] = q.astype(np.float16)
    return out


# --- windows -----------------------------------------------------------------------------------------------
def daylight(raw):
    return (raw["geom"][..., 0] < 88) & (raw["geom"][..., 2] > 10)


def make_inputs(src, sites, starts, m1, m2):
    raw = F.gather(src, sites, starts, "site", "nowcast", m1=m1, m2=m2)
    inp = F.site_inputs(raw)
    lab = window_labels(raw["fault"], daylight(raw))
    return raw, inp, lab


def baseline_scores(raw, cus, starts, sites, factor):
    """Threshold rule and CUSUM scores for a batch of windows (higher = more likely faulty)."""
    pv = raw["pv"].astype(np.float32)
    exp = raw["pvlib"].astype(np.float32) * factor[sites][:, None]
    kwp = raw["kwp_recorded"].astype(np.float32)
    day = (exp > 0.05 * kwp[:, None]) & raw.get("valid", True)  # logger dropouts (0-filled) are not evidence
    ratio = (pv * day).sum(1) / np.maximum((exp * day).sum(1), 1e-3)
    # gating a tuned operator rule would use: no energy-ratio verdict on snow days or dark days
    snow = raw["wx"][..., 6].max(1) > 0.5
    dark = (exp * day).sum(1) * 0.25 < 0.5 * kwp  # < 0.5 kWh per kWp expected in the window
    ratio_term = np.where(snow | dark, 0.0, np.abs(1 - ratio))
    fz = (F.frozen_flags(pv, raw["hist_pv"][:, -1]) * day).sum(1) / np.maximum(day.sum(1), 1)
    trips = (raw["status"] == 3).sum(1)
    thr = np.maximum.reduce([ratio_term, fz, np.minimum(1.0, trips / 2.0)])
    end = starts + H - 1
    return thr, cus[sites, end]


def site_factor(src, days=60):
    """Per-site pvlib scale from the first `days` of data (commissioning check), median of daily ratios."""
    pv = np.asarray(src.array("pv_ac")[:, : days * H], np.float32).reshape(len(src.sites), days, H).sum(-1)
    pl = np.asarray(src.array("pvlib_rec")[:, : days * H], np.float32).reshape(len(src.sites), days, H).sum(-1)
    r = pv / np.maximum(pl, 1e-3)
    r = np.where(pl > 1.0, r, np.nan)
    return np.clip(np.nan_to_num(np.nanmedian(r, 1), nan=1.0), 0.3, 1.6).astype(np.float32)


def cusum_all(src, factor, k):
    pv = np.asarray(src.array("pv_ac"), np.float32)
    exp = np.asarray(src.array("pvlib_rec"), np.float32) * factor[:, None]
    kwp = src.site_meta("kwp_recorded")[:, None]
    snow = np.asarray(src.array("weather_act")[:, :, 6], np.float32)[src.loc_idx()] > 0.5
    valid = (exp > 0.05 * kwp) & ~snow  # skip snow-covered steps
    if src.has("valid"):
        valid &= np.asarray(src.array("valid")) > 0  # and logger dropouts
    r = np.where(valid, (exp - pv) / kwp, 0.0)
    return M.cusum(r, k, valid)


# --- main --------------------------------------------------------------------------------------------------
def run(argv=None):
    import torch

    from . import forecast
    from .nets import SiteHealthNet, focal_loss, n_params

    args = parse_args(argv)
    args.m1_minutes = args.m1_minutes if args.m1_minutes is not None else 0.28 * args.max_minutes
    args.m2_minutes = args.m2_minutes if args.m2_minutes is not None else 0.16 * args.max_minutes
    ctx = RunContext("sitehealth", vars(args), seed=args.seed, device=args.device)
    dev, amp = ctx.device, ctx.amp_dtype
    rng = np.random.default_rng(args.seed)
    common = ["--seed", str(args.seed), "--data", args.data, "--device", args.device]
    if args.synth_sites:
        common += ["--synth-sites", str(args.synth_sites)]
    # upstream M1/M2 are inputs here (their headline numbers come from the pvnet/loadnet runs): no LightGBM
    # reference and a capped eval keep the M4 budget for M4
    common += ["--no-lgbm", "--eval-limit", str(args.eval_limit or 20000)]

    with ctx.stage("upstream"):
        ups = {}
        for model, ck, mins, pre in (("pvnet", args.m1_ckpt, args.m1_minutes, "m1_"),
                                     ("loadnet", args.m2_ckpt, args.m2_minutes, "m2_")):
            if ck:
                net, stats, mode = _load_ckpt(ck, model, dev)
                src = None
            else:
                r = forecast.run(model, common + ["--max-minutes", str(mins), "--epochs", "60"], prefix=pre)
                net, stats, mode = r["net"], r["stats"], "nowcast"
                for k, v in r["ctx"].metrics.items():
                    ctx.metrics.setdefault(k, v)
            ups[model] = (net, stats, mode)
        src = open_source(args.data, args.synth_sites)
        sp = make_splits(src)
        m1 = upstream_predictions(src, "pvnet", *ups["pvnet"][:2], ups["pvnet"][2], dev, amp)
        m2 = upstream_predictions(src, "loadnet", *ups["loadnet"][:2], ups["loadnet"][2], dev, amp)[..., 1]

    n_days = src.n_days
    tr_s, tr_d = sp.site_days("train")
    train_set = set(sp.train_sites.tolist())
    tr_days = set(sp.days["train"].tolist())
    eps_tr = [e for e in src.episodes if e["site"] in train_set and (e["onset"] // H) in tr_days]
    lo, hi = F.HIST_DAYS * H + H, (n_days - 1) * H

    def sample_train(n):
        n_f = int(n * args.fault_frac) if eps_tr else 0
        k = rng.integers(0, len(tr_s), n - n_f)
        ends = tr_d[k] * H + rng.integers(0, H, n - n_f)
        sites = tr_s[k]
        if n_f:
            ei = rng.integers(0, len(eps_tr), n_f)
            on = np.array([eps_tr[i]["onset"] for i in ei])
            en = np.array([eps_tr[i]["end"] for i in ei])
            early = rng.random(n_f) < 0.5
            span = np.where(early, np.minimum(en - on, 3 * H), en - on + H // 2)
            e_end = on + 4 + (rng.random(n_f) * np.maximum(span, 1)).astype(int)
            sites = np.concatenate([sites, np.array([eps_tr[i]["site"] for i in ei])])
            ends = np.concatenate([ends, e_end])
        ends = np.clip(ends, lo, hi)
        return sites, ends - H + 1

    def daily(split, limit=args.eval_limit):
        s, d = sp.site_days(split)
        if limit and len(s) > limit:
            sel = np.sort(rng.choice(len(s), limit, replace=False))
            s, d = s[sel], d[sel]
        return s, d * H + 72 - H + 1  # window = 24 h ending 18:00 UTC of the target day

    with ctx.stage("stats"):
        probe = [make_inputs(src, *sample_train(512), m1, m2)[1] for _ in range(12)]
        stats = F.fit_stats(probe)
        dims = {k: probe[0][k].shape[-1] for k in ("seq", "summ", "stat")}
        # class priors: natural (uniform windows) vs our sampling mix -> logit adjustment at inference
        nat = np.concatenate([make_inputs(src, tr_s[i], tr_d[i] * H + rng.integers(0, H, len(i)) - H + 1, m1, m2)[2]
                              for i in [rng.integers(0, len(tr_s), 4096) for _ in range(3)]])
        mix = np.concatenate([make_inputs(src, *sample_train(4096), m1, m2)[2] for _ in range(3)])
        p_nat = (np.bincount(nat, minlength=N_CLS) + 1) / (len(nat) + N_CLS)
        p_mix = (np.bincount(mix, minlength=N_CLS) + 1) / (len(mix) + N_CLS)
        logit_bias = np.log(p_nat / p_mix).astype(np.float32)
        alpha = torch.tensor(np.clip((1 / p_mix) / (1 / p_mix).mean(), 0.25, 4.0) ** 0.5, dtype=torch.float32, device=dev)

    net = SiteHealthNet(dims["seq"], dims["summ"], dims["stat"], width=args.width, drop=args.dropout).to(dev)
    ctx.emit("n_params", n_params(net))
    opt = torch.optim.AdamW(net.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    total = args.epochs * args.steps_per_epoch
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=args.lr, total_steps=total, pct_start=0.08)
    scaler = torch.amp.GradScaler("cuda", enabled=(amp == torch.float16))

    def to_t(x):
        x = F.normalise(x, stats)
        return [torch.from_numpy(x[k]).to(dev) for k in ("seq", "summ", "stat")]

    def infer(sites, starts, batch=2048):
        net.eval()
        P, E, L, R = [], [], [], []
        with torch.no_grad():
            for k in range(0, len(sites), batch):
                raw, inp, lab = make_inputs(src, sites[k:k + batch], starts[k:k + batch], m1, m2)
                with ctx.autocast():
                    lg, emb = net(*to_t(inp))
                lg = lg.float().cpu().numpy() + logit_bias
                P.append(np.exp(lg - lg.max(1, keepdims=True)) / np.exp(lg - lg.max(1, keepdims=True)).sum(1, keepdims=True))
                E.append(emb.float().cpu().numpy())
                L.append(lab)
                R.append(raw)
        return np.concatenate(P), np.concatenate(E), np.concatenate(L), R

    vs, vst = daily("val", limit=args.eval_limit or 8000)
    deadline = args.max_minutes * 0.8
    best, best_state, bad, history = -math.inf, None, 0, []
    pool = ThreadPoolExecutor(max_workers=3)

    def make_batch():
        s, st = sample_train(args.batch)
        _, inp, lab = make_inputs(src, s, st, m1, m2)
        if args.stat_noise > 0:  # in raw units scaled by the train std, so it matches the normalised-space noise
            sd = np.asarray(stats["stat"]["std"], np.float32)
            inp["stat"] = inp["stat"] + np.random.default_rng().normal(0, args.stat_noise, inp["stat"].shape) * sd
        return inp, lab

    with ctx.stage("train"):
        futs = [pool.submit(make_batch) for _ in range(6)]
        step = 0
        for ep in range(args.epochs):
            net.train()
            t_ep, losses = time.time(), []
            for _ in range(args.steps_per_epoch):
                inp, lab = futs.pop(0).result()
                futs.append(pool.submit(make_batch))
                with ctx.autocast():
                    lg, _ = net(*to_t(inp))
                loss = focal_loss(lg.float(), torch.from_numpy(lab).to(dev), args.gamma, alpha)
                opt.zero_grad(set_to_none=True)
                scaler.scale(loss).backward()
                scaler.unscale_(opt)
                torch.nn.utils.clip_grad_norm_(net.parameters(), 1.0)
                scaler.step(opt)
                scaler.update()
                if step < total - 1:
                    sched.step()
                step += 1
                losses.append(float(loss.detach()))
            P, _, L, _ = infer(vs, vst)
            mac, _ = M.prauc_macro(P, L)
            binp = M.ap(L > 0, 1 - P[:, 0])
            history.append({"epoch": ep, "loss": float(np.mean(losses)), "val_prauc": mac, "val_prauc_binary": binp,
                            "sec": round(time.time() - t_ep, 1)})
            print(f"[train] ep {ep} loss {np.mean(losses):.4f} val_prauc {mac:.4f} binary {binp:.4f} "
                  f"({time.time() - t_ep:.1f}s, {ctx.elapsed_min():.1f} min)", flush=True)
            score = mac if not np.isnan(mac) else -1
            if score > best + 1e-4:
                best, bad = score, 0
                best_state = {k: v.detach().clone() for k, v in net.state_dict().items()}
            else:
                bad += 1
            if bad >= args.patience or ctx.elapsed_min() > deadline:
                print("[train] stop (patience/time)", flush=True)
                break
        for f in futs:
            f.cancel()
        pool.shutdown(wait=False, cancel_futures=True)
    if best_state is not None:
        net.load_state_dict(best_state)
    ctx.emit("epochs_run", len(history))

    report = {"model": "sitehealth", "model_version": ctx.model_version, "data_label": src.label,
              "classes": FAULTS, "history": history, "logit_bias": logit_bias.tolist(), "definitions": M.__doc__}
    with ctx.stage("eval"):
        factor = site_factor(src)
        # CUSUM allowance k tuned on validation (binary PR-AUC)
        vP, vE, vL, vR = infer(vs, vst)
        best_k, best_ap, cus = None, -1, None
        for k in (0.02, 0.05, 0.1):
            c = cusum_all(src, factor, k)
            sc = np.concatenate([baseline_scores(r, c, vst[i * 2048:(i + 1) * 2048], vs[i * 2048:(i + 1) * 2048], factor)[1]
                                 for i, r in enumerate(vR)])
            a = M.ap(vL > 0, sc)
            if a > best_ap:
                best_k, best_ap, cus = k, a, c
        thr_v = np.concatenate([baseline_scores(r, cus, vst[i * 2048:(i + 1) * 2048], vs[i * 2048:(i + 1) * 2048], factor)[0]
                                for i, r in enumerate(vR)])
        cus_v = np.concatenate([baseline_scores(r, cus, vst[i * 2048:(i + 1) * 2048], vs[i * 2048:(i + 1) * 2048], factor)[1]
                                for i, r in enumerate(vR)])
        # thresholds frozen on validation at a fixed false-alarm budget on healthy windows (same for all methods)
        hv = vL == 0
        th = {"model": float(np.quantile(1 - vP[hv, 0], 1 - args.fa_budget)),
              "threshold": float(np.quantile(thr_v[hv], 1 - args.fa_budget)),
              "cusum": float(np.quantile(cus_v[hv], 1 - args.fa_budget))}
        report["thresholds_val"] = {"rule": f"{args.fa_budget:.0%} false alarms on healthy val windows", **th}
        report["best_f1_val"] = {"model": M.best_f1_threshold(vL > 0, 1 - vP[:, 0]),
                                 "threshold": M.best_f1_threshold(vL > 0, thr_v),
                                 "cusum": M.best_f1_threshold(vL > 0, cus_v)}
        report["cusum_k"] = best_k
        mac_v, per_v = M.prauc_macro(vP, vL)
        ctx.emit("val_prauc", mac_v)
        ctx.emit("val_prauc_binary", M.ap(vL > 0, 1 - vP[:, 0]))
        ctx.emit("val_baseline_threshold_prauc_binary", M.ap(vL > 0, thr_v))
        ctx.emit("val_baseline_cusum_prauc_binary", M.ap(vL > 0, cus_v))

        ts, tst = daily("test")
        tP, tE, tL, tR = infer(ts, tst)
        thr_t = np.concatenate([baseline_scores(r, cus, tst[i * 2048:(i + 1) * 2048], ts[i * 2048:(i + 1) * 2048], factor)[0]
                                for i, r in enumerate(tR)])
        cus_t = np.concatenate([baseline_scores(r, cus, tst[i * 2048:(i + 1) * 2048], ts[i * 2048:(i + 1) * 2048], factor)[1]
                                for i, r in enumerate(tR)])
        mac_t, per_t = M.prauc_macro(tP, tL)
        ctx.emit("test_prauc", mac_t)
        ctx.emit("test_prauc_binary", M.ap(tL > 0, 1 - tP[:, 0]))
        ctx.emit("baseline_threshold_prauc_binary", M.ap(tL > 0, thr_t))
        ctx.emit("baseline_cusum_prauc_binary", M.ap(tL > 0, cus_t))
        seen = sp.seen_mask[ts]
        for g, sel in (("seen", seen), ("unseen", ~seen)):
            if sel.any():
                ctx.emit(f"test_prauc_{g}", M.prauc_macro(tP[sel], tL[sel])[0])
        pred = tP.argmax(1)
        cm = M.confusion(pred, tL)
        report["test_confusion_matrix"] = {"labels": FAULTS, "rows=true, cols=pred": cm.tolist()}
        report["test_ap_per_class"] = {FAULTS[c]: v for c, v in per_t.items()}
        report["val_ap_per_class"] = {FAULTS[c]: v for c, v in per_v.items()}
        healthy = tL == 0
        fa = {"model": float((1 - tP[healthy, 0] > th["model"]).mean()),
              "threshold": float((thr_t[healthy] > th["threshold"]).mean()),
              "cusum": float((cus_t[healthy] > th["cusum"]).mean())}
        report["false_alarms_per_healthy_site_day"] = fa
        ctx.emit("false_alarm_rate", fa["model"])
        ctx.emit("baseline_threshold_false_alarm_rate", fa["threshold"])
        ctx.emit("baseline_cusum_false_alarm_rate", fa["cusum"])

    with ctx.stage("detect"):
        test_days = set(sp.days["test"].tolist())
        eps_te = [e for e in src.episodes if (e["onset"] // H) in test_days and e["onset"] + DETECT_HORIZON_H * 4 < src.n_steps]
        if args.max_episodes and len(eps_te) > args.max_episodes:
            eps_te = [eps_te[i] for i in sorted(rng.choice(len(eps_te), args.max_episodes, replace=False))]
        hrs = np.arange(1, DETECT_HORIZON_H + 1)
        res = {"model": [], "threshold": [], "cusum": []}
        per_cls = {}
        for k0 in range(0, len(eps_te), 24):
            chunk = eps_te[k0:k0 + 24]
            s = np.repeat([e["site"] for e in chunk], len(hrs))
            ends = np.concatenate([e["onset"] + hrs * 4 - 1 for e in chunk])
            starts = ends - H + 1
            P, _, _, R = infer(s, starts)
            thr = np.concatenate([baseline_scores(r, cus, starts[i * 2048:(i + 1) * 2048], s[i * 2048:(i + 1) * 2048], factor)[0]
                                  for i, r in enumerate(R)])
            cu = np.concatenate([baseline_scores(r, cus, starts[i * 2048:(i + 1) * 2048], s[i * 2048:(i + 1) * 2048], factor)[1]
                                 for i, r in enumerate(R)])
            scores = {"model": 1 - P[:, 0], "threshold": thr, "cusum": cu}
            for j, e in enumerate(chunk):
                active = hrs * 4 <= (e["end"] - e["onset"]) + H  # stop searching a day after the episode ended
                for name, sc in scores.items():
                    v = sc[j * len(hrs):(j + 1) * len(hrs)]
                    hit = np.nonzero((v > th[name]) & active)[0]
                    h = float(hrs[hit[0]]) if len(hit) else float("nan")
                    res[name].append(h)
                    per_cls.setdefault(e["fault"], {}).setdefault(name, []).append(h)
        det = {n: M.summarize_detection(v, DETECT_HORIZON_H) for n, v in res.items()}
        report["detection_test"] = det
        report["detection_test_by_fault"] = {f: {n: M.summarize_detection(v, DETECT_HORIZON_H) for n, v in d.items()}
                                             for f, d in per_cls.items()}
        ctx.emit("hours_to_detect", det["model"]["median_hours"])
        ctx.emit("detect_rate_72h", det["model"]["detected_frac"])
        ctx.emit("hours_to_detect_mean_capped", det["model"]["mean_hours_capped"])
        for n in ("threshold", "cusum"):
            ctx.emit(f"baseline_{n}_hours_to_detect", det[n]["median_hours"])
            ctx.emit(f"baseline_{n}_detect_rate_72h", det[n]["detected_frac"])
            ctx.emit(f"baseline_{n}_hours_to_detect_mean_capped", det[n]["mean_hours_capped"])

    print("[analysis] test confusion (rows=true, cols=pred): " + " ".join(f[:6] for f in FAULTS), flush=True)
    for f, row in zip(FAULTS, cm):
        print(f"[analysis]   {f:17s} " + " ".join(f"{v:6d}" for v in row), flush=True)
    for f, d in report["detection_test_by_fault"].items():
        print(f"[analysis] detect {f:17s} " + "  ".join(
            f"{n}: rate={v['detected_frac']:.2f} med_h={v['median_hours']:.1f}" for n, v in d.items()), flush=True)
    print(f"[analysis] AP per class test {report['test_ap_per_class']}", flush=True)
    print(f"[analysis] thresholds {report['thresholds_val']} false alarms {fa}", flush=True)
    tr_s2, tr_st2 = sample_train(8192)
    _, emb_tr, _, _ = infer(tr_s2, tr_st2)
    ood = forecast.fit_ood(emb_tr)
    report["ood_test_frac_over_3"] = float((forecast.ood_score(ood, tE) > 3).mean())
    calib = {"model": "sitehealth", "model_version": ctx.model_version, "classes": FAULTS, "data_label": src.label,
             "thresholds": {"p_fault": th["model"], "ood": 3.0, "width": None, "member_std": None},
             "baseline_thresholds": {"threshold_rule": th["threshold"], "cusum": th["cusum"], "cusum_k": best_k},
             "logit_bias": logit_bias.tolist(), "ood": ood, "dims": dims,
             "channels": {"seq": list(F.SITE_SEQ), "summ": list(F.SITE_SUM), "stat": list(F.SITE_STATIC)},
             "inputs": ["seq", "summ", "stat"], "outputs": ["probs", "emb"],
             "upstream": {"pvnet": "pvnet.onnx", "loadnet": "loadnet.onnx"},
             "metrics": {k: v for k, v in ctx.metrics.items() if not k.startswith(("t_", "mem_"))}}
    if not args.no_export:
        with ctx.stage("export"):
            wrapped = _wrap(net, stats, logit_bias)
            s8, st8 = sample_train(8)
            _, inp, _ = make_inputs(src, s8, st8, m1, m2)
            info = export_onnx(wrapped, {k: inp[k] for k in ("seq", "summ", "stat")}, ctx.out / "sitehealth.onnx",
                               ["probs", "emb"])
            calib["onnx"] = info
            ctx.emit("onnx_parity_ok", float(info["parity_ok"]))
            torch.save({"state": net.state_dict(), "stats": stats, "dims": dims, "args": vars(args),
                        "logit_bias": logit_bias}, ctx.out / "sitehealth.pt")
    ctx.write_json("sitehealth.calibration.json", calib)
    report["metrics"] = ctx.metrics
    ctx.write_json("sitehealth.report.json", report)
    ctx.finish({"data_root": str(src.root), "data_digest": src.index.get("digest")})


def _wrap(net, stats, logit_bias):
    import torch

    class W(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.net = net
            for k in ("seq", "summ", "stat"):
                self.register_buffer(f"{k}_mu", torch.tensor(stats[k]["mean"], dtype=torch.float32))
                self.register_buffer(f"{k}_sd", torch.tensor(stats[k]["std"], dtype=torch.float32))
            self.register_buffer("bias", torch.tensor(logit_bias, dtype=torch.float32))

        def forward(self, seq, summ, stat):
            lg, emb = self.net((seq - self.seq_mu) / self.seq_sd, (summ - self.summ_mu) / self.summ_sd,
                               (stat - self.stat_mu) / self.stat_sd)
            return torch.softmax(lg + self.bias, -1), emb

    return W()


def main(argv=None):
    run(argv)
