"""M1 PVNet and M2 LoadNet: training, evaluation against baselines on the same split, calibration, ONNX export.

    python train.py --model pvnet  [--mode nowcast|forecast] [--epochs 40 --lr 1e-3 --batch 256 --d-model 64 ...]
    python train.py --model loadnet [...]

Prints LAB_METRIC val_nmae, val_pinball, test_nmae, test_pinball, coverage_10_90, baseline_* and t_<stage>_s.
Writes to $LAB_OUTPUT_DIR: <model>.onnx, <model>.calibration.json, <model>.report.json, <model>.manifest.json.
"""

from __future__ import annotations

import argparse
import math
import time
from concurrent.futures import ThreadPoolExecutor
from contextlib import nullcontext

import numpy as np

from . import features as F
from . import metrics as M
from .data import H, make_splits, open_source
from .harness import RunContext, Throughput, compile_net, export_onnx, pin, resolve_compile

SPECS = {
    "pvnet": dict(kind="pv", d=96, layers=3, heads=4, ff=192, unit="kW", scale="kwp_recorded",
                  channels=dict(seq=F.PV_SEQ, summ=F.PV_SUM, stat=F.PV_STATIC)),
    "loadnet": dict(kind="load", d=80, layers=3, heads=4, ff=160, unit="kW", scale="load_hist_mean",
                    channels=dict(seq=F.LOAD_SEQ, summ=F.LOAD_SUM, stat=F.LOAD_STATIC)),
}
BUCKETS = [(0, 24), (24, 48), (48, 72), (72, 96)]  # horizon buckets (6 h each) for quantile offsets


def parse_args(model: str, argv=None):
    sp = SPECS[model]
    ap = argparse.ArgumentParser(prog=f"train.py --model {model}")
    ap.add_argument("--epochs", type=int, default=40)
    ap.add_argument("--steps-per-epoch", type=int, default=300)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--weight-decay", type=float, default=1e-4)
    ap.add_argument("--batch", type=int, default=256)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--d-model", type=int, default=sp["d"])
    ap.add_argument("--layers", type=int, default=sp["layers"])
    ap.add_argument("--heads", type=int, default=sp["heads"])
    ap.add_argument("--ff", type=int, default=sp["ff"])
    ap.add_argument("--dropout", type=float, default=0.1)
    ap.add_argument("--pointwise", type=int, default=64, help="per-step MLP branch width (0 = off)")
    ap.add_argument("--residual", action=argparse.BooleanOptionalAction, default=True,
                    help="q50 = residual on pvlib (PV) / 7-day mean profile (load)")
    ap.add_argument("--mode", choices=["nowcast", "forecast"], default="nowcast")
    ap.add_argument("--offset-frac", type=float, default=0.3, help="share of train windows with a random t0 offset")
    ap.add_argument("--stat-noise", type=float, default=0.3,
                    help="Gaussian noise (train-normalised units) on static features in training, so the static token "
                         "can't act as a site id and unseen sites rely on their own reference window")
    ap.add_argument("--patience", type=int, default=10)
    ap.add_argument("--max-minutes", type=float, default=20.0, help="wall budget; training stops early to export")
    ap.add_argument("--data", default="auto", help="auto | lab | synth | <layout dir>")
    ap.add_argument("--synth-sites", type=int, default=None)
    ap.add_argument("--device", default="auto")
    ap.add_argument("--eval-limit", type=int, default=40000,
                    help="windows per eval split, a fixed seeded sample shared by all runs and baselines (0 = all)")
    ap.add_argument("--no-export", action="store_true")
    ap.add_argument("--no-lgbm", action="store_true", help="skip the LightGBM baseline on the same split")
    ap.add_argument("--lgbm-sites", type=int, default=150, help="train sites for the in-run LightGBM baseline")
    ap.add_argument("--tag", default="", help="suffix for artifact names (ensembles: seed tag)")
    ap.add_argument("--profile", action=argparse.BooleanOptionalAction, default=False,
                    help="exact data-wait vs compute split (cuda sync per step)")
    ap.add_argument("--loader-workers", type=int, default=0, help="batch-building threads (0 = auto from CPU count)")
    ap.add_argument("--compile", default="auto", choices=["auto", "off", "default", "reduce-overhead",
                                                          "max-autotune-no-cudagraphs"],
                    help="torch.compile mode for the train forward; auto = reduce-overhead (CUDA graphs) on CUDA "
                         "sm>=7.5, else off. Compile time is logged as t_compile_s")
    ap.add_argument("--fused-adam", action=argparse.BooleanOptionalAction, default=True,
                    help="fused AdamW kernel on CUDA (one launch per step instead of one per tensor group)")
    return ap.parse_args(argv)


# --- sampling ----------------------------------------------------------------------------------------------
def step_weights(kind: str, raw: dict, inp: dict, strict: bool = False) -> np.ndarray:
    """Loss / metric mask per target step (only logged steps, `valid`).
    PV: daylight (solar zenith < 90°) and no fault on that step; strict (secondary metric) also needs the clear-sky
    profile >= 5% of recorded kWp, so dawn/dusk zeros don't count. Load: every logged step."""
    valid = raw.get("valid", np.ones(raw["y"].shape, bool))
    if kind == "pv":
        m = (raw["geom"][..., 0] < 90) & (raw["fault"] == 0) & valid
        if strict:
            m &= raw["pvclear"] >= 0.05 * raw["kwp_recorded"][:, None]
        return m.astype(np.float32)
    return valid.astype(np.float32)


def batch_arrays(src, kind, sites, starts, mode, stats):
    raw = F.gather(src, sites, starts, kind, mode)
    inp = F.build(kind, raw)
    w = step_weights(kind, raw, inp)
    y = raw["y"] / inp["scale"][:, None]
    x = F.normalise(inp, stats) if stats else inp
    mask = (~inp["night"]).astype(np.float32)
    return raw, inp, x, y.astype(np.float32), w, mask


BASE_CHANNEL = {"pvnet": "pvlib", "loadnet": "mean7"}  # residual targets (see QuantileEncoder base)


def build_model(model: str, args, dims, stats=None):
    from .nets import QuantileEncoder

    base = None
    if getattr(args, "residual", False):
        chans = list(SPECS[model]["channels"]["seq"])
        ch = chans.index(BASE_CHANNEL[model])
        mu, sd = (stats["seq"]["mean"][ch], stats["seq"]["std"][ch]) if stats else (0.0, 1.0)
        base = (ch, mu, sd)
    return QuantileEncoder(dims["seq"], dims["ref"], dims["summ"], dims["stat"], d=args.d_model, layers=args.layers,
                           heads=args.heads, ff=args.ff, drop=args.dropout, nonneg=True,
                           pointwise=getattr(args, "pointwise", 0), base=base)


class Wrapped:
    """torch module factory: normalisation baked in, so ONNX takes raw features."""

    @staticmethod
    def make(net, stats):
        import torch

        class W(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.net = net
                for k in ("seq", "ref", "summ", "stat"):
                    self.register_buffer(f"{k}_mu", torch.tensor(stats[k]["mean"], dtype=torch.float32))
                    self.register_buffer(f"{k}_sd", torch.tensor(stats[k]["std"], dtype=torch.float32))

            def forward(self, seq, ref, summ, stat, mask):
                return self.net((seq - self.seq_mu) / self.seq_sd, (ref - self.ref_mu) / self.ref_sd,
                                (summ - self.summ_mu) / self.summ_sd, (stat - self.stat_mu) / self.stat_sd, mask)

        return W()


def loader_workers() -> int:
    """Batch-building threads: the Rust gather / feature kernels release the GIL, so threads scale with cores."""
    import os

    n = len(os.sched_getaffinity(0)) if hasattr(os, "sched_getaffinity") else (os.cpu_count() or 4)
    return int(max(2, min(8, n // 2)))


def val_inputs(src, kind, sites, starts, mode, stats, device, batch=1024) -> dict:
    """Normalised inputs, targets and masks of a fixed window set, built once and kept on `device`
    (same arrays as collect(), so val_score() matches the collect()-based numbers)."""
    import torch

    parts = {k: [] for k in ("seq", "ref", "summ", "stat", "mask", "y", "w", "scale")}
    for k in range(0, len(sites), batch):
        raw, inp, x, y, w, mask = batch_arrays(src, kind, sites[k:k + batch], starts[k:k + batch], mode, stats)
        for n in ("seq", "ref", "summ", "stat"):
            parts[n].append(x[n])
        parts["mask"].append(mask)
        parts["y"].append(raw["y"].astype(np.float32))
        parts["w"].append(w)
        parts["scale"].append(inp["scale"].astype(np.float32))
    out = {k: np.concatenate(v) for k, v in parts.items()}
    dev = {k: torch.from_numpy(np.ascontiguousarray(out[k])).to(device) for k in ("seq", "ref", "summ", "stat", "mask")}
    return {"dev": dev, "y": out["y"], "w": out["w"], "scale": out["scale"]}


def val_score(net, vc: dict, amp=None, batch=2048) -> tuple[float, float]:
    """(val_pinball, val_nmae) on a val_inputs() cache, same definitions as the collect()-based epoch check."""
    import torch

    net.eval()
    d, qs = vc["dev"], []
    with torch.no_grad():
        for k in range(0, len(vc["y"]), batch):
            with _amp(amp):
                q, _ = net(*(d[n][k:k + batch] for n in ("seq", "ref", "summ", "stat", "mask")))
            qs.append(q.float())
    qn = torch.cat(qs).cpu().numpy()  # normalised units
    sc = vc["scale"]
    q = qn * sc[:, None, None]
    m = vc["w"] > 0
    return M.pinball(q / sc[:, None, None], vc["y"] / sc[:, None], m), M.nmae(q[..., 1], vc["y"], m)


def predict(net, src, kind, sites, starts, mode, stats, device, batch=1024, amp=None, with_emb=False):
    """Run a trained (unwrapped) net over windows; returns q in kW [N,96,3] (+ embeddings)."""
    import torch

    net.eval()
    qs, embs = [], []
    with torch.no_grad():
        for k in range(0, len(sites), batch):
            raw, inp, x, y, w, mask = batch_arrays(src, kind, sites[k:k + batch], starts[k:k + batch], mode, stats)
            t = [torch.from_numpy(x[n]).to(device) for n in ("seq", "ref", "summ", "stat")]
            m = torch.from_numpy(mask).to(device)
            with _amp(amp):
                q, e = net(*t, m)
            qs.append(q.float().cpu().numpy() * inp["scale"][:, None, None])
            if with_emb:
                embs.append(e.float().cpu().numpy())
    q = np.concatenate(qs) if qs else np.zeros((0, H, 3), np.float32)
    return (q, np.concatenate(embs)) if with_emb else q


def _amp(amp):
    import torch

    return torch.autocast("cuda", dtype=amp) if amp is not None else nullcontext()


def fit_ood(emb: np.ndarray) -> dict:
    """Mahalanobis on embeddings (Ledoit-Wolf shrinkage), standardised as a robust z-score of the train
    distances: score = max(0, (D - median_train) / (1.4826 MAD_train)). Same dict layout as MahalanobisOOD."""
    x = emb.astype(np.float64)
    mu = x.mean(0)
    xc = x - mu
    n, d = xc.shape
    s = xc.T @ xc / n
    m = np.trace(s) / d
    delta = s.copy()
    delta.flat[:: d + 1] -= m
    d2 = (delta**2).sum() / d
    x2 = xc**2
    beta = min(((x2.T @ x2) / n - s**2).sum() / (d * n), d2)
    alpha = float(np.clip(beta / d2 if d2 > 0 else 1.0, 0, 1))
    cov = (1 - alpha) * s + alpha * m * np.eye(d)
    cov.flat[:: d + 1] += 1e-9 * max(m, 1e-12)
    prec = np.linalg.inv(cov)
    dist = np.sqrt(np.maximum(np.einsum("ij,jk,ik->i", xc, prec, xc), 0))
    med = float(np.median(dist))
    mad = float(np.median(np.abs(dist - med)))
    return {"version": 1, "mean": mu.tolist(), "precision": prec.tolist(), "shrinkage": alpha, "d_median": med,
            "d_scale": 1.4826 * mad if mad > 0 else float(dist.std() or 1.0), "extra": {"fit_on": "train", "n": n}}


def ood_score(ood: dict, emb: np.ndarray) -> np.ndarray:
    xc = emb.astype(np.float64) - np.asarray(ood["mean"])
    dist = np.sqrt(np.maximum(np.einsum("ij,jk,ik->i", xc, np.asarray(ood["precision"]), xc), 0))
    return np.maximum(0.0, (dist - ood["d_median"]) / ood["d_scale"])


def conformal_offsets(qn: np.ndarray, yn: np.ndarray, w: np.ndarray) -> dict:
    """Asymmetric CQR offsets per horizon bucket (normalised units): P(y < q10 - lo) ≈ P(y > q90 + hi) ≈ 10 %."""
    lo, hi = [], []
    for a, b in BUCKETS:
        m = w[:, a:b] > 0
        e_lo = (qn[:, a:b, 0] - yn[:, a:b])[m]
        e_hi = (yn[:, a:b] - qn[:, a:b, 2])[m]
        if e_lo.size < 50:
            lo.append(0.0), hi.append(0.0)
            continue
        n = e_lo.size
        lvl = min(1.0, 0.9 * (1 + 1 / n))
        lo.append(float(np.quantile(e_lo, lvl)))
        hi.append(float(np.quantile(e_hi, lvl)))
    return {"buckets": BUCKETS, "q10": lo, "q90": hi}


def apply_offsets(q: np.ndarray, off: dict, scale: np.ndarray, nonneg: bool, night: np.ndarray | None = None):
    q = q.copy()
    for (a, b), lo, hi in zip(off["buckets"], off["q10"], off["q90"]):
        q[:, a:b, 0] -= lo * scale[:, None]
        q[:, a:b, 2] += hi * scale[:, None]
    if nonneg:
        q[..., 0] = np.maximum(q[..., 0], 0)
    q[..., 0] = np.minimum(q[..., 0], q[..., 1])
    q[..., 2] = np.maximum(q[..., 2], q[..., 1])
    if night is not None:
        q[night] = 0.0
    return q


# --- baselines on the same windows -------------------------------------------------------------------------
def baselines(kind: str, raw: dict) -> dict[str, np.ndarray]:
    days = F._hist_days(raw["hist_pv" if kind == "pv" else "hist_load"], F.HIST_DAYS)
    out = {"persistence_yesterday": days[:, -1], "persistence_7d_mean": days[:, -7:].mean(1)}
    if kind == "pv":
        out["pvlib"] = raw["pvlib"]
        hp = F._hist_days(raw["hist_pvlib"], 7).sum((1, 2))
        fac = np.clip(days[:, -7:].sum((1, 2)) / np.maximum(hp, 1e-3), 0.3, 1.6)
        out["pvlib_scaled"] = raw["pvlib"] * fac[:, None]
    else:
        out["persistence_week"] = days[:, -7]
    return out


def collect(src, kind, net, sites, starts, mode, stats, device, amp, batch=1024):
    """Model q (kW), targets, masks, baselines and grouping keys over a set of windows."""
    import torch

    net.eval()
    res = {k: [] for k in ("q", "y", "y_healthy", "w", "w_strict", "night", "scale", "fault", "emb", "csi", "month",
                           "offday", "snow")}
    bl: dict[str, list] = {}
    with torch.no_grad():
        for k in range(0, len(sites), batch):
            s, st = sites[k:k + batch], starts[k:k + batch]
            raw = F.gather(src, s, st, kind, mode)
            if kind == "pv":
                hidx = st[:, None] + np.arange(-7 * H, 0)[None]
                raw["hist_pvlib"] = np.asarray(src.array("pvlib_rec")[s[:, None], hidx], np.float32)
            inp = F.build(kind, raw)
            x = F.normalise(inp, stats)
            t = [torch.from_numpy(x[n]).to(device) for n in ("seq", "ref", "summ", "stat")]
            m = torch.from_numpy((~inp["night"]).astype(np.float32)).to(device)
            with _amp(amp):
                q, e = net(*t, m)
            res["q"].append(q.float().cpu().numpy() * inp["scale"][:, None, None])
            res["emb"].append(e.float().cpu().numpy())
            res["y"].append(raw["y"])
            res["y_healthy"].append(raw.get("y_healthy", raw["y"]))
            res["w"].append(step_weights(kind, raw, inp))
            res["w_strict"].append(step_weights(kind, raw, inp, strict=True))
            res["night"].append(inp["night"])
            res["scale"].append(inp["scale"])
            res["fault"].append(raw["fault"].max(1))
            csi = raw["wx"][..., 0].sum(1) / np.maximum(raw["geom"][..., 2].sum(1), 1)
            res["csi"].append(csi)
            res["month"].append(raw["cal"][:, 48, 6:8].copy())
            res["offday"].append(raw["cal"][:, 48, 4])
            res["snow"].append(raw["wx"][..., 6].max(1))
            for name, v in baselines(kind, raw).items():
                bl.setdefault(name, []).append(v)
    out = {k: np.concatenate(v) for k, v in res.items()}
    out["baselines"] = {k: np.concatenate(v) for k, v in bl.items()}
    return out


def season_of(doy_sin, doy_cos):
    doy = (np.arctan2(doy_sin, doy_cos) / (2 * np.pi) * 365.25) % 365.25
    m = np.clip((doy / 30.44).astype(int), 0, 11) + 1
    return np.select([np.isin(m, [12, 1, 2]), np.isin(m, [3, 4, 5]), np.isin(m, [6, 7, 8])], ["DJF", "MAM", "JJA"],
                     "SON")


def group_metrics(c, groups: dict[str, np.ndarray], mask_key="w", truth="y") -> dict:
    out = {}
    for gname, keys in groups.items():
        out[gname] = {}
        for g in np.unique(keys):
            sel = keys == g
            if sel.sum() < 5:
                continue
            w = c[mask_key][sel] > 0
            y = c[truth][sel]
            d = {"n_windows": int(sel.sum()), "nmae": M.nmae(c["q"][sel][..., 1], y, w),
                 "pinball": M.pinball(c["q"][sel], y, w), "coverage": M.coverage(c["qc"][sel], y, w)}
            for b, v in c["baselines"].items():
                d[f"{b}_nmae"] = M.nmae(v[sel], y, w)
            out[gname][str(g)] = d
    return out


def run(model: str, argv=None, prefix: str = "") -> dict:
    import torch

    args = parse_args(model, argv)
    spec = SPECS[model]
    kind = spec["kind"]
    name = model + (f".{args.tag}" if args.tag else "")
    ctx = RunContext(model, vars(args), seed=args.seed, device=args.device, prefix=prefix)
    dev, amp = ctx.device, ctx.amp_dtype
    rng = np.random.default_rng(args.seed)

    with ctx.stage("data"):
        src = open_source(args.data, args.synth_sites)
        sp = make_splits(src)
        n_days = src.n_days
        tr_s, tr_d = sp.site_days("train")
        print(f"[data] {src.label}: {len(src.sites)} sites ({len(sp.train_sites)} train / "
              f"{len(sp.unseen_sites)} unseen), {n_days} days; train windows {len(tr_s)}", flush=True)

    def sample_train(n):
        k = rng.integers(0, len(tr_s), n)
        st = tr_d[k] * H
        off = rng.random(n) < args.offset_frac
        st = st + np.where(off, rng.integers(-H // 2, H // 2, n), 0)
        st = np.clip(st, F.HIST_DAYS * H, (n_days - 1) * H)
        return tr_s[k], st

    def eval_set(split, limit=args.eval_limit):
        s, d = sp.site_days(split)
        if split == "cal":  # conformal set: cal days on ALL sites. Unseen sites are never trained on and cal days
            # never overlap test days, so this leaks nothing, and the offsets also cover never-seen sites
            # (calibrating on train sites only left unseen-site coverage far below target)
            s, d = np.meshgrid(np.arange(len(src.sites)), sp.days["cal"], indexing="ij")
            s, d = s.ravel(), d.ravel()
        if limit and len(s) > limit:  # fixed sample (independent of --seed): every run / seed / baseline scores
            sel = np.sort(np.random.default_rng(1234 + len(split)).choice(len(s), limit, replace=False))
            s, d = s[sel], d[sel]
        return s, d * H

    with ctx.stage("stats"):
        probe = [F.build(kind, F.gather(src, *sample_train(512), kind, args.mode)) for _ in range(16)]
        stats = F.fit_stats(probe)
        dims = {"seq": probe[0]["seq"].shape[-1], "ref": probe[0]["ref"].shape[-1],
                "summ": probe[0]["summ"].shape[-1], "stat": probe[0]["stat"].shape[-1]}

    from .nets import n_params, pinball

    net = build_model(model, args, dims, stats).to(dev)
    n_par = n_params(net)
    ctx.emit("n_params", n_par)
    cuda = str(dev).startswith("cuda")
    opt = torch.optim.AdamW(net.parameters(), lr=args.lr, weight_decay=args.weight_decay,
                            **({"fused": True} if cuda and args.fused_adam else {}))
    total = args.epochs * args.steps_per_epoch
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=args.lr, total_steps=total, pct_start=0.08,
                                                anneal_strategy="cos")
    scaler = torch.amp.GradScaler("cuda", enabled=(amp == torch.float16))
    vs, vst = eval_set("val", limit=args.eval_limit or 6000)
    val_fixed = (vs[:6000], vst[:6000])
    deadline = args.max_minutes * 0.72
    best, best_state, bad, history = math.inf, None, 0, []
    with ctx.stage("valcache"):  # the early-stopping subset is fixed: build its inputs once, keep them on the device
        vcache = val_inputs(src, kind, *val_fixed, args.mode, stats, dev)
    fwd = compile_net(net, args.compile)

    def make_batch(s, st):
        raw, inp, x, y, w, mask = batch_arrays(src, kind, s, st, args.mode, stats)
        if args.stat_noise > 0:
            noise = np.random.default_rng().normal(0, args.stat_noise, x["stat"].shape)
            x["stat"] = x["stat"] + noise.astype(np.float32)
        # pinned host tensors built in the worker, so the main thread only issues async H2D copies
        return [pin(torch.from_numpy(a)) for a in (x["seq"], x["ref"], x["summ"], x["stat"], mask, y, w)]

    n_workers = args.loader_workers or loader_workers()
    pool = ThreadPoolExecutor(max_workers=n_workers)
    ctx.emit("loader_workers", n_workers)
    tp = Throughput(ctx.emit, dev, sync=args.profile)
    with ctx.stage("train"):
        futs = [pool.submit(make_batch, *sample_train(args.batch)) for _ in range(2 * n_workers)]
        step = 0
        for ep in range(args.epochs):
            net.train()
            t_ep, losses = time.time(), []
            for _ in range(args.steps_per_epoch):
                with tp.wait():
                    host = futs.pop(0).result()
                futs.append(pool.submit(make_batch, *sample_train(args.batch)))
                t_c = time.perf_counter()
                with tp.compute(len(host[0])):
                    seq, ref, summ, stat, mt, yt, wt = (h.to(dev, non_blocking=True) for h in host)
                    with ctx.autocast():
                        q, _ = fwd(seq, ref, summ, stat, mt)
                    loss = pinball(q.float(), yt, wt)
                    opt.zero_grad(set_to_none=True)
                    scaler.scale(loss).backward()
                    scaler.unscale_(opt)
                    torch.nn.utils.clip_grad_norm_(net.parameters(), 1.0)
                    scaler.step(opt)
                    scaler.update()
                    if step < total - 1:
                        sched.step()
                    step += 1
                    losses.append(loss.detach())  # no per-step GPU sync
                if resolve_compile(args.compile) != "off" and step == 1:
                    if cuda:
                        torch.cuda.synchronize()
                    ctx.emit("t_compile_s", time.perf_counter() - t_c)
                    tp.warmup(time.perf_counter() - t_c, len(host[0]))
            # validation (fixed subset) for early stopping
            with tp.val():
                vp, vn = val_score(net, vcache, amp)
            train_loss = float(torch.stack(losses).mean())
            history.append({"epoch": ep, "train_loss": train_loss, "val_pinball": vp, "val_nmae": vn,
                            "sec": round(time.time() - t_ep, 1)})
            print(f"[train] ep {ep} loss {train_loss:.4f} val_pinball {vp:.4f} val_nmae {vn:.4f} "
                  f"({time.time() - t_ep:.1f}s, {ctx.elapsed_min():.1f} min)", flush=True)
            if vp < best - 1e-5:
                best, bad = vp, 0
                best_state = {k: v.detach().clone() for k, v in net.state_dict().items()}
            else:
                bad += 1
            if bad >= args.patience:
                print("[train] early stop", flush=True)
                break
            if ctx.elapsed_min() > deadline:
                print(f"[train] time budget reached ({ctx.elapsed_min():.1f} min)", flush=True)
                break
        for f in futs:
            f.cancel()
        pool.shutdown(wait=False, cancel_futures=True)
    tp.finish()
    if best_state is not None:
        net.load_state_dict(best_state)
    ctx.emit("epochs_run", len(history))

    report = {"model": model, "model_version": ctx.model_version, "data_label": src.label, "mode": args.mode,
              "n_params": n_par, "history": history, "definitions": M.__doc__}
    with ctx.stage("eval"):
        cs, cst = eval_set("cal")
        cal = collect(src, kind, net, cs, cst, args.mode, stats, dev, amp)
        sc = cal["scale"][:, None]
        offsets = conformal_offsets(cal["q"] / sc[..., None], cal["y"] / sc, cal["w"])
        evals = {}
        for split in ("val", "test"):
            s, st = eval_set(split)
            c = collect(src, kind, net, s, st, args.mode, stats, dev, amp)
            c["qc"] = apply_offsets(c["q"], offsets, c["scale"], True, c["night"])
            c["site"] = s
            c["start"] = st
            evals[split] = c
        from . import lgbm_ref

        if not args.no_lgbm and lgbm_ref.available():
            # the reference LightGBM (its own feature code / params) on the SAME windows, split and mask -> "lgbm"
            with ctx.stage("lgbm"):
                try:
                    bundle = lgbm_ref.fit(src, sp, kind, args.mode, max_train_sites=args.lgbm_sites, seed=args.seed,
                                          log=lambda m: print(f"[lgbm] {m}", flush=True))
                    for split, c in evals.items():
                        lq = lgbm_ref.predict(bundle, src, c["site"], c["start"] // H, args.mode)
                        c["baselines"]["lgbm"] = lq[..., 1]
                        w = c["w"] > 0
                        ctx.emit(f"{split}_baseline_lgbm_pinball", M.pinball(lq, c["y"], w))
                        ctx.emit(f"{split}_baseline_lgbm_coverage", M.coverage(lq, c["y"], w))
                except Exception as e:  # never lose the M1/M2 run to the reference baseline
                    print(f"[lgbm] skipped: {type(e).__name__}: {e}", flush=True)
        tr_s2, tr_st2 = sample_train(min(20000, 40 * args.batch))
        _, emb_tr = predict(net, src, kind, tr_s2, tr_st2, args.mode, stats, dev, amp=amp, with_emb=True)
        ood = fit_ood(emb_tr)

        for split, c in evals.items():
            w = c["w"] > 0
            ctx.emit(f"{split}_nmae", M.nmae(c["q"][..., 1], c["y"], w))
            ctx.emit(f"{split}_pinball", M.pinball(c["q"], c["y"], w))
            if kind == "pv":
                ws = c["w_strict"] > 0
                ctx.emit(f"{split}_nmae_strict", M.nmae(c["q"][..., 1], c["y"], ws))
                ctx.emit(f"{split}_coverage_cal_strict", M.coverage(c["qc"], c["y"], ws))
            ctx.emit(f"{split}_coverage_raw", M.coverage(c["q"], c["y"], w))
            ctx.emit(f"{split}_coverage_cal", M.coverage(c["qc"], c["y"], w))
            ctx.emit(f"{split}_pinball_cal", M.pinball(c["qc"], c["y"], w))
            for b, v in c["baselines"].items():
                ctx.emit(f"{split}_baseline_{b}_nmae", M.nmae(v, c["y"], w))
        t = evals["test"]
        ctx.emit("coverage_10_90", M.coverage(t["qc"], t["y"], t["w"] > 0))
        for b, v in t["baselines"].items():
            ctx.emit(f"baseline_{b}_nmae", M.nmae(v, t["y"], t["w"] > 0))
        main_b = "pvlib" if kind == "pv" else "persistence_yesterday"
        ctx.emit(f"test_gain_vs_{main_b}", 1 - ctx.get("test_nmae") / max(ctx.get(f"baseline_{main_b}_nmae"), 1e-9))
        if "lgbm" in t["baselines"]:
            ctx.emit("test_gain_vs_lgbm", 1 - ctx.get("test_nmae") / max(ctx.get("baseline_lgbm_nmae"), 1e-9))

        # error analysis on test
        seen = sp.seen_mask[t["site"]]
        groups = {"seen_vs_unseen": np.where(seen, "seen", "unseen"),
                  "season": season_of(t["month"][:, 0], t["month"][:, 1])}
        if kind == "pv":
            wrong = np.array([src.sites[i].get("meta_wrong", False) for i in t["site"]])
            groups["metadata"] = np.where(wrong, "wrong_recorded_meta", "right_recorded_meta")
            groups["snow"] = np.where(t["snow"] > 0.5, "snow_on_ground", "no_snow")
            groups["cloudiness"] = np.select([t["csi"] < 0.4, t["csi"] < 0.7], ["overcast(csi<0.4)", "mixed"],
                                             "clear(csi>=0.7)")
            ood_test = ood_score(ood, t["emb"])
            groups["ood"] = np.where(ood_test > 3.0, "ood>3", "in_dist")
        else:
            arch = np.array([src.sites[i].get("archetype", "?") for i in t["site"]])
            groups["archetype"] = arch
            groups["day_type"] = np.where(t["offday"] > 0.5, "weekend_or_holiday", "workday")
        report["test_groups"] = group_metrics(t, groups)
        if kind == "pv":  # M1 on fault days vs the true healthy power (what M4 measures against)
            from .synth import FAULTS

            tt = dict(t)
            tt["w_all"] = (~t["night"]).astype(np.float32)
            fd = group_metrics(tt, {"fault_in_window": np.array([FAULTS[i] for i in t["fault"]])},
                               mask_key="w_all", truth="y_healthy")
            report["test_vs_true_healthy_by_fault"] = fd["fault_in_window"]
            for wname, wsel in (("wrong", True), ("right", False)):
                sel = groups["metadata"] == ("wrong_recorded_meta" if wsel else "right_recorded_meta")
                ww = t["w"][sel] > 0
                ctx.emit(f"test_nmae_meta_{wname}", M.nmae(t["q"][sel][..., 1], t["y"][sel], ww))
                ctx.emit(f"baseline_pvlib_nmae_meta_{wname}", M.nmae(t["baselines"]["pvlib"][sel], t["y"][sel], ww))
                if "lgbm" in t["baselines"]:
                    ctx.emit(f"baseline_lgbm_nmae_meta_{wname}", M.nmae(t["baselines"]["lgbm"][sel], t["y"][sel], ww))
            ns = groups["snow"] == "no_snow"
            ctx.emit("test_nmae_nosnow", M.nmae(t["q"][ns][..., 1], t["y"][ns], t["w"][ns] > 0))
            ctx.emit("baseline_pvlib_nmae_nosnow", M.nmae(t["baselines"]["pvlib"][ns], t["y"][ns], t["w"][ns] > 0))
            ctx.emit("baseline_pvlib_scaled_nmae_nosnow",
                     M.nmae(t["baselines"]["pvlib_scaled"][ns], t["y"][ns], t["w"][ns] > 0))
            if "lgbm" in t["baselines"]:
                ctx.emit("baseline_lgbm_nmae_nosnow", M.nmae(t["baselines"]["lgbm"][ns], t["y"][ns], t["w"][ns] > 0))
                sn = ~ns
                if sn.any():
                    ctx.emit("test_nmae_snow", M.nmae(t["q"][sn][..., 1], t["y"][sn], t["w"][sn] > 0))
                    ctx.emit("baseline_lgbm_nmae_snow", M.nmae(t["baselines"]["lgbm"][sn], t["y"][sn], t["w"][sn] > 0))
        for gname in ("seen", "unseen"):
            sel = seen if gname == "seen" else ~seen
            if sel.any():
                ctx.emit(f"test_nmae_{gname}", M.nmae(t["q"][sel][..., 1], t["y"][sel], t["w"][sel] > 0))
                ctx.emit(f"test_coverage_cal_{gname}", M.coverage(t["qc"][sel], t["y"][sel], t["w"][sel] > 0))
                if "lgbm" in t["baselines"]:
                    ctx.emit(f"baseline_lgbm_nmae_{gname}", M.nmae(t["baselines"]["lgbm"][sel], t["y"][sel],
                                                                    t["w"][sel] > 0))
        ood_val = ood_score(ood, evals["val"]["emb"])
        ood_t = ood_score(ood, t["emb"])
        report["ood"] = {"val_p50": float(np.median(ood_val)), "val_p99": float(np.quantile(ood_val, 0.99)),
                         "test_seen_p50": float(np.median(ood_t[seen])) if seen.any() else None,
                         "test_unseen_p50": float(np.median(ood_t[~seen])) if (~seen).any() else None,
                         "test_frac_over_3": float((ood_t > 3).mean())}
        widths = ((t["qc"][..., 2] - t["qc"][..., 0]) / t["scale"][:, None])[t["w"] > 0]
        cal_qc = apply_offsets(cal["q"], offsets, cal["scale"], True, cal["night"])
        cal_w = (cal_qc[..., 2] - cal_qc[..., 0]) / cal["scale"][:, None]
        report["test_mean_width_norm"] = float(widths.mean()) if widths.size else None

    for gname, g in report["test_groups"].items():
        for k, d in g.items():
            bl = " ".join(f"{b.replace('_nmae', '')}={v:.3f}" for b, v in d.items()
                          if b.endswith("_nmae") and b != "nmae")
            print(f"[analysis] {gname:16s} {k:24s} n={d['n_windows']:6d} nmae={d['nmae']:.3f} "
                  f"cov={d['coverage']:.3f} {bl}", flush=True)
    for k, d in report.get("test_vs_true_healthy_by_fault", {}).items():
        print(f"[analysis] vs_true_healthy  {k:24s} n={d['n_windows']:6d} nmae={d['nmae']:.3f}", flush=True)
    print(f"[analysis] ood {report['ood']}", flush=True)
    calib = {
        "model": model, "model_version": ctx.model_version, "mode": args.mode, "unit": spec["unit"],
        "scale": spec["scale"], "data_label": src.label, "quantiles": [0.1, 0.5, 0.9],
        "quantile_offsets": offsets, "offset_units": f"normalised (multiply by {spec['scale']})",
        "thresholds": {"ood": 3.0,
                       "width": float(np.quantile(cal_w[cal["w"] > 0], 0.95)) if (cal["w"] > 0).any() else None,
                       "member_std": None, "p_fault": None},
        "ood": ood, "channels": {k: list(v) for k, v in spec["channels"].items()}, "dims": dims,
        "inputs": ["seq", "ref", "summ", "stat", "mask"], "outputs": ["q", "emb"],
        "metrics": {k: v for k, v in ctx.metrics.items() if not k.startswith(("t_", "mem_"))},
    }
    if not args.no_export:
        with ctx.stage("export"):
            wrapped = Wrapped.make(net, stats)
            ex = {k: v[:8] for k, v in batch_arrays(src, kind, *sample_train(8), args.mode, None)[2].items()
                  if k in ("seq", "ref", "summ", "stat")}
            ex["mask"] = (~F.build(kind, F.gather(src, *sample_train(8), kind, args.mode))["night"]).astype(np.float32)
            info = export_onnx(wrapped, ex, ctx.out / f"{name}.onnx", ["q", "emb"])
            calib["onnx"] = info
            report["onnx"] = info
            ctx.emit("onnx_parity_ok", float(info["parity_ok"]))
            torch.save({"state": net.state_dict(), "stats": stats, "dims": dims, "args": vars(args)},
                       ctx.out / f"{name}.pt")
    ctx.write_json(f"{name}.calibration.json", calib)
    report["metrics"] = ctx.metrics
    report["calibration"] = {k: calib[k] for k in ("quantile_offsets", "thresholds")}
    ctx.write_json(f"{name}.report.json", report)
    ctx.finish({"data_root": str(src.root), "data_digest": src.index.get("digest"), "n_params": n_par})
    return {"net": net, "stats": stats, "dims": dims, "src": src, "splits": sp, "ctx": ctx, "calib": calib}


def main_pvnet(argv=None):
    run("pvnet", argv)


def main_loadnet(argv=None):
    run("loadnet", argv)
