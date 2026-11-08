"""Temperature residuals from M6's WindNBM GRU, trained in-process (torch; `--residuals nbm`).

Reuses wattsentry.model.wind (M6, #41) as a library: `nbm_inputs` (inputs / targets), `WindNBM` (GRU with turbine
embedding), `pinball`, `run_nbm` (chunked causal inference). Only the thermal objective is trained (M7 needs the
expected component temperatures, not power).

Normal selection is LABEL-FREE: a temperature step enters the loss when it is finite and no integral error code is
present. We deliberately do NOT pass the injector labels (M6's default masks with labels == normal, which Stefan's
review on #41 flagged as optimistic) and do NOT use #70's temperature drift mask (it would drop the slow drifts M7
must detect). The held-out park runs with turbine embedding 0 (unknown turbine), as in M6.

The trained net is exported to ONNX with M6's signature (x, tid) → (pq, pstd, tq), so predictor.py can run it with
onnxruntime only.
"""

from __future__ import annotations

import time
from pathlib import Path

import numpy as np

from wattsentry.model.wind.data import TEMP_SENSORS

from . import COMPONENTS


def nbm_residuals(data, sp, train_t: np.ndarray, args, log=print, out: Path | None = None):
    """→ r [T, N, C] (°C, COMPONENTS order), meta dict (incl. turbine_index for the predictor)."""
    import torch

    from wattsentry.model.wind.features import NBM_IN, T_RISE_SCALE, nbm_inputs, run_nbm
    from wattsentry.model.wind.nbm import WindNBM, pinball

    t_start = time.time()
    torch.manual_seed(args.seed)
    dev = torch.device(args.device or ("cuda" if torch.cuda.is_available() else "cpu"))
    N, T = len(data.turbines), data.n_steps
    rated = data.rated_kw.astype(np.float64)
    is_large = (rated >= 4000).astype(np.float32)
    pres = data.weather["pressure_hpa"][:, data.park_of]
    X, Y, M = nbm_inputs(data.sensors, data.meteo, pres, data.ts, rated, is_large, labels=None)
    M &= (data.err.T == 0)[..., None]  # label-free: finite + no error code (no labels, no drift mask)
    emb_idx = np.zeros(N, dtype=np.int64)
    emb_idx[sp.train_turbines] = np.arange(1, len(sp.train_turbines) + 1)
    L, warm, B = 128, 32, 256

    def starts_in(mask):
        c = np.concatenate([[0], np.cumsum(mask.astype(np.int64))])
        s = np.arange(0, T - L + 1)
        return s[(c[s + L] - c[s]) == L]

    # inner train / val split of the pre-test time for early stopping: every 5th week is validation
    week = (np.arange(T) // (7 * 96)) % 5
    tr_starts, va_starts = starts_in(train_t & (week != 4)), starts_in(train_t & (week == 4))
    Xg, Yg, Mg = (torch.from_numpy(a).to(dev) for a in (X, Y, M))
    ar = torch.arange(L, device=dev)
    tt = torch.from_numpy(sp.train_turbines).to(dev)
    embg = torch.from_numpy(emb_idx).to(dev)

    def batch(starts, n, gen):
        j = tt[torch.from_numpy(gen.integers(0, len(sp.train_turbines), n)).to(dev)]
        s = torch.from_numpy(starts[gen.integers(0, len(starts), n)]).to(dev)
        idx = s[:, None] + ar[None]
        return Xg[j[:, None], idx], Yg[j[:, None], idx], Mg[j[:, None], idx], embg[j]

    vgen = np.random.default_rng(1234)
    val_batches = [batch(va_starts, B, vgen) for _ in range(8)]
    model = WindNBM(len(sp.train_turbines), hidden=args.nbm_hidden).to(dev)
    opt = torch.optim.AdamW(model.parameters(), lr=args.nbm_lr, weight_decay=1e-4)
    total = args.nbm_epochs * args.nbm_steps
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=args.nbm_lr, total_steps=max(total, 2), pct_start=0.05)

    def loss_fn(xb, yb, mb, tb):
        _, tq = model.forward_all(xb, tb)
        return pinball(tq[:, warm:], yb[:, warm:, 1:], mb[:, warm:, 1:])

    gen = np.random.default_rng(args.seed)
    best, best_state, epochs = float("inf"), None, 0
    deadline = t_start + args.nbm_max_minutes * 60
    for ep in range(args.nbm_epochs):
        model.train()
        for _ in range(args.nbm_steps):
            loss = loss_fn(*batch(tr_starts, B, gen))
            opt.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            sched.step()
        model.eval()
        with torch.no_grad():
            vl = float(np.mean([float(loss_fn(*b)) for b in val_batches]))
        epochs = ep + 1
        log(f"[windfailure] nbm epoch {epochs}/{args.nbm_epochs} val temp pinball {vl:.5f}")
        if vl < best:
            best, best_state = vl, {k: v.detach().clone() for k, v in model.state_dict().items()}
        if time.time() > deadline:
            log("[windfailure] nbm time budget reached")
            break
    model.load_state_dict(best_state)
    model.eval()
    del Xg, Yg, Mg, val_batches

    def fn(xb, tb):
        with torch.no_grad():
            a, b, c = model(torch.from_numpy(xb).to(dev), torch.from_numpy(tb).to(dev))
        return a.float().cpu().numpy(), b.float().cpu().numpy(), c.float().cpu().numpy()

    _, _, tq = run_nbm(fn, X, emb_idx, batch=512)
    amb = np.nan_to_num(data.sensors["turbine_t"], nan=10.0)
    r = np.full((T, N, len(COMPONENTS)), np.nan, np.float32)
    for ci, c in enumerate(COMPONENTS):
        k = TEMP_SENSORS.index(c)
        y = data.sensors[c]
        r[..., ci] = np.where(np.isfinite(y) & (data.err == 0), y - (tq[:, :, k, 1] * T_RISE_SCALE + amb), np.nan)
    meta = {"kind": "m6_windnbm_gru", "hidden": args.nbm_hidden, "epochs_done": epochs, "val_temp_pinball": best,
            "turbine_index": {data.turbines[j]["turbine_id"]: int(emb_idx[j]) for j in range(N) if emb_idx[j] > 0},
            "t_train_s": round(time.time() - t_start, 1), "device": str(dev)}
    if out is not None:  # ONNX with M6's signature for the onnxruntime-only predictor
        mc = WindNBM(len(sp.train_turbines), hidden=args.nbm_hidden)
        mc.load_state_dict({k: v.float().cpu() for k, v in model.state_dict().items()})
        mc.eval()
        p = Path(out) / "windfailure_nbm.onnx"
        torch.onnx.export(
            mc, (torch.zeros(2, 128, len(NBM_IN)), torch.tensor([0, 1], dtype=torch.int64)), str(p),
            input_names=["x", "tid"], output_names=["pq", "pstd", "tq"],
            dynamic_axes={k: {0: "B", 1: "L"} for k in ("x", "pq", "pstd", "tq")} | {"tid": {0: "B"}},
            opset_version=17, dynamo=False,
        )
        meta["onnx"] = p.name
    return r, meta
