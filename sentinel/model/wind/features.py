"""M6 feature code shared by training and the onnxruntime predictor (numpy only, no torch).

- `nbm_inputs`: per-step NBM inputs [N, T, F_IN] and targets (density-normalised winds per plan §3.5).
- `run_nbm`: chunked causal inference through any backend (torch in train, onnxruntime in the predictor).
- `clf_features`: residual + park-context features for the anomaly classifier, one park at a time.
- `curtail_ok` / `decide`: per-class thresholds + park-wide curtailment rule (common ceiling or park consensus)
  + run-length smoothing → WindAnomaly index per step. Training, eval and the predictor all go through these.
"""

from __future__ import annotations

import itertools
from typing import Callable, Optional

import numpy as np

from sentinel.kernels import rolling_mean

from .data import AMBIENT, CLASSES, CLASS_INDEX, POWER, TEMP_SENSORS, WIND
from .physics import density_normalise, air_density

K = len(CLASSES)
V_SCALE = 15.0
T_RISE_SCALE = 40.0
NBM_IN = [
    "v_nac_norm", "nac_valid", "v_met_norm", "met_valid", "dir_sin", "dir_cos", "amb", "hour_sin", "hour_cos",
    "is_large",
    "p_frac", "p_valid",  # thermal path only (load history), never seen by the power path
]
N_POWER_IN = 10
NBM_TARGETS = ["p_frac"] + [f"{s}_rise" for s in TEMP_SENSORS]


def _nan0(x):
    return np.where(np.isfinite(x), x, 0.0)


def density_inputs(sensors, meteo, pressure_hpa_turb):
    """v_nac and v_met (m/s, density normalised) with ambient from turbine_t."""
    amb = sensors[AMBIENT].astype(np.float64)
    p = np.where(np.isfinite(pressure_hpa_turb), pressure_hpa_turb, 1013.25)
    t = np.where(np.isfinite(amb), amb, 10.0)
    rho = air_density(p, t)
    v_nac = sensors[WIND].astype(np.float64) / 3.6
    v_met = meteo["wind_ms"].astype(np.float64)
    return density_normalise(v_nac, rho), density_normalise(v_met, rho), rho


def nbm_inputs(sensors, meteo, pressure_turb, ts, rated, is_large, labels=None, train_mask_labels=True):
    """Returns X [N, T, F_IN] float32, Y [N, T, 6] float32, M [N, T, 6] bool (loss mask)."""
    vn, vm, _ = density_inputs(sensors, meteo, pressure_turb)
    T, N = vn.shape
    hour = (ts - ts.astype("datetime64[D]")).astype(int) / 60.0
    d = np.deg2rad(meteo["wdir_deg"].astype(np.float64))
    dvalid = np.isfinite(d)
    amb = sensors[AMBIENT].astype(np.float64)
    p = sensors[POWER].astype(np.float64) / rated[None]
    X = np.zeros((T, N, len(NBM_IN)), dtype=np.float32)
    X[..., 0] = _nan0(vn) / V_SCALE
    X[..., 1] = np.isfinite(vn)
    X[..., 2] = _nan0(vm) / V_SCALE
    X[..., 3] = np.isfinite(vm)
    X[..., 4] = np.where(dvalid, np.sin(_nan0(d)), 0.0)
    X[..., 5] = np.where(dvalid, np.cos(_nan0(d)), 0.0)
    X[..., 6] = _nan0(amb) / 20.0
    X[..., 7] = np.sin(2 * np.pi * hour / 24)[:, None]
    X[..., 8] = np.cos(2 * np.pi * hour / 24)[:, None]
    X[..., 9] = np.asarray(is_large, dtype=np.float32)[None]
    X[..., 10] = _nan0(p)
    X[..., 11] = np.isfinite(p)
    Y = np.zeros((T, N, len(NBM_TARGETS)), dtype=np.float32)
    M = np.zeros((T, N, len(NBM_TARGETS)), dtype=bool)
    Y[..., 0] = _nan0(p)
    M[..., 0] = np.isfinite(p)
    for k, s in enumerate(TEMP_SENSORS):
        r = (sensors[s].astype(np.float64) - amb) / T_RISE_SCALE
        Y[..., 1 + k] = _nan0(r)
        M[..., 1 + k] = np.isfinite(r)
    if labels is not None and train_mask_labels:
        # Normal-behaviour training: power only on normal steps; temps also during stops / curtailment
        # (the thermal path sees actual power) but never under overheating or gaps.
        M[..., 0] &= labels == 0
        bad_t = np.isin(labels, [CLASS_INDEX["overheating"], CLASS_INDEX["data_gap"]])
        M[..., 1:] &= ~bad_t[..., None]
    return X.transpose(1, 0, 2).copy(), Y.transpose(1, 0, 2).copy(), M.transpose(1, 0, 2).copy()


def run_nbm(fn: Callable, X: np.ndarray, tidx: np.ndarray, chunk: int = 672, warm: int = 96, batch: int = 256):
    """Chunked causal inference. fn(x [B, L, F] f32, tid [B] i64) -> (pq [B,L,3], pstd [B,L], tq [B,L,5,3]).

    Returns pq [T, N, 3], pstd [T, N], tq [T, N, 5, 3] in model units (fractions of rated / rise / 40)."""
    N, T, F = X.shape
    L = chunk + warm
    starts = np.arange(0, T, chunk)
    jobs = [(j, s) for j in range(N) for s in starts]
    pq = np.zeros((T, N, 3), np.float32)
    ps = np.zeros((T, N), np.float32)
    tq = np.zeros((T, N, len(TEMP_SENSORS), 3), np.float32)
    for b0 in range(0, len(jobs), batch):
        jb = jobs[b0 : b0 + batch]
        xb = np.zeros((len(jb), L, F), np.float32)
        for i, (j, s) in enumerate(jb):
            a = s - warm
            src = X[j, max(0, a) : min(T, s + chunk)]
            off = max(0, -a)
            xb[i, off : off + len(src)] = src
        tb = tidx[[j for j, _ in jb]].astype(np.int64)
        o_pq, o_ps, o_tq = fn(xb, tb)
        for i, (j, s) in enumerate(jb):
            n = min(chunk, T - s)
            pq[s : s + n, j] = o_pq[i, warm : warm + n]
            ps[s : s + n, j] = o_ps[i, warm : warm + n]
            tq[s : s + n, j] = o_tq[i, warm : warm + n]
    return pq, ps, tq


# --------------------------------------------------------------------------------------------------------------
BASELINE_STEPS = 21 * 96
BASELINE_LAG = 7 * 96
MIN_HISTORY_STEPS = BASELINE_STEPS + BASELINE_LAG  # 28 days: full per-turbine temperature baseline


def rmean(x: np.ndarray, w: int) -> np.ndarray:
    """Causal NaN-aware rolling mean over the last w steps along axis 0 (0 where no data); Rust kernel #95."""
    return rolling_mean(x, w)


def shift(x: np.ndarray, k: int) -> np.ndarray:
    out = np.zeros_like(x)
    out[k:] = x[:-k]
    return out


def frozen_steps(x: np.ndarray, cap: int = 16) -> np.ndarray:
    same = np.zeros(x.shape, bool)
    same[1:] = (x[1:] == x[:-1]) & np.isfinite(x[1:])
    run = np.zeros(x.shape, np.float32)
    for t in range(1, x.shape[0]):
        run[t] = np.where(same[t], np.minimum(run[t - 1] + 1, cap), 0)
    return run / cap


def met_curve_eval(curve: dict, v_norm: np.ndarray, is_large: np.ndarray) -> np.ndarray:
    """Binned empirical (meteo wind → power fraction) curve per class fitted on training data."""
    edges = np.asarray(curve["edges"])
    vals = np.asarray(curve["values"])  # [2, n_bins]
    b = np.clip(np.digitize(np.nan_to_num(v_norm), edges) - 1, 0, vals.shape[1] - 1)
    return np.where(np.asarray(is_large)[None].astype(bool), vals[1][b], vals[0][b])


def fit_met_curve(v_norm: np.ndarray, p_frac: np.ndarray, is_large: np.ndarray, ok: np.ndarray) -> dict:
    edges = np.arange(0, 30.5, 0.5)
    vals = np.zeros((2, len(edges) - 1))
    for c in (0, 1):
        cols = np.asarray(is_large).astype(bool) == bool(c)
        if not cols.any():
            cols = np.ones_like(cols)
        v = v_norm[:, cols][ok[:, cols]]
        p = p_frac[:, cols][ok[:, cols]]
        b = np.clip(np.digitize(v, edges) - 1, 0, len(edges) - 2)
        med = np.zeros(len(edges) - 1)
        for i in range(len(edges) - 1):
            m = b == i
            med[i] = np.median(p[m]) if m.sum() >= 20 else np.nan
        # fill empty bins by interpolation / edge values
        good = np.isfinite(med)
        med = np.interp(np.arange(len(med)), np.flatnonzero(good), med[good]) if good.any() else np.zeros_like(med)
        vals[c] = med
    return {"edges": edges.tolist(), "values": vals.tolist()}


def clf_features(
    sensors: dict, err: np.ndarray, meteo: dict, weather_park: dict, rated: np.ndarray, is_large: np.ndarray,
    neighbours: np.ndarray, pq: np.ndarray, pstd: np.ndarray, tq: np.ndarray, met_curve: dict,
) -> tuple[np.ndarray, list[str]]:
    """Features [T, n, F] for the turbines of ONE park (park context needs the whole park).

    pq/pstd in fractions of rated; tq in °C (absolute). weather_park: 1-D [T] series for the park."""
    T, n = sensors[POWER].shape
    feats: dict[str, np.ndarray] = {}
    p = sensors[POWER].astype(np.float64) / rated[None]
    valid = np.isfinite(p)
    q10, q50, q90 = pq[..., 0].astype(np.float64), pq[..., 1].astype(np.float64), pq[..., 2].astype(np.float64)
    res = np.where(valid, p - q50, np.nan)
    width = np.maximum(q90 - q10, 0.02)
    z = np.clip(res / width, -10, 10)
    feats.update(p=_nan0(p), q50=q50, q10=q10, q90=q90, res=_nan0(res), z=_nan0(z))
    for w in (4, 16, 96, 672):
        feats[f"res_m{w}"] = rmean(res, w)
    feats["z_m16"] = rmean(z, 16)
    below = np.where(valid, (p < q10).astype(float), np.nan)
    feats["below_p10_m16"] = rmean(below, 16)
    feats["below_p10_m96"] = rmean(below, 96)
    # partial-load deficit only (yaw / power-curve loss shows between cut-in and rated)
    part = np.where(valid & (q50 > 0.08) & (q50 < 0.85), res, np.nan)
    feats["res_partial_m96"] = rmean(part, 96)
    feats["res_partial_m672"] = rmean(part, 672)
    feats["above_p90_m16"] = rmean(np.where(valid, (p > q90).astype(float), np.nan), 16)
    feats["pstd"] = pstd.astype(np.float64)
    pres = np.broadcast_to(weather_park.get("pressure_hpa", np.full(T, np.nan))[:, None], (T, n))
    vn, vm, _ = density_inputs(sensors, meteo, pres)
    lr = np.log((vn + 0.5) / (vm + 0.5))
    feats.update(v_nac=_nan0(vn), v_met=_nan0(vm), lr=_nan0(lr), lr_m16=rmean(lr, 16), lr_m96=rmean(lr, 96))
    pm = met_curve_eval(met_curve, vm, is_large)
    feats["p_met"] = pm
    dm = np.where(valid, p - pm, np.nan)
    feats.update(p_minus_met=_nan0(dm), p_minus_met_m16=rmean(dm, 16))
    feats["nac_low_met_high_m16"] = rmean(np.where(np.isfinite(vn), ((vn < 1.0) & (vm > 5)).astype(float), np.nan), 16)
    feats["wd_frozen"] = frozen_steps(sensors[WIND])
    amb = sensors[AMBIENT].astype(np.float64)
    rh = np.broadcast_to(weather_park.get("rh_pct", np.full(T, np.nan))[:, None], (T, n))
    pr = np.broadcast_to(weather_park.get("precip_mm", np.full(T, np.nan))[:, None], (T, n))
    ice = np.where(np.isfinite(amb), (amb < 1.5) * np.nan_to_num(rh, nan=80.0) / 100.0, np.nan)
    feats.update(amb=_nan0(amb), rh=np.nan_to_num(rh, nan=-1.0), precip=np.nan_to_num(pr, nan=-1.0),
                 icing_cond=_nan0(ice), icing_cond_m16=rmean(ice, 16))
    for k, s in enumerate(TEMP_SENSORS):
        x = sensors[s].astype(np.float64)
        tr_raw = x - tq[:, :, k, 1]
        # the turbine's own baseline (21 days ending 7 days ago): unseen turbines that run hot by design are
        # not "overheating"; only a residual rising above the turbine's own baseline is (plan §3.5)
        base = shift(rmean(tr_raw, BASELINE_STEPS), BASELINE_LAG)
        tr = tr_raw - base
        tw = np.maximum(tq[:, :, k, 2] - tq[:, :, k, 0], 0.5)
        m96 = rmean(tr, 96)
        feats[f"{s}_res"] = _nan0(tr)
        feats[f"{s}_res_m96"] = m96
        feats[f"{s}_z_m16"] = rmean(tr / tw, 16)
        if s in ("gearbox_t", "gen1_t", "gen2_t"):
            feats[f"{s}_trend3d"] = m96 - shift(m96, 288)
            feats[f"{s}_res_m672"] = rmean(tr, 672)
    miss = (~valid).astype(float)
    feats.update(err=(err != 0).astype(float), missing=miss, missing_m16=rmean(miss, 16), missing_m96=rmean(miss, 96))
    d = np.deg2rad(meteo["wdir_deg"].astype(np.float64))
    feats.update(dir_sin=_nan0(np.sin(d)), dir_cos=_nan0(np.cos(d)))
    # park context
    with np.errstate(all="ignore"):
        import warnings

        with warnings.catch_warnings():
            warnings.simplefilter("ignore", RuntimeWarning)
            med_p = np.nanmedian(np.where(valid, p, np.nan), axis=1)
            med_q = np.nanmedian(q50, axis=1)
            pmax = np.nanmax(np.where(valid, p, np.nan), axis=1)
            park_ratio = np.nan_to_num(med_p / np.maximum(med_q, 0.02), nan=1.0)
            zero = np.where(valid, ((p < 0.01) & (q50 > 0.1)).astype(float), np.nan)
            park_zero = np.nan_to_num(np.nanmean(zero, axis=1))
            deficit = np.nan_to_num(med_q - pmax) > 0.05
            near = np.where(valid, (np.abs(p - pmax[:, None]) < 0.02).astype(float), np.nan)
            cap_frac = np.nan_to_num(np.nanmean(near, axis=1)) * deficit
            park_res = np.nan_to_num(np.nanmedian(res, axis=1))
            low = np.where(valid, (z < -1.5).astype(float), np.nan)
            park_low = np.nan_to_num(np.nanmean(low, axis=1))
            park_miss = miss.mean(axis=1)
            nb = np.nanmean(res[:, neighbours], axis=2) if n > 1 else np.zeros((T, n))
            nb_p = np.nanmean(np.where(valid, p, np.nan)[:, neighbours], axis=2) if n > 1 else np.zeros((T, n))
    feats["park_ratio"] = np.broadcast_to(park_ratio[:, None], (T, n))
    feats["park_zero_frac"] = np.broadcast_to(park_zero[:, None], (T, n))
    feats["park_cap_frac"] = np.broadcast_to(cap_frac[:, None], (T, n))
    feats["park_cap_frac_m4"] = np.broadcast_to(rmean(cap_frac[:, None], 4), (T, n))
    feats["at_ceiling"] = _nan0(near) * deficit[:, None]
    feats["park_res_med"] = np.broadcast_to(park_res[:, None], (T, n))
    feats["res_minus_park"] = _nan0(res - park_res[:, None])
    feats["nb_res"] = _nan0(nb)
    feats["res_minus_nb"] = _nan0(res - nb)
    feats["res_minus_nb_m96"] = rmean(res - nb, 96)
    feats["res_minus_nb_m672"] = rmean(res - nb, 672)
    feats["p_minus_nb_p"] = _nan0(p - nb_p)
    feats["park_low_frac"] = np.broadcast_to(park_low[:, None], (T, n))
    feats["park_missing_frac"] = np.broadcast_to(park_miss[:, None], (T, n))
    feats["is_large"] = np.broadcast_to(np.asarray(is_large, dtype=float)[None], (T, n))
    names = list(feats)
    out = np.empty((T, n, len(names)), dtype=np.float32)
    for i, k in enumerate(names):
        out[..., i] = feats[k]
    return out, names


def neighbours_for(lat: np.ndarray, lon: np.ndarray, k: int = 4) -> np.ndarray:
    """k nearest turbines (indices within the park) by position."""
    n = len(lat)
    if n <= 1:
        return np.zeros((n, 1), dtype=np.int64)
    xy = np.stack([np.asarray(lat) * 111.32, np.asarray(lon) * 111.32 * np.cos(np.deg2rad(np.mean(lat)))], 1)
    d = ((xy[:, None] - xy[None]) ** 2).sum(-1)
    np.fill_diagonal(d, np.inf)
    kk = min(k, n - 1)
    return np.argsort(d, axis=1)[:, :kk]


# --------------------------------------------------------------------------------------------------------------
DEFAULT_MIN_RUN = {"fault_stop": 2, "curtailment": 2, "icing": 4, "underperformance": 16, "overheating": 16,
                   "anemometer_fault": 8, "data_gap": 1}
DEFAULT_MERGE = {"fault_stop": 4, "curtailment": 4, "icing": 8, "underperformance": 32, "overheating": 32,
                 "anemometer_fault": 16, "data_gap": 32}


CURTAIL_MIN_PARK_FRAC = 0.15
# Park consensus (#99): a cap to 0 % leaves noisy near-zero power with no common ceiling, so the ceiling test alone
# turns a park-wide cap the classifier is sure about into "normal". Curtailment is therefore also allowed where at
# least this share of the park's reporting turbines has p_curtailment >= θ_curtailment in the same step ...
CURTAIL_CONSENSUS_FRAC = 0.5
# ... and at least this many turbines agree, so a lone turbine (also in a 1- or 2-turbine park) never qualifies.
CURTAIL_CONSENSUS_MIN_TURBINES = 2
# A vote needs p_curtailment >= max(θ_curtailment, this). The consensus lifts the veto without any physical
# ceiling, so it only counts confident turbines: once the veto stops binding, the tuner lowers θ_curtailment to
# 0.15-0.25 (#99 train#554-557), and votes that weak would let a diffuse weather-driven signal form a consensus.
CURTAIL_CONSENSUS_MIN_P = 0.5

# What `curtail_ok` returns: one park's curtailment context per step and turbine. Structured so that the masks of
# several parks can be concatenated along the turbine axis (train.py tunes on all validation parks at once) and
# `decide` still counts the consensus within each park.
CURTAIL_CTX = np.dtype([("ceiling", "?"), ("reporting", "?"), ("park", "<i8")])
_park_ids = itertools.count()


def curtail_ok(F: np.ndarray, names: list[str]) -> np.ndarray:
    """Curtailment context [T, n] (dtype CURTAIL_CTX) for the turbines of ONE park, F as from `clf_features`.

    Curtailment is park-wide by definition (plan §3.5). `decide` allows it where `ceiling` holds (>= 15 % of the
    park sits at one ceiling below its expected power, park_cap_frac over the last hour) OR where the park agrees
    (>= CURTAIL_CONSENSUS_FRAC of the `reporting` turbines, i.e. those with a valid power reading, have
    p_curtailment >= max(θ_curtailment, CURTAIL_CONSENSUS_MIN_P); only voting turbines with no other anomaly class
    at its threshold are released).
    Each call gets its own `park` id."""
    out = np.empty(F.shape[:2], dtype=CURTAIL_CTX)
    out["ceiling"] = F[..., names.index("park_cap_frac_m4")] >= CURTAIL_MIN_PARK_FRAC
    out["reporting"] = F[..., names.index("missing")] < 0.5
    out["park"] = next(_park_ids)
    return out


def curtail_allowed(p_curt: np.ndarray, theta: float, ctx: np.ndarray,
                    other_hit: Optional[np.ndarray] = None) -> np.ndarray:
    """[T, N] bool: where curtailment may be called. `ctx` from `curtail_ok` (possibly several parks concatenated
    on axis 1); a plain bool [T, N] is read as the ceiling test of one park whose turbines all report.
    `other_hit` [T, N]: another anomaly class is at or above its own threshold there. The consensus never releases
    such a turbine (no physical ceiling backs it), so it cannot hide an actionable class the veto would have kept."""
    if ctx.dtype.names is None:
        ceiling = np.asarray(ctx, dtype=bool)
        reporting = np.ones_like(ceiling)
        park = np.zeros(ceiling.shape[1], dtype=np.int64)
    else:
        ceiling, reporting = ctx["ceiling"], ctx["reporting"]
        park = ctx["park"][0] if len(ctx) else np.zeros(ctx.shape[1], dtype=np.int64)
    vote = (p_curt >= max(theta, CURTAIL_CONSENSUS_MIN_P)) & reporting
    agree = np.zeros_like(ceiling)
    for g in np.unique(park):
        cols = park == g
        n_vote = vote[:, cols].sum(1)
        n_rep = reporting[:, cols].sum(1)
        ok = (n_vote >= CURTAIL_CONSENSUS_MIN_TURBINES) & (n_vote >= CURTAIL_CONSENSUS_FRAC * np.maximum(n_rep, 1))
        agree[:, cols] = ok[:, None]
    release = agree & vote
    if other_hit is not None:
        release &= ~other_hit
    return ceiling | release


def decide(probs: np.ndarray, thresholds: dict, min_run: Optional[dict] = None, merge: Optional[dict] = None,
           curtail_mask: Optional[np.ndarray] = None) -> np.ndarray:
    """probs [T, N, K] → class index [T, N] (uint8): argmax of p_c/θ_c over anomaly classes if p_c ≥ θ_c, then
    fill short holes inside same-class runs and drop runs shorter than min_run. `curtail_mask` [T, N] (from
    `curtail_ok`) vetoes curtailment unless the park has a common ceiling or a consensus of its reporting
    turbines (`curtail_allowed`), so a lone underperforming turbine is never hidden as curtailment."""
    min_run = min_run or DEFAULT_MIN_RUN
    merge = merge or DEFAULT_MERGE
    th = np.array([1.0] + [thresholds[c] for c in CLASSES[1:]], dtype=np.float32)
    r = probs[..., 1:] / th[None, None, 1:]
    if curtail_mask is not None:
        ci = CLASS_INDEX["curtailment"]
        other_hit = np.delete(r, ci - 1, axis=-1).max(-1) >= 1.0
        ok = curtail_allowed(probs[..., ci], float(th[ci]), curtail_mask, other_hit)
        r = r.copy()
        r[..., ci - 1] = np.where(ok, r[..., ci - 1], 0.0)
    c = r.argmax(-1) + 1
    keep = np.take_along_axis(r, (c - 1)[..., None], -1)[..., 0] >= 1.0
    out = np.where(keep, c, 0).astype(np.uint8)
    T, N = out.shape
    for j in range(N):
        col = out[:, j]
        for ci in range(1, K):
            name = CLASSES[ci]
            m = col == ci
            if not m.any():
                continue
            d = np.diff(np.concatenate([[0], m.astype(np.int8), [0]]))
            s, e = np.flatnonzero(d == 1), np.flatnonzero(d == -1)
            g = merge.get(name, 0)
            for a, b in zip(e[:-1], s[1:]):  # fill holes (only over normal steps)
                if b - a <= g and (col[a:b] == 0).all():
                    col[a:b] = ci
            m = col == ci
            d = np.diff(np.concatenate([[0], m.astype(np.int8), [0]]))
            s, e = np.flatnonzero(d == 1), np.flatnonzero(d == -1)
            for a, b in zip(s, e):
                if b - a < min_run.get(name, 1):
                    col[a:b] = 0
        out[:, j] = col
    return out


# --------------------------------------------------------------------------------------------------------------
OOD_FEATURES = (0, 2, 6)  # v_nac_norm, v_met_norm, amb (NBM input columns)


def input_ranges(X: np.ndarray, valid_cols=(1, 3, None)) -> dict:
    """0.1 / 99.9 percentiles of the OOD features on training inputs X [N, T, F]."""
    out = {}
    for f, v in zip(OOD_FEATURES, valid_cols):
        x = X[..., f]
        if v is not None:
            x = x[X[..., v] > 0.5]
        x = np.asarray(x, dtype=np.float64).ravel()
        if x.size > 2_000_000:
            x = x[:: x.size // 2_000_000]
        out[NBM_IN[f]] = [float(np.percentile(x, 0.1)), float(np.percentile(x, 99.9))]
    return out


def ood_steps(X: np.ndarray, ranges: dict) -> np.ndarray:
    """Per-step OOD score [N, T] >= 0: largest exceedance of the training range, in units of the range width."""
    s = np.zeros(X.shape[:2], dtype=np.float32)
    for f in OOD_FEATURES:
        lo, hi = ranges[NBM_IN[f]]
        w = max(hi - lo, 1e-6)
        x = X[..., f]
        s = np.maximum(s, np.maximum((x - hi) / w, (lo - x) / w).clip(0))
    return s
