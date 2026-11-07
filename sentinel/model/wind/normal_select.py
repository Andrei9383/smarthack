"""Label-free selection of 'normal' steps for WindNBM training (card #70, follow-up to #41).

Real SCADA has no fault labels, so the normal-behaviour model cannot be trained on `labels == normal`. This module
picks normal steps from the telemetry alone, the way an operator would clean a power curve:

1. drop steps where any WindMon rule fires (zero power in wind, temperature spike, gap, suspect / error code);
2. per-turbine binned power curve on the density-normalised meteo wind (robust median + MAD per 0.5 m/s bin,
   fitted on training time only, refitted twice on the kept steps): drop steps far below the curve (stops,
   curtailment, icing, derating) and far above it (wind reading too low);
3. anemometer check: nacelle / meteo wind ratio far from the turbine's own median ratio;
4. slow losses: a 2-day rolling median of the turbine's relative residual (partial load, after removing its
   typical loss per wind sector, i.e. wakes) below -6 %;
5. dilate every exclusion by a few steps (ramps around stops) and drop kept runs shorter than an hour.

The temperature targets get a lighter mask (as with labels, the thermal path also learns from stops and
curtailment): gaps, suspect data, temperature spikes, and steps whose temperature rise is far above the turbine's
own typical rise at that load, or whose 1-day median rise drifts > 3.5 °C above its two-week baseline.

Mild slow drifts (early bearing overheating, losses under ~6 %) still survive. That is the gap #70
measures: compare the label-free run against the label-masked one with `--normal-select`.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from .data import AMBIENT, CLASSES, POWER, TEMP_SENSORS
from .windmon_rules import windmon_flags

V_EDGES = np.arange(0.0, 30.5, 0.5)
P_EDGES = np.linspace(0.0, 1.1, 12)
MAD_TO_STD = 1.4826
N_SECTORS = 16


def _binned_median_mad(b: np.ndarray, x: np.ndarray, ok: np.ndarray, n_bins: int, min_n: int):
    """Per-bin median and MAD of x over the steps in ok. b, x, ok are 1-D; returns (med, mad), NaN when < min_n."""
    med = np.full(n_bins, np.nan)
    mad = np.full(n_bins, np.nan)
    bb, xx = b[ok], x[ok]
    if bb.size == 0:
        return med, mad
    order = np.argsort(bb, kind="stable")
    bb, xx = bb[order], xx[order]
    cuts = np.flatnonzero(np.diff(bb)) + 1
    for seg_b, seg_x in zip(np.split(bb, cuts), np.split(xx, cuts)):
        if seg_x.size >= min_n:
            m = np.median(seg_x)
            med[seg_b[0]] = m
            mad[seg_b[0]] = np.median(np.abs(seg_x - m))
    return med, mad


def _dilate(bad: np.ndarray, n: int) -> np.ndarray:
    """Extend True runs by n steps on both sides along time (axis 0)."""
    if n <= 0:
        return bad
    out = bad.copy()
    for s in range(1, n + 1):
        out[s:] |= bad[:-s]
        out[:-s] |= bad[s:]
    return out


def _drop_short_runs(ok: np.ndarray, min_run: int) -> np.ndarray:
    """Set to False every True run (along time) shorter than min_run steps."""
    if min_run <= 1:
        return ok
    T, N = ok.shape
    pad = np.zeros((1, N), dtype=np.int8)
    d = np.diff(np.concatenate([pad, ok.astype(np.int8), pad]), axis=0)
    out = ok.copy()
    for j in range(N):
        starts = np.flatnonzero(d[:, j] == 1)
        ends = np.flatnonzero(d[:, j] == -1)
        for s, e in zip(starts, ends):
            if e - s < min_run:
                out[s:e, j] = False
    return out


def _rolling_median(x: np.ndarray, window: int, min_periods: int) -> np.ndarray:
    """Centred rolling median along time of a [T, N] array with NaNs (NaN where too few values)."""
    return pd.DataFrame(x).rolling(window, center=True, min_periods=min_periods).median().to_numpy()


def select_normal(
    sensors: dict[str, np.ndarray],
    err: np.ndarray,
    rated_kw: np.ndarray,
    v_nac: np.ndarray,
    v_met: np.ndarray,
    fit_t: np.ndarray,
    wdir_deg: np.ndarray | None = None,
    *,
    k_below: float = 3.0,
    k_above: float = 4.0,
    floor: float = 0.03,
    k_ratio: float = 4.0,
    ratio_floor: float = 0.05,
    k_temp: float = 4.0,
    temp_floor_c: float = 1.5,
    dilate: int = 2,
    min_run: int = 4,
    min_bin: int = 30,
    n_iter: int = 2,
    roll_steps: int = 192,
    roll_loss: float = 0.06,
    roll_temp_c: float = 3.5,
) -> tuple[np.ndarray, np.ndarray]:
    """Label-free normal masks.

    sensors: PPC sensor -> [T, N] (NaN = gap); err: int [T, N]; rated_kw: [N];
    v_nac, v_met: density-normalised nacelle / meteo wind, m/s, [T, N] (NaN = missing); fit_t: bool [T], the steps
    the curves are fitted on (training time only); wdir_deg: wind direction [T, N] (optional, removes the
    direction-dependent wake loss before the slow-loss check).
    Returns (power_ok, temp_ok), bool [T, N].
    """
    T, N = err.shape
    rated = np.asarray(rated_kw, dtype=np.float64)
    p = sensors[POWER].astype(np.float64) / rated[None]
    flags = windmon_flags(sensors, err, rated)
    rule_any = np.zeros((T, N), dtype=bool)
    for f in flags.values():
        rule_any |= f
    base = np.isfinite(p) & np.isfinite(v_met) & (err == 0) & ~_dilate(rule_any, dilate)

    vb = np.clip(np.digitize(np.nan_to_num(v_met, nan=0.0), V_EDGES) - 1, 0, len(V_EDGES) - 2)
    nb = len(V_EDGES) - 1
    with np.errstate(divide="ignore", invalid="ignore"):
        lr = np.log(v_nac / v_met)
    ratio_valid = np.isfinite(lr) & (v_met > 3.0)

    ok = base.copy()
    rel = np.full((T, N), np.nan)
    for _ in range(max(1, n_iter)):
        new_ok = base.copy()
        for j in range(N):
            fit = ok[:, j] & fit_t
            med, mad = _binned_median_mad(vb[:, j], p[:, j], fit, nb, min_bin)
            m, s = med[vb[:, j]], np.maximum(MAD_TO_STD * mad[vb[:, j]], floor)
            r = p[:, j] - m
            curve_ok = np.isfinite(m) & (r >= -k_below * s) & (r <= k_above * s)
            part = curve_ok & (m > 0.1) & (m < 0.9)  # partial load: where a fractional loss is visible
            rel[:, j] = np.where(part, r / np.where(part, m, 1.0), np.nan)
            rf = fit & ratio_valid[:, j]
            if rf.sum() >= min_bin:
                lm = np.median(lr[rf, j])
                ls = max(MAD_TO_STD * np.median(np.abs(lr[rf, j] - lm)), ratio_floor)
                anem_ok = ~ratio_valid[:, j] | (np.abs(lr[:, j] - lm) <= k_ratio * ls)
            else:
                anem_ok = np.ones(T, dtype=bool)
            new_ok[:, j] &= curve_ok & anem_ok
        ok = new_ok
    # slow losses (yaw misalignment, degraded power curve) hide in the per-step scatter but shift the multi-day
    # median of the turbine's own relative residual
    if wdir_deg is not None:  # wakes: subtract the turbine's typical relative loss for that wind sector
        sec = (np.nan_to_num(wdir_deg, nan=0.0) // (360 / N_SECTORS)).astype(int) % N_SECTORS
        for j in range(N):
            med, _ = _binned_median_mad(sec[:, j], rel[:, j], np.isfinite(rel[:, j]) & fit_t, N_SECTORS, min_bin)
            rel[:, j] -= np.nan_to_num(med, nan=0.0)[sec[:, j]]
    slow = _rolling_median(rel, roll_steps, roll_steps // 8) < -roll_loss
    ok &= ~slow
    power_ok = _drop_short_runs(~_dilate(~ok, dilate) & base, min_run)

    # temperature targets: gaps / suspect / spikes, and a rise far above the turbine's own rise at that load
    amb = sensors[AMBIENT].astype(np.float64)
    hard = _dilate(flags["DATA_GAP"] | flags["SUSPECT_DATA"] | flags["TEMPERATURE_SPIKE"], dilate)
    temp_ok = np.isfinite(amb) & np.isfinite(p) & (err == 0) & ~hard
    pb = np.clip(np.digitize(np.nan_to_num(p, nan=0.0), P_EDGES) - 1, 0, len(P_EDGES) - 2)
    hot = np.zeros((T, N), dtype=bool)
    drift = np.zeros((T, N), dtype=bool)
    for k in TEMP_SENSORS:
        rise = sensors[k].astype(np.float64) - amb
        fin = np.isfinite(rise)
        for j in range(N):
            fit = temp_ok[:, j] & fin[:, j] & fit_t
            med, mad = _binned_median_mad(pb[:, j], rise[:, j], fit, len(P_EDGES) - 1, min_bin)
            m = med[pb[:, j]]
            s = np.maximum(MAD_TO_STD * mad[pb[:, j]], temp_floor_c)
            hot[:, j] |= fin[:, j] & np.isfinite(m) & (rise[:, j] - m > k_temp * s)
            res = np.where(fin[:, j] & np.isfinite(m), rise[:, j] - m, np.nan)[:, None]
            day = _rolling_median(res, roll_steps // 2, roll_steps // 16)[:, 0]
            local = _rolling_median(res, 14 * 96, 96)[:, 0]  # two-week baseline absorbs seasonal cooling changes
            drift[:, j] |= (day - np.nan_to_num(local) > roll_temp_c) | (day > 2.5 * roll_temp_c)
    temp_ok &= ~_dilate(hot, dilate) & ~drift
    return power_ok, temp_ok


def selection_quality(sel: np.ndarray, labels: np.ndarray, where: np.ndarray | None = None) -> dict:
    """How well a normal mask matches the true labels (diagnostics only; labels are never used to select).

    precision = share of selected steps that are truly normal; recall = share of truly normal steps kept;
    kept_<class> = share of that class's steps that slipped into the training set."""
    w = np.ones_like(sel, dtype=bool) if where is None else where
    s, lab = sel[w], labels[w]
    normal = lab == 0
    out = {
        "precision": float(normal[s].mean()) if s.any() else 0.0,
        "recall": float(s[normal].mean()) if normal.any() else 0.0,
        "selected_frac": float(s.mean()) if s.size else 0.0,
    }
    for c, name in enumerate(CLASSES):
        if c == 0:
            continue
        m = lab == c
        if m.any():
            out[f"kept_{name}"] = float(s[m].mean())
    return out
