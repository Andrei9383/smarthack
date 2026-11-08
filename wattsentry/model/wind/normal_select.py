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

from wattsentry.kernels import binned_median_mad, rolling_median

from .data import AMBIENT, CLASSES, POWER, TEMP_SENSORS
from .windmon_rules import windmon_flags

V_EDGES = np.arange(0.0, 30.5, 0.5)
P_EDGES = np.linspace(0.0, 1.1, 12)
MAD_TO_STD = 1.4826
N_SECTORS = 16

# All turbines are processed at once (#136): the per-bin medians / MADs and the centred rolling medians go through the
# dispatched kernels in wattsentry.kernels.median (exact; Rust when built, else the numpy/pandas reference), and
# `_lookup` gathers each step's bin statistic. The masks are bit-identical to the
# per-turbine loop this replaced (train#593: normal_sel_precision 0.956275, recall 0.912301).


def _flat(b: np.ndarray) -> np.ndarray:
    """Bin index b int [T, N] -> flat index into a [n_bins, N] table (computed once per binning)."""
    return b * b.shape[1] + np.arange(b.shape[1])


def _lookup(stat: np.ndarray, flat: np.ndarray) -> np.ndarray:
    """stat [n_bins, N], flat = _flat(b) -> stat[b[t, j], j] as [T, N] (a plain gather; faster than take_along_axis)."""
    return np.ascontiguousarray(stat).ravel().take(flat)


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
    edge = np.zeros((N, T + 2), dtype=np.int8)
    edge[:, 1:-1] = ok.T
    d = np.diff(edge, axis=1).ravel()  # per turbine row: +1 at a run's first step, -1 one past its last
    starts, ends = np.flatnonzero(d == 1), np.flatnonzero(d == -1)  # runs never cross rows (zero padding)
    short = ends - starts < min_run
    cut = np.zeros(d.size + 1, dtype=np.int32)
    cut[starts[short]] += 1
    cut[ends[short]] -= 1
    inside = np.cumsum(cut[:-1]).reshape(N, T + 1)[:, :T] > 0
    return ok & ~inside.T


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
    fit_t = np.asarray(fit_t, dtype=bool)[:, None]
    rated = np.asarray(rated_kw, dtype=np.float64)
    p = sensors[POWER].astype(np.float64) / rated[None]
    flags = windmon_flags(sensors, err, rated)
    rule_any = np.zeros((T, N), dtype=bool)
    for f in flags.values():
        rule_any |= f
    base = np.isfinite(p) & np.isfinite(v_met) & (err == 0) & ~_dilate(rule_any, dilate)

    vb = np.clip(np.digitize(np.nan_to_num(v_met, nan=0.0), V_EDGES) - 1, 0, len(V_EDGES) - 2)
    vbf = _flat(vb)
    nb = len(V_EDGES) - 1
    with np.errstate(divide="ignore", invalid="ignore"):
        lr = np.log(v_nac / v_met)
    ratio_valid = np.isfinite(lr) & (v_met > 3.0)
    one_bin = np.zeros((T, N), dtype=np.int64)  # the anemometer ratio: one median per turbine

    ok = base.copy()
    rel = np.full((T, N), np.nan)
    for _ in range(max(1, n_iter)):
        fit = ok & fit_t
        med, mad = binned_median_mad(vb, p, fit, nb, min_bin)
        m, s = _lookup(med, vbf), np.maximum(MAD_TO_STD * _lookup(mad, vbf), floor)
        r = p - m
        curve_ok = np.isfinite(m) & (r >= -k_below * s) & (r <= k_above * s)
        part = curve_ok & (m > 0.1) & (m < 0.9)  # partial load: where a fractional loss is visible
        rel = np.where(part, r / np.where(part, m, 1.0), np.nan)
        rf = fit & ratio_valid
        lm, lmad = binned_median_mad(one_bin, lr, rf, 1, min_bin)
        ls = np.maximum(MAD_TO_STD * lmad, ratio_floor)
        anem_ok = ~ratio_valid | (np.abs(lr - lm) <= k_ratio * ls) | (rf.sum(axis=0) < min_bin)[None]
        ok = base & curve_ok & anem_ok
    # slow losses (yaw misalignment, degraded power curve) hide in the per-step scatter but shift the multi-day
    # median of the turbine's own relative residual
    if wdir_deg is not None:  # wakes: subtract the turbine's typical relative loss for that wind sector
        sec = (np.nan_to_num(wdir_deg, nan=0.0) // (360 / N_SECTORS)).astype(int) % N_SECTORS
        med, _ = binned_median_mad(sec, rel, np.isfinite(rel) & fit_t, N_SECTORS, min_bin)
        rel -= _lookup(np.nan_to_num(med, nan=0.0), _flat(sec))
    slow = rolling_median(rel, roll_steps, roll_steps // 8) < -roll_loss
    ok &= ~slow
    power_ok = _drop_short_runs(~_dilate(~ok, dilate) & base, min_run)

    # temperature targets: gaps / suspect / spikes, and a rise far above the turbine's own rise at that load
    amb = sensors[AMBIENT].astype(np.float64)
    hard = _dilate(flags["DATA_GAP"] | flags["SUSPECT_DATA"] | flags["TEMPERATURE_SPIKE"], dilate)
    temp_ok = np.isfinite(amb) & np.isfinite(p) & (err == 0) & ~hard
    pb = np.clip(np.digitize(np.nan_to_num(p, nan=0.0), P_EDGES) - 1, 0, len(P_EDGES) - 2)
    pbf = _flat(pb)
    temp_fit = temp_ok & fit_t
    hot = np.zeros((T, N), dtype=bool)
    drift = np.zeros((T, N), dtype=bool)
    for k in TEMP_SENSORS:
        rise = sensors[k].astype(np.float64) - amb
        fin = np.isfinite(rise)
        med, mad = binned_median_mad(pb, rise, temp_fit & fin, len(P_EDGES) - 1, min_bin)
        m = _lookup(med, pbf)
        s = np.maximum(MAD_TO_STD * _lookup(mad, pbf), temp_floor_c)
        fm = fin & np.isfinite(m)
        hot |= fm & (rise - m > k_temp * s)
        res = np.where(fm, rise - m, np.nan)
        day = rolling_median(res, roll_steps // 2, roll_steps // 16)
        local = rolling_median(res, 14 * 96, 96)  # two-week baseline absorbs seasonal cooling changes
        drift |= (day - np.nan_to_num(local) > roll_temp_c) | (day > 2.5 * roll_temp_c)
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
