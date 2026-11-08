"""The Grader: compare every prediction with what actually happened (team/judge-design.md §5.1-5.7).

`grade(predictions, obs, reports, run_id=...)` takes a replay's predictions (Prediction envelopes) and observations
(SiteDay / TurbineDay packed into a `ReplayObs`) plus the crew / audit reports, and returns `outcomes` rows (§8):

    run_id, model, model_version, asset_id, day, score_name, score, weight, source, observed_at, gated,
    kind, seen, battery, season, quality, ood_high

- `day` is the local (Europe/Bucharest) date of the graded day; `observed_at` a UTC ISO time (sortable as text): the
  moment the outcome could first be known (§5.5). Telemetry scores arrive at d+1, reports at the visit, M7 at the
  failure or d+14.
- `gated` = 1: the asset-day failed the physics-only gate (§5.4); `score` is None and the row is ungraded.
- Bucket columns (§6.1): kind (site|turbine|zone|park), seen (bool), battery (bool), season (warm|cold),
  quality (clean|gappy: >10 % missing steps), ood_high (bool, prediction OOD above the model's threshold).
- Scores (§5.3): forecasts `calibration`, `sharpness` (interval-score skill vs the substitute) and `norm_err`
  (10 quantiles of the day's normalised errors, for the ledger's band scale, §6.4; not in 0..1); detectors
  `alarm_precision` / `miss_score`; M7 `hit` (+ `brier` detail); M8 `flag` / `miss_score`.

Truth never enters here: the grader reads telemetry, pvlib physics, reports (CrewReport) and observable events only.
Scoring is vectorised over [asset-days, 96]; the only Python loops run over predictions, assets or reports.
"""

from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass, field
from typing import Any, Callable, Iterable, Mapping, Optional, Sequence

import numpy as np

from wattsentry.agent.audit import iso, local_dt
from wattsentry.agent.audit import parse_iso as _ts

from . import gate as G
from . import scores as S

DAY_S = 86400
STEPS = 96
STEP_H = 0.25

OUTCOME_COLUMNS = ("run_id", "model", "model_version", "asset_id", "day", "score_name", "score", "weight", "source",
                   "observed_at", "gated", "kind", "seen", "battery", "season", "quality", "ood_high")
BUCKET_COLUMNS = ("kind", "seen", "battery", "season", "quality", "ood_high")

# witness -> substitutes in order of preference (§3); substitutes are graded against their witness
# model ids as in models/judge_priors.json (#147): lgbm_pv, lgbm_load, persistence_load, persistence_zone, ...
SUBSTITUTES = {"pvnet": ("lgbm_pv", "lgbm", "pvlib", "persistence_pv"),
               "loadnet": ("lgbm_load", "lgbm", "persistence_load", "persistence"),
               "demandnet": ("persistence_zone", "persistence"), "windpower": ("binned_curve", "persistence_park")}
WITNESS_TARGET = {"pvnet": "pv", "loadnet": "load", "demandnet": "zone", "windpower": "park", "pvlib": "pv",
                  "lgbm": "pv", "lgbm_pv": "pv", "lgbm_load": "load", "persistence": "load", "persistence_pv": "pv",
                  "persistence_load": "load", "persistence_zone": "zone", "persistence_park": "park",
                  "binned_curve": "park", "mock": "pv"}
SITE_DETECTORS = ("sitehealth", "threshold_cusum")
TURBINE_DETECTORS = ("windnbm", "windmon_rules")
CLIM_VERSION, CLIM_DAYS = "grader-clim7", 7  # internal substitute when none is given: model persistence_<target>
OOD_DEFAULT = 3.0  # #26 thresholds: ood_max
GAPPY = 0.10  # §6.1 data quality
PROD_FRAC = 0.02  # production step: pvlib clear-sky above 2 % of recorded kWp
RESID_LO, RESID_HI, RESID_DAYS = 0.80, 1.30, 3  # §5.3 residual check: 3-day energy vs pvlib x shared-sky index
RESID_FAIL3, RESID_FAIL1 = 0.60, 0.75  # fail: 3-day ratio below 0.6 and the claimed day itself below 0.75
REPORT_WINDOW_DAYS = 3  # an alarm is confirmed / a healthy call passed by a report within 3 days
M7_HORIZON_DAYS = 14
FAULT_STOP_CODE, CURTAIL_CODE = 211, 40  # observable status codes (SCADA / operator curtailment log)
NORM_ERR_Q = (np.arange(10) + 0.5) / 10  # 10 quantiles of the day's normalised errors per asset-day (J2 #147)


# ------------------------------------------------------------------------------------------ observations
@dataclass
class ReplayObs:
    """Observed telemetry of a replay as arrays over [asset, H + D days, 96] (NaN = not reported).

    Days -H..-1 are the assets' own history (gate unit-jump median, internal persistence substitute); grading covers
    days 0..D-1. `cs` = pvlib clear-sky power from recorded metadata (gate.clearsky_power).
    """

    day0_ts: int
    days: int
    hist: int
    site_ids: list[str]
    site_rows: list[Mapping[str, Any]]
    pv: np.ndarray
    load: np.ndarray
    turbine_ids: list[str] = field(default_factory=list)
    turbine_rows: list[Mapping[str, Any]] = field(default_factory=list)
    tpow: Optional[np.ndarray] = None
    tcode: Optional[np.ndarray] = None
    zone_ids: list[str] = field(default_factory=list)
    zone: Optional[np.ndarray] = None  # [Z, H + D, 96] realised zone load (MW)
    cs: Optional[np.ndarray] = None

    def __post_init__(self) -> None:
        H, D = self.hist, self.days
        self.site_index = {s: i for i, s in enumerate(self.site_ids)}
        self.turbine_index = {t: i for i, t in enumerate(self.turbine_ids)}
        self.zone_index = {z: i for i, z in enumerate(self.zone_ids)}
        if self.tpow is None:
            self.tpow = np.zeros((0, H + D, STEPS))
            self.tcode = np.zeros((0, H + D, STEPS), np.int64)
        if self.cs is None:
            self.cs = G.clearsky_power(self.site_rows, self.day0_ts - H * DAY_S, H + D)
        self.kwp = np.array([float(r.get("kwp_recorded") or r.get("kwp", 0) or 0) for r in self.site_rows])
        self.fuse = np.array([float(r.get("fuse_kw") or G.DEFAULT_FUSE_KW) for r in self.site_rows])
        self.rated = np.array([float(r["rated_kw"]) for r in self.turbine_rows])
        self.day_ts = np.array([self.day0_ts + d * DAY_S for d in range(D + 1)])
        loc = [local_dt(int(t)) for t in self.day_ts[:D]]
        self.day_label = [x.date().isoformat() for x in loc]
        self.day_iso = [iso(t) for t in self.day_ts]
        self.season = ["warm" if 4 <= x.month <= 9 else "cold" for x in loc]
        # parks: sum of their turbines (NaN if any turbine is unreported at that step)
        parks: dict[str, list[int]] = defaultdict(list)
        for i, r in enumerate(self.turbine_rows):
            parks[str(r.get("park_id", ""))].append(i)
        self.park_ids = sorted(parks)
        self.park_index = {p: i for i, p in enumerate(self.park_ids)}
        self.park = np.stack([self.tpow[parks[p]].sum(0) for p in self.park_ids]) if self.park_ids \
            else np.zeros((0, H + D, STEPS))
        self._gates: dict[str, np.ndarray] = {}

    # gates are computed once per replay, from telemetry + physics + own history only (never a prediction)
    def gate(self, channel: str) -> np.ndarray:
        if channel not in self._gates:
            if channel == "pv":
                g = G.gate_pv(self.pv, self.cs, self.kwp)
            elif channel == "load":
                g = G.gate_load(self.load, self.fuse)
            elif channel == "turbine":
                g = G.gate_turbine(self.tpow, self.rated)
            elif channel == "park":
                g = np.zeros(self.park.shape[:2], np.int64)
                for p, pid in enumerate(self.park_ids):
                    idx = [self.turbine_index[r["turbine_id"]] for r in self.turbine_rows if r.get("park_id") == pid]
                    g[p] = np.bitwise_or.reduce(self.gate("turbine")[idx], axis=0)
            else:  # zone: realised load only needs to be reported
                z = self.zone if self.zone is not None else np.zeros((0, self.hist + self.days, STEPS))
                g = np.where(np.isfinite(z).mean(-1) < G.MIN_REPORTED, G.GAP, 0).astype(np.int64)
            self._gates[channel] = g
        return self._gates[channel]

    def series(self, target: str) -> tuple[np.ndarray, dict[str, int]]:
        return {"pv": (self.pv, self.site_index), "load": (self.load, self.site_index),
                "park": (self.park, self.park_index),
                "zone": (self.zone if self.zone is not None else np.zeros((0, self.hist + self.days, STEPS)),
                         self.zone_index)}[target]

    def residual_pass(self) -> np.ndarray:
        """§5.3 residual check per site-day [S, H + D] over days d..d+2: 1 pass, 0 fail, NaN unknown.

        Expected PV = pvlib clear-sky (recorded metadata) x the locality-day clear-sky index (median measured /
        clear-sky energy of the ungated sites under the same sky, >= 3 sites). Never reads a witness (not PVNet's
        band). Over the 3-day window (>= 2 ungated days): measured / expected energy in [0.8, 1.3] -> pass;
        < 0.6 with day d itself < 0.75 -> fail; otherwise unknown (ungraded).
        """
        if "_resid" in self._gates:
            return self._gates["_resid"]
        valid = np.isfinite(self.pv) & (self.cs > PROD_FRAC * self.kwp[:, None, None])
        e_obs = np.where(valid, self.pv, 0).sum(-1)
        e_cs = np.where(valid, self.cs, 0).sum(-1)
        ok = (e_cs > 0) & (valid.sum(-1) >= S.MIN_STEPS) & (self.gate("pv") == 0)
        with np.errstate(invalid="ignore", divide="ignore"):
            r = np.where(ok, e_obs / e_cs, np.nan)
        loc = np.array([str(x.get("locality", x.get("gis_feeder_id", ""))) for x in self.site_rows])
        k = np.full_like(r, np.nan)
        for name in np.unique(loc):
            m = loc == name
            if m.sum() >= 3:
                k[m] = S.nanquantile(r[m], 0.5, axis=0)[..., 0]
        ok &= np.isfinite(k) & (k > 0)
        obs_e = np.where(ok, e_obs, 0.0)
        exp_e = np.where(ok, e_cs * np.nan_to_num(k), 0.0)
        A, T = r.shape
        pad = lambda x: np.concatenate([x, np.zeros((A, RESID_DAYS - 1))], 1)
        win = lambda x: np.lib.stride_tricks.sliding_window_view(pad(x), RESID_DAYS, axis=1).sum(-1)[:, :T]
        n_ok, so, se = win(ok.astype(float)), win(obs_e), win(exp_e)
        with np.errstate(invalid="ignore", divide="ignore"):
            ratio3 = np.where((n_ok >= 2) & (se > 0), so / se, np.nan)
            ratio1 = np.where(ok, e_obs / (e_cs * k), np.nan)
        out = np.full_like(r, np.nan)
        out[(ratio3 >= RESID_LO) & (ratio3 <= RESID_HI)] = 1.0
        out[(ratio3 < RESID_FAIL3) & (ratio1 < RESID_FAIL1)] = 0.0
        self._gates["_resid"] = out
        return out

    def missing_share(self, target: str) -> np.ndarray:
        y, _ = self.series(target)
        return 1.0 - np.isfinite(y).mean(-1)


def _arr(xs: Sequence[Optional[float]]) -> np.ndarray:
    return np.array([np.nan if x is None else x for x in xs], np.float64)


def build_obs(day0_ts: int, days: int, site_rows: Sequence[Mapping], site_day: Callable[[str, int], Any],
              turbine_rows: Sequence[Mapping] = (), turbine_day: Optional[Callable[[str, int], Any]] = None,
              hist: int = 0, zone_obs: Optional[Mapping[str, np.ndarray]] = None) -> ReplayObs:
    """Pack SiteDay / TurbineDay (wattsentry.agent.world) into arrays. Steps with telemetry_age_s > 0 are stale
    (held values, not readings) and become NaN, like a missing turbine reading (None)."""
    D, H = int(days), int(hist)
    site_rows, turbine_rows = list(site_rows), list(turbine_rows)
    nS, nT = len(site_rows), len(turbine_rows)
    pv = np.full((nS, H + D, STEPS), np.nan)
    load = np.full((nS, H + D, STEPS), np.nan)
    for i, r in enumerate(site_rows):
        for j, d in enumerate(range(-H, D)):
            try:
                sd = site_day(str(r["site_id"]), d)
            except Exception:
                if d >= 0:
                    raise
                continue
            stale = np.asarray(sd.telemetry_age_s, np.float64) > 0
            pv[i, j] = np.where(stale, np.nan, _arr(sd.pv_kw))
            load[i, j] = np.where(stale, np.nan, _arr(sd.load_kw))
    tpow = np.full((nT, H + D, STEPS), np.nan)
    tcode = np.zeros((nT, H + D, STEPS), np.int64)
    for i, r in enumerate(turbine_rows):
        for j, d in enumerate(range(-H, D)):
            try:
                td = turbine_day(str(r["turbine_id"]), d)
            except Exception:
                if d >= 0:
                    raise
                continue
            tpow[i, j] = _arr(td.act_pwt_kw)
            tcode[i, j] = np.asarray(td.error_code, np.int64)
    zids, z = [], None
    if zone_obs:
        zids = sorted(zone_obs)
        z = np.stack([np.asarray(zone_obs[k], np.float64).reshape(H + D, STEPS) for k in zids])
    return ReplayObs(int(day0_ts), D, H, [str(r["site_id"]) for r in site_rows], site_rows, pv, load,
                     [str(r["turbine_id"]) for r in turbine_rows], turbine_rows, tpow, tcode, zids, z)


def obs_from_world(world: Any, hist: int = CLIM_DAYS) -> ReplayObs:
    """ReplayObs from a World: what the agent observed (`observed_site_day`), plus `hist` days of own history when
    the world can produce days before the replay (MockWorld can; others just skip them)."""
    sids, tids = sorted(world.sites), sorted(world.turbines)
    return build_obs(world.day_ts(0), world.days, [world.sites[s].row for s in sids], world.observed_site_day,
                     [world.turbines[t].row for t in tids], world.turbine_day, hist=hist)


# ------------------------------------------------------------------------------------------ row assembly
class _Rows:
    def __init__(self, run_id: str):
        self.run_id = run_id
        self.cols: dict[str, list] = {c: [] for c in OUTCOME_COLUMNS}

    def add(self, n: int, **kw: Any) -> None:
        """Append n rows; each kw is a scalar or a length-n sequence."""
        if n <= 0:
            return
        kw["run_id"] = self.run_id
        for c in OUTCOME_COLUMNS:
            v = kw[c]
            if isinstance(v, np.ndarray):
                v = v.tolist()  # native dtypes only (no object arrays) -> plain Python str / bool / float
            elif not isinstance(v, (list, tuple)):
                v = [v] * n
            self.cols[c].extend(v)

    def rows(self) -> list[dict[str, Any]]:
        c = self.cols
        c["score"] = [None if (x is None or x != x) else x for x in c["score"]]
        return [dict(zip(OUTCOME_COLUMNS, t)) for t in zip(*(self.cols[c] for c in OUTCOME_COLUMNS))]


@dataclass
class _Asset:
    kind: str
    seen: bool
    battery: bool


def _asset_info(obs: ReplayObs) -> dict[str, _Asset]:
    info = {}
    for r in obs.site_rows:
        info[str(r["site_id"])] = _Asset("site", r.get("split", "train") != "unseen",
                                         float(r.get("battery_kwh") or 0) > 0)
    for r in obs.turbine_rows:
        info[str(r["turbine_id"])] = _Asset("turbine", r.get("split", "train") != "unseen", False)
    for p in obs.park_ids:
        info[p] = _Asset("park", True, False)
    for z in obs.zone_ids:
        info[z] = _Asset("zone", True, False)
    return info


# ------------------------------------------------------------------------------------------ forecasts
@dataclass
class _FGroup:
    model: str
    version: str
    target: str
    aidx: np.ndarray  # asset index into the target series
    day: np.ndarray  # replay day 0..D-1
    q: np.ndarray  # [P, 3, 96]
    ood: np.ndarray
    asset_ids: list[str]


def _target_of(p: Any) -> str:
    t = (p.extra or {}).get("target")
    return str(t) if t else WITNESS_TARGET.get(p.model, "pv")


def _forecast_groups(preds: Sequence[Any], obs: ReplayObs) -> list[_FGroup]:
    by: dict[tuple[str, str, str], list] = defaultdict(list)
    for p in preds:
        by[(p.model, p.model_version, _target_of(p))].append(p)
    out = []
    for (m, v, tgt), ps in sorted(by.items()):
        _, index = obs.series(tgt)
        keep = []
        for p in ps:
            off = _ts(p.t0) - obs.day0_ts
            d = off // DAY_S
            if off % DAY_S or not (0 <= d < obs.days) or p.asset_id not in index:
                continue  # not a day-ahead window of this replay
            keep.append((p, int(d)))
        if not keep:
            continue
        P = len(keep)
        q = np.full((P, 3, STEPS), np.nan)
        for i, (p, _) in enumerate(keep):
            h = min(int(p.horizon), STEPS)
            q[i, 0, :h], q[i, 1, :h], q[i, 2, :h] = p.q10[:h], p.q50[:h], p.q90[:h]
        out.append(_FGroup(m, v, tgt, np.array([index[p.asset_id] for p, _ in keep]),
                           np.array([d for _, d in keep]), q, np.array([p.ood_score for p, _ in keep]),
                           [p.asset_id for p, _ in keep]))
    return out


def _climatology(y: np.ndarray, hist: int, days: int) -> np.ndarray:
    """Internal persistence substitute [A, D, 3, 96]: per-step P10/P50/P90 of the asset's own previous <= 7 days
    (causal, needs >= 2 reported days), so a skill score exists even when no baseline ran in the replay."""
    A = y.shape[0]
    pad = np.concatenate([np.full((A, CLIM_DAYS, STEPS), np.nan), y], axis=1)
    out = np.full((A, days, 3, STEPS), np.nan)
    for d in range(days):  # D iterations, each vectorised over [A, 7, 96]
        j = CLIM_DAYS + hist + d
        win = pad[:, j - CLIM_DAYS:j]
        cnt = np.isfinite(win).sum(1)
        qs = S.nanquantile(win, (0.1, 0.5, 0.9), axis=1)  # [A, 96, 3]
        out[:, d] = np.where(cnt[:, None, :] >= 2, np.moveaxis(qs, -1, 1), np.nan)
    return out


def _norm_err(q: np.ndarray, y: np.ndarray, valid: np.ndarray) -> np.ndarray:
    """[P, 10] quantiles of e = (q50 - y)/(q50 - q10) below the median, (y - q50)/(q90 - q50) above (J2, #147)."""
    q10, q50, q90 = q[:, 0], q[:, 1], q[:, 2]
    with np.errstate(invalid="ignore", divide="ignore"):
        lo = (q50 - y) / np.maximum(q50 - q10, 1e-6)
        hi = (y - q50) / np.maximum(q90 - q50, 1e-6)
    e = np.where(y < q50, lo, hi)
    e = np.where(valid & np.isfinite(e), e, np.nan)
    v = S.nanquantile(e, NORM_ERR_Q, axis=1)  # [P, 10]
    return np.where((valid.sum(1) >= S.MIN_STEPS)[:, None], v, np.nan)


def _grade_forecasts(groups: list[_FGroup], obs: ReplayObs, info: Mapping[str, _Asset], rows: _Rows,
                     ood_thr: Mapping[str, float], diag: dict) -> None:
    H = obs.hist
    by_target: dict[str, list[_FGroup]] = defaultdict(list)
    for g in groups:
        by_target[g.target].append(g)
    for tgt, gs in by_target.items():
        y_all, index = obs.series(tgt)
        if tgt == "pv":
            prod = obs.cs > PROD_FRAC * obs.kwp[:, None, None]
        else:
            prod = np.ones(y_all.shape, bool)
        gate = obs.gate(tgt)
        miss = obs.missing_share(tgt)
        witnesses = [g for g in gs if g.model in SUBSTITUTES]
        lookup = {(g.model, g.version): {(int(a), int(d)): i for i, (a, d) in enumerate(zip(g.aidx, g.day))}
                  for g in gs}
        clim = None
        has_sub = {g.model for g in gs if g.model not in SUBSTITUTES}
        extra: list[_FGroup] = []
        for w in witnesses:  # no baseline in the replay -> the grader's own persistence substitute
            if not any(s in has_sub for s in SUBSTITUTES[w.model]):
                if clim is None:
                    clim = _climatology(y_all, H, obs.days)
                q = clim[w.aidx, w.day]
                ok = np.isfinite(q).any(axis=(1, 2))
                clim_model = f"persistence_{tgt}"
                if ok.any() and not any(e.model == clim_model and e.version == CLIM_VERSION for e in extra):
                    extra.append(_FGroup(clim_model, CLIM_VERSION, tgt, w.aidx[ok], w.day[ok], q[ok],
                                         np.zeros(int(ok.sum())), [w.asset_ids[i] for i in np.flatnonzero(ok)]))
                    lookup[(clim_model, CLIM_VERSION)] = {(int(a), int(d)): i for i, (a, d) in
                                                         enumerate(zip(w.aidx[ok], w.day[ok]))}
        for g in gs + extra:
            if g.model in SUBSTITUTES:
                refs = [r for s in SUBSTITUTES[g.model] for r in gs if r.model == s] or extra
            else:
                refs = witnesses
            ref_q = np.full((len(g.day), 2, STEPS), np.nan)
            for r in refs[:1]:
                lk = lookup[(r.model, r.version)]
                ix = np.array([lk.get((int(a), int(d)), -1) for a, d in zip(g.aidx, g.day)])
                hit = ix >= 0
                ref_q[hit] = r.q[ix[hit]][:, [0, 2]]
            j = H + g.day
            y = y_all[g.aidx, j]
            valid = np.isfinite(y) & np.isfinite(g.q).all(1) & prod[g.aidx, j]
            ref_valid = np.isfinite(ref_q).all(1)
            sc = S.forecast_scores(g.q[:, 0], g.q[:, 2], y, valid, ref_q[:, 0], ref_q[:, 1], ref_valid)
            gated = gate[g.aidx, j] != 0
            calib = np.where(gated, np.nan, sc["calibration"])
            sharp = np.where(gated, np.nan, sc["skill"])
            thr = float(ood_thr.get(g.model, OOD_DEFAULT))
            a_inf = [info.get(a, _Asset(tgt, True, False)) for a in g.asset_ids]
            common = dict(model=g.model, model_version=g.version, asset_id=g.asset_ids,
                          day=[obs.day_label[d] for d in g.day], source="event" if tgt == "zone" else "telemetry",
                          observed_at=[obs.day_iso[d + 1] for d in g.day], kind=[a.kind for a in a_inf],
                          seen=[a.seen for a in a_inf], battery=[a.battery for a in a_inf],
                          season=[obs.season[d] for d in g.day],
                          quality=np.where(miss[g.aidx, j] > GAPPY, "gappy", "clean"), ood_high=(g.ood > thr))
            for name, val in (("calibration", calib), ("sharpness", sharp)):
                keep = gated | np.isfinite(val)
                sub = {k: (np.asarray(v)[keep] if isinstance(v, (list, np.ndarray)) else v)
                       for k, v in common.items()}
                rows.add(int(keep.sum()), score_name=name, score=val[keep].astype(float), weight=1.0,
                         gated=gated[keep].astype(int), **sub)
            ne = _norm_err(g.q, y, valid)
            ok = ~gated & np.isfinite(ne).all(1)
            if ok.any():
                rep = {k: (np.repeat(np.asarray(v)[ok], len(NORM_ERR_Q)) if isinstance(v, (list, np.ndarray))
                           else v) for k, v in common.items()}
                rows.add(int(ok.sum()) * len(NORM_ERR_Q), score_name="norm_err", score=ne[ok].reshape(-1),
                         weight=1.0 / len(NORM_ERR_Q), gated=0, **rep)
            diag.setdefault("forecast_groups", []).append(
                {"model": g.model, "version": g.version, "target": tgt, "asset_days": int(len(g.day)),
                 "gated": int(gated.sum()), "reference": refs[0].model if refs else None})


# ------------------------------------------------------------------------------------------ detectors
@dataclass
class _Claim:
    alarm: bool
    cls: str
    ood: float


def _site_claims(preds: Sequence[Any], obs: ReplayObs) -> dict[tuple[str, str, str, int], _Claim]:
    """Per (model, version, site, day): alarm if any prediction that day calls a fault (P(fault) >= 0.5)."""
    out: dict[tuple, _Claim] = {}
    best: dict[tuple, float] = {}
    for p in preds:
        d = (_ts(p.t0) - obs.day0_ts) // DAY_S
        if not (0 <= d < obs.days) or p.asset_id not in obs.site_index:
            continue
        k = (p.model, p.model_version, p.asset_id, int(d))
        pf = p.p_fault
        alarm = pf >= 0.5 and p.top_fault != "healthy"
        c = out.get(k)
        if c is None:
            out[k], best[k] = _Claim(alarm, p.top_fault if alarm else "healthy", p.ood_score), pf
        else:
            c.ood = max(c.ood, p.ood_score)
            if alarm and (not c.alarm or pf > best[k]):
                c.alarm, c.cls, best[k] = True, p.top_fault, pf
    return out


def _turbine_claims(preds: Sequence[Any], obs: ReplayObs) -> dict[tuple[str, str, str, int], _Claim]:
    """Per (model, version, turbine, day): alarm if any step / prediction is not 'normal'; class = most frequent."""
    counts: dict[tuple, dict[str, int]] = defaultdict(lambda: defaultdict(int))
    ood: dict[tuple, float] = defaultdict(float)
    for p in preds:
        d = (_ts(p.t0) - obs.day0_ts) // DAY_S
        if not (0 <= d < obs.days) or p.asset_id not in obs.turbine_index:
            continue
        k = (p.model, p.model_version, p.asset_id, int(d))
        steps = (p.extra or {}).get("step_classes")
        cls = list(steps) if steps is not None else [p.anomaly_class or "normal"]
        for c in cls:
            counts[k][str(c)] += 1
        ood[k] = max(ood[k], p.ood_score)
    out = {}
    for k, cnt in counts.items():
        faults = {c: n for c, n in cnt.items() if c != "normal"}
        out[k] = _Claim(bool(faults), max(faults, key=faults.get) if faults else "normal", ood[k])
    return out


def _turbine_events(obs: ReplayObs) -> list[list[set]]:
    """Observable per turbine-day classes from telemetry: fault_stop (code 211), curtailment (code 40, operator
    log), data_gap (>= 4 unreported steps). Returns [N][H+D] sets of classes."""
    tc, tp = obs.tcode, obs.tpow
    stop = (tc == FAULT_STOP_CODE).any(-1)
    curt = (tc == CURTAIL_CODE).any(-1)
    gap = (~np.isfinite(tp)).sum(-1) >= 4
    ev = [[set() for _ in range(tc.shape[1])] for _ in range(tc.shape[0])]
    for arr, name in ((stop, "fault_stop"), (curt, "curtailment"), (gap, "data_gap")):
        for i, j in zip(*np.nonzero(arr)):
            ev[i][j].add(name)
    return ev


def _grade_detectors(claims: Mapping[tuple, _Claim], kind: str, obs: ReplayObs, reports_by: Mapping[str, list],
                     info: Mapping[str, _Asset], rows: _Rows, ood_thr: Mapping[str, float], diag: dict) -> None:
    H = obs.hist
    gate = obs.gate("pv" if kind == "site" else "turbine")
    index = obs.site_index if kind == "site" else obs.turbine_index
    resid = obs.residual_pass() if kind == "site" else None
    tev = _turbine_events(obs) if kind == "turbine" else None
    win = REPORT_WINDOW_DAYS * DAY_S
    miss_all = obs.missing_share("pv") if kind == "site" else 1.0 - np.isfinite(obs.tpow).mean(-1)
    cols: dict[str, list] = defaultdict(list)
    for (m, v, aid, d), c in claims.items():
        a = index[aid]
        ds, de = int(obs.day_ts[d]), int(obs.day_ts[d + 1])
        gated = int(gate[a, H + d] != 0)
        name = "alarm_precision" if c.alarm else "miss_score"
        ev: Optional[tuple[float, int, str, float]] = None  # (score, observed_at, source, weight)
        if not gated:
            reps = reports_by.get(aid, ())
            cands: list[tuple[int, int, float, str, float]] = []  # (tier, observed, score, source, weight)
            if c.alarm:
                for r in reps:
                    if ds <= r.visit_ts < ds + win:
                        existed = r.onset_ts is not None and r.onset_ts < de
                        sc = float(S.alarm_precision(np.array([c.cls]), np.array([r.found]), np.array([existed]))[0])
                        cands.append((0, r.visit_ts, sc, r.source, r.weight))
                if kind == "turbine" and c.cls in tev[a][H + d]:
                    cands.append((0, de, 1.0, "event" if c.cls == "curtailment" else "telemetry", 1.0))
            else:
                for r in reps:
                    if r.visit_ts >= ds and r.is_fault and r.onset_ts is not None and r.onset_ts < ds:
                        cands.append((0, r.visit_ts, 0.0, r.source, r.weight))
                    elif ds <= r.visit_ts < ds + win and not r.is_fault:
                        cands.append((0, r.visit_ts, 1.0, r.source, r.weight))
                if kind == "turbine" and tev[a][H + d]:
                    cands.append((0, de, 0.0, "telemetry", 1.0))
                if kind == "site" and np.isfinite(resid[a, H + d]):
                    # known once the 3-day window has been observed (clipped at the end of the replay)
                    ob = int(obs.day_ts[min(d + RESID_DAYS, obs.days)])
                    cands.append((1, ob, float(resid[a, H + d]), "residual", 1.0))
            if cands:
                t, ob, sc, src, w = min(cands, key=lambda x: (x[0], x[1], x[2]))
                ev = (sc, ob, src, w)
        if ev is None and not gated:
            continue  # no outcome: ungraded, trust unchanged (§11)
        inf = info[aid]
        cols["model"].append(m)
        cols["model_version"].append(v)
        cols["asset_id"].append(aid)
        cols["day"].append(obs.day_label[d])
        cols["score_name"].append(name)
        cols["score"].append(np.nan if gated else ev[0])
        cols["weight"].append(1.0 if gated else ev[3])
        cols["source"].append("telemetry" if gated else ev[2])
        cols["observed_at"].append(iso(de if gated else ev[1]))
        cols["gated"].append(gated)
        cols["kind"].append(inf.kind)
        cols["seen"].append(inf.seen)
        cols["battery"].append(inf.battery)
        cols["season"].append(obs.season[d])
        cols["quality"].append("gappy" if miss_all[a, H + d] > GAPPY else "clean")
        cols["ood_high"].append(bool(c.ood > float(ood_thr.get(m, OOD_DEFAULT))))
    n = len(cols["model"])
    rows.add(n, **cols)
    diag.setdefault("detectors", {})[kind] = {"asset_days": len(claims), "graded_rows": n}


# ------------------------------------------------------------------------------------------ slow witnesses
def _failure_onsets(obs: ReplayObs, events: Iterable[Mapping]) -> dict[str, list[int]]:
    """Observable component failures: the start of each fault-stop run in SCADA (code 211) + explicit events."""
    out: dict[str, list[int]] = defaultdict(list)
    if obs.tcode.size:
        flat = (obs.tcode == FAULT_STOP_CODE).reshape(len(obs.turbine_ids), -1)
        start = flat & ~np.concatenate([np.zeros((flat.shape[0], 1), bool), flat[:, :-1]], 1)
        t0 = obs.day0_ts - obs.hist * DAY_S
        for i, k in zip(*np.nonzero(start)):
            out[obs.turbine_ids[i]].append(int(t0 + k * DAY_S // STEPS))
    for e in events:
        out[str(e["asset_id"])].append(int(e["ts"]))
    return {k: sorted(v) for k, v in out.items()}


def _grade_m7(preds: Sequence[Any], obs: ReplayObs, events: Iterable[Mapping], horizon_end: int,
              info: Mapping[str, _Asset], rows: _Rows) -> None:
    fails = _failure_onsets(obs, events)
    hz = M7_HORIZON_DAYS * DAY_S
    cols: dict[str, list] = defaultdict(list)
    for p in preds:
        d = (_ts(p.t0) - obs.day0_ts) // DAY_S
        if not (0 <= d < obs.days) or p.asset_id not in obs.turbine_index:
            continue
        ds = int(obs.day_ts[d])
        ex = p.extra or {}
        risk = float(ex.get("risk", p.q50[0]))
        alarm = bool(ex.get("alarm", risk >= 0.5))
        f = [t for t in fails.get(p.asset_id, ()) if ds <= t < ds + hz]
        if f:
            failed, ob = True, f[0]
        elif horizon_end >= ds + hz:
            failed, ob = False, ds + hz
        else:
            continue  # 14 days have not passed yet: not knowable
        inf = info[p.asset_id]
        for name, sc in (("hit", float(S.m7_hit([alarm], [failed])[0])),
                         ("brier", float(S.brier_score([risk], [failed])[0]))):
            for k, v in (("model", p.model), ("model_version", p.model_version), ("asset_id", p.asset_id),
                         ("day", obs.day_label[d]), ("score_name", name), ("score", sc), ("weight", 1.0),
                         ("source", "event"), ("observed_at", iso(ob)), ("gated", 0), ("kind", inf.kind),
                         ("seen", inf.seen), ("battery", False), ("season", obs.season[d]), ("quality", "clean"),
                         ("ood_high", False)):
                cols[k].append(v)
    rows.add(len(cols["model"]), **cols)


def _grade_m8(flags: Iterable[Mapping], version: str, reports: Sequence[Any], obs: ReplayObs,
              info: Mapping[str, _Asset], rows: _Rows) -> None:
    flagged = {str(f["site_id"]) for f in flags if f.get("site_id") is not None}
    seen: set = set()
    cols: dict[str, list] = defaultdict(list)
    for r in sorted(reports, key=lambda r: r.visit_ts):
        if r.gis_wrong is None or r.asset_id not in obs.site_index:
            continue
        d = (r.visit_ts - obs.day0_ts) // DAY_S
        if not (0 <= d < obs.days) or (r.asset_id, d) in seen:
            continue
        seen.add((r.asset_id, d))
        is_flag = r.asset_id in flagged
        sc = float(r.gis_wrong) if is_flag else float(not r.gis_wrong)
        inf = info[r.asset_id]
        for k, v in (("model", "gridtwin"), ("model_version", version), ("asset_id", r.asset_id),
                     ("day", obs.day_label[d]), ("score_name", "flag" if is_flag else "miss_score"), ("score", sc),
                     ("weight", r.weight), ("source", r.source), ("observed_at", iso(r.visit_ts)), ("gated", 0),
                     ("kind", inf.kind), ("seen", inf.seen), ("battery", inf.battery), ("season", obs.season[d]),
                     ("quality", "clean"), ("ood_high", False)):
            cols[k].append(v)
    rows.add(len(cols["model"]), **cols)


# ------------------------------------------------------------------------------------------ entry points
def grade(predictions: Iterable[Any], obs: ReplayObs, reports: Sequence[Any] = (), *, run_id: str = "replay",
          gis_flags: Optional[Iterable[Mapping]] = None, gis_version: str = "mock-1", events: Iterable[Mapping] = (),
          ood_thresholds: Optional[Mapping[str, float]] = None, horizon_end_ts: Optional[int] = None,
          diag: Optional[dict] = None) -> list[dict[str, Any]]:
    """Grade a replay's predictions against observations and reports; returns `outcomes` rows (module doc).

    predictions: Prediction envelopes (any mix of witnesses and substitutes; detectors may have several per day).
    reports: CrewReport objects (world.crew_report / world.audit_checks). gis_flags: M8 output (world.gis_flags()).
    events: extra observable failure events {asset_id, ts} for M7. horizon_end_ts: last moment observed (default:
    end of the replay), so M7 'no failure in 14 days' is only graded once it is knowable.
    """
    diag = {} if diag is None else diag
    ood_thr = dict(ood_thresholds or {})
    rows = _Rows(run_id)
    info = _asset_info(obs)
    preds = list(predictions)
    fc, site_det, turb_det, m7 = [], [], [], []
    for p in preds:
        if p.model == "windfailure":
            m7.append(p)
        elif p.fault_probs is not None or p.model in SITE_DETECTORS:
            site_det.append(p)
        elif p.anomaly_class is not None or p.model in TURBINE_DETECTORS or (p.extra or {}).get("step_classes"):
            turb_det.append(p)
        elif p.model != "gridtwin":
            fc.append(p)
    reports = sorted(reports, key=lambda r: r.visit_ts)
    reports_by: dict[str, list] = defaultdict(list)
    for r in reports:
        reports_by[r.asset_id].append(r)
    _grade_forecasts(_forecast_groups(fc, obs), obs, info, rows, ood_thr, diag)
    _grade_detectors(_site_claims(site_det, obs), "site", obs, reports_by, info, rows, ood_thr, diag)
    _grade_detectors(_turbine_claims(turb_det, obs), "turbine", obs, reports_by, info, rows, ood_thr, diag)
    _grade_m7(m7, obs, events, int(obs.day_ts[obs.days] if horizon_end_ts is None else horizon_end_ts), info, rows)
    if gis_flags is not None:
        _grade_m8(gis_flags, gis_version, reports, obs, info, rows)
    out = rows.rows()
    out.sort(key=lambda r: (r["observed_at"], r["model"], r["asset_id"], r["score_name"]))
    return out


def grade_world(world: Any, predictions: Iterable[Any], *, run_id: str = "replay", crew_visits: Iterable = (),
                audit: bool = True, audit_weights: Optional[Callable[[int], Mapping[str, float]]] = None,
                hist: int = CLIM_DAYS, obs: Optional[ReplayObs] = None, **kw: Any) -> list[dict[str, Any]]:
    """Convenience: observations from the world, crew reports for `crew_visits` [(asset_id, visit_ts)] (action
    visits, weight 0.5), the Judge's audit checks for each day, M8 flags, then `grade`. Simulator truth is only
    reached through `world.crew_report` (anti-goal §2)."""
    obs = obs if obs is not None else obs_from_world(world, hist=hist)
    reports = [world.crew_report(a, t) for a, t in crew_visits]
    if audit:
        for d in range(world.days):
            reports += world.audit_checks(d, weights=audit_weights(d) if audit_weights else None)
    kw.setdefault("gis_flags", world.gis_flags())
    return grade(predictions, obs, reports, run_id=run_id, **kw)


def bucket_days(rows: Iterable[Mapping[str, Any]], as_of: Optional[str] = None,
                skip: Sequence[str] = ("norm_err",)) -> list[dict[str, Any]]:
    """§5.7: average the asset-day scores of one day within a context bucket into one bucket-day.

    Only ungated, scored rows with observed_at <= as_of (UTC ISO) are used, so calling it per day is causal.
    Bucket-day score = weight-weighted mean; weight = mean row weight (IPW / action down-weighting carried over);
    n = asset-days; observed_at = the latest contributing row.
    """
    acc: dict[tuple, list[float]] = {}
    last: dict[tuple, str] = {}
    for r in rows:
        if r["gated"] or r["score"] is None or r["score_name"] in skip:
            continue
        if as_of is not None and r["observed_at"] > as_of:
            continue
        k = (r["model"], r["model_version"], r["score_name"], *(r[c] for c in BUCKET_COLUMNS), r["day"])
        a = acc.setdefault(k, [0.0, 0.0, 0])
        w = float(r["weight"])
        a[0] += w * float(r["score"])
        a[1] += w
        a[2] += 1
        if last.get(k, "") < r["observed_at"]:
            last[k] = r["observed_at"]
    out = []
    for k, (sw, w, n) in acc.items():
        d = dict(zip(("model", "model_version", "score_name", *BUCKET_COLUMNS, "day"), k))
        d.update(score=sw / w if w > 0 else None, weight=w / n, n=n, observed_at=last[k])
        out.append(d)
    out.sort(key=lambda r: (r["day"], r["model"], r["score_name"]))
    return out
