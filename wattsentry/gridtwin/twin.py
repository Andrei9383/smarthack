"""Grid-twin (M8, card #13): check the GIS feeder / phase of every site against its own voltage, score how far we
trust each feeder's records (tau), and give the LP its effective export limit G = tau*G_max + (1-tau)*G_cons.

Physics it relies on (fleet/grid.py, real LV feeders behave the same): a site's inverter voltage = MV base of its
feeder + a rise driven by the net export on its own phase (and a little from the other phases) + noise. So:
  - remove each feeder's common MV component (median across its sites per step);
  - at night (PV off) sites on the same phase then share the remaining load-driven signal: starting from the GIS
    phases (90 % right), assign every site to the phase centroid it correlates with best, iterate -> a site that
    ends on another phase is a phase flag;
  - a site whose raw voltage follows another feeder's MV base better than its GIS feeder's -> a feeder flag.
Flags are review requests with a score ("likely L2 of F07, GIS says L1 of F06, 0.83"), never silent corrections.
Phase flags need a higher score than feeder flags (0.98 vs 0.7) and one assignment pass over 180 nights (#214): on the
600-home fleet (#211, ~8 sites per phase) the iterated assignment drifted near-transformer homes, whose phase signal is
weak, onto other phases (precision 0.37). Now 0.68 there and 0.91 on the default fleet (was 0.69).
tau per feeder = share of its sites whose record agrees with the data (scored), times a decay since the last field
verification; low trust only ever tightens the limit.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

from wattsentry.contracts.opt import FeederSpec

PHASES = ("A", "B", "C")
LABEL = {"A": "L1", "B": "L2", "C": "L3"}


@dataclass
class TwinResult:
    flags: pd.DataFrame        # one row per flagged site
    sites: pd.DataFrame        # every site: gis vs likely feeder / phase + scores
    feeders: pd.DataFrame      # feeder_id, tau, g_max_kw, g_conservative_kw, g_kw, n_flags


DAYS = 180  # nights of voltage the twin reads (#214: 60 left phase flags noisy)
PHASE_THRESHOLD = 0.98  # phase flags; feeder flags keep run_twin's `threshold`


def load_voltages(fleet_dir: Path, start: str = "2024-05-01", days: int = DAYS) -> tuple[pd.DataFrame, pd.DataFrame]:
    """(sites, voltage matrix [site x step]) for a window of daytime steps from a fleet dataset."""
    fleet_dir = Path(fleet_dir)
    sites = pd.read_parquet(fleet_dir / "sites.parquet")
    t0 = pd.Timestamp(start, tz="UTC")
    t1 = t0 + pd.Timedelta(days=days)
    parts = [pd.read_parquet(f, columns=["ts", "site_id", "inv_voltage_v", "pv_ac_w"],
                             filters=[("ts", ">=", t0), ("ts", "<", t1)])
             for f in sorted((fleet_dir / "telemetry").glob("year=*.parquet"))]
    tel = pd.concat(parts)
    tel["site_id"] = tel.site_id.astype(str)
    v = tel.pivot(index="site_id", columns="ts", values="inv_voltage_v").reindex(sites.site_id)
    pv = tel.pivot(index="site_id", columns="ts", values="pv_ac_w").reindex(sites.site_id)
    v.attrs["night"] = ((pv.fillna(0) > 0).mean(axis=0) < 0.02).to_numpy()  # sun down for (almost) every site
    return sites, v


def _corr(x: np.ndarray) -> np.ndarray:
    x = x - np.nanmean(x, 1, keepdims=True)
    x = np.nan_to_num(x)
    n = np.linalg.norm(x, axis=1, keepdims=True)
    x = x / np.maximum(n, 1e-9)
    return x @ x.T


def recover_feeders(sites: pd.DataFrame, v: pd.DataFrame) -> pd.DataFrame:
    """Per site: the feeder whose MV base its voltage follows best (correlation with each feeder's median voltage)."""
    vm = v.to_numpy(float)
    feeders = sorted(sites.gis_feeder_id.unique())
    base = np.stack([np.nanmedian(vm[(sites.gis_feeder_id == f).to_numpy()], 0) for f in feeders])
    d = np.diff(np.vstack([vm, base]), axis=1)  # step changes: the slow daily swing is shared by all feeders
    c = _corr(d)[: len(vm), len(vm):]  # [site, feeder]
    best = c.argmax(1)
    gis = np.array([feeders.index(f) for f in sites.gis_feeder_id])
    own = c[np.arange(len(vm)), gis]
    top = c[np.arange(len(vm)), best]
    # softmax-style confidence between the best feeder and the GIS one
    score = 1 / (1 + np.exp(-(top - own) * 40))
    return pd.DataFrame({"site_id": sites.site_id.to_numpy(), "likely_feeder_id": np.array(feeders)[best],
                         "feeder_corr_gis": own.round(3), "feeder_corr_best": top.round(3),
                         "feeder_score": np.where(best == gis, 1 - score, score).round(3)})


def _phase_features(vm: np.ndarray, night: np.ndarray | None) -> np.ndarray:
    """Night-time voltage levels minus the feeder median per step: PV is off, so what is left is the load on each
    phase (the MV base and anything feeder-wide cancel). Falls back to step changes if there is no night mask."""
    x = vm - np.nanmedian(vm, 0, keepdims=True)
    x = x[:, night] if night is not None and night.sum() > 50 else np.diff(x, axis=1)
    x = np.nan_to_num(x - np.nanmean(x, 1, keepdims=True))
    # Every phase's load follows the same evening/night shape with its own amplitude, which leaves a shared
    # component scaled by position x amplitude: project it out (top principal component), keep what is phase-specific.
    u, sv, vt = np.linalg.svd(x, full_matrices=False)
    return x - (u[:, :1] * sv[:1]) @ vt[:1]


def recover_phases(sites: pd.DataFrame, v: pd.DataFrame, feeder_col: str = "likely_feeder_id",
                   seed: int = 0, night: np.ndarray | None = None, iters: int = 1) -> pd.DataFrame:
    """Per feeder, GIS-seeded centroid assignment: each phase's centroid = mean normalised profile of the sites the
    current assignment puts on it (starting from the GIS, ~90 % right); every site moves to the centroid it
    correlates with best (its own profile left out); repeat. Averaging ~10+ sites per centroid removes most of
    the measurement noise, which pairwise clustering cannot. One pass by default (#214): repeating it lets
    centroids on small feeders drift with the weak near-transformer homes. Score = how significant the margin of the best phase
    over the GIS phase is (correlation difference in units of its standard error)."""
    out = []
    vm_all = v.to_numpy(float)
    for f, idx in sites.groupby(feeder_col).indices.items():
        sub = sites.iloc[idx]
        gis = sub.gis_phase.to_numpy()
        if len(idx) < 6:
            out.append(pd.DataFrame({"site_id": sub.site_id, "likely_phase": gis, "phase_score": 0.0}))
            continue
        x = _phase_features(vm_all[idx], night)
        x = x / np.maximum(np.linalg.norm(x, axis=1, keepdims=True), 1e-9)
        n_t = x.shape[1]
        assign = np.array([PHASES.index(p) for p in gis])
        for _ in range(iters):
            sums = np.stack([x[assign == k].sum(0) for k in range(3)])  # [3, T]
            cnt = np.array([(assign == k).sum() for k in range(3)])
            # leave-one-out centroids: a site's own profile must not pull its centroid
            c = np.empty((len(idx), 3))
            for k in range(3):
                own = assign == k
                cen = (sums[k][None] - np.where(own[:, None], x, 0)) / np.maximum(cnt[k] - own, 1)[:, None]
                c[:, k] = (x * cen).sum(1) / np.maximum(np.linalg.norm(cen, axis=1), 1e-9)
            new = c.argmax(1)
            if np.array_equal(new, assign):
                break
            assign = new
        g = np.array([PHASES.index(p) for p in gis])
        margin = c[np.arange(len(idx)), assign] - c[np.arange(len(idx)), g]  # >= 0
        z = margin * np.sqrt(n_t)
        score = np.where(assign != g, 1 - np.exp(-z / 3), 1.0)
        out.append(pd.DataFrame({"site_id": sub.site_id.to_numpy(), "likely_phase": np.array(PHASES)[assign],
                                 "phase_score": np.clip(score, 0, 1).round(3)}))
    return pd.concat(out, ignore_index=True)


def trust(sites_res: pd.DataFrame, feeders: pd.DataFrame, days_since_verified: pd.Series | None = None,
          half_life_days: float = 365.0) -> pd.Series:
    """tau per feeder in [0, 1]: agreement of its GIS records with the data x decay since field verification."""
    flagged = sites_res.phase_flag | sites_res.feeder_flag
    agree = 1 - flagged.groupby(sites_res.gis_feeder_id).mean()
    tau = (agree.reindex(feeders.feeder_id).fillna(1.0)) ** 2  # a few bad records already cost trust
    if days_since_verified is not None:
        tau = tau * 0.5 ** (days_since_verified.reindex(feeders.feeder_id).fillna(0) / half_life_days)
    return tau.clip(0, 1)


def run_twin(sites: pd.DataFrame, v: pd.DataFrame, feeders: pd.DataFrame, threshold: float = 0.7,
             days_since_verified: pd.Series | None = None, phase_threshold: float = PHASE_THRESHOLD) -> TwinResult:
    fr = recover_feeders(sites, v)
    s = sites[["site_id", "gis_feeder_id", "gis_phase", "lat", "lon"]].merge(fr, on="site_id")
    pr = recover_phases(s, v.loc[s.site_id.to_numpy()], seed=0, night=v.attrs.get("night"))  # rows aligned with s
    s = s.merge(pr, on="site_id")
    s["feeder_flag"] = (s.likely_feeder_id != s.gis_feeder_id) & (s.feeder_score >= threshold)
    s["phase_flag"] = (s.likely_phase != s.gis_phase) & (s.phase_score >= phase_threshold) & ~s.feeder_flag
    s["message"] = [
        f"likely {LABEL[r.likely_phase]} of {r.likely_feeder_id}, GIS says {LABEL[r.gis_phase]} of {r.gis_feeder_id}, "
        f"{max(r.feeder_score if r.feeder_flag else 0, r.phase_score if r.phase_flag else 0):.2f}"
        for r in s.itertuples()]
    tau = trust(s, feeders, days_since_verified)
    f = feeders[["feeder_id", "g_max_kw", "g_conservative_kw"]].copy()
    f["tau"] = f.feeder_id.map(tau).round(3)
    f["g_kw"] = [FeederSpec(r.feeder_id, r.g_max_kw, r.g_conservative_kw, r.tau).g_kw for r in f.itertuples()]
    f["n_flags"] = f.feeder_id.map((s.phase_flag | s.feeder_flag).groupby(s.gis_feeder_id).sum()).fillna(0).astype(int)
    flags = s[s.phase_flag | s.feeder_flag].copy()
    flags["kind"] = np.where(flags.feeder_flag, "feeder", "phase")
    flags["score"] = np.where(flags.feeder_flag, flags.feeder_score, flags.phase_score)
    return TwinResult(flags.reset_index(drop=True), s, f)


def to_geojson(res: TwinResult) -> dict:
    feats = [{
        "type": "Feature",
        "geometry": {"type": "Point", "coordinates": [float(r.lon), float(r.lat)]},
        "properties": {"site_id": r.site_id, "kind": r.kind, "score": float(r.score), "message": r.message,
                       "gis_feeder_id": r.gis_feeder_id, "gis_phase": r.gis_phase,
                       "likely_feeder_id": r.likely_feeder_id, "likely_phase": r.likely_phase,
                       "status": "review_requested"},
    } for r in res.flags.itertuples()]
    return {"type": "FeatureCollection", "features": feats}
