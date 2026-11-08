"""The Judge slot in the agent loop (team/judge-design.md §7.5, §7.7, §8, §11, §13; card #149).

    slot = JudgeSlot.load(models_dir)                  # never raises; ledger-only mode when the Judge is missing
    slot.start_run(world, run_id, sink)
    slot.at_t0(d, t0, pv_p, ld_p, hl_p, ages_t0_s)     # PV / load / SiteHealth verdicts, before decide
    slot.observe_wind_tick(k, wind_preds)              # M6 classes as the loop sees them, tick by tick
    slot.at_end_of_day(d, oday, tday, pv_p, ld_p)      # M6 day verdicts, grading hook, ledger update, trust rows

What it does: one verdict per (witness, asset, day), written to `judge_verdicts` next to the predictions. Actions
are still open (§10): nothing in the loop reads the verdicts, so decisions and KPIs are unchanged with the judge on.

Pieces, each optional (guarded imports, §11 "Judge model cannot load -> ledger-only mode"):
- Judge model (J3, #148): `wattsentry.judge.model.load_judge(models_dir) -> Judge | None`; `Judge.predict(rows)` ->
  p_valid, p_valid_sd, reasons; thresholds tau_use / tau_widen / widen_factor and judge_version. Missing module or
  weights -> P(valid) = ledger trust, judge_version "ledger-only".
- Ledger (J2, #147): `wattsentry.judge.ledger.Ledger` with `lookup(model, model_version, score_name, bucket, day)`
  and `update(outcomes)`. Missing -> `PriorLedger`: trust = the model's validation prior (the `priors` block of its
  calibration JSON, models/judge_priors.json, windnbm.priors.json), n_eff = 0, so every verdict is rule 1 or 3
  (§11 "ledger empty": the agent behaves as today, at validated strength).
- Grader (J1, #146): `wattsentry.judge.grader.grade` (via `_J1Grader`) -> §8 outcome rows at the end of each day.
  Missing -> no outcomes, trust stays at the prior.

Causality (§5.1, §5.5): at t0 the slot reads only the predictions, telemetry age at t0, and the observed telemetry
of past days (kept from earlier `at_end_of_day` calls). Never fault labels or simulator truth.

The per-prediction rows frame (`judge_rows`, also what J3 trains on) is documented on card #149 and in ROW_COLUMNS.
"""

from __future__ import annotations

import copy
import importlib
import json
import math
import sys
import time
import traceback
from collections import Counter
from datetime import datetime
from pathlib import Path
from typing import Any, Callable, Mapping, Optional, Sequence

import numpy as np
import pandas as pd

from .buckets import bucket_key as _bucket_key
from .buckets import season_of
from .verdict import LEDGER_ONLY, SAFETY_CLASSES, Thresholds, apply_rules, verdict_rows

REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_MODELS = REPO_ROOT / "models"
STALE_S = 3600.0  # qualify's stale limit (models/thresholds.json site.stale_minutes = 60)
GAPPY = 0.10  # §6.1: > 10 % missing steps = gappy
FROZEN_STEPS = 8  # §5.4: the same value for >= 8 steps
PV_MAX_KWP = 1.2  # §5.4: PV <= 1.2 x kWp
TURBINE_MAX_RATED = 1.1  # §5.4: turbine power <= 1.1 x rated
UNIT_JUMP = (0.2, 5.0)  # §5.4: daily energy outside 0.2-5 x the asset's own median
ALARM_P = 0.6  # models/thresholds.json site.pfault_confident
OOD_MAX = 3.0  # models/thresholds.json site/turbine ood_max

# witness -> (family, substitute model id (J2 priors names), primary score for P(valid) in ledger-only mode)
WITNESSES = {
    "pvnet": ("forecast", "lgbm_pv"),
    "loadnet": ("forecast", "lgbm_load"),
    "demandnet": ("forecast", "persistence_zone"),
    "sitehealth": ("detector", "threshold_cusum"),
    "windnbm": ("detector", "windmon_rules"),
    "windfailure": ("detector", "residual_trend"),
    "gridtwin": ("detector", "gis_recorded"),
}
LEDGER_ONLY_WITNESSES = frozenset({"windfailure", "gridtwin"})  # §7.4: too few graded outcomes to train on
SCORES = {"forecast": ("calibration", "sharpness"), "detector": ("alarm_precision", "miss_score")}
ROW_COLUMNS = (
    # id / context
    "run_id", "day", "t0", "asset_id", "asset_kind", "witness", "model_version", "family", "substitute", "bucket",
    "battery", "seen", "season", "month", "quality", "ood_high",
    # own signals
    "ood_score", "member_std", "rel_spread", "band_width_rel", "is_fallback", "telemetry_age_min", "gap_share",
    "frozen_input", "unit_jump", "input_bad",
    # detectors
    "p_fault", "top_fault", "alarm", "anomaly_class", "anomaly_score", "safety_call",
    # cross-witness agreement
    "pv_vs_pvlib", "pv_residual_d1", "park_consensus", "m8_flag",
    # ledger
    "prior", "trust", "n_eff", "lo80", "hi80", "trust_trend_3d", "band_scale", "health_alert", "sub_trust",
    "sub_n_eff",
)


def log(msg: str) -> None:
    print(f"[judge] {msg}", file=sys.stderr, flush=True)


# ------------------------------------------------------------------------------------------- buckets (§6.1)
def bucket_ancestors(key: str) -> list[str]:
    parts = key.split("|") if key and key != "*" else []
    return ["|".join(parts[:lv]) for lv in range(len(parts), 0, -1)] + ["*"]


# ------------------------------------------------------------------------------------------- built-in ledger
class PriorLedger:
    """No live record: trust = the validation prior of that model version (§6.2), n_eff = 0, band scale 1."""

    version = "builtin:prior-only"

    def __init__(self, models_dir: Optional[Path] = None):
        self.priors: dict[str, dict[str, dict[str, float]]] = {}  # model -> score -> bucket -> prior
        d = Path(models_dir or DEFAULT_MODELS)
        files = sorted(d.glob("*.calibration.json")) + sorted(d.glob("*.priors.json")) + \
            [p for p in (d / "judge_priors.json",) if p.exists()]
        for f in files:
            try:
                obj = json.loads(f.read_text())
            except Exception:
                continue
            if f.name == "judge_priors.json":
                for mid, block in (obj.get("models") or {}).items():
                    self._add(mid, block)
                continue
            block = obj.get("priors") if isinstance(obj, dict) and "priors" in obj else obj
            mid = (obj.get("model") if isinstance(obj, dict) else None) or f.name.split(".")[0]
            self._add(str(mid).split(".")[0], block)

    def _add(self, model: str, block: Any) -> None:
        scores = block.get("scores") if isinstance(block, dict) else None
        if not isinstance(scores, dict):
            return
        m = self.priors.setdefault(model, {})
        for score, by_bucket in scores.items():
            if isinstance(by_bucket, dict):
                for b, v in by_bucket.items():
                    pv = v.get("prior") if isinstance(v, dict) else v
                    try:
                        m.setdefault(score, {}).setdefault(b, float(pv))
                    except (TypeError, ValueError):
                        pass

    def prior(self, model: str, score: str, bucket: str) -> float:
        by_bucket = self.priors.get(model, {}).get(score, {})
        for key in bucket_ancestors(bucket):
            if key in by_bucket:
                return by_bucket[key]
        return float("nan")

    def lookup(self, model: str, model_version: Optional[str], score_name: str, bucket: str, day: Any = None
               ) -> dict[str, Any]:
        p = self.prior(model, score_name, bucket)
        return {"prior": p, "trust": p, "n_eff": 0.0, "lo80": float("nan"), "hi80": float("nan"),
                "band_scale": 1.0, "health_alert": False}

    def update(self, outcomes: Any) -> None:  # no live record kept
        return None


class _LedgerAdapter:
    """Duck-typed wrapper over J2's ledger: lookup -> the mapping the slot needs, update(outcomes)."""

    def __init__(self, obj: Any, fallback: PriorLedger):
        self.obj, self.fallback = obj, fallback
        self.version = getattr(obj, "version", None) or f"real:wattsentry.judge.ledger.{type(obj).__name__}"
        self.errors = 0

    def lookup(self, model, model_version, score_name, bucket, day=None) -> dict[str, Any]:
        try:
            r = self.obj.lookup(model, model_version, score_name, bucket, day)
            r = dict(r) if isinstance(r, Mapping) else dict(vars(r))
        except Exception as e:
            self._err("lookup", e)
            return self.fallback.lookup(model, model_version, score_name, bucket, day)
        base = self.fallback.lookup(model, model_version, score_name, bucket, day)
        base.update({k: v for k, v in r.items() if v is not None})
        return base

    def update(self, outcomes: Any) -> None:
        fn = getattr(self.obj, "update", None) or getattr(self.obj, "add_outcomes", None)
        if fn is None:
            return
        try:
            fn(outcomes)
        except Exception as e:
            self._err("update", e)

    def _err(self, what: str, e: BaseException) -> None:
        self.errors += 1
        if self.errors == 1:
            log(f"WARNING ledger.{what} raised {type(e).__name__}: {e}; prior-only for this call (further errors "
                f"counted, not printed)")


class _J2Ledger:
    """J2's batch ledger (#147, `wattsentry.judge.ledger.Ledger(outcomes, priors, start)`) used day by day: outcomes
    accumulate as the replay grades them, the ledger is rebuilt when they change, and `lookup(day=D)` reads only
    outcomes with observed_at <= D (J2's causal grid). No outcomes yet -> the version's prior (J2 `Priors`)."""

    def __init__(self, mod: Any, models_dir: Path, start: Optional[str] = None, run_id: str = ""):
        self.mod, self.start, self.run_id = mod, start, run_id
        self.priors = mod.Priors.from_dir(models_dir)
        self.rows: list[dict[str, Any]] = []
        self.L: Any = None
        self.version = "real:wattsentry.judge.ledger.Ledger"

    def update(self, outcomes: Any) -> None:
        recs = outcomes.to_dict("records") if hasattr(outcomes, "to_dict") else list(outcomes)
        if recs:
            self.rows.extend({k: _clean(v) for k, v in r.items()} for r in recs)
            self.L = None

    def lookup(self, model, model_version, score_name, bucket, day=None) -> dict[str, Any]:
        if not self.rows:
            p = float(self.priors.get(model, model_version, score_name, bucket))
            lo, hi = self.mod.beta_interval(np.array([p]), np.array([0.0]))
            return {"prior": p, "trust": p, "n_eff": 0.0, "lo80": float(lo[0]), "hi80": float(hi[0]),
                    "band_scale": 1.0, "health_alert": False}
        if self.L is None:
            import polars as pl

            self.L = self.mod.Ledger(pl.DataFrame(self.rows, infer_schema_length=None), self.priors,
                                     start=self.start, run_id=self.run_id)
        return dict(self.L.lookup(model, model_version, score_name, bucket, day))


def _load_ledger(models_dir: Path, start: Optional[str] = None, run_id: str = "") -> Any:
    fallback = PriorLedger(models_dir)
    try:
        mod = importlib.import_module("wattsentry.judge.ledger")
    except ImportError:
        return fallback
    except Exception as e:
        log(f"WARNING wattsentry.judge.ledger failed to import ({type(e).__name__}: {e}); prior-only ledger")
        return fallback
    try:  # J2 (#147)
        return _LedgerAdapter(_J2Ledger(mod, models_dir, start, run_id), fallback)
    except Exception as e:
        log(f"WARNING J2 ledger could not be built ({type(e).__name__}: {e}); prior-only ledger")
        return fallback


def _load_model(models_dir: Path) -> tuple[Any, str]:
    try:
        mod = importlib.import_module("wattsentry.judge.model")
    except ImportError:
        return None, "no wattsentry.judge.model"
    except Exception as e:
        return None, f"wattsentry.judge.model failed to import ({type(e).__name__}: {e})"
    try:
        j = mod.load_judge(models_dir)
    except Exception as e:
        return None, f"load_judge raised {type(e).__name__}: {e}"
    return (j, "loaded") if j is not None else (None, "no Judge weights")


class _J1Grader:
    """J1's grader (#146, `wattsentry.judge.grader.grade`) called once per replay day on that day's predictions.

    Observations come from `obs_from_world` (what the agent observed, built once per run); reports are the Judge's
    audit checks for the day (`world.audit_checks`, sampled towards uncertain verdicts, §5.6); M8 flags on day 0.
    Outcomes that need later data (residual check, visits) carry their own later observed_at, and the ledger only
    reads an outcome from the day it was observable (§5.5)."""

    def __init__(self, mod: Any):
        self.mod = mod
        self._obs: dict[int, Any] = {}
        self.version = "real:wattsentry.judge.grader.grade"

    def __call__(self, ctx: dict[str, Any]) -> Any:
        w, d = ctx["world"], int(ctx["d"])
        if id(w) not in self._obs:
            self._obs = {id(w): self.mod.obs_from_world(w)}
        obs = self._obs[id(w)]
        preds = [p for k in ("pvnet", "loadnet", "sitehealth") for p in ctx["predictions"].get(k, {}).values()]
        calls = ctx.get("wind_calls") or {}
        for t, p in ctx["predictions"].get("windnbm", {}).items():
            if (getattr(p, "extra", None) or {}).get("step_classes") is None and t in calls:
                p = copy.copy(p)  # the loop's per-tick M6 stand-in classes, summarised as the day's call
                p.anomaly_class, p.anomaly_score = calls[t]
            preds.append(p)
        reports = []
        if hasattr(w, "audit_checks"):
            weights = ctx.get("audit_weights") or None
            try:
                reports = list(w.audit_checks(d, weights=weights))
            except TypeError:
                reports = list(w.audit_checks(d))
        kw: dict[str, Any] = {"run_id": ctx["run_id"], "horizon_end_ts": int(w.day_ts(d)) + 86400}
        if d == 0 and ctx.get("gis_flags"):
            kw["gis_flags"] = ctx["gis_flags"]
            kw["gis_version"] = str(getattr(w, "gridtwin_version", "mock-1"))
        return self.mod.grade(preds, obs, reports, **kw)


def _load_grader() -> Optional[Callable[[dict[str, Any]], Any]]:
    try:
        mod = importlib.import_module("wattsentry.judge.grader")
    except Exception:
        return None
    return _J1Grader(mod)


# ------------------------------------------------------------------------------------------- input gate (§5.4)
def _runs_frozen(x: np.ndarray, ok: np.ndarray, n: int = FROZEN_STEPS) -> np.ndarray:
    """[A, T] -> [A] bool: >= n consecutive equal observed values > 0.05, ignoring runs at the day's max (a clipped
    inverter is flat at its limit; that is clipping, not a frozen sensor)."""
    if x.size == 0:
        return np.zeros(x.shape[0], bool)
    mx = np.nanmax(np.where(ok, x, -np.inf), axis=1, keepdims=True)
    same = ok[:, 1:] & ok[:, :-1] & (np.abs(np.diff(x, axis=1)) < 1e-9) & (x[:, 1:] > 0.05) & \
        (x[:, 1:] < 0.99 * mx)
    run = np.zeros(x.shape[0], int)
    best = np.zeros(x.shape[0], int)
    for t in range(same.shape[1]):  # 95 steps, vectorised over assets
        run = np.where(same[:, t], run + 1, 0)
        best = np.maximum(best, run)
    return best + 1 >= n


def _unit_jump(energy_today: np.ndarray, history: np.ndarray) -> np.ndarray:
    """history [A, D] of past daily energies (NaN = unknown); needs >= 2 past days."""
    if history.shape[1] < 2:
        return np.zeros(len(energy_today), bool)
    with np.errstate(invalid="ignore", divide="ignore"):
        med = np.nanmedian(history, axis=1)
        r = energy_today / med
    lo, hi = UNIT_JUMP
    return np.isfinite(r) & (med > 0.5) & ((r < lo) | (r > hi))


def _f(v: Any, default: float = float("nan")) -> float:
    try:
        f = float(v)
    except (TypeError, ValueError):
        return default
    return f if math.isfinite(f) else default


# ------------------------------------------------------------------------------------------- the slot
class JudgeSlot:
    """Per-run state: past observed days (input gate + residuals), ledger, trust trend, timings."""

    def __init__(self, models_dir: Optional[Path] = None, judge: Any = None, model_status: str = "",
                 grader: Optional[Callable] = None):
        self.models_dir = Path(models_dir or DEFAULT_MODELS)
        self.judge = judge
        self.model_status = model_status
        self.grader = grader
        self.thresholds = Thresholds.from_judge(judge)
        self.ledger: Any = None
        self.errors: dict[str, int] = {}
        self._reset()

    # -- construction -------------------------------------------------------------------------------
    @classmethod
    def load(cls, models_dir: Optional[Path] = None) -> "JudgeSlot":
        d = Path(models_dir or DEFAULT_MODELS)
        judge, status = _load_model(d)
        slot = cls(d, judge, status, _load_grader())
        log(f"{slot.source}")
        return slot

    @property
    def mode(self) -> str:
        return "judge" if self.judge is not None else LEDGER_ONLY

    @property
    def source(self) -> str:
        led = getattr(self.ledger, "version", None) or "ledger chosen per run"
        gr = (getattr(self.grader, "version", None) or "real:wattsentry.judge.grader.grade_loop_day") if self.grader \
            else "absent"
        head = f"real:wattsentry.judge.model {self.thresholds.judge_version}" if self.judge is not None else \
            f"ledger-only ({self.model_status})"
        return f"{head}; {led}; grader {gr}"

    def _reset(self) -> None:
        self.run_id = "local"
        self.sink: Any = None
        self.world: Any = None
        self.tables: set[str] = set()
        self.hist_pv: dict[str, list[np.ndarray]] = {}  # site -> past observed days (NaN = not observed)
        self.hist_load: dict[str, list[np.ndarray]] = {}
        self.residual_d1: dict[str, float] = {}
        self.trust_hist: dict[tuple[str, str, str, str], list[tuple[int, float]]] = {}
        self.wind_counts: dict[str, Counter] = {}
        self.wind_calls: dict[str, tuple[str, float]] = {}
        self.wind_score: dict[tuple[str, str], float] = {}
        self.gis_sites: set[str] = set()
        self.gis_flags: list[dict[str, Any]] = []
        self.seconds: dict[int, float] = {}
        self.counts: Counter = Counter()
        self.n_outcomes = 0
        self.n_trust = 0
        self.day_rows: dict[int, pd.DataFrame] = {}

    def start_run(self, world: Any, run_id: str, sink: Any, gis_flags: Sequence[Mapping[str, Any]] = ()) -> None:
        self._reset()
        self.world, self.run_id, self.sink = world, run_id, sink
        self.ledger = _load_ledger(self.models_dir, self._context(world.day_t0(0))[0], run_id)
        for t in ("judge_verdicts", "trust", "outcomes"):
            ok = sink.has_table(t) if hasattr(sink, "has_table") else True
            if ok:
                self.tables.add(t)
            else:
                log(f"table {t!r} not in this DB's schema: its rows are not written (verdicts still computed)")
        self.gis_flags = [dict(f) for f in gis_flags]
        self.gis_sites = {str(f.get("site_id")) for f in self.gis_flags if f.get("site_id") is not None}

    def _timed(self, d: int, fn: Callable, *a: Any) -> Any:
        t = time.perf_counter()
        try:
            return fn(*a)
        except Exception as e:  # the judge never stops the loop: it is an observer until actions read it (§10)
            n = self.errors.get(fn.__name__, 0)
            self.errors[fn.__name__] = n + 1
            if n == 0:
                log(f"WARNING {fn.__name__} raised {type(e).__name__}: {e} (further errors counted)")
                traceback.print_exc(limit=4, file=sys.stderr)
            return None
        finally:
            self.seconds[d] = self.seconds.get(d, 0.0) + time.perf_counter() - t

    # -- loop hooks ---------------------------------------------------------------------------------
    def at_t0(self, d: int, t0: str, pv_p: Mapping[str, Any], ld_p: Mapping[str, Any], hl_p: Mapping[str, Any],
              ages_t0_s: Mapping[str, float]) -> Optional[pd.DataFrame]:
        return self._timed(d, self._at_t0, d, t0, pv_p, ld_p, hl_p, ages_t0_s)

    def observe_wind_tick(self, d: int, k: int, wind_preds: Mapping[str, Any]) -> None:
        self._timed(d, self._observe_wind, k, wind_preds)

    def at_end_of_day(self, d: int, t0: str, oday: Mapping[str, Any], tday: Mapping[str, Any],
                      wd_p: Mapping[str, Any], pv_p: Mapping[str, Any], ld_p: Mapping[str, Any],
                      hl_p: Mapping[str, Any]) -> Optional[pd.DataFrame]:
        return self._timed(d, self._at_end, d, t0, oday, tday, wd_p, pv_p, ld_p, hl_p)

    # -- rows ---------------------------------------------------------------------------------------
    def _context(self, t0: str) -> tuple[str, int, str]:
        from wattsentry.agent.audit import local_dt, parse_iso

        loc = local_dt(parse_iso(t0))
        return loc.strftime("%Y-%m-%d"), loc.month, season_of(loc.month)

    def _site_signals(self, sids: list[str], ages_t0_s: Mapping[str, float]) -> dict[str, np.ndarray]:
        """Input-window gate per site from yesterday's observed telemetry (vectorised over [sites, 96])."""
        A = len(sids)
        rows = [self.world.sites[s].row for s in sids]
        kwp = np.array([_f(r.get("kwp_recorded") or r.get("kwp"), np.inf) for r in rows])
        have = np.array([bool(self.hist_pv.get(s)) for s in sids])
        T = 96
        pv = np.full((A, T), np.nan)
        ld = np.full((A, T), np.nan)
        for i, s in enumerate(sids):
            if have[i]:
                pv[i] = self.hist_pv[s][-1]
                ld[i] = self.hist_load[s][-1]
        ok = np.isfinite(pv)
        gap = np.where(have, 1.0 - ok.mean(axis=1), np.nan)
        rng_bad = have & np.any(ok & ((pv < -1e-6) | (pv > PV_MAX_KWP * kwp[:, None])), axis=1)
        frozen = have & _runs_frozen(np.nan_to_num(pv), ok)
        e_pv = np.nansum(pv, axis=1) * 0.25
        e_ld = np.nansum(ld, axis=1) * 0.25
        D = max((len(self.hist_pv.get(s, [])) for s in sids), default=0)
        hpv = np.full((A, max(D - 1, 0)), np.nan)
        hld = np.full((A, max(D - 1, 0)), np.nan)
        for i, s in enumerate(sids):
            past = self.hist_pv.get(s, [])[:-1]
            pastl = self.hist_load.get(s, [])[:-1]
            for j, (a, b) in enumerate(zip(past[-(D - 1):] if D > 1 else [], pastl[-(D - 1):] if D > 1 else [])):
                fa, fb = np.isfinite(a), np.isfinite(b)
                hpv[i, j] = np.nansum(a) * 0.25 if fa.mean() > 0.5 else np.nan
                hld[i, j] = np.nansum(b) * 0.25 if fb.mean() > 0.5 else np.nan
        jump_pv = have & (ok.mean(axis=1) > 0.5) & _unit_jump(e_pv, hpv)
        jump_ld = have & (np.isfinite(ld).mean(axis=1) > 0.5) & _unit_jump(e_ld, hld)
        age = np.array([_f(ages_t0_s.get(s), 0.0) for s in sids])
        return {"gap": gap, "range": rng_bad, "frozen": frozen, "jump_pv": jump_pv, "jump_ld": jump_ld,
                "stale": age > STALE_S, "age_min": age / 60.0}

    def _base(self, d_iso: str, t0: str, month: int, season: str, asset_id: str, kind: str, witness: str,
              p: Any) -> dict[str, Any]:
        fam, sub = WITNESSES.get(witness, ("detector", None))
        q10, q50, q90 = (np.asarray(getattr(p, a), float) for a in ("q10", "q50", "q90"))
        m50 = float(q50.mean()) if q50.size else float("nan")
        wid = float((q90 - q10).mean()) if q50.size else float("nan")
        ext = getattr(p, "extra", None) or {}
        fb = ext.get("is_fallback")
        return {"run_id": self.run_id, "day": d_iso, "t0": t0, "asset_id": asset_id, "asset_kind": kind,
                "witness": witness, "model_version": str(getattr(p, "model_version", "?")), "family": fam,
                "substitute": sub, "month": month, "season": season,
                "ood_score": _f(getattr(p, "ood_score", 0.0)), "member_std": _f(getattr(p, "member_std", 0.0)),
                "rel_spread": _f(getattr(p, "member_std", 0.0)) / m50 if m50 > 1e-6 else float("nan"),
                "band_width_rel": wid / m50 if m50 > 1e-6 else float("nan"),
                "is_fallback": int(fb) if fb is not None else int(str(getattr(p, "model_version", "")).startswith("mock")),
                "pv_vs_pvlib": float("nan"), "pv_residual_d1": float("nan"), "park_consensus": float("nan"),
                "m8_flag": False, "p_fault": float("nan"), "top_fault": None, "alarm": False, "anomaly_class": None,
                "anomaly_score": float("nan"), "safety_call": False}

    def judge_rows(self, d: int, t0: str, pv_p: Mapping[str, Any], ld_p: Mapping[str, Any],
                   hl_p: Mapping[str, Any], ages_t0_s: Mapping[str, float]) -> pd.DataFrame:
        """Rows for the site witnesses at t0 (PV, load, SiteHealth), ledger columns not yet filled."""
        d_iso, month, season = self._context(t0)
        sids = sorted(set(pv_p) | set(ld_p) | set(hl_p))
        if not sids:
            return pd.DataFrame(columns=list(ROW_COLUMNS))
        sig = self._site_signals(sids, ages_t0_s)
        pvlib = getattr(self.world, "pvlib_day", None)
        out = []
        for i, s in enumerate(sids):
            row = self.world.sites[s].row
            batt = self.world.sites[s].battery is not None
            seen = str(row.get("split", "train")) != "unseen"
            gap = sig["gap"][i]
            common = {"battery": batt, "seen": seen, "quality": "gappy" if (gap == gap and gap > GAPPY) else "clean",
                      "telemetry_age_min": float(sig["age_min"][i]), "gap_share": float(gap),
                      "frozen_input": bool(sig["frozen"][i]), "m8_flag": s in self.gis_sites}
            for witness_key, preds in (("pv", pv_p), ("load", ld_p), ("sitehealth", hl_p)):
                p = preds.get(s)
                if p is None:
                    continue
                witness = str(getattr(p, "model", witness_key))
                if witness in ("mock", "pv", "load"):
                    witness = {"pv": "pvnet", "load": "loadnet"}.get(witness_key, witness_key)
                r = self._base(d_iso, t0, month, season, s, "site", witness, p)
                r.update(common)
                why = []
                if sig["stale"][i]:
                    why.append("telemetry stale")
                if witness_key == "load":
                    jump = bool(sig["jump_ld"][i])
                    if jump:
                        why.append("load energy jump vs own median")
                else:
                    jump = bool(sig["jump_pv"][i])
                    if sig["range"][i]:
                        why.append("PV outside 0..1.2 x kWp")
                    if sig["frozen"][i]:
                        why.append("frozen PV reading")
                    if jump:
                        why.append("PV energy jump vs own median")
                r["unit_jump"] = jump
                r["input_bad"] = bool(why)
                r["bad_why"] = ", ".join(why)
                if witness_key == "pv":
                    r["pv_residual_d1"] = self.residual_d1.get(s, float("nan"))
                    if pvlib is not None:
                        try:
                            ref = float(np.sum(pvlib(s, d)))
                            r["pv_vs_pvlib"] = float(np.sum(p.q50)) / ref - 1.0 if ref > 1e-6 else float("nan")
                        except Exception:
                            pass
                if witness_key == "sitehealth":
                    pf = float(p.p_fault) if hasattr(p, "p_fault") else float("nan")
                    r.update({"p_fault": pf, "top_fault": getattr(p, "top_fault", None), "alarm": pf >= ALARM_P})
                r["ood_high"] = r["ood_score"] > OOD_MAX
                out.append(r)
        return pd.DataFrame(out)

    def wind_rows(self, d: int, t0: str, tday: Mapping[str, Any], wd_p: Mapping[str, Any]) -> pd.DataFrame:
        """M6 rows at the end of the day: the day's call = the safety class if any tick called one, else the most
        frequent non-normal class, else normal; input gate on the day's own SCADA."""
        d_iso, month, season = self._context(t0)
        w = self.world
        tids = sorted(wd_p)
        calls: dict[str, str] = {}
        for t in tids:
            c = self.wind_counts.get(t, Counter())
            safe = [(n, k) for k, n in c.items() if k in SAFETY_CLASSES]
            other = [(n, k) for k, n in c.items() if k not in ("normal", None)]
            calls[t] = max(safe)[1] if safe else (max(other)[1] if other else "normal")
        self.wind_calls = {t: (c, self.wind_score.get((t, c), 0.5 if c != "normal" else 0.05))
                           for t, c in calls.items()}
        by_park: dict[str, Counter] = {}
        for t in tids:
            by_park.setdefault(str(w.turbines[t].park_id), Counter())[calls[t]] += 1
        out = []
        for t in tids:
            p = wd_p[t]
            tb = w.turbines[t]
            act = tday[t].act_pwt_kw
            ok = np.array([a is not None for a in act])
            x = np.array([a if a is not None else np.nan for a in act], float)
            tail = 0
            for a in act[::-1]:
                if a is not None:
                    break
                tail += 1
            rated = _f(getattr(tb, "rated_kw", None), np.inf)
            why = []
            if tail * 900 > STALE_S:
                why.append("SCADA stale")
            if np.any(ok & (np.nan_to_num(x) > TURBINE_MAX_RATED * rated)):
                why.append("power over 1.1 x rated")
            if _runs_frozen(np.nan_to_num(x)[None, :], ok[None, :])[0]:
                why.append("frozen power reading")
            witness = str(getattr(p, "model", "windnbm"))
            if witness == "mock" or witness == "wind":
                witness = "windnbm"
            r = self._base(d_iso, t0, month, season, t, "turbine", witness, p)
            gap = 1.0 - float(ok.mean()) if ok.size else float("nan")
            park = by_park[str(tb.park_id)]
            cls = calls[t]
            row = getattr(tb, "row", {}) or {}
            r.update({"battery": False, "seen": str(row.get("split", "train")) != "unseen",
                      "quality": "gappy" if gap > GAPPY else "clean", "telemetry_age_min": tail * 15.0,
                      "gap_share": gap, "frozen_input": "frozen power reading" in why, "unit_jump": False,
                      "input_bad": bool(why), "bad_why": ", ".join(why), "anomaly_class": cls,
                      "anomaly_score": self.wind_score.get((t, cls), float("nan")), "alarm": cls != "normal",
                      "safety_call": cls in SAFETY_CLASSES,
                      "park_consensus": park[cls] / max(1, sum(park.values()))})
            r["ood_high"] = r["ood_score"] > OOD_MAX
            out.append(r)
        return pd.DataFrame(out)

    def gis_rows(self, t0: str) -> pd.DataFrame:
        """M8 flags (ledger-only, §7.4): one row per flagged site, judged on day 0 when the twin runs."""
        d_iso, month, season = self._context(t0)
        out = []
        for fl in self.gis_flags:
            sid = fl.get("site_id")
            if sid is None or str(sid) not in self.world.sites:
                continue
            site = self.world.sites[str(sid)]
            tau = _f(fl.get("tau"))
            r = {"run_id": self.run_id, "day": d_iso, "t0": t0, "asset_id": str(sid), "asset_kind": "site",
                 "witness": "gridtwin", "model_version": str(getattr(self.world, "gridtwin_version", "gridtwin:mock-1")),
                 "family": "detector", "substitute": WITNESSES["gridtwin"][1], "month": month, "season": season,
                 "battery": site.battery is not None, "seen": str(site.row.get("split", "train")) != "unseen",
                 "quality": "clean", "ood_score": 0.0, "ood_high": False, "alarm": True, "m8_flag": True,
                 "anomaly_score": 1.0 - tau if tau == tau else float("nan"), "input_bad": False,
                 "safety_call": False, "is_fallback": int(str(getattr(self.world, "gridtwin_version", "mock")).find("mock") >= 0)}
            out.append(r)
        return pd.DataFrame(out)

    # -- ledger + model + rules ---------------------------------------------------------------------
    def _score_name(self, r: Mapping[str, Any]) -> str:
        if r["witness"] == "gridtwin":
            return "flag"
        if r["witness"] == "windfailure":
            return "hit"
        if r["family"] == "forecast":
            return "calibration"
        return "alarm_precision" if bool(r.get("alarm")) else "miss_score"

    def _fill_ledger(self, d: int, rows: pd.DataFrame) -> pd.DataFrame:
        if rows.empty:
            return rows
        kinds = rows["asset_kind"].to_numpy(object)
        buckets = [_bucket_key(kind=k, seen=bool(s), battery=bool(b), season=se, quality=q, ood_high=bool(o))
                   for k, s, b, se, q, o in zip(kinds, rows["seen"], rows["battery"], rows["season"],
                                                rows["quality"], rows["ood_high"])]
        rows = rows.assign(bucket=buckets)
        cache: dict[tuple, dict[str, Any]] = {}

        def look(model, version, score, bucket):
            key = (model, version, score, bucket)
            if key not in cache:
                cache[key] = self.ledger.lookup(model, version, score, bucket, rows["day"].iat[0])
            return cache[key]

        cols: dict[str, list] = {c: [] for c in ("prior", "trust", "n_eff", "lo80", "hi80", "trust_trend_3d",
                                                 "band_scale", "health_alert", "sub_trust", "sub_n_eff",
                                                 "score_name", "ledger_reason")}
        for r in rows.to_dict("records"):
            sc = self._score_name(r)
            L = look(r["witness"], r["model_version"], sc, r["bucket"])
            S = look(r["substitute"], None, sc, r["bucket"]) if r.get("substitute") else {}
            tr = _f(L.get("trust"))
            hk = (r["witness"], r["model_version"], sc, r["bucket"])
            trend = _f(L.get("trend_3d"))
            if not math.isfinite(trend):
                h = [v for dd, v in self.trust_hist.get(hk, []) if dd <= d - 3]
                trend = tr - h[-1] if h and math.isfinite(tr) and math.isfinite(h[-1]) else float("nan")
            cols["prior"].append(_f(L.get("prior")))
            cols["trust"].append(tr)
            cols["n_eff"].append(_f(L.get("n_eff"), 0.0))
            cols["lo80"].append(_f(L.get("lo80")))
            cols["hi80"].append(_f(L.get("hi80")))
            cols["trust_trend_3d"].append(trend)
            cols["band_scale"].append(_f(L.get("band_scale"), 1.0))
            cols["health_alert"].append(bool(L.get("health_alert", False)))
            cols["sub_trust"].append(_f(S.get("trust")))
            cols["sub_n_eff"].append(_f(S.get("n_eff"), 0.0))
            cols["score_name"].append(sc)
            cols["ledger_reason"].append(
                (f"trust {tr:.2f} ({'little evidence yet' if _f(L.get('n_eff'), 0) < 5 else 'n_eff %.0f' % _f(L.get('n_eff'))})"
                 + (", health alert" if L.get("health_alert") else "")) if math.isfinite(tr) and _f(L.get("n_eff"), 0) > 0
                else None)
        for key, v in cache.items():
            self.trust_hist.setdefault(key, [])
            if not self.trust_hist[key] or self.trust_hist[key][-1][0] != d:
                self.trust_hist[key].append((d, _f(v.get("trust"))))
        return rows.assign(**cols)

    def _predict(self, rows: pd.DataFrame) -> pd.DataFrame:
        rows = rows.assign(p_valid=np.nan, p_valid_sd=np.nan, reasons=[None] * len(rows))
        if self.judge is None or rows.empty:
            return rows
        m = ~rows["witness"].isin(LEDGER_ONLY_WITNESSES)
        if not m.any():
            return rows
        try:
            x = rows.loc[m, [c for c in rows.columns if c != "reasons"]].copy()
            if "model" not in x:  # the Judge model keys families on "model"; the slot calls it "witness"
                x["model"] = x["witness"]
            out = self.judge.predict(x)
            out = pd.DataFrame(out)
            out.index = rows.index[m.to_numpy()]
            rows.loc[m, "p_valid"] = pd.to_numeric(out.get("p_valid"), errors="coerce")
            if "p_valid_sd" in out:
                rows.loc[m, "p_valid_sd"] = pd.to_numeric(out["p_valid_sd"], errors="coerce")
            if "reasons" in out:
                rs = rows["reasons"].to_list()
                pos = {ix: i for i, ix in enumerate(rows.index)}
                for ix, v in out["reasons"].items():
                    rs[pos[ix]] = list(v) if isinstance(v, (list, tuple, np.ndarray)) else v
                rows["reasons"] = rs
        except Exception as e:
            n = self.errors.get("judge.predict", 0)
            self.errors["judge.predict"] = n + 1
            if n == 0:
                log(f"WARNING Judge.predict raised {type(e).__name__}: {e}; ledger-only for this call")
        return rows

    def decide(self, d: int, rows: pd.DataFrame) -> pd.DataFrame:
        """Ledger columns -> P(valid) -> §7.5 rules. Returns rows + the §7.7 output columns."""
        if rows.empty:
            return rows
        rows = self._fill_ledger(d, rows)
        rows = self._predict(rows)
        out = apply_rules(rows, self.thresholds)
        merged = rows.drop(columns=[c for c in out.columns if c in rows.columns and c != "reasons"])
        merged = merged.rename(columns={"reasons": "model_reasons"})
        return pd.concat([merged, out], axis=1)

    def _write(self, d: int, judged: pd.DataFrame) -> None:
        if judged is None or judged.empty:
            return
        self.counts.update(str(v) for v in judged["verdict"])
        self.counts.update(f"rule:{r}" for r in judged["rule"])
        self.day_rows[d] = pd.concat([self.day_rows[d], judged]) if d in self.day_rows else judged
        if "judge_verdicts" in self.tables:
            self.sink.add_many("judge_verdicts", verdict_rows(judged, self.run_id))

    def _trust_rows(self, d: int, judged: pd.DataFrame) -> None:
        if judged is None or judged.empty or "trust" not in self.tables:
            return
        day = judged["day"].iat[0]
        seen = set()
        rows = []
        for r in judged[["witness", "model_version", "bucket", "family", "substitute"]].drop_duplicates() \
                .to_dict("records"):
            fam = r["family"]
            scores = ("flag",) if r["witness"] == "gridtwin" else SCORES.get(fam, ())
            for model, ver in ((r["witness"], r["model_version"]), (r["substitute"], None)):
                if not model:
                    continue
                for sc in scores:
                    key = (model, ver, r["bucket"], sc)
                    if key in seen:
                        continue
                    seen.add(key)
                    L = self.ledger.lookup(model, ver, sc, r["bucket"], day)
                    rows.append({"run_id": self.run_id, "day": day, "model": model, "model_version": ver or "*",
                                 "bucket": r["bucket"], "score_name": sc, "prior": _rf(L.get("prior")),
                                 "n_eff": _rf(L.get("n_eff"), 0.0), "trust": _rf(L.get("trust")),
                                 "lo80": _rf(L.get("lo80")), "hi80": _rf(L.get("hi80")),
                                 "band_scale": _rf(L.get("band_scale"), 1.0),
                                 "health_alert": int(bool(L.get("health_alert", False)))})
        self.sink.add_many("trust", rows)
        self.n_trust += len(rows)

    def _at_t0(self, d, t0, pv_p, ld_p, hl_p, ages_t0_s):
        rows = self.judge_rows(d, t0, pv_p, ld_p, hl_p, ages_t0_s)
        if d == 0 and self.gis_flags:
            g = self.gis_rows(t0)
            if not g.empty:
                rows = pd.concat([rows, g], ignore_index=True)
        judged = self.decide(d, rows)
        self._write(d, judged)
        self._trust_rows(d, judged)
        self.wind_counts = {}
        self.wind_score = {}
        return judged

    def _observe_wind(self, k, wind_preds):
        for t, p in wind_preds.items():
            cls = getattr(p, "anomaly_class", None) or "normal"
            self.wind_counts.setdefault(t, Counter())[cls] += 1
            sc = _f(getattr(p, "anomaly_score", None))
            if math.isfinite(sc):
                key = (t, cls)
                self.wind_score[key] = max(self.wind_score.get(key, sc), sc)

    def _at_end(self, d, t0, oday, tday, wd_p, pv_p, ld_p, hl_p):
        judged = None
        if wd_p:
            judged = self.decide(d, self.wind_rows(d, t0, tday, wd_p))
            self._write(d, judged)
            self._trust_rows(d, judged)
        # the day is over: its telemetry becomes input history for tomorrow's gate and residual (observed only)
        for s, od in oday.items():
            ok = np.asarray(od.telemetry_age_s, float) == 0
            pv = np.where(ok, np.asarray(od.pv_kw, float), np.nan)
            ld = np.where(ok, np.asarray(od.load_kw, float), np.nan)
            self.hist_pv.setdefault(s, []).append(pv)
            self.hist_load.setdefault(s, []).append(ld)
            del self.hist_pv[s][:-31], self.hist_load[s][:-31]
            p = pv_p.get(s)
            if p is not None and ok.sum() >= 48:
                e_fc = float(np.asarray(p.q50, float)[ok].sum())
                self.residual_d1[s] = float(np.nansum(pv[ok])) / e_fc - 1.0 if e_fc > 1e-3 else float("nan")
        # grading hook (J1) -> outcomes table + ledger update; observed_at ordering is the grader's job (§5.5)
        if self.grader is not None:
            rows = self.day_rows.get(d)
            ctx = {"run_id": self.run_id, "d": d, "day": rows["day"].iat[0] if rows is not None and len(rows) else None,
                   "t0": t0, "rows": rows, "predictions": {"pvnet": dict(pv_p), "loadnet": dict(ld_p),
                                                           "sitehealth": dict(hl_p), "windnbm": dict(wd_p)},
                   "observed": dict(oday), "turbines": dict(tday), "world": self.world,
                   "wind_calls": dict(self.wind_calls), "gis_flags": self.gis_flags,
                   "audit_weights": self._audit_weights(rows)}
            try:
                outc = self.grader(ctx)
            except Exception as e:
                outc = None
                n = self.errors.get("grader", 0)
                self.errors["grader"] = n + 1
                if n == 0:
                    log(f"WARNING grade_loop_day raised {type(e).__name__}: {e} (further errors counted)")
            if outc is not None and len(outc):
                outc = led_rows = pd.DataFrame(outc)  # the ledger gets the grader's rows as they are
                if "outcomes" in self.tables:
                    if "bucket" not in outc.columns and "kind" in outc.columns:  # J1 rows carry the conditions
                        outc = outc.assign(bucket=[
                            _bucket_key(kind=r.get("kind", "site"), seen=bool(r.get("seen", True)),
                                        battery=bool(r.get("battery", False)),
                                        season=r.get("season") or season_of(int(str(r["day"])[5:7])),
                                        quality=r.get("quality") or "clean", ood_high=bool(r.get("ood_high", False)))
                            for r in outc.to_dict("records")])
                    keep = [c for c in OUTCOME_COLUMNS if c in outc.columns]
                    recs = outc[keep].assign(run_id=self.run_id).to_dict("records")
                    self.sink.add_many("outcomes", [{k: _clean(v) for k, v in r.items()} for r in recs])
                self.n_outcomes += len(outc)
                self.ledger.update(led_rows)
        self.day_rows.pop(d - 1, None)
        return judged

    def _audit_weights(self, rows: Optional[pd.DataFrame]) -> dict[str, float]:
        """§5.6: audit draws favour assets whose least-trusted verdict sits near a threshold (weight 0.1 + bump)."""
        if rows is None or rows.empty or "p_valid" not in rows:
            return {}
        p = rows.groupby("asset_id")["p_valid"].min().dropna()
        th = (self.thresholds.widen("forecast"), self.thresholds.use("forecast"))
        return {str(a): 0.1 + max(math.exp(-(((v - t) / 0.15) ** 2)) for t in th) for a, v in p.items()}

    # -- report -------------------------------------------------------------------------------------
    def summary(self) -> dict[str, Any]:
        days = sorted(self.seconds)
        return {"mode": self.mode, "source": self.source, "judge_version": self.thresholds.judge_version,
                "thresholds": {"tau_use": self.thresholds.tau_use, "tau_widen": self.thresholds.tau_widen,
                               "widen_factor": self.thresholds.widen_factor, "source": self.thresholds.source},
                "ledger": getattr(self.ledger, "version", None), "grader": bool(self.grader),
                "verdicts": {k: v for k, v in sorted(self.counts.items())}, "outcomes": self.n_outcomes,
                "trust_rows": self.n_trust, "t_judge_s": round(sum(self.seconds.values()), 4),
                "t_judge_day_s": {d: round(self.seconds[d], 4) for d in days}, "errors": dict(self.errors),
                "tables_missing": sorted({"judge_verdicts", "trust", "outcomes"} - self.tables)}


OUTCOME_COLUMNS = ("run_id", "model", "model_version", "asset_id", "day", "score_name", "score", "weight", "source",
                   "observed_at", "gated", "bucket")


def _rf(v: Any, default: Optional[float] = None) -> Optional[float]:
    f = _f(v)
    if not math.isfinite(f):
        return default
    return round(f, 5)


def _clean(v: Any) -> Any:
    if isinstance(v, (np.generic,)):
        v = v.item()
    if isinstance(v, float) and not math.isfinite(v):
        return None
    if isinstance(v, (pd.Timestamp, datetime)):
        return v.isoformat()
    return v
