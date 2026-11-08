"""How every feature reads validated information: raw witness prediction in, prediction + Judge verdict out, with
the band scale applied and, on "don't use", the named substitute swapped in.

    v = validated("1272", "pvnet", "2025-06-02", raw=pred, substitutes={"lgbm": lgbm_pred})
    if not v.known:            # dont_use and no substitute data: treat as unknown
        ...
    use(v.q10, v.q50, v.q90)   # already widened / substituted

    b = validated_many("pvnet", asset_ids, days, raw=RawBatch(q=q_raw))   # q_raw: (N, 96, 3) = q10, q50, q90

Verdict sources: `NoJudge()` (default: everything passes as "use", rule "no_judge"), `MemoryVerdicts(rows)`,
`SqliteVerdicts(db, run_id)` (the `judge_verdicts` table). A missing (asset, model, day) also passes as "use".
Band scale: q10' = q50 - s (q50 - q10), q90' = q50 + s (q90 - q50); `band_scale` already includes the widen factor.
Non-negative quantities stay non-negative. Pure numpy + stdlib: features run without Judge weights.
"""

from __future__ import annotations

import json
import math
import os
import sqlite3
from dataclasses import dataclass, field
from datetime import date, datetime
from pathlib import Path
from typing import Any, Iterable, Mapping, Optional, Protocol, Sequence, Union
from zoneinfo import ZoneInfo

import numpy as np

H = 96  # 15-min steps per day
VERDICTS = ("use", "widen", "dont_use")
NO_JUDGE_RULE = "no_judge"
NO_JUDGE_REASON = "no Judge verdict: model used at its validated strength"
LOCAL_TZ = "Europe/Bucharest"


# ============================================================================================ verdicts
@dataclass(frozen=True)
class Verdict:
    """One Judge verdict (= a `judge_verdicts` row without run_id/asset/model/day)."""

    verdict: str = "use"
    p_valid: Optional[float] = None
    trust: Optional[float] = None
    n_eff: float = 0.0
    band_scale: float = 1.0
    substitute: Optional[str] = None
    reasons: tuple[str, ...] = ()
    rule: str = NO_JUDGE_RULE
    judge_version: Optional[str] = None

    def __post_init__(self) -> None:
        if self.verdict not in VERDICTS:
            raise ValueError(f"verdict must be one of {VERDICTS}, got {self.verdict!r}")


PASSTHROUGH = Verdict(reasons=(NO_JUDGE_REASON,))


def _f(x: Any) -> Optional[float]:
    if x is None:
        return None
    x = float(x)
    return None if math.isnan(x) else x


def as_verdict(x: Any) -> Verdict:
    """Coerce a slot verdict object, a `judge_verdicts` row dict (reasons_json) or a Verdict into a Verdict."""
    if isinstance(x, Verdict):
        return x
    get = x.get if isinstance(x, Mapping) else (lambda k, d=None: getattr(x, k, d))
    reasons = get("reasons", None)
    if reasons is None:
        rj = get("reasons_json", None)
        reasons = json.loads(rj) if isinstance(rj, str) and rj else (rj or ())
    if isinstance(reasons, str):
        reasons = (reasons,)
    bs = _f(get("band_scale", 1.0))
    sub = get("substitute", None)
    return Verdict(verdict=str(get("verdict", "use")), p_valid=_f(get("p_valid", None)), trust=_f(get("trust", None)),
                   n_eff=float(get("n_eff", 0.0) or 0.0),
                   band_scale=bs if bs is not None and bs > 0 and math.isfinite(bs) else 1.0,
                   substitute=str(sub) if sub else None, reasons=tuple(str(r) for r in reasons)[:3],
                   rule=str(get("rule", None) or NO_JUDGE_RULE), judge_version=get("judge_version", None))


def day_key(day: Union[str, date, datetime]) -> str:
    """Local (Europe/Bucharest) calendar day as 'YYYY-MM-DD', the `judge_verdicts.day` key."""
    if isinstance(day, datetime):
        if day.tzinfo is not None:
            day = day.astimezone(ZoneInfo(LOCAL_TZ))
        return day.date().isoformat()
    if isinstance(day, date):
        return day.isoformat()
    return str(day)[:10]


class VerdictSource(Protocol):
    def get(self, asset_id: str, model: str, day: str) -> Optional[Verdict]: ...


class NoJudge:
    """No Judge in the loop: every lookup misses, so everything passes through as 'use'."""

    def get(self, asset_id: str, model: str, day: str) -> Optional[Verdict]:
        return None


class MemoryVerdicts:
    """Verdicts in memory. `rows`: iterable of mappings / objects with asset_id, model (or witness), day and the
    verdict fields, or a mapping {(asset_id, model, day): verdict-like}."""

    def __init__(self, rows: Union[Iterable[Any], Mapping[tuple, Any]] = ()):
        self._d: dict[tuple[str, str, str], Verdict] = {}
        if isinstance(rows, Mapping):
            for (a, m, d), v in rows.items():
                self.put(a, m, d, v)
        else:
            for r in rows:  # slot rows name the model "witness"
                get = r.get if isinstance(r, Mapping) else (lambda k, r=r: getattr(r, k, None))
                self.put(get("asset_id"), get("model") or get("witness"), get("day"), r)

    def put(self, asset_id: Any, model: str, day: Any, v: Any) -> None:
        self._d[(str(asset_id), str(model), day_key(day))] = as_verdict(v)

    def get(self, asset_id: str, model: str, day: str) -> Optional[Verdict]:
        return self._d.get((str(asset_id), str(model), day_key(day)))


class SqliteVerdicts(MemoryVerdicts):
    """The `judge_verdicts` table of a run, loaded once. Missing table -> empty (no-Judge mode)."""

    def __init__(self, db: Union[str, Path, sqlite3.Connection], run_id: str, models: Optional[Sequence[str]] = None):
        super().__init__()
        con = db if isinstance(db, sqlite3.Connection) else sqlite3.connect(f"file:{db}?mode=ro", uri=True)
        try:
            con.row_factory = sqlite3.Row
            if not con.execute("SELECT 1 FROM sqlite_master WHERE type='table' AND name='judge_verdicts'").fetchone():
                return
            q, args = "SELECT * FROM judge_verdicts WHERE run_id = ?", [run_id]
            if models:
                q += f" AND model IN ({','.join('?' * len(models))})"
                args += list(models)
            for r in con.execute(q, args):
                self.put(r["asset_id"], r["model"], r["day"], dict(r))
        finally:
            if not isinstance(db, sqlite3.Connection):
                con.close()


_DEFAULT: Optional[VerdictSource] = None


def set_default_judge(src: Optional[VerdictSource]) -> None:
    """Verdict source used when a call passes judge=None (None restores no-Judge mode)."""
    global _DEFAULT
    _DEFAULT = src


def default_judge() -> VerdictSource:
    """`set_default_judge` value, else WATTSENTRY_JUDGE_DB + WATTSENTRY_JUDGE_RUN (env), else NoJudge."""
    global _DEFAULT
    if _DEFAULT is None:
        db, run = os.environ.get("WATTSENTRY_JUDGE_DB"), os.environ.get("WATTSENTRY_JUDGE_RUN")
        _DEFAULT = SqliteVerdicts(db, run) if db and run else NoJudge()
    return _DEFAULT


# ============================================================================================ raw predictions
@dataclass(frozen=True)
class RawPrediction:
    """A witness output in adapter form. Forecasts: q (H, 3) = q10, q50, q90. Detectors: cls / score (and q =
    the expected power they measured against, when they have one). Build from a `contracts.Prediction` with
    `RawPrediction.of(pred)`."""

    model: str
    q: Optional[np.ndarray] = None
    cls: Optional[str] = None
    score: Optional[float] = None
    fault_probs: Optional[Mapping[str, float]] = None
    model_version: Optional[str] = None
    extra: Mapping[str, Any] = field(default_factory=dict)

    @classmethod
    def of(cls, p: Any, model: Optional[str] = None) -> "RawPrediction":
        if isinstance(p, RawPrediction):
            return p
        get = p.get if isinstance(p, Mapping) else (lambda k, d=None: getattr(p, k, d))
        q = get("q", None)
        if q is None and get("q50", None) is not None:
            q = np.stack([np.asarray(get(k), dtype=np.float64) for k in ("q10", "q50", "q90")], axis=-1)
        fp = get("fault_probs", None)
        c = get("cls", None) or get("anomaly_class", None)
        s = get("score", None)
        if s is None:
            s = get("anomaly_score", None)
        if c is None and fp:  # M4: top fault class, score = P(any fault)
            c = max(fp.items(), key=lambda kv: kv[1])[0]
            s = 1.0 - float(fp.get("healthy", 0.0))
        return cls(model=str(model or get("model", "")), q=None if q is None else np.asarray(q, dtype=np.float64),
                   cls=c, score=_f(s), fault_probs=dict(fp) if fp else None,
                   model_version=get("model_version", None), extra=dict(get("extra", None) or {}))


@dataclass(frozen=True)
class RawBatch:
    """N raw outputs of one model, row-aligned with validated_many's asset_ids/days. Any field may be None.
    q: (N, H, 3) float; cls: (N,) str/object; score: (N,) float; NaN rows in q = prediction missing."""

    q: Optional[np.ndarray] = None
    cls: Optional[np.ndarray] = None
    score: Optional[np.ndarray] = None


def scale_band(q: np.ndarray, s: Union[float, np.ndarray]) -> np.ndarray:
    """Rescale P10-P90 around P50 by s (scalar or one per row). q: (..., H, 3). Non-negative bands stay >= 0."""
    q = np.asarray(q, dtype=np.float64)
    s = np.asarray(s, dtype=np.float64)
    s = s.reshape(s.shape + (1,) * (q.ndim - 1 - s.ndim)) if s.ndim else s
    lo, mid, hi = q[..., 0], q[..., 1], q[..., 2]
    lo2 = mid - s * (mid - lo)
    nonneg = np.all(np.nan_to_num(lo, nan=0.0) >= 0, axis=-1, keepdims=True)
    lo2 = np.where(nonneg, np.maximum(lo2, 0.0), lo2)
    return np.stack([np.minimum(lo2, mid), mid, np.maximum(mid + s * (hi - mid), mid)], axis=-1)


# ============================================================================================ envelope
@dataclass(frozen=True)
class Validated:
    """What a feature acts on, for one (asset, model, day).

    known=False means "no reliable source" (dont_use without substitute data): the numbers are NaN / None and
    the action must treat the quantity as unknown (e.g. human review, or the feature's no-information baseline).
    `raw` keeps the original witness output for display only; actions must not read it.
    """

    asset_id: str
    model: str  # the witness asked for
    day: str
    source_model: Optional[str]  # whose numbers q/cls/score are (model, the substitute, or None if unknown)
    known: bool
    q10: Optional[np.ndarray]  # (H,) band_scale applied; None for outputs without a band
    q50: Optional[np.ndarray]
    q90: Optional[np.ndarray]
    cls: Optional[str]  # detector class (M4 top fault, M6 anomaly class, M7/M8 flag)
    score: Optional[float]  # detector score / probability (M4 P(fault), M6 anomaly score, M7 risk, M8 tau)
    fault_probs: Optional[Mapping[str, float]]
    verdict: str
    p_valid: Optional[float]
    trust: Optional[float]
    n_eff: float
    band_scale: float  # the scale actually applied to q10/q90 (substitute's own when swapped)
    substitute: Optional[str]
    reasons: tuple[str, ...]
    rule: str
    judge_version: Optional[str]
    raw: Optional[RawPrediction] = None


def _lookup(judge: VerdictSource, asset_id: str, model: str, day: str) -> Verdict:
    return judge.get(asset_id, model, day) or PASSTHROUGH


def validated(asset_id: Any, model: str, day: Union[str, date, datetime], *, raw: Any,
              substitutes: Optional[Mapping[str, Any]] = None, judge: Optional[VerdictSource] = None) -> Validated:
    """The one call a feature makes per (asset, model, day). See the module docstring.

    raw: the witness's output (contracts.Prediction, mapping, RawPrediction) or None if it produced nothing.
    substitutes: {model id: output} for the baselines the caller can supply (only the named one is used).
    """
    judge = judge or default_judge()
    aid, d = str(asset_id), day_key(day)
    v = _lookup(judge, aid, model, d)
    r = None if raw is None else RawPrediction.of(raw, model)
    src, scale, reasons = r, v.band_scale, v.reasons
    if v.verdict == "dont_use":
        src = None
        sub = (substitutes or {}).get(v.substitute) if v.substitute else None
        if sub is not None:
            src = RawPrediction.of(sub, v.substitute)
            scale = _lookup(judge, aid, v.substitute, d).band_scale
        else:
            scale = 1.0
            if v.substitute:
                reasons = (reasons + (f"substitute {v.substitute} not supplied",))[:4]
    if src is None:
        return Validated(aid, model, d, None, False, *(None,) * 3, None, None, None, v.verdict, v.p_valid, v.trust,
                         v.n_eff, scale, v.substitute, reasons, v.rule, v.judge_version, r)
    q = None if src.q is None else scale_band(src.q, scale)
    return Validated(aid, model, d, src.model, True, *((None,) * 3 if q is None else (q[:, 0], q[:, 1], q[:, 2])),
                     src.cls, src.score, src.fault_probs, v.verdict, v.p_valid, v.trust, v.n_eff, scale,
                     v.substitute, reasons, v.rule, v.judge_version, r)


# ============================================================================================ batch
@dataclass(frozen=True)
class ValidatedBatch:
    """validated() for N (asset, day) rows of one model, as arrays. q*: (N, H) NaN where unknown/missing."""

    model: str
    asset_ids: np.ndarray  # (N,) object
    days: np.ndarray  # (N,) object 'YYYY-MM-DD'
    known: np.ndarray  # (N,) bool
    source_model: np.ndarray  # (N,) object (None where unknown)
    q10: Optional[np.ndarray]
    q50: Optional[np.ndarray]
    q90: Optional[np.ndarray]
    cls: Optional[np.ndarray]  # (N,) object
    score: Optional[np.ndarray]  # (N,) float
    verdict: np.ndarray  # (N,) '<U8'
    p_valid: np.ndarray  # (N,) float, NaN = none
    trust: np.ndarray
    n_eff: np.ndarray
    band_scale: np.ndarray
    substitute: np.ndarray  # (N,) object
    reasons: list[tuple[str, ...]]
    rule: np.ndarray  # (N,) object
    judge_version: np.ndarray  # (N,) object

    @property
    def q(self) -> Optional[np.ndarray]:
        return None if self.q50 is None else np.stack([self.q10, self.q50, self.q90], axis=-1)

    def counts(self) -> dict[str, int]:
        """Verdict counts + unknown + substituted (for run notes and the UI)."""
        out = {k: int((self.verdict == k).sum()) for k in VERDICTS}
        out["unknown"] = int((~self.known).sum())
        out["substituted"] = sum(1 for k, s in zip(self.known, self.source_model) if k and s not in (None, self.model))
        return out

def validated_many(model: str, asset_ids: Sequence[Any], days: Union[Sequence[Any], str, date], *,
                   raw: RawBatch, substitutes: Optional[Mapping[str, RawBatch]] = None,
                   judge: Optional[VerdictSource] = None) -> ValidatedBatch:
    """Vectorised validated() for thousands of (asset, day) rows of one model.

    asset_ids: (N,); days: (N,) or one day for all rows; raw: RawBatch row-aligned with asset_ids.
    substitutes: {model id: RawBatch} row-aligned too. Verdict lookup is O(N) dict access; the band maths is numpy.
    """
    judge = judge or default_judge()
    ids = np.asarray([str(a) for a in asset_ids], dtype=object)
    n = len(ids)
    dd = (np.asarray([day_key(days)] * n, dtype=object) if isinstance(days, (str, date))
          else np.asarray([day_key(d) for d in days], dtype=object))
    if len(dd) != n:
        raise ValueError(f"{len(dd)} days for {n} assets")
    vs = [_lookup(judge, ids[i], model, dd[i]) for i in range(n)]
    verdict = np.asarray([v.verdict for v in vs], dtype="<U8")
    scale = np.asarray([v.band_scale for v in vs], dtype=np.float64)
    sub_name = np.asarray([v.substitute for v in vs], dtype=object)
    reasons = [v.reasons for v in vs]
    src = np.asarray([model] * n, dtype=object)
    subs = substitutes or {}

    def has(f: str) -> bool:
        return getattr(raw, f) is not None or any(getattr(s, f) is not None for s in subs.values())

    q = np.full((n, H, 3), np.nan) if has("q") else None
    cls = np.full(n, None, dtype=object) if has("cls") else None
    score = np.full(n, np.nan) if has("score") else None

    def fill(rows: np.ndarray, b: RawBatch) -> None:
        if q is not None and b.q is not None:
            q[rows] = np.asarray(b.q, dtype=np.float64)[rows]
        if cls is not None and b.cls is not None:
            cls[rows] = np.asarray(b.cls, dtype=object)[rows]
        if score is not None and b.score is not None:
            score[rows] = np.asarray(b.score, dtype=np.float64)[rows]

    known = verdict != "dont_use"
    fill(np.flatnonzero(known), raw)
    for name in {s for s in sub_name[~known] if s}:
        rows = np.flatnonzero((~known) & (sub_name == name))
        if name in subs:
            fill(rows, subs[name])
            known[rows] = True
            src[rows] = name
            scale[rows] = [_lookup(judge, ids[i], name, dd[i]).band_scale for i in rows]
        else:
            for i in rows:
                reasons[i] = (reasons[i] + (f"substitute {name} not supplied",))[:4]
    src[~known] = None
    scale[~known] = 1.0
    if q is not None:
        q[~known] = np.nan
        q = scale_band(q, scale)
    if cls is not None:
        cls[~known] = None
    if score is not None:
        score[~known] = np.nan
    nan = (lambda x: np.nan if x is None else x)
    return ValidatedBatch(
        model=model, asset_ids=ids, days=dd, known=known, source_model=src,
        q10=None if q is None else q[..., 0], q50=None if q is None else q[..., 1],
        q90=None if q is None else q[..., 2], cls=cls, score=score, verdict=verdict,
        p_valid=np.asarray([nan(v.p_valid) for v in vs], dtype=np.float64),
        trust=np.asarray([nan(v.trust) for v in vs], dtype=np.float64),
        n_eff=np.asarray([v.n_eff for v in vs], dtype=np.float64), band_scale=scale, substitute=sub_name,
        reasons=reasons, rule=np.asarray([v.rule for v in vs], dtype=object),
        judge_version=np.asarray([v.judge_version for v in vs], dtype=object))
