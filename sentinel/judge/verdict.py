"""Judge verdict rules (team/judge-design.md §7.5) and the per-prediction output contract (§7.7).

    out = apply_rules(rows, thresholds=Thresholds.from_judge(judge))   # rows: one per (witness, asset, day)

`rows` needs these columns (the loop slot, `sentinel.judge.slot.judge_rows`, builds them):
    family        forecast | detector
    input_bad     bool: the input window failed the physics gate (§5.4) or the telemetry is stale
    safety_call   bool: an icing / overheating call (M6)
    p_valid       the Judge model's P(valid), NaN where it gave none (ledger-only mode fills it with `trust`)
    trust, n_eff  ledger trust and effective evidence of this witness in this bucket (§6.3)
    band_scale    the bucket's band scale (§6.4), 1.0 when unknown
    substitute    the substitute model id (§3), sub_trust its ledger trust in the same bucket (its P(valid))
    reasons       optional list of plain-word strings from the model (feature contributions)

Rules, in order; the first match wins:
    1 bad input              -> dont_use ("input unreliable")
    2 safety pass-through    -> use (an icing / overheating call is never suppressed; "low trust" when p < tau_use)
    3 no record (n_eff = 0)  -> use, at the prior (user decision: no record -> trust the model)
    4 p_valid >= tau_use     -> use
    5 forecast, tau_widen <= p_valid < tau_use -> widen (band_scale x widen_factor)
    6 otherwise              -> dont_use, naming the substitute if its P(valid) here is higher, else
                                "no reliable source" (actions must treat it as unknown)

Output (§7.7), one row per prediction: verdict, p_valid, p_valid_sd, trust, n_eff, band_scale, substitute,
reasons (<= 3 strings), rule, judge_version.

Default thresholds (ledger-only mode, no calibrated Judge yet; my call): tau_use 0.70, tau_widen 0.50,
widen_factor 1.25. A trained Judge brings its own from models/judge.calibration.json (§7.5).
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any, Mapping, Optional

import numpy as np
import pandas as pd

USE, WIDEN, DONT_USE = "use", "widen", "dont_use"
VERDICTS = (USE, WIDEN, DONT_USE)
RULE_BAD_INPUT = "1_bad_input"
RULE_SAFETY = "2_safety"
RULE_NO_RECORD = "3_no_record"
RULE_USE = "4_use"
RULE_WIDEN = "5_widen"
RULE_DONT_USE = "6_dont_use"
SAFETY_CLASSES = frozenset({"icing", "overheating"})
LEDGER_ONLY = "ledger-only"
MAX_REASONS = 3
OUTPUT_COLUMNS = ("verdict", "p_valid", "p_valid_sd", "trust", "n_eff", "band_scale", "substitute", "reasons",
                  "rule", "judge_version")


def _per_family(v: Any, family: str, default: float) -> float:
    if isinstance(v, Mapping):
        v = v.get(family, v.get("*", default))
    try:
        f = float(v)
    except (TypeError, ValueError):
        return default
    return f if math.isfinite(f) else default


@dataclass
class Thresholds:
    """tau_use / tau_widen / widen_factor: a float for every family or a {family: float} mapping."""

    tau_use: Any = 0.70
    tau_widen: Any = 0.50
    widen_factor: Any = 1.25
    judge_version: str = LEDGER_ONLY
    source: str = "default (ledger-only)"
    extra: dict[str, Any] = field(default_factory=dict)

    @classmethod
    def from_judge(cls, judge: Any) -> "Thresholds":
        if judge is None:
            return cls()
        d = cls()
        return cls(tau_use=getattr(judge, "tau_use", d.tau_use), tau_widen=getattr(judge, "tau_widen", d.tau_widen),
                   widen_factor=getattr(judge, "widen_factor", d.widen_factor),
                   judge_version=str(getattr(judge, "judge_version", "judge")), source="judge calibration JSON")

    def use(self, family: str) -> float:
        return _per_family(self.tau_use, family, 0.70)

    def widen(self, family: str) -> float:
        return _per_family(self.tau_widen, family, 0.50)

    def factor(self, family: str) -> float:
        return _per_family(self.widen_factor, family, 1.25)


def _col(rows: pd.DataFrame, name: str, default: Any) -> pd.Series:
    if name in rows.columns:
        return rows[name]
    return pd.Series([default] * len(rows), index=rows.index)


def _num(rows: pd.DataFrame, name: str, default: float = np.nan) -> np.ndarray:
    return pd.to_numeric(_col(rows, name, default), errors="coerce").to_numpy(float)


def _flag(rows: pd.DataFrame, name: str) -> np.ndarray:
    s = _col(rows, name, False)
    return s.fillna(False).astype(bool).to_numpy()


def _pct(x: float) -> str:
    return f"{x:.2f}"


def apply_rules(rows: pd.DataFrame, thresholds: Optional[Thresholds] = None) -> pd.DataFrame:
    """§7.5 over every row at once. Returns a frame with OUTPUT_COLUMNS, same index as `rows`."""
    th = thresholds or Thresholds()
    n = len(rows)
    if n == 0:
        return pd.DataFrame(columns=list(OUTPUT_COLUMNS))
    family = _col(rows, "family", "forecast").fillna("forecast").astype(str).to_numpy()
    bad = _flag(rows, "input_bad")
    safety = _flag(rows, "safety_call")
    trust = _num(rows, "trust")
    n_eff = np.nan_to_num(_num(rows, "n_eff", 0.0), nan=0.0)
    p = _num(rows, "p_valid")
    model_p = np.isfinite(p)
    p = np.where(model_p, p, trust)  # ledger-only: P(valid) = trust (§7.4, §11)
    sd = _num(rows, "p_valid_sd")
    scale = np.nan_to_num(_num(rows, "band_scale", 1.0), nan=1.0)
    sub_trust = _num(rows, "sub_trust")
    subs = _col(rows, "substitute", None).to_numpy(object)
    jv = np.where(model_p, th.judge_version, LEDGER_ONLY) if th.judge_version != LEDGER_ONLY else \
        np.full(n, LEDGER_ONLY, dtype=object)
    tau_u = np.array([th.use(f) for f in family])
    tau_w = np.array([th.widen(f) for f in family])
    fac = np.array([th.factor(f) for f in family])
    no_record = n_eff <= 0
    pv = np.where(np.isfinite(p), p, -np.inf)
    is_fc = family == "forecast"
    conds = [bad, safety, no_record, pv >= tau_u, is_fc & (pv >= tau_w) & (pv < tau_u)]
    rule = np.select(conds, [RULE_BAD_INPUT, RULE_SAFETY, RULE_NO_RECORD, RULE_USE, RULE_WIDEN], RULE_DONT_USE)
    verdict = np.select(conds, [DONT_USE, USE, USE, USE, WIDEN], DONT_USE)
    out_scale = np.where(rule == RULE_WIDEN, scale * fac, scale)
    sub_better = np.isfinite(sub_trust) & (sub_trust > np.where(np.isfinite(p), p, -np.inf))
    substitute = np.where((rule == RULE_DONT_USE) & sub_better, subs, None)

    model_reasons = _col(rows, "reasons", None).to_numpy(object)
    ledger_reason = _col(rows, "ledger_reason", None).to_numpy(object)
    reasons = []
    for i in range(n):
        r: list[str] = []
        if rule[i] == RULE_BAD_INPUT:
            r.append("input unreliable" + (f" ({rows['bad_why'].iat[i]})" if "bad_why" in rows.columns
                                           and isinstance(rows["bad_why"].iat[i], str) and rows["bad_why"].iat[i]
                                           else ""))
        elif rule[i] == RULE_SAFETY:
            r.append("safety call, never suppressed" + (", low trust" if not pv[i] >= tau_u[i] else ""))
        elif rule[i] == RULE_NO_RECORD:
            r.append("no record yet: used at validated strength")
        elif rule[i] == RULE_WIDEN:
            r.append(f"band widened x{out_scale[i]:.2f}")
        elif rule[i] == RULE_DONT_USE:
            r.append(f"use {substitute[i]} instead" if substitute[i] is not None else "no reliable source")
        mr = model_reasons[i]
        if isinstance(mr, (list, tuple, np.ndarray)):
            r.extend(str(x) for x in mr if x)
        elif isinstance(mr, str) and mr:
            r.append(mr)
        lr = ledger_reason[i]
        if isinstance(lr, str) and lr:
            r.append(lr)
        elif np.isfinite(trust[i]):
            r.append(f"trust {_pct(trust[i])} over {n_eff[i]:.1f} graded days")
        reasons.append(list(dict.fromkeys(r))[:MAX_REASONS])
    idx = rows.index

    def obj(v):  # object dtype keeps None as None (pandas' string inference would turn it into NaN)
        return pd.Series(list(v), index=idx, dtype=object)

    return pd.DataFrame({
        "verdict": obj(verdict), "p_valid": np.where(np.isfinite(p), p, np.nan),
        "p_valid_sd": np.where(model_p, sd, np.nan), "trust": trust, "n_eff": n_eff, "band_scale": out_scale,
        "substitute": obj(substitute), "reasons": obj(reasons), "rule": obj(rule), "judge_version": obj(jv),
    }, index=idx)


def verdict_rows(judged: pd.DataFrame, run_id: str, out: Optional[pd.DataFrame] = None) -> list[dict[str, Any]]:
    """`judge_verdicts` table rows (packages/db/src/schema/sentinel.ts) from judged rows (rows + apply_rules
    output in one frame, or the two frames as `judged` and `out`)."""
    if out is not None:
        judged = pd.concat([judged.drop(columns=[c for c in out.columns if c in judged.columns]), out], axis=1)
    res = []
    for r in judged.to_dict("records"):
        res.append({
            "run_id": run_id, "day": r["day"], "t0": r.get("t0"), "asset_id": r["asset_id"], "model": r["witness"],
            "model_version": r.get("model_version"), "bucket": r.get("bucket"), "verdict": r["verdict"],
            "p_valid": _f(r["p_valid"]), "p_valid_sd": _f(r["p_valid_sd"]), "trust": _f(r["trust"]),
            "n_eff": _f(r["n_eff"]), "band_scale": _f(r["band_scale"]), "substitute": r["substitute"],
            "reasons_json": list(r["reasons"]), "rule": r["rule"], "judge_version": r["judge_version"]})
    return res


def _f(v: Any) -> Optional[float]:
    try:
        f = float(v)
    except (TypeError, ValueError):
        return None
    return round(f, 5) if math.isfinite(f) else None
