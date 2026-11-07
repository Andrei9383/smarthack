"""Split-conformal calibration of q10/q50/q90, per ensemble member (plan §3.5 "P10 commitment", R8).

Method: normalised one-sided conformal shift per quantile (asymmetric CQR, Romano et al. 2019). For quantile
tau with forecast q_tau and interval scale w = max(q90 - q10, eps_abs) the score is s = (y - q_tau) / w; the
calibrated quantile is q_tau + delta_tau * w with delta_tau the finite-sample tau-quantile of s on the
calibration slice. Scaling by the interval width keeps night steps (q10 = q90 = 0) at zero and lets the band
breathe with the forecast. Calibrated 10-90 bands should cover ~80% (target 78-82%).

Offsets can be split by horizon bucket (step index // (H / n_buckets)), as model.md's calibration JSON allows.
Members: the envelope carries one q10/q50/q90; a model that keeps members puts them in
`Prediction.extra["members"] = [{"q10": [...], "q50": [...], "q90": [...]}, ...]` (see `members_of`).
"""

from __future__ import annotations

import json
import math
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Mapping, Optional, Sequence, Union

import numpy as np

from sentinel.contracts import Prediction

QUANTILES = {"q10": 0.10, "q50": 0.50, "q90": 0.90}
Z90 = 1.2815515655446004  # Phi^-1(0.9)
EPS_ABS = 1e-3  # kW; floor for the interval scale

ArrayLike = Union[np.ndarray, Sequence[float]]
Triple = tuple[np.ndarray, np.ndarray, np.ndarray]


# --- metrics -------------------------------------------------------------------------------------------------
def default_mask(y: np.ndarray, q10: np.ndarray, q90: np.ndarray, eps: float = EPS_ABS) -> np.ndarray:
    """Steps that carry information: forecast band or actual above eps (drops PV nights)."""
    return (np.asarray(q90) > eps) | (np.asarray(y) > eps)


def coverage(y: ArrayLike, lo: ArrayLike, hi: ArrayLike, mask: Optional[np.ndarray] = None) -> float:
    """Empirical share of y inside [lo, hi] (inclusive, 1e-9 tolerance)."""
    y, lo, hi = (np.asarray(a, dtype=float) for a in (y, lo, hi))
    inside = (y >= lo - 1e-9) & (y <= hi + 1e-9)
    if mask is not None:
        inside = inside[np.asarray(mask, dtype=bool)]
    return float(inside.mean()) if inside.size else float("nan")


def pinball_loss(y: ArrayLike, q: ArrayLike, tau: float, mask: Optional[np.ndarray] = None) -> float:
    """Mean pinball (quantile) loss of q at level tau."""
    y, q = np.asarray(y, dtype=float), np.asarray(q, dtype=float)
    d = y - q
    loss = np.maximum(tau * d, (tau - 1) * d)
    if mask is not None:
        loss = loss[np.asarray(mask, dtype=bool)]
    return float(loss.mean()) if loss.size else float("nan")


def mean_pinball(y: ArrayLike, q10: ArrayLike, q50: ArrayLike, q90: ArrayLike,
                 mask: Optional[np.ndarray] = None) -> float:
    """Average pinball loss over the three quantiles."""
    return float(np.mean([pinball_loss(y, q, tau, mask) for q, tau in zip((q10, q50, q90), QUANTILES.values())]))


def interval_report(y: ArrayLike, q10: ArrayLike, q50: ArrayLike, q90: ArrayLike,
                    mask: Optional[np.ndarray] = None) -> dict[str, float]:
    """{coverage_10_90, pinball, width} on the masked steps (default mask: daylight / non-zero)."""
    y, q10, q50, q90 = (np.asarray(a, dtype=float) for a in (y, q10, q50, q90))
    if mask is None:
        mask = default_mask(y, q10, q90)
    return {
        "coverage_10_90": coverage(y, q10, q90, mask),
        "pinball": mean_pinball(y, q10, q50, q90, mask),
        "width": float((q90 - q10)[mask].mean()) if mask.any() else float("nan"),
        "n": int(mask.sum()),
    }


# --- members -------------------------------------------------------------------------------------------------
def members_of(pred: Prediction) -> list[Triple]:
    """Per-member (q10, q50, q90) arrays; falls back to the envelope itself as the single member."""
    raw = (pred.extra or {}).get("members")
    if raw:
        return [tuple(np.asarray(m[k], dtype=float) for k in ("q10", "q50", "q90")) for m in raw]  # type: ignore
    return [(np.asarray(pred.q10, float), np.asarray(pred.q50, float), np.asarray(pred.q90, float))]


def sort_quantiles(q10: np.ndarray, q50: np.ndarray, q90: np.ndarray, floor: Optional[float] = 0.0,
                   cap: Optional[ArrayLike] = None) -> Triple:
    """Monotone rearrangement (sort per step) + optional physical floor / cap."""
    stack = np.sort(np.stack([q10, q50, q90]), axis=0)
    if floor is not None:
        stack = np.maximum(stack, floor)
    if cap is not None:
        stack = np.minimum(stack, np.asarray(cap, dtype=float))
    return stack[0], stack[1], stack[2]


# --- calibration ---------------------------------------------------------------------------------------------
def _conformal_quantile(scores: np.ndarray, tau: float) -> float:
    """Finite-sample split-conformal quantile: upper side uses ceil((n+1)tau), lower side floor((n+1)tau)."""
    s = np.sort(scores)
    n = s.size
    if n == 0:
        return 0.0
    if tau >= 0.5:
        k = min(n, math.ceil((n + 1) * tau))
    else:
        k = max(1, math.floor((n + 1) * tau))
    return float(s[k - 1])


def _bucket_index(shape: tuple[int, ...], n_buckets: int) -> np.ndarray:
    """Horizon bucket per element; the last axis is the horizon."""
    h = shape[-1]
    b = (np.arange(h) * n_buckets) // h
    return np.broadcast_to(b, shape)


@dataclass
class MemberCalibration:
    """Offsets[q][bucket] in units of the interval scale w = max(q90 - q10, eps_abs)."""

    offsets: dict[str, list[float]]
    n_cal: int = 0
    raw_coverage: float = float("nan")

    def apply(self, q10: ArrayLike, q50: ArrayLike, q90: ArrayLike, floor: Optional[float] = 0.0,
              cap: Optional[ArrayLike] = None, eps_abs: float = EPS_ABS) -> Triple:
        q = {k: np.asarray(v, dtype=float) for k, v in zip(QUANTILES, (q10, q50, q90))}
        w = np.maximum(q["q90"] - q["q10"], 0.0)
        live = w > eps_abs  # zero-width steps (PV night) stay put
        n_buckets = len(self.offsets["q10"])
        b = _bucket_index(q["q10"].shape, n_buckets)
        out = []
        for k in QUANTILES:
            delta = np.asarray(self.offsets[k], dtype=float)[b]
            out.append(np.where(live, q[k] + delta * np.maximum(w, eps_abs), q[k]))
        return sort_quantiles(*out, floor=floor, cap=cap)


@dataclass
class ConformalCalibration:
    """Per-member split-conformal offsets for one model. `fit` on the calibration slice, `apply` anywhere."""

    model: str = ""
    model_version: str = ""
    n_buckets: int = 1
    eps_abs: float = EPS_ABS
    members: dict[str, MemberCalibration] = field(default_factory=dict)
    version: int = 1

    # -- fit ----------------------------------------------------------------------------------------------
    @classmethod
    def fit(cls, y: ArrayLike, members: Union[Triple, Sequence[Triple], Mapping[str, Triple]], *,
            mask: Optional[np.ndarray] = None, n_buckets: int = 1, model: str = "", model_version: str = "",
            eps_abs: float = EPS_ABS) -> "ConformalCalibration":
        """y and each member's q10/q50/q90: same shape, last axis = horizon (e.g. (n_windows, 96)).

        `members` may be one (q10, q50, q90) triple, a list of triples or {member_id: triple}. `mask` selects the
        calibration steps (default: daylight / non-zero, see `default_mask`).
        """
        mem = _as_member_dict(members)
        y = np.asarray(y, dtype=float)
        cal = cls(model=model, model_version=model_version, n_buckets=n_buckets, eps_abs=eps_abs)
        for mid, (q10, q50, q90) in mem.items():
            q10, q50, q90 = (np.asarray(a, dtype=float) for a in (q10, q50, q90))
            if not (q10.shape == q50.shape == q90.shape == y.shape):
                raise ValueError(f"member {mid}: shapes {q10.shape} {q50.shape} {q90.shape} vs y {y.shape}")
            m = default_mask(y, q10, q90, eps_abs) if mask is None else np.asarray(mask, dtype=bool)
            w = np.maximum(q90 - q10, eps_abs)
            b = _bucket_index(y.shape, n_buckets)
            offsets: dict[str, list[float]] = {}
            for k, tau in QUANTILES.items():
                q = {"q10": q10, "q50": q50, "q90": q90}[k]
                s = (y - q) / w
                offsets[k] = [_conformal_quantile(s[m & (b == j)], tau) for j in range(n_buckets)]
            cal.members[mid] = MemberCalibration(offsets=offsets, n_cal=int(m.sum()),
                                                 raw_coverage=coverage(y, q10, q90, m))
        return cal

    # -- apply --------------------------------------------------------------------------------------------
    def member(self, member: Union[str, int, None] = None) -> MemberCalibration:
        if not self.members:
            raise ValueError("calibration is empty")
        if member is None:
            return next(iter(self.members.values()))
        key = str(member)
        if key in self.members:
            return self.members[key]
        # unseen member id: fall back to positional, then the first member
        vals = list(self.members.values())
        if isinstance(member, int) and member < len(vals):
            return vals[member]
        return vals[0]

    def apply(self, q10: ArrayLike, q50: ArrayLike, q90: ArrayLike, member: Union[str, int, None] = None,
              floor: Optional[float] = 0.0, cap: Optional[ArrayLike] = None) -> Triple:
        return self.member(member).apply(q10, q50, q90, floor=floor, cap=cap, eps_abs=self.eps_abs)

    def apply_members(self, members: Sequence[Triple], floor: Optional[float] = 0.0,
                      cap: Optional[ArrayLike] = None) -> list[Triple]:
        return [self.apply(*m, member=i, floor=floor, cap=cap) for i, m in enumerate(members)]

    def apply_prediction(self, pred: Prediction, floor: Optional[float] = 0.0,
                         cap: Optional[ArrayLike] = None) -> Prediction:
        """New Prediction with calibrated envelope (member 0's offsets) and calibrated extra["members"]."""
        q10, q50, q90 = self.apply(pred.q10, pred.q50, pred.q90, member=None, floor=floor, cap=cap)
        d = pred.to_dict()
        d.update(q10=q10.tolist(), q50=q50.tolist(), q90=q90.tolist())
        extra = dict(d.get("extra") or {})
        if extra.get("members"):
            cal_m = self.apply_members(members_of(pred), floor=floor, cap=cap)
            extra["members"] = [{"q10": a.tolist(), "q50": b.tolist(), "q90": c.tolist()} for a, b, c in cal_m]
        extra["calibrated"] = True
        d["extra"] = extra
        return Prediction.from_dict(d)

    def report(self, y: ArrayLike, members: Union[Triple, Sequence[Triple], Mapping[str, Triple]],
               mask: Optional[np.ndarray] = None) -> dict[str, Any]:
        """Raw vs calibrated coverage and pinball per member on a (test) slice."""
        out: dict[str, Any] = {}
        y = np.asarray(y, dtype=float)
        for i, (mid, trip) in enumerate(_as_member_dict(members).items()):
            raw = tuple(np.asarray(a, dtype=float) for a in trip)
            m = default_mask(y, raw[0], raw[2], self.eps_abs) if mask is None else mask
            cal = self.apply(*raw, member=mid if mid in self.members else i)
            out[mid] = {"raw": interval_report(y, *raw, mask=m), "calibrated": interval_report(y, *cal, mask=m)}
        return out

    # -- io -----------------------------------------------------------------------------------------------
    def to_dict(self) -> dict[str, Any]:
        return {
            "version": self.version,
            "method": "split_conformal_normalised_shift",
            "model": self.model,
            "model_version": self.model_version,
            "n_buckets": self.n_buckets,
            "eps_abs": self.eps_abs,
            "members": {k: {"offsets": v.offsets, "n_cal": v.n_cal, "raw_coverage": v.raw_coverage}
                        for k, v in self.members.items()},
        }

    @classmethod
    def from_dict(cls, d: Mapping[str, Any]) -> "ConformalCalibration":
        return cls(
            model=d.get("model", ""),
            model_version=d.get("model_version", ""),
            n_buckets=int(d.get("n_buckets", 1)),
            eps_abs=float(d.get("eps_abs", EPS_ABS)),
            version=int(d.get("version", 1)),
            members={k: MemberCalibration(offsets={q: [float(x) for x in v["offsets"][q]] for q in QUANTILES},
                                          n_cal=int(v.get("n_cal", 0)),
                                          raw_coverage=float(v.get("raw_coverage", float("nan"))))
                     for k, v in d["members"].items()},
        )

    def save(self, path: Union[str, Path]) -> None:
        p = Path(path)
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(json.dumps(self.to_dict(), indent=2) + "\n")

    @classmethod
    def load(cls, path: Union[str, Path]) -> "ConformalCalibration":
        return cls.from_dict(json.loads(Path(path).read_text()))


def _as_member_dict(members: Union[Triple, Sequence[Triple], Mapping[str, Triple]]) -> dict[str, Triple]:
    if isinstance(members, Mapping):
        return {str(k): v for k, v in members.items()}
    # a single (q10, q50, q90) triple: a tuple of three arrays / lists (not of tuples)
    if isinstance(members, tuple) and len(members) == 3 and not isinstance(members[0], tuple):
        return {"0": members}
    seq = list(members)
    if len(seq) == 3 and isinstance(seq[0], np.ndarray):
        return {"0": (seq[0], seq[1], seq[2])}
    return {str(i): tuple(m) for i, m in enumerate(seq)}  # type: ignore[misc]


# --- quantile function (used by scenarios) -------------------------------------------------------------------
def quantile_at(q10: np.ndarray, q50: np.ndarray, q90: np.ndarray, z: np.ndarray) -> np.ndarray:
    """Value at standard-normal score z from three quantiles: piecewise linear in z through (−z90, q10), (0, q50),
    (z90, q90), extended linearly past the 10/90 points (a split-normal tail). Broadcasts."""
    lo = (q50 - q10) / Z90
    hi = (q90 - q50) / Z90
    return q50 + np.where(z < 0, z * lo, z * hi)
