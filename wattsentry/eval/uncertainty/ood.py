"""Mahalanobis OOD score on a model embedding (plan §3.5 "Qualify": OOD above a validation-tuned threshold
→ human review).

`fit` on train-slice embeddings: mean + shrunk covariance (Ledoit-Wolf by default, or a fixed shrinkage
alpha towards (tr(S)/d)·I). The raw Mahalanobis distance is standardised against the train distances, so the
score is unit-free and matches the envelope's convention (`ood_score >= 0`, 0 = in-distribution):

    ood_score = max(0, (D(x) - median_train(D)) / (1.4826 · MAD_train(D)))

≈ "robust z-score of the distance"; the default threshold `ood_max = 3.0` lives in models/thresholds.json.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Mapping, Optional, Union

import numpy as np


def ledoit_wolf_shrinkage(x: np.ndarray) -> float:
    """Ledoit-Wolf optimal shrinkage intensity towards mu·I (centred data, rows = samples)."""
    n, d = x.shape
    s = x.T @ x / n
    mu = np.trace(s) / d
    delta = s.copy()
    delta.flat[:: d + 1] -= mu
    d2 = (delta ** 2).sum() / d
    x2 = x ** 2
    beta = ((x2.T @ x2) / n - s ** 2).sum() / (d * n)
    beta = min(beta, d2)
    return float(beta / d2) if d2 > 0 else 1.0


@dataclass
class MahalanobisOOD:
    mean: Optional[np.ndarray] = None
    precision: Optional[np.ndarray] = None
    shrinkage: float = 0.0
    d_median: float = 0.0
    d_scale: float = 1.0
    extra: dict[str, Any] = field(default_factory=dict)

    def fit(self, x_train: np.ndarray, shrinkage: Union[str, float] = "ledoit_wolf") -> "MahalanobisOOD":
        x = np.asarray(x_train, dtype=float)
        if x.ndim != 2 or x.shape[0] < 2:
            raise ValueError("x_train must be (n >= 2, d)")
        self.mean = x.mean(axis=0)
        xc = x - self.mean
        n, d = xc.shape
        s = xc.T @ xc / n
        alpha = ledoit_wolf_shrinkage(xc) if shrinkage == "ledoit_wolf" else float(shrinkage)
        alpha = float(np.clip(alpha, 0.0, 1.0))
        mu = np.trace(s) / d if d else 1.0
        cov = (1 - alpha) * s + alpha * mu * np.eye(d)
        cov.flat[:: d + 1] += 1e-9 * max(mu, 1e-12)  # numerical floor
        self.precision = np.linalg.inv(cov)
        self.shrinkage = alpha
        dist = self.distance(x)
        self.d_median = float(np.median(dist))
        mad = float(np.median(np.abs(dist - self.d_median)))
        self.d_scale = 1.4826 * mad if mad > 0 else float(dist.std() or 1.0)
        return self

    def distance(self, x: np.ndarray) -> np.ndarray:
        """Raw Mahalanobis distance sqrt((x-mu)^T P (x-mu)); x is (n, d) or (d,)."""
        if self.mean is None or self.precision is None:
            raise RuntimeError("fit first")
        x = np.atleast_2d(np.asarray(x, dtype=float)) - self.mean
        d2 = np.einsum("ij,jk,ik->i", x, self.precision, x)
        return np.sqrt(np.maximum(d2, 0.0))

    def score(self, x: np.ndarray) -> np.ndarray:
        """Standardised OOD score >= 0 (0 = typical of train)."""
        return np.maximum(0.0, (self.distance(x) - self.d_median) / self.d_scale)

    # -- io -----------------------------------------------------------------------------------------------
    def to_dict(self) -> dict[str, Any]:
        if self.mean is None or self.precision is None:
            raise RuntimeError("fit first")
        return {"version": 1, "mean": self.mean.tolist(), "precision": self.precision.tolist(),
                "shrinkage": self.shrinkage, "d_median": self.d_median, "d_scale": self.d_scale,
                "extra": self.extra}

    @classmethod
    def from_dict(cls, d: Mapping[str, Any]) -> "MahalanobisOOD":
        return cls(mean=np.asarray(d["mean"], dtype=float), precision=np.asarray(d["precision"], dtype=float),
                   shrinkage=float(d["shrinkage"]), d_median=float(d["d_median"]), d_scale=float(d["d_scale"]),
                   extra=dict(d.get("extra") or {}))

    def save(self, path: Union[str, Path]) -> None:
        p = Path(path)
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(json.dumps(self.to_dict()) + "\n")

    @classmethod
    def load(cls, path: Union[str, Path]) -> "MahalanobisOOD":
        return cls.from_dict(json.loads(Path(path).read_text()))
