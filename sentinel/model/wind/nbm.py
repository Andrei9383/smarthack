"""M6 normal-behaviour model (torch): two small causal GRUs with a turbine embedding.

- power path: density-normalised nacelle + meteo wind, direction, ambient, hour, class → expected power
  quantiles (q10/q50/q90) from H bootstrap heads (multi-head ensemble → member_std);
- thermal path: same inputs + actual power history (load) → quantiles of the five component temperature rises.
Turbine embedding index 0 = unknown turbine (held-out park); embedding dropout trains it.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from .features import N_POWER_IN, NBM_IN, TEMP_SENSORS

QUANTILES = (0.1, 0.5, 0.9)


class WindNBM(nn.Module):
    def __init__(self, n_turbines: int, hidden: int = 96, emb: int = 8, heads: int = 3, emb_dropout: float = 0.2):
        super().__init__()
        self.n_turbines = n_turbines
        self.heads = heads
        self.emb_dropout = emb_dropout
        self.n_temp = len(TEMP_SENSORS)
        self.emb = nn.Embedding(n_turbines + 1, emb)
        self.inp_p = nn.Linear(N_POWER_IN + emb, hidden)
        self.gru_p = nn.GRU(hidden, hidden, batch_first=True)
        self.head_p = nn.Sequential(nn.Linear(hidden, hidden), nn.GELU(), nn.Linear(hidden, heads * 3))
        self.inp_t = nn.Linear(len(NBM_IN) + emb, hidden)
        self.gru_t = nn.GRU(hidden, hidden, batch_first=True)
        self.head_t = nn.Linear(hidden, self.n_temp * 3)

    @staticmethod
    def _quantiles(raw: torch.Tensor) -> torch.Tensor:
        """raw [..., 3] → monotone (q10, q50, q90)."""
        q50 = raw[..., 1]
        q10 = q50 - F.softplus(raw[..., 0]) - 1e-4
        q90 = q50 + F.softplus(raw[..., 2]) + 1e-4
        return torch.stack([q10, q50, q90], -1)

    def forward_all(self, x: torch.Tensor, tid: torch.Tensor):
        """x [B, L, F_IN], tid [B] → power quantiles per head [B, L, H, 3], temp quantiles [B, L, 5, 3]."""
        if self.training and self.emb_dropout > 0:
            drop = torch.rand(tid.shape, device=tid.device) < self.emb_dropout
            tid = torch.where(drop, torch.zeros_like(tid), tid)
        B, L, _ = x.shape
        e = self.emb(tid)[:, None, :].expand(B, L, -1)
        hp, _ = self.gru_p(F.gelu(self.inp_p(torch.cat([x[..., :N_POWER_IN], e], -1))))
        pq = self._quantiles(self.head_p(hp).view(B, L, self.heads, 3))
        ht, _ = self.gru_t(F.gelu(self.inp_t(torch.cat([x, e], -1))))
        tq = self._quantiles(self.head_t(ht).view(B, L, self.n_temp, 3))
        return pq, tq

    def forward(self, x: torch.Tensor, tid: torch.Tensor):
        """Export / inference signature: (pq mean over heads [B,L,3], q50 head std [B,L], tq [B,L,5,3])."""
        pq, tq = self.forward_all(x, tid)
        return pq.mean(2), pq[..., 1].std(2, unbiased=False), tq


def pinball(q: torch.Tensor, y: torch.Tensor, m: torch.Tensor) -> torch.Tensor:
    """q [..., 3], y [...], m [...] bool → mean pinball loss over masked elements."""
    taus = torch.tensor(QUANTILES, device=q.device, dtype=q.dtype)
    d = y[..., None] - q
    l = torch.maximum(taus * d, (taus - 1) * d).mean(-1)
    mf = m.to(l.dtype)
    return (l * mf).sum() / mf.sum().clamp_min(1.0)


def n_params(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters())
