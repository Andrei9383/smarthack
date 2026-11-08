"""Torch modules: PatchTST-style quantile encoder (M1 PVNet, M2 LoadNet) and the 1D-CNN fault head (M4).

Attention is written out by hand (no nn.TransformerEncoder fast path) so the legacy ONNX exporter (opset 17)
handles it everywhere.
"""

from __future__ import annotations

import math

import torch
import torch.nn as nn
import torch.nn.functional as F

QUANTILES = (0.1, 0.5, 0.9)


class Block(nn.Module):
    def __init__(self, d: int, heads: int, ff: int, drop: float):
        super().__init__()
        self.h, self.dh = heads, d // heads
        self.ln1, self.ln2 = nn.LayerNorm(d), nn.LayerNorm(d)
        self.qkv = nn.Linear(d, 3 * d)
        self.proj = nn.Linear(d, d)
        self.ff = nn.Sequential(nn.Linear(d, ff), nn.GELU(), nn.Dropout(drop), nn.Linear(ff, d))
        self.drop = nn.Dropout(drop)

    def forward(self, x):
        B, N, D = x.shape
        q, k, v = self.qkv(self.ln1(x)).reshape(B, N, 3, self.h, self.dh).permute(2, 0, 3, 1, 4)
        att = torch.softmax(q @ k.transpose(-1, -2) / math.sqrt(self.dh), dim=-1)
        y = (att @ v).transpose(1, 2).reshape(B, N, D)
        x = x + self.drop(self.proj(y))
        return x + self.drop(self.ff(self.ln2(x)))


def _mlp(i, d, drop=0.0):
    return nn.Sequential(nn.Linear(i, d), nn.GELU(), nn.Dropout(drop), nn.Linear(d, d))


class QuantileEncoder(nn.Module):
    """96 steps -> 12 patch tokens of 8 steps (all channels) + reference, 14-day summary and static tokens.

    Each patch token decodes its own 8 steps x 3 quantiles; q10 <= q50 <= q90 by construction (softplus gaps).
    forward(seq [B,96,C], ref [B,R], summ [B,14,Cs], stat [B,Cst], mask [B,96]) -> (q [B,96,3], emb [B,d]).
    Output is in normalised units (PV: kW per recorded kWp; load: per site scale); mask zeroes night steps.
    """

    def __init__(self, c_seq, r_dim, c_sum, c_stat, d=64, layers=3, heads=4, ff=128, patch=8, horizon=96,
                 drop=0.1, nonneg=True, pointwise=0, base=None):
        """pointwise > 0: a per-step MLP over the seq channels (hidden size `pointwise`) added to the head, so the
        step-wise physics mapping (weather/geometry/pvlib -> power, what a GBM learns) is direct instead of going
        through 8-step patch embeddings. base = (channel, mean, std): q50 is a residual on that (un-normalised) seq
        channel, e.g. pvlib-with-recorded-metadata per kWp for PV, the 7-day mean profile for load."""
        super().__init__()
        self.patch, self.n_patch, self.nonneg = patch, horizon // patch, nonneg
        self.patch_emb = nn.Linear(patch * c_seq, d)
        self.pos = nn.Parameter(torch.zeros(1, self.n_patch + 3, d))
        nn.init.normal_(self.pos, std=0.02)
        self.ref_tok = _mlp(r_dim, d, drop)
        self.sum_tok = _mlp(14 * c_sum, d, drop)
        self.stat_tok = _mlp(c_stat, d, drop)
        self.blocks = nn.ModuleList(Block(d, heads, ff, drop) for _ in range(layers))
        self.ln = nn.LayerNorm(d)
        self.head = nn.Linear(d, patch * 3)
        self.pw = (nn.Sequential(nn.Linear(c_seq, pointwise), nn.GELU(), nn.Linear(pointwise, pointwise), nn.GELU(),
                                 nn.Linear(pointwise, 3)) if pointwise else None)
        self.base_ch = None if base is None else int(base[0])
        if base is not None:
            self.register_buffer("base_mu", torch.tensor(float(base[1])))
            self.register_buffer("base_sd", torch.tensor(float(base[2])))

    def forward(self, seq, ref, summ, stat, mask):
        B = seq.shape[0]
        p = seq.reshape(B, self.n_patch, -1)  # [B, 12, 8*C] (steps-major within a patch)
        toks = torch.cat([
            self.patch_emb(p),
            self.ref_tok(ref).unsqueeze(1),
            self.sum_tok(summ.reshape(B, -1)).unsqueeze(1),
            self.stat_tok(stat).unsqueeze(1),
        ], 1) + self.pos
        for blk in self.blocks:
            toks = blk(toks)
        toks = self.ln(toks)
        out = self.head(toks[:, : self.n_patch]).reshape(B, -1, 3)  # [B, 96, 3]
        if self.pw is not None:
            out = out + self.pw(seq)
        mid = out[..., 1]
        if self.base_ch is not None:
            mid = mid + seq[..., self.base_ch] * self.base_sd + self.base_mu
        if self.nonneg:
            mid = F.softplus(mid, beta=8.0)
        lo = mid - F.softplus(out[..., 0])
        hi = mid + F.softplus(out[..., 2])
        if self.nonneg:
            lo = torch.clamp(lo, min=0.0)
        q = torch.stack([lo, mid, hi], -1) * mask.unsqueeze(-1)
        return q, toks.mean(1)


def pinball(q, y, w=None):
    """Mean pinball loss over (q10, q50, q90); q [B,H,3], y [B,H], optional weights [B,H]."""
    taus = torch.tensor(QUANTILES, device=q.device, dtype=q.dtype)
    e = y.unsqueeze(-1) - q
    l = torch.maximum(taus * e, (taus - 1) * e).mean(-1)
    if w is None:
        return l.mean()
    return (l * w).sum() / w.sum().clamp(min=1e-6)


class SiteHealthNet(nn.Module):
    """1D-CNN over the 24 h window of residual / telemetry channels + 14-day summary + static -> 8 logits."""

    def __init__(self, c_seq, c_sum, c_stat, width=48, n_cls=8, drop=0.15):
        super().__init__()
        w = width
        self.conv = nn.Sequential(
            nn.Conv1d(c_seq, w, 5, padding=2), nn.GELU(), nn.BatchNorm1d(w),
            nn.Conv1d(w, w, 5, padding=4, dilation=2), nn.GELU(), nn.BatchNorm1d(w),
            nn.Conv1d(w, w, 5, padding=8, dilation=4), nn.GELU(), nn.BatchNorm1d(w),
            nn.Conv1d(w, w, 3, padding=8, dilation=8), nn.GELU(),
        )
        self.sum = nn.Sequential(nn.Conv1d(c_sum, 32, 3, padding=1), nn.GELU(),
                                 nn.Conv1d(32, 32, 3, padding=1), nn.GELU())
        self.stat = nn.Sequential(nn.Linear(c_stat, 16), nn.GELU())
        self.head = nn.Sequential(nn.Linear(2 * w + 64 + 16, 96), nn.GELU(), nn.Dropout(drop), nn.Linear(96, n_cls))

    def forward(self, seq, summ, stat):
        h = self.conv(seq.transpose(1, 2))
        s = self.sum(summ.transpose(1, 2))
        emb = torch.cat([h.mean(-1), h.amax(-1), s.mean(-1), s[..., -1], self.stat(stat)], -1)
        return self.head(emb), emb


def focal_loss(logits, target, gamma=2.0, alpha=None):
    logp = F.log_softmax(logits, -1)
    lp = logp.gather(1, target[:, None]).squeeze(1)
    p = lp.exp()
    loss = -((1 - p) ** gamma) * lp
    if alpha is not None:
        loss = loss * alpha[target]
    return loss.mean()


def n_params(m: nn.Module) -> int:
    return sum(p.numel() for p in m.parameters())
