"""Shared building blocks: conv block, attention blocks, BEV sampling, positional codes, cubic curves.

Axes everywhere: rows = x (right), columns = y (forward), metres, origin at the ego, the maps span -50 .. 50 m.
"""
import math

import torch
import torch.nn as nn
import torch.nn.functional as F

HALF = 50.0                                                   # half extent of every map (m)


def block(cin: int, cout: int, stride: int = 1) -> nn.Sequential:
    """two 3 x 3 convolutions with GroupNorm and ReLU"""
    return nn.Sequential(nn.Conv2d(cin, cout, 3, stride, 1, bias=False), nn.GroupNorm(8, cout), nn.ReLU(inplace=True),
                         nn.Conv2d(cout, cout, 3, 1, 1, bias=False), nn.GroupNorm(8, cout), nn.ReLU(inplace=True))


def unit(v: torch.Tensor, dim: int = 1, eps: float = 1e-6) -> torch.Tensor:
    return v / v.norm(dim=dim, keepdim=True).clamp_min(eps)


def cell_centres(n: int, device) -> torch.Tensor:
    """(n,) centres of n cells over -HALF .. HALF"""
    return -HALF + (torch.arange(n, device=device, dtype=torch.float32) + 0.5) * (2 * HALF / n)


def fourier_pts(p: torch.Tensor, nf: int = 4) -> torch.Tensor:
    """(..., 2) metres -> (..., 4 nf) sin / cos code"""
    k = (2.0 ** torch.arange(nf, device=p.device, dtype=p.dtype)) * math.pi / HALF
    a = p[..., None] * k
    return torch.cat([a.sin(), a.cos()], -1).flatten(-2)


def sample(maps: torch.Tensor, pts: torch.Tensor, border: bool = False) -> torch.Tensor:
    """bilinear samples of maps (B, C, H, W) at pts (B, ..., 2) metres -> (B, ..., C); outside the map: zero, or the border value"""
    B, shp = maps.shape[0], pts.shape[1:-1]
    g = torch.stack([pts[..., 1], pts[..., 0]], -1).reshape(B, -1, 1, 2).float() / HALF     # grid x = column = y
    out = F.grid_sample(maps.float(), g, mode="bilinear", padding_mode="border" if border else "zeros", align_corners=False)
    return out[..., 0].transpose(1, 2).reshape(B, *shp, maps.shape[1])


def bezier(a, ta, b, tb, sa, sb, n: int):
    """(..., n, 2) points of the cubic curve a -> b with control points a + sa L ta and b - sb L tb (L = the chord); sa = sb = 1/3 is the
    chord-tangent Hermite curve. Differentiable in all inputs."""
    L = (b - a).norm(dim=-1, keepdim=True)
    s = torch.linspace(0, 1, n, device=a.device, dtype=a.dtype).view(*([1] * (a.dim() - 1)), n, 1)
    q1, q2 = (a + sa * L * ta)[..., None, :], (b - sb * L * tb)[..., None, :]
    return (1 - s) ** 3 * a[..., None, :] + 3 * (1 - s) ** 2 * s * q1 + 3 * (1 - s) * s ** 2 * q2 + s ** 3 * b[..., None, :]


class SetBlock(nn.Module):
    """self-attention over a set (padding masked), pre-norm, residual, feed-forward"""

    def __init__(self, d: int, heads: int = 4):
        super().__init__()
        self.n1, self.n2 = nn.LayerNorm(d), nn.LayerNorm(d)
        self.att = nn.MultiheadAttention(d, heads, batch_first=True)
        self.ffn = nn.Sequential(nn.Linear(d, 2 * d), nn.GELU(), nn.Linear(2 * d, d))

    def forward(self, x: torch.Tensor, pad: torch.Tensor) -> torch.Tensor:
        h = self.n1(x)
        x = x + torch.nan_to_num(self.att(h, h, h, key_padding_mask=pad, need_weights=False)[0])
        return x + self.ffn(self.n2(x))


class LocalAttn(nn.Module):
    """neighbourhood attention on a grid (B, d, H, W): every cell attends to the k x k cells around it, with a learned relative-position bias"""

    def __init__(self, d: int, n: int, heads: int = 4, k: int = 7):
        super().__init__()
        self.d, self.h, self.k = d, heads, k
        self.n1, self.n2 = nn.GroupNorm(1, d), nn.GroupNorm(1, d)
        self.qkv, self.proj = nn.Conv2d(d, 3 * d, 1), nn.Conv2d(d, d, 1)
        self.ffn = nn.Sequential(nn.Conv2d(d, 2 * d, 1), nn.GELU(), nn.Conv2d(2 * d, d, 1))
        self.bias = nn.Parameter(torch.zeros(heads, k * k))
        outside = F.unfold(torch.ones(1, 1, n, n), k, padding=k // 2).view(1, 1, k * k, n * n) <= 0.5      # window cells beyond the n x n grid
        self.register_buffer("outside", outside, persistent=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, d, H, W = x.shape
        h, k, dh = self.h, self.k, self.d // self.h
        q, kk, v = self.qkv(self.n1(x)).chunk(3, 1)
        unf = lambda t: F.unfold(t, k, padding=k // 2).view(B, h, dh, k * k, H * W)  # noqa: E731
        att = (q.view(B, h, dh, 1, H * W) * unf(kk)).sum(2) / math.sqrt(dh) + self.bias.view(1, h, k * k, 1)
        att = att.masked_fill(self.outside, float("-inf")).softmax(2)
        x = x + self.proj((att.unsqueeze(2) * unf(v)).sum(3).reshape(B, d, H, W))
        return x + self.ffn(self.n2(x))
