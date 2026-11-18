"""The junction pair head (E287 / E324 s): which entry keypoint joins which exit keypoint, and the shape of the connector.

    out = PairHead(cfg)(kp, P, jm) # tok (B, 2K, 64) keypoint tokens; logit (B, K, K) pair logit entry i -> exit j; allowed (B, K, K); scale (B, K, K, 2) handle scales
Pair features: the two keypoint tokens (after two set-attention layers over all keypoints of the frame), 13 geometry numbers, 6 evidence numbers
sampled along the chord-tangent Hermite curve of the pair (lane centre of B, junction zone and core, stop line and crossing maxima) and 5 numbers of
the ORDER RULE (entry 1 joins exit 1 inside a group of parallel lanes: rank inside its group / group size of the entry and of the exit, same rank).
The groups (single linkage: heading within group_angle, group_along and group_across apart) are a function of the keypoint positions without gradient.
"""
import math

import torch
import torch.nn as nn

from lg_config import Config
from lg_layers import HALF, SetBlock, sample


def hermite(a, ta, b, tb, n: int):
    """(..., n, 2) the cubic Hermite curve a -> b with tangent length = chord (the evidence curve; equal to `lg_layers.bezier` with scales 1/3, but summed in the Hermite form as the original)"""
    L = (b - a).norm(dim=-1, keepdim=True)
    s = torch.linspace(0, 1, n, device=a.device).view(*([1] * (a.dim() - 1)), n, 1)
    h00, h10, h01, h11 = 2 * s ** 3 - 3 * s ** 2 + 1, s ** 3 - 2 * s ** 2 + s, -2 * s ** 3 + 3 * s ** 2, s ** 3 - s ** 2
    return h00 * a[..., None, :] + h10 * (L * ta)[..., None, :] + h01 * b[..., None, :] + h11 * (L * tb)[..., None, :]


@torch.no_grad()
def rank_features(pos: torch.Tensor, D: torch.Tensor, valid: torch.Tensor, cfg: Config) -> torch.Tensor:
    """(B, K, K, 5) order-rule features from the keypoints pos, D (B, 2K, 2) and valid (B, 2, K), on the device (no host round trip):
    per kind the groups (connected components by matrix squaring), the group size n and the rank r inside the group (0 = leftmost looking along the
    group's mean heading, ties by index); features r_entry / max(n - 1, 1), r_exit / max(n - 1, 1), n_entry / 4, n_exit / 4, [n and r equal]"""
    B, K = valid.shape[0], valid.shape[2]
    P, T = pos.view(B, 2, K, 2), D.view(B, 2, K, 2)
    ti, tj = T[:, :, :, None, :], T[:, :, None, :, :]
    ang = torch.rad2deg(torch.arccos((ti[..., 0] * tj[..., 0] + ti[..., 1] * tj[..., 1]).clamp(-1, 1)))
    s = ti + tj
    sn = s + 1e-9
    t = s / (sn[..., 0] * sn[..., 0] + sn[..., 1] * sn[..., 1]).sqrt()[..., None]                    # mean heading of the pair
    d = P[:, :, None, :, :] - P[:, :, :, None, :]
    along = (d[..., 0] * t[..., 0] + d[..., 1] * t[..., 1]).abs()
    across = (d[..., 0] * -t[..., 1] + d[..., 1] * t[..., 0]).abs()
    vv = valid[:, :, :, None] & valid[:, :, None, :]
    A = (ang <= cfg.group_angle) & (along <= cfg.group_along) & (across <= cfg.group_across) & vv
    R = (A | A.transpose(-1, -2) | torch.eye(K, dtype=torch.bool, device=pos.device)).float()
    for _ in range(math.ceil(math.log2(K))):                                                          # transitive closure: paths of up to K - 1 links
        R = (R @ R > 0).float()
    n = R.sum(-1)                                                                                     # group size (1 for an invalid keypoint)
    tg = (R @ T) / n[..., None]
    tg = tg / (tg.norm(dim=-1, keepdim=True) + 1e-9)
    lat = ((P - (R @ P) / n[..., None]) * torch.stack([-tg[..., 1], tg[..., 0]], -1)).sum(-1)       # lateral position inside the group
    li, lj = lat[..., :, None], lat[..., None, :]
    before = (lj > li) | ((lj == li) & (torch.arange(K, device=pos.device)[None, :] < torch.arange(K, device=pos.device)[:, None]))
    r = (R.bool() & before).sum(-1).float()
    na, nb, ra, rb = n[:, 0, :, None], n[:, 1, None, :], r[:, 0, :, None], r[:, 1, None, :]
    rk = [ra / (na - 1).clamp_min(1.0), rb / (nb - 1).clamp_min(1.0), na / 4.0, nb / 4.0, ((na == nb) & (ra == rb)).float()]
    return torch.stack([x.expand(B, K, K) for x in rk], -1)


class PairHead(nn.Module):
    def __init__(self, cfg: Config, d: int = 64, f1: int = 64, hid: int = 64):
        super().__init__()
        self.cfg = cfg
        self.kp = nn.Linear(f1 + 6, d)
        self.blocks = nn.ModuleList([SetBlock(d), SetBlock(d)])
        nin = 2 * d + 13 + 6 + 5
        self.pair = nn.Sequential(nn.Linear(nin, hid), nn.GELU(), nn.Linear(hid, hid), nn.GELU(), nn.Linear(hid, 1))
        self.scale = nn.Sequential(nn.Linear(nin, 32), nn.GELU(), nn.Linear(32, 2))

    def forward(self, kp: dict, P: dict, jm: dict) -> dict:
        cfg, pos, D, pr, valid = self.cfg, kp["pos"], kp["dir"], kp["p"], kp["valid"]
        K = valid.shape[2]
        t = self.kp(torch.cat([kp["feat"], pos / HALF, D, pr[..., None], kp["kind"][..., None]], -1))
        for blk in self.blocks:
            t = blk(t, ~valid.flatten(1))
        pa, pb, ta, tb = pos[:, :K], pos[:, K:], D[:, :K], D[:, K:]
        ch = pb[:, None, :, :] - pa[:, :, None, :]                                                    # [b, i, j]: exit j - entry i
        L = ch.norm(dim=-1)
        ta_, tb_ = ta[:, :, None, :].expand_as(ch), tb[:, None, :, :].expand_as(ch)
        na_, nb_ = torch.stack([-ta_[..., 1], ta_[..., 0]], -1), torch.stack([-tb_[..., 1], tb_[..., 0]], -1)
        dot = lambda u, v: (u * v).sum(-1)  # noqa: E731
        cross = lambda u, v: u[..., 0] * v[..., 1] - u[..., 1] * v[..., 0]  # noqa: E731
        cu = ch / L.clamp_min(1e-6)[..., None]
        cos_turn = dot(ta_, tb_)
        geo = torch.stack([cos_turn, cross(ta_, tb_), L / 20.0, dot(ch, ta_) / 20.0, dot(ch, na_) / 20.0, dot(ch, tb_) / 20.0, dot(ch, nb_) / 20.0,
                           dot(cu, ta_), cross(ta_, cu), dot(cu, tb_), cross(tb_, cu), pr[:, :K, None].expand_as(L), pr[:, None, K:].expand_as(L)], -1)
        allowed = (cos_turn >= cfg.cos_turn) & (L >= cfg.chord_min) & (dot(ch, ta_) > 0) & (dot(ch, tb_) > 0) & valid[:, 0, :, None] & valid[:, 1, None, :]
        pts = hermite(pa[:, :, None, :].expand_as(ch), ta_, pb[:, None, :, :].expand_as(ch), tb_, cfg.evidence_samples)   # (B, K, K, n, 2)
        lane = sample(P["centre1"][:, :1].sigmoid(), pts)[..., 0]
        zone, paint = sample(jm["j1"][:, :2].sigmoid(), pts), sample(jm["j05"].sigmoid(), pts)
        ev = torch.stack([lane.mean(-1), lane.min(-1)[0], zone[..., 0].mean(-1), zone[..., 1].mean(-1), paint[..., 0].max(-1)[0], paint[..., 1].max(-1)[0]], -1)
        f = torch.cat([t[:, :K, None].expand(-1, -1, K, -1), t[:, None, K:].expand(-1, K, -1, -1), geo, ev, rank_features(pos, D, valid, cfg)], -1)
        return dict(tok=t, logit=self.pair(f)[..., 0], allowed=allowed, scale=0.05 + 1.2 * self.scale(f).sigmoid())     # trained range of the handle scales: 0.05 .. 1.25 chords
