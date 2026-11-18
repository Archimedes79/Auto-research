"""The junction maps head (E286 / E307) and its keypoints.

    jm = JunctionMaps()(P)        # j1 (B, 8, 100, 100) 1 m logits: zone, conflict core, entry heat, exit heat, entry heading (2), exit heading (2)
                                  # j05 (B, 2, 200, 200) 0.5 m logits: stop lines, pedestrian crossings; f1 (B, 64, 100, 100) the 1 m features
    kp = keypoints(jm, cfg)       # the K strongest 3 x 3 peaks of the entry and of the exit heat, position by soft-argmax over the 3 x 3 window
The keypoint position is differentiable in the heat; the choice of the K peaks is an index choice (the gradient flows through the values).
"""
import torch
import torch.nn as nn
import torch.nn.functional as F

from lg_config import Config
from lg_layers import block, cell_centres, unit

N1 = 100                                     # 1 m grid


class ResDil(nn.Module):
    def __init__(self, d: int, dil: int):
        super().__init__()
        self.f = nn.Sequential(nn.Conv2d(d, d, 3, 1, dil, dilation=dil, bias=False), nn.GroupNorm(8, d), nn.ReLU(inplace=True),
                               nn.Conv2d(d, d, 3, 1, 1, bias=False), nn.GroupNorm(8, d))

    def forward(self, x):
        return F.relu(x + self.f(x))


class JunctionMaps(nn.Module):
    """2 m context (four dilated residual blocks, reach about 30 m), upsampled into the 1 m and 0.5 m features of the pyramid"""

    def __init__(self, c2: int = 144, c1: int = 48, c05: int = 32, d: int = 64):
        super().__init__()
        self.p2 = nn.Sequential(nn.Conv2d(c2, d, 1, bias=False), nn.GroupNorm(8, d), nn.ReLU(inplace=True))
        self.blocks = nn.ModuleList([ResDil(d, k) for k in (1, 2, 4, 8)])
        self.m1, self.head1 = block(c1 + d, d), nn.Conv2d(d, 8, 1)
        self.m05, self.head05 = block(c05 + d, 32), nn.Conv2d(32, 2, 1)

    def forward(self, P: dict) -> dict:
        up = lambda t: F.interpolate(t, scale_factor=2, mode="bilinear", align_corners=False)  # noqa: E731
        c = self.p2(P["feat2"])
        for b in self.blocks:
            c = b(c)
        x1 = self.m1(torch.cat([P["feat1"], up(c)], 1))
        return dict(j1=self.head1(x1), j05=self.head05(self.m05(torch.cat([P["feat05"], up(x1)], 1))), f1=x1)


def keypoints(jm: dict, cfg: Config) -> dict:
    """entry keypoints (first K) and exit keypoints (last K) of every frame:
    pos (B, 2K, 2) m, dir (B, 2K, 2) unit heading, p (B, 2K) heat probability, feat (B, 2K, 64), valid (B, 2, K), kind (B, 2K) 0 entry / 1 exit"""
    j1, f1 = jm["j1"], jm["f1"]
    B, K, dev = j1.shape[0], cfg.keypoints, j1.device
    c = cell_centres(N1, dev)
    off = torch.tensor([-1.0, 0.0, 1.0], device=dev)
    dx, dy = (t.flatten() for t in torch.meshgrid(off, off, indexing="ij"))
    out = dict(pos=[], dir=[], p=[], feat=[], valid=[])
    for k in (0, 1):
        pm = j1[:, 2 + k].sigmoid()
        peak = (pm >= F.max_pool2d(pm[:, None], 3, 1, 1)[:, 0]) & (pm > cfg.peak_min)
        v, ix = torch.where(peak, pm, torch.zeros_like(pm)).flatten(1).topk(K, dim=1)
        w = F.unfold(pm[:, None], 3, padding=1).gather(2, ix[:, None, :].expand(B, 9, -1))       # (B, 9, K) the 3 x 3 window, zero outside the map
        w = w / w.sum(1, keepdim=True).clamp_min(1e-6)
        x, y = c[ix // N1] + (w * dx[None, :, None]).sum(1), c[ix % N1] + (w * dy[None, :, None]).sum(1)
        out["pos"].append(torch.stack([x, y], -1))
        out["dir"].append(unit(j1[:, 4 + 2 * k: 6 + 2 * k]).flatten(2).gather(2, ix[:, None, :].expand(B, 2, -1)).transpose(1, 2))
        out["p"].append(pm.flatten(1).gather(1, ix))
        out["feat"].append(f1.flatten(2).gather(2, ix[:, None, :].expand(B, f1.shape[1], -1)).transpose(1, 2))
        out["valid"].append(v > 0)
    kp = {n: torch.cat(t, 1) for n, t in out.items() if n != "valid"}
    kp["valid"] = torch.stack(out["valid"], 1)
    kp["kind"] = torch.cat([torch.zeros(B, K, device=dev), torch.ones(B, K, device=dev)], 1)
    return kp
