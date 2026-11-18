"""The pixel stage (E273 stage A): one encoder-decoder on the 39 cached BEV channels, read at three resolutions.

    Pyramid()(x (B, 39, 200, 200)) -> dict
        centre1 (B, 12, 100, 100)  1 m logits: lane-centre probability, heading, offsets, keep gate (0-5) and six event heats (6-11)
        road2   (B, 2, 50, 50)     2 m logits: road, intersection
        feat2   (B, 144, 50, 50)   2 m features (decoder + pooled 1 m level)
        feat1   (B, 48, 100, 100)  1 m features
        feat05  (B, 32, 200, 200)  0.5 m features
The 0.5 m line maps the stage was also trained on are not computed (nothing in the champion reads them).
"""
import torch
import torch.nn as nn
import torch.nn.functional as F

from lg_layers import block

C_IN = 39


class Pyramid(nn.Module):
    def __init__(self, width: int = 48, f05: int = 32):
        super().__init__()
        w = width
        self.stem = block(C_IN, w, stride=2)          # 1 m
        self.e1 = block(w, 2 * w, stride=2)           # 2 m
        self.e2 = block(2 * w, 3 * w, stride=2)       # 4 m
        self.e3 = block(3 * w, 4 * w, stride=2)       # 8 m
        self.d2 = block(7 * w, 3 * w)
        self.d1 = block(5 * w, 2 * w)
        self.d0 = block(3 * w, w)
        self.centre1 = nn.Conv2d(w, 12, 1)
        self.road2 = nn.Conv2d(2 * w, 2, 1)
        self.s05 = nn.Sequential(nn.Conv2d(C_IN, f05, 3, 1, 1, bias=False), nn.GroupNorm(8, f05), nn.ReLU(inplace=True))
        self.m05 = block(w + f05, f05)

    def forward(self, x: torch.Tensor) -> dict:
        x = x[:, :C_IN]
        up = lambda t, ref: F.interpolate(t, size=ref.shape[-2:], mode="bilinear", align_corners=False)  # noqa: E731
        s = self.stem(x)
        a = self.e1(s)
        b = self.e2(a)
        c = self.e3(b)
        d1 = self.d1(torch.cat([up(self.d2(torch.cat([up(c, b), b], 1)), a), a], 1))
        d0 = self.d0(torch.cat([up(d1, s), s], 1))
        f05 = self.m05(torch.cat([up(d0, x), self.s05(x)], 1))
        return dict(centre1=self.centre1(d0), road2=self.road2(d1), feat2=torch.cat([d1, F.avg_pool2d(d0, 2)], 1), feat1=d0, feat05=f05)
