"""FLOPs and parameters of BEV trunk candidates on the 120 x 104 grid (hardware independent), plus the loading check of
the pre-trained DualBEV BEV encoder and an indicative GPU time.

FLOPs = 2 x multiply-adds, counted by torch.utils.flop_counter on one sample (conv / matmul only; upsampling,
BatchNorm, ReLU not counted). The GaussianCaR-style two-stream fuser is a convolution-only skeleton of the stages
described in literature/code/gaussiancar_fusion.md (stride-2 conv + 2 / 4 / 4 ResBlocks at 128 channels, per modality):
a LOWER bound, its rectify / cross-attention modules and the DPT head are not included.
Timing: the GPU is shared with the lane project's jobs, so the milliseconds are an upper bound, not a timing number.
Run from bev_fusion: wsl ~/miniconda3/envs/gcar/bin/python temp/checks/trunk_flops.py
"""
import sys
import time

import torch
import torch.nn as nn
from torch.utils.flop_counter import FlopCounterMode

sys.path.insert(0, "training")
from bev_trunk import BEVTrunk, n_params  # noqa: E402


def gflops(model, x):
    with torch.no_grad(), FlopCounterMode(display=False) as fc:
        model(x)
    return fc.get_total_flops() / 1e9


class ResBlock(nn.Module):
    def __init__(self, c):
        super().__init__()
        self.net = nn.Sequential(nn.Conv2d(c, c, 3, 1, 1), nn.ReLU(), nn.Conv2d(c, c, 3, 1, 1))

    def forward(self, x):
        return torch.relu(x + self.net(x))


class TwoStreamConvSkeleton(nn.Module):
    def __init__(self, c=128):
        super().__init__()

        def stream():
            blocks = [nn.Sequential(nn.Conv2d(c, c, 3, 2, 1), *[ResBlock(c) for _ in range(n)]) for n in (2, 4, 4)]
            return nn.Sequential(*blocks)
        self.cam, self.rad = stream(), stream()

    def forward(self, x):
        return self.cam(x), self.rad(x)


def ms(model, x, n=20):
    if not torch.cuda.is_available():
        return float("nan")
    model, x = model.cuda().eval().half(), x.cuda().half()
    with torch.no_grad():
        for _ in range(5):
            model(x)
        torch.cuda.synchronize()
        t = time.perf_counter()
        for _ in range(n):
            model(x)
        torch.cuda.synchronize()
    return (time.perf_counter() - t) / n * 1e3


rows = []
for name, cin, kw in [("T0 DualBEV trunk, width 1 (pre-trained shape)", 80, {}),
                      ("T1 width 1/2", 40, {"width": 0.5}),
                      ("T2 width 1/2 + light up", 40, {"width": 0.5, "light_up": True}),
                      ("T3 width 1/4 + light up", 32, {"width": 0.25, "light_up": True}),
                      ("T0 + light up (width 1)", 80, {"light_up": True}),
                      ("T2 + stride-1 detail path (32 ch)", 40, {"width": 0.5, "light_up": True, "detail": 32}),
                      ("T3 + stride-1 detail path (32 ch)", 32, {"width": 0.25, "light_up": True, "detail": 32}),
                      ("T2 + stride-1 detail path (16 ch)", 40, {"width": 0.5, "light_up": True, "detail": 16})]:
    m = BEVTrunk(cin=cin, n_classes=14, **kw).eval()
    x = torch.randn(1, cin, 120, 104)
    rows.append((name, gflops(m, x), n_params(m) / 1e6, ms(m, x)))
sk = TwoStreamConvSkeleton().eval()
rows.append(("G GaussianCaR two-stream fuser, conv skeleton (lower bound)", gflops(sk, torch.randn(1, 128, 120, 104)),
             n_params(sk) / 1e6, ms(sk, torch.randn(1, 128, 120, 104))))
print(f"{'candidate':62s} {'GFLOPs':>8s} {'M params':>9s} {'ms (shared GPU, fp16, upper bound)':>36s}")
for r in rows:
    print(f"{r[0]:62s} {r[1]:8.1f} {r[2]:9.2f} {r[3]:36.1f}")
t0 = BEVTrunk(cin=80, n_classes=14).eval()
sd_flops = {}
for pname, mod in [("backbone", t0.img_bev_encoder_backbone), ("neck", t0.img_bev_encoder_neck), ("head", t0.head)]:
    pass
x80 = torch.randn(1, 80, 120, 104)
with torch.no_grad():
    f = t0.img_bev_encoder_backbone(x80)
    gb = gflops(t0.img_bev_encoder_backbone, x80)
    gn = gflops(t0.img_bev_encoder_neck, f)
    gh = gflops(t0.head, torch.randn(1, 256, 120, 104))
print(f"\nT0 split: backbone {gb:.1f} GFLOPs, FPN neck {gn:.1f}, head {gh:.1f}")
if len(sys.argv) > 1:
    rep = BEVTrunk(cin=80, n_classes=14).load_pretrained("models/pretrained/dualbev-r50-cbgs.pth")
    print(f"pre-trained DualBEV BEV encoder: loaded {rep['loaded']}, missing {len(rep['missing'])}, "
          f"unexpected {len(rep['unexpected'])}, shape mismatch {len(rep['shape_mismatch'])}")
