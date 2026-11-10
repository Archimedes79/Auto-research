"""One-off: FLOPs (2 x MACs, as plan 6) of light image encoders with a stride-8 output at 256 x 704, against the current
DualBEV image side (ResNet-50 + CustomFPN + DepthNet at stride 16 = 71 GFLOPs, plan 6). Random weights (FLOPs do not depend on them).
Candidate = backbone to stride 32 + FPN (128 ch, output stride 8) + light heads at stride 8: depth (3x3 conv 128, 3x3 conv 128,
1x1 -> 118) and context (1x1 -> 80). Also the backbone alone and the heads at stride 16. 2 threads (shared box).
Run: wsl ~/miniconda3/envs/gcar/bin/python temp/checks/encoder_flops.py"""
import torch
import torch.nn as nn
import torchvision.models as tvm
from torch.utils.flop_counter import FlopCounterMode

torch.set_num_threads(2)


class Feats(nn.Module):
    """backbone returning (stride 8, 16, 32) maps"""
    def __init__(self, name):
        super().__init__()
        if name.startswith("resnet"):
            m = getattr(tvm, name)(weights=None)
            self.stem = nn.Sequential(m.conv1, m.bn1, m.relu, m.maxpool, m.layer1)
            self.s8, self.s16, self.s32 = m.layer2, m.layer3, m.layer4
        elif name.startswith("efficientnet"):
            m = getattr(tvm, name)(weights=None).features
            cut = {"efficientnet_b0": (3, 5, 9), "efficientnet_b2": (3, 5, 9), "efficientnet_b4": (3, 5, 9)}[name]
            self.stem, self.s8 = m[:cut[0]], m[cut[0]:cut[0] + 1]
            self.s16, self.s32 = m[cut[0] + 1:cut[1] + 1], m[cut[1] + 1:cut[2]]
        elif name.startswith("regnet"):
            m = getattr(tvm, name)(weights=None)
            self.stem = nn.Sequential(m.stem, m.trunk_output.block1)
            self.s8, self.s16, self.s32 = m.trunk_output.block2, m.trunk_output.block3, m.trunk_output.block4
        elif name.startswith("mobilenet"):
            m = tvm.mobilenet_v3_large(weights=None).features
            self.stem, self.s8, self.s16, self.s32 = m[:4], m[4:7], m[7:13], m[13:17]
        elif name.startswith("convnext"):
            m = tvm.convnext_tiny(weights=None).features
            self.stem, self.s8, self.s16, self.s32 = m[:2], m[2:4], m[4:6], m[6:8]
        self.name = name

    def forward(self, x):
        c2 = self.stem(x)
        c3 = self.s8(c2)
        c4 = self.s16(c3)
        c5 = self.s32(c4)
        return c3, c4, c5


class Head(nn.Module):
    def __init__(self, chans, c=128, stride=8, n_depth=118, n_ctx=80):
        super().__init__()
        self.lat = nn.ModuleList([nn.Conv2d(ch, c, 1) for ch in chans])
        self.smooth = nn.Conv2d(c, c, 3, padding=1)
        self.stride = stride
        self.depth = nn.Sequential(nn.Conv2d(c, c, 3, padding=1), nn.BatchNorm2d(c), nn.ReLU(), nn.Conv2d(c, c, 3, padding=1),
                                   nn.BatchNorm2d(c), nn.ReLU(), nn.Conv2d(c, n_depth, 1))
        self.ctx = nn.Conv2d(c, n_ctx, 1)

    def forward(self, c3, c4, c5):
        p5 = self.lat[2](c5)
        p4 = self.lat[1](c4) + nn.functional.interpolate(p5, size=c4.shape[-2:], mode="nearest")
        p3 = self.lat[0](c3) + nn.functional.interpolate(p4, size=c3.shape[-2:], mode="nearest")
        f = self.smooth(p3 if self.stride == 8 else p4)
        return self.depth(f), self.ctx(f)


def flops(fn, *a):
    with FlopCounterMode(display=False) as fc, torch.no_grad():
        fn(*a)
    return fc.get_total_flops() / 1e9


x = torch.randn(1, 3, 256, 704)
print("DualBEV image side now (plan 6): ResNet-50 29.4 + neck 4.4 + DepthNet 35.6 = 71.4 GFLOPs at stride 16 (16 x 44 cells)")
print(f"{'backbone':22s} {'params M':>8s} {'backbone GF':>11s} {'+ FPN/heads s8':>14s} {'(s8 cells)':>10s} {'+ heads s16':>11s}")
for name in ("resnet18", "resnet34", "resnet50", "efficientnet_b0", "efficientnet_b2", "efficientnet_b4", "regnet_y_800mf",
             "regnet_y_1_6gf", "mobilenet_v3_large", "convnext_tiny"):
    try:
        f = Feats(name).eval()
        chans = [t.shape[1] for t in f(x)]
        shapes = [tuple(t.shape[-2:]) for t in f(x)]
        gb = flops(f, x)
        h8, h16 = Head(chans, stride=8).eval(), Head(chans, stride=16).eval()
        feats = f(x)
        g8, g16 = flops(h8, *feats), flops(h16, *feats)
        npar = sum(p.numel() for p in f.parameters()) / 1e6 + sum(p.numel() for p in h8.parameters()) / 1e6
        print(f"{name:22s} {npar:8.1f} {gb:11.1f} {gb + g8:14.1f} {str(shapes[0]):>10s} {gb + g16:11.1f}   chans {chans}")
    except Exception as e:                                                      # noqa: BLE001
        print(name, "failed:", str(e)[:100])
