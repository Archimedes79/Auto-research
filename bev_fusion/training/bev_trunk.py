"""BEV trunk after the fusion: ResNet pyramid + FPN decoder + semantic head - the standard flow of the camera(-radar)
BEV networks (BEVDet / BEVDepth / CRN / DualBEV: `CustomResNet` + `FPN_LSS`), in plain PyTorch with a width factor.

Layer names follow DualBEV's `img_bev_encoder_backbone.` / `img_bev_encoder_neck.` (26.7 M parameters at width 1), so
at width 1 the pre-trained trunk can be loaded (`load_pretrained`; detection-trained, an ablation, not a given).

Flow (grid 120 x 104 at 0.5 m; input = the fused map, c_in channels, 80 for the pre-trained trunk):
  stage 0: BasicBlock stride 2 + BasicBlock        c0 = 160 w   60 x 52   (stride 2)
  stage 1: BasicBlock stride 2 + BasicBlock        c1 = 320 w   30 x 26   (stride 4)
  stage 2: BasicBlock stride 2 + BasicBlock        c2 = 640 w   15 x 13   (stride 8)
  FPN_LSS: up(stage 2, x4) ++ stage 0 -> 3x3 conv -> 3x3 conv (mid = 512 w) -> up x2 -> 3x3 conv (256 w) -> 1x1
  head:    3x3 conv BN ReLU + 1x1 -> n_classes at the full grid (0.5 m)
`light_up=True` replaces the full-resolution 3x3 conv of the last upsampling (the most expensive layer) by a
sub-pixel conv at stride 2 (a 1x1 to 4 x channels at stride 2, then pixel shuffle), the same output at about 1/6 of
the cost of that layer; not loadable from the pre-trained trunk.
Run the self-test: python training/bev_trunk.py   (FLOP table: temp/checks/trunk_flops.py)
"""
import torch
import torch.nn as nn
import torch.nn.functional as F


class BasicBlock(nn.Module):
    """BEVDet / mmdet BasicBlock (keys conv1, bn1, conv2, bn2, downsample as a plain 3x3 stride-2 conv with bias)."""

    def __init__(self, cin, cout, stride=1, downsample=None):
        super().__init__()
        self.conv1 = nn.Conv2d(cin, cout, 3, stride, 1, bias=False)
        self.bn1 = nn.BatchNorm2d(cout)
        self.conv2 = nn.Conv2d(cout, cout, 3, 1, 1, bias=False)
        self.bn2 = nn.BatchNorm2d(cout)
        self.downsample = downsample

    def forward(self, x):
        idt = x if self.downsample is None else self.downsample(x)
        y = F.relu(self.bn1(self.conv1(x)))
        return F.relu(self.bn2(self.conv2(y)) + idt)


class BEVBackbone(nn.Module):
    """CustomResNet: three stages of two BasicBlocks, stride 2 each; returns the three stage outputs."""

    def __init__(self, cin=80, width=1.0):
        super().__init__()
        chans = [int(round(c * width)) for c in (160, 320, 640)]
        self.chans = chans
        layers, cur = [], cin
        for c in chans:
            ds = nn.Conv2d(cur, c, 3, 2, 1)
            layers.append(nn.Sequential(BasicBlock(cur, c, 2, ds), BasicBlock(c, c)))
            cur = c
        self.layers = nn.Sequential(*layers)

    def forward(self, x):
        feats = []
        for layer in self.layers:
            x = layer(x)
            feats.append(x)
        return feats


class FPNLSS(nn.Module):
    """BEVDet FPN_LSS: deepest stage upsampled x4 and concatenated with stage 0, two 3x3 convs, upsample x2, 3x3 conv
    and 1x1 conv to out channels at the full grid (light_up: sub-pixel conv instead of the full-resolution 3x3)."""

    def __init__(self, chans, width=1.0, light_up=False):
        super().__init__()
        c0, _, c2 = chans
        mid, out = int(round(512 * width)), int(round(256 * width))
        self.up1 = nn.Upsample(scale_factor=4, mode="bilinear", align_corners=True)
        self.conv = nn.Sequential(nn.Conv2d(c0 + c2, mid, 3, 1, 1, bias=False), nn.BatchNorm2d(mid), nn.ReLU(True),
                                  nn.Conv2d(mid, mid, 3, 1, 1, bias=False), nn.BatchNorm2d(mid), nn.ReLU(True))
        if light_up:
            self.up2 = nn.Sequential(nn.Conv2d(mid, 4 * out, 1, bias=False), nn.PixelShuffle(2),
                                     nn.BatchNorm2d(out), nn.ReLU(True), nn.Conv2d(out, out, 1))
        else:
            self.up2 = nn.Sequential(nn.Upsample(scale_factor=2, mode="bilinear", align_corners=True),
                                     nn.Conv2d(mid, out, 3, 1, 1, bias=False), nn.BatchNorm2d(out), nn.ReLU(True),
                                     nn.Conv2d(out, out, 1))
        self.out_channels = out

    def forward(self, feats):
        x = torch.cat([self.up1(feats[2]), feats[0]], 1)
        return self.up2(self.conv(x))


class SemanticHead(nn.Module):
    """3x3 conv BN ReLU + 1x1 to n_classes (logits, multi-label: sigmoid)."""

    def __init__(self, cin, n_classes, hidden=None):
        super().__init__()
        hidden = hidden or cin
        self.net = nn.Sequential(nn.Conv2d(cin, hidden, 3, 1, 1, bias=False), nn.BatchNorm2d(hidden), nn.ReLU(True),
                                 nn.Conv2d(hidden, n_classes, 1))

    def forward(self, x):
        return self.net(x)


class BEVTrunk(nn.Module):
    def __init__(self, cin=80, n_classes=14, width=1.0, light_up=False, head_hidden=None):
        super().__init__()
        self.img_bev_encoder_backbone = BEVBackbone(cin, width)
        self.img_bev_encoder_neck = FPNLSS(self.img_bev_encoder_backbone.chans, width, light_up)
        self.head = SemanticHead(self.img_bev_encoder_neck.out_channels, n_classes, head_hidden)

    def features(self, x):
        return self.img_bev_encoder_neck(self.img_bev_encoder_backbone(x))

    def forward(self, x):
        return self.head(self.features(x))

    def load_pretrained(self, path):
        """Load img_bev_encoder_backbone. / img_bev_encoder_neck. from a DualBEV checkpoint (width 1, no light_up);
        returns the report (loaded, missing, unexpected, shape mismatches), as camera_encoder.load_pretrained."""
        sd = torch.load(path, map_location="cpu", weights_only=True)
        sd = sd.get("state_dict", sd)
        mine = self.state_dict()
        take, mism, unexp = {}, [], []
        for k, v in sd.items():
            if k.startswith(("img_bev_encoder_backbone.", "img_bev_encoder_neck.")):
                if k not in mine:
                    unexp.append(k)
                elif mine[k].shape != v.shape:
                    mism.append((k, tuple(v.shape), tuple(mine[k].shape)))
                else:
                    take[k] = v
        self.load_state_dict(take, strict=False)
        missing = [k for k in mine if k.startswith(("img_bev_encoder_backbone.", "img_bev_encoder_neck.")) and k not in take]
        return {"loaded": len(take), "missing": missing, "unexpected": unexp, "shape_mismatch": mism}


def n_params(m):
    return sum(p.numel() for p in m.parameters())


def _selftest():
    torch.manual_seed(0)
    x = torch.randn(2, 80, 120, 104)
    trunk = BEVTrunk(cin=80, n_classes=14).eval()
    with torch.no_grad():
        feats = trunk.img_bev_encoder_backbone(x)
        assert [tuple(f.shape[1:]) for f in feats] == [(160, 60, 52), (320, 30, 26), (640, 15, 13)]
        y = trunk(x)
    assert y.shape == (2, 14, 120, 104)
    enc = n_params(trunk.img_bev_encoder_backbone) + n_params(trunk.img_bev_encoder_neck)
    stats = sum(v.numel() for k, v in trunk.state_dict().items()
                if k.startswith(("img_bev_encoder_backbone.", "img_bev_encoder_neck.")) and "num_batches" not in k)
    assert stats == 26668896, stats          # parameters + BatchNorm statistics = the tensors of DualBEV's BEV encoder
    # the light variants keep the output shape; widths scale the parameters roughly with w^2
    for kw in ({"width": 0.5}, {"width": 0.5, "light_up": True}, {"width": 0.25, "light_up": True}):
        t = BEVTrunk(cin=48, n_classes=14, **kw).eval()
        with torch.no_grad():
            assert t(torch.randn(1, 48, 120, 104)).shape == (1, 14, 120, 104)
    # gradient reaches the input and every parameter
    trunk.train()
    trunk(x).mean().backward()
    assert all(p.grad is not None for p in trunk.parameters())
    print(f"bev_trunk selftest ok: shapes of the three stages, output (2, 14, 120, 104), {enc / 1e6:.2f} M encoder "
          f"parameters = DualBEV's, light variants, gradients")


if __name__ == "__main__":
    _selftest()
