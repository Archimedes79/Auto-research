"""Front-camera encoder with pre-trained camera heads: image -> context features + depth distribution -> BEV (lift PD).

Layers and state-dict names follow the DualBEV single-frame R50 model (dualbev-r50-cbgs, Apache-2.0 code; its
DepthNet comes from BEVDepth, MIT), so the pre-trained image backbone, neck and DepthNet can be loaded and frozen;
only the BEV side is trained. Re-implemented in plain PyTorch (no mmcv / mmdet3d); see
bev_fusion/literature/code/dualbev.md (sections 2.1, 2.5) for the source.

Pipeline (DualBEV main setting): image 256 x 704 (nuScenes 1600 x 900 resized by 0.44, top rows cropped) ->
ResNet-50 (stages 3 and 4, strides 16 / 32) -> CustomFPN to 512 channels at stride 16 (16 x 44) -> DepthNet with the
camera-aware gate (27-vector) -> 118 depth logits (1-60 m, 0.5 m bins, left edges) + 2 semantic logits + 80 context
channels -> lift PD (net/model/lift_pd.py) with the softmax depth at bin centres 1.25 + 0.5 i -> BEV (8 x 80 channels).

The key names are derived from the DualBEV code, not yet read from a checkpoint: load_pretrained() reports missing,
unexpected and shape-mismatched keys, and the first load of the real file is the check.
Tests: tests/test_camera_encoder.py
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision

from bev_fusion.net.model.lift_pd import Grid, LiftPD
from bev_fusion.net.prep.geometry import DEPTH_BINS, depth_centres  # noqa: F401
from bev_fusion.net.prep.images import IMG_MEAN, IMG_STD, nuscenes_front_ida  # noqa: F401


# ------------------------------------------------------------------------------------------------ image neck
class ConvModule(nn.Module):
    """mmcv ConvModule without norm: keys <name>.conv.weight / .conv.bias."""

    def __init__(self, cin, cout, k, padding=0):
        super().__init__()
        self.conv = nn.Conv2d(cin, cout, k, padding=padding)

    def forward(self, x):
        return self.conv(x)


class CustomFPN(nn.Module):
    """BEVDet CustomFPN, in [1024, 2048] -> 512, one output at stride 16 (out_ids=[0])."""

    def __init__(self, in_channels=(1024, 2048), out_channels=512):
        super().__init__()
        self.lateral_convs = nn.ModuleList([ConvModule(c, out_channels, 1) for c in in_channels])
        self.fpn_convs = nn.ModuleList([ConvModule(out_channels, out_channels, 3, padding=1)])

    def forward(self, c3, c4):
        lat3, lat4 = self.lateral_convs[0](c3), self.lateral_convs[1](c4)
        lat3 = lat3 + F.interpolate(lat4, size=lat3.shape[-2:], mode="nearest")
        return self.fpn_convs[0](lat3)


# ------------------------------------------------------------------------------------------------ DepthNet
class Mlp(nn.Module):
    def __init__(self, cin, hidden, cout):
        super().__init__()
        self.fc1, self.act, self.drop1 = nn.Linear(cin, hidden), nn.ReLU(), nn.Dropout(0.0)
        self.fc2, self.drop2 = nn.Linear(hidden, cout), nn.Dropout(0.0)

    def forward(self, x):
        return self.drop2(self.fc2(self.drop1(self.act(self.fc1(x)))))


class SELayer(nn.Module):
    def __init__(self, c):
        super().__init__()
        self.conv_reduce, self.act1 = nn.Conv2d(c, c, 1, bias=True), nn.ReLU()
        self.conv_expand, self.gate = nn.Conv2d(c, c, 1, bias=True), nn.Sigmoid()

    def forward(self, x, x_se):
        return x * self.gate(self.conv_expand(self.act1(self.conv_reduce(x_se))))


class _ASPPModule(nn.Module):
    def __init__(self, cin, cout, k, padding, dilation):
        super().__init__()
        self.atrous_conv = nn.Conv2d(cin, cout, k, stride=1, padding=padding, dilation=dilation, bias=False)
        self.bn, self.relu = nn.BatchNorm2d(cout), nn.ReLU()

    def forward(self, x):
        return self.relu(self.bn(self.atrous_conv(x)))


class ASPP(nn.Module):
    def __init__(self, inplanes, mid):
        super().__init__()
        self.aspp1 = _ASPPModule(inplanes, mid, 1, 0, 1)
        self.aspp2 = _ASPPModule(inplanes, mid, 3, 6, 6)
        self.aspp3 = _ASPPModule(inplanes, mid, 3, 12, 12)
        self.aspp4 = _ASPPModule(inplanes, mid, 3, 18, 18)
        self.global_avg_pool = nn.Sequential(nn.AdaptiveAvgPool2d((1, 1)), nn.Conv2d(inplanes, mid, 1, bias=False),
                                             nn.BatchNorm2d(mid), nn.ReLU())
        self.conv1, self.bn1 = nn.Conv2d(mid * 5, inplanes, 1, bias=False), nn.BatchNorm2d(inplanes)
        self.relu, self.dropout = nn.ReLU(), nn.Dropout(0.5)

    def forward(self, x):
        x5 = F.interpolate(self.global_avg_pool(x), size=x.shape[-2:], mode="bilinear", align_corners=True)
        x = torch.cat([self.aspp1(x), self.aspp2(x), self.aspp3(x), self.aspp4(x), x5], 1)
        return self.dropout(self.relu(self.bn1(self.conv1(x))))


class DepthNet(nn.Module):
    """BEVDepth DepthNet as used by DualBEV: camera-aware SE gates on depth and context branches."""

    def __init__(self, cin=512, mid=512, context=80, depth=120, cam_dim=27):
        super().__init__()
        self.reduce_conv = nn.Sequential(nn.Conv2d(cin, mid, 3, padding=1), nn.BatchNorm2d(mid), nn.ReLU())
        self.context_conv = nn.Conv2d(mid, context, 1)
        self.bn = nn.BatchNorm1d(cam_dim)
        self.depth_mlp, self.depth_se = Mlp(cam_dim, mid, mid), SELayer(mid)
        self.context_mlp, self.context_se = Mlp(cam_dim, mid, mid), SELayer(mid)
        blocks = [torchvision.models.resnet.BasicBlock(mid, mid) for _ in range(3)]
        self.depth_conv = nn.Sequential(*blocks, ASPP(mid, mid), nn.Conv2d(mid, depth, 1))

    def forward(self, x, cam_vec):
        cam = self.bn(cam_vec)
        x = self.reduce_conv(x)
        context = self.context_conv(self.context_se(x, self.context_mlp(cam)[..., None, None]))
        depth = self.depth_conv(self.depth_se(x, self.depth_mlp(cam)[..., None, None]))
        return depth, context


# ------------------------------------------------------------------------------------------------ the encoder
class FrontCameraEncoder(nn.Module):
    """Image (B, 3, 256, 704), normalised -> BEV (B, out_channels, 120, 104) of the front camera.

    Pre-trained parts (load_pretrained, then freeze_camera_heads): img_backbone, img_neck, depth_net.
    Trained parts: out_conv (after height-into-channels) and, optionally, learned elevations of the lift.
    """

    def __init__(self, grid=None, out_channels=128, learn_heights=False):
        super().__init__()
        r = torchvision.models.resnet50(weights=None)
        self.img_backbone = nn.Module()
        for name in ("conv1", "bn1", "relu", "maxpool", "layer1", "layer2", "layer3", "layer4"):
            setattr(self.img_backbone, name, getattr(r, name))
        self.img_neck = CustomFPN()
        self.depth_net = DepthNet()
        self.n_depth = int(round((DEPTH_BINS[1] - DEPTH_BINS[0]) / DEPTH_BINS[2]))
        self.grid = grid or Grid()
        self.lift = LiftPD(self.grid, depth_centres(), learn_heights=learn_heights)
        nz = len(self.grid.z_levels)
        self.out_conv = nn.Sequential(nn.Conv2d(nz * 80, out_channels, 1, bias=False),
                                      nn.BatchNorm2d(out_channels), nn.ReLU())

    def image_features(self, img, cam_vec):
        b = self.img_backbone
        x = b.maxpool(b.relu(b.bn1(b.conv1(img))))
        c3 = b.layer3(b.layer2(b.layer1(x)))
        c4 = b.layer4(c3)
        depth, context = self.depth_net(self.img_neck(c3, c4), cam_vec)
        return depth[:, :self.n_depth].softmax(1), context            # channels n_depth.. are DualBEV's semantics

    def forward(self, img, K, T_cam_car, ida, cam_vec):
        """K (3x3) original intrinsics, T_cam_car (4x4), ida (3x3) image augmentation of this batch (one rig per
        batch: the table is shared), cam_vec (B, 27) from camera_vector()."""
        prob, context = self.image_features(img, cam_vec)
        bev = self.lift(context, prob, K, T_cam_car, img.shape[-2:], ida)
        return self.out_conv(bev)

    def freeze_camera_heads(self):
        """Pre-trained image backbone, neck and DepthNet: no gradient, BatchNorm in eval mode (the BatchNorm1d of
        the camera vector was fitted on six nuScenes cameras: keep its statistics)."""
        for m in (self.img_backbone, self.img_neck, self.depth_net):
            m.eval()
            for p in m.parameters():
                p.requires_grad_(False)

    def train(self, mode=True):
        super().train(mode)
        if not any(p.requires_grad for p in self.depth_net.parameters()):
            for m in (self.img_backbone, self.img_neck, self.depth_net):
                m.eval()
        return self

    def load_pretrained(self, path):
        """Load img_backbone., img_neck., img_view_transformer.depth_net. from a DualBEV checkpoint; returns a report
        (loaded, missing, unexpected-with-prefix, shape mismatches)."""
        try:                                       # third-party file: tensors only, no code; mmap keeps RAM low
            ck = torch.load(path, map_location="cpu", weights_only=True, mmap=True)
        except RuntimeError:                       # legacy (non-zip) format cannot be memory-mapped
            ck = torch.load(path, map_location="cpu", weights_only=True)
        sd = ck.get("state_dict", ck)
        mine = self.state_dict()
        take, mism, unexp = {}, [], []
        for k, v in sd.items():
            if k.startswith("img_view_transformer.depth_net."):
                k2 = "depth_net." + k[len("img_view_transformer.depth_net."):]
            elif k.startswith(("img_backbone.", "img_neck.")):
                k2 = k
            else:
                continue
            if k2 not in mine:
                unexp.append(k)
            elif mine[k2].shape != v.shape:
                mism.append((k, tuple(v.shape), tuple(mine[k2].shape)))
            else:
                take[k2] = v
        self.load_state_dict(take, strict=False)
        prefixes = ("img_backbone.", "img_neck.", "depth_net.")
        missing = [k for k in mine if k.startswith(prefixes) and k not in take]
        return {"loaded": len(take), "missing": missing, "unexpected": unexp, "shape_mismatch": mism}
