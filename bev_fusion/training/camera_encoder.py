"""Front-camera encoder with pre-trained camera heads: image -> context features + depth distribution -> BEV (lift PD).

Layers and state-dict names follow the DualBEV single-frame R50 model (dualbev-r50-cbgs, Apache-2.0 code; its
DepthNet comes from BEVDepth, MIT), so the pre-trained image backbone, neck and DepthNet can be loaded and frozen;
only the BEV side is trained. Re-implemented in plain PyTorch (no mmcv / mmdet3d); see
bev_fusion/literature/code/dualbev.md (sections 2.1, 2.5) for the source and bev_fusion/plan.md section 2.

Pipeline (DualBEV main setting): image 256 x 704 (nuScenes 1600 x 900 resized by 0.44, top rows cropped) ->
ResNet-50 (stages 3 and 4, strides 16 / 32) -> CustomFPN to 512 channels at stride 16 (16 x 44) -> DepthNet with the
camera-aware gate (27-vector) -> 118 depth logits (1-60 m, 0.5 m bins, left edges) + 2 semantic logits + 80 context
channels -> lift PD (training/lift_pd.py) with the softmax depth at bin centres 1.25 + 0.5 i -> BEV (8 x 80 channels).

The key names are derived from the DualBEV code, not yet read from a checkpoint: load_pretrained() reports missing,
unexpected and shape-mismatched keys, and the first load of the real file is the check.
Run the self-test: python training/camera_encoder.py
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision

try:
    from .lift_pd import Grid, LiftPD
except ImportError:                                   # run as a script
    from lift_pd import Grid, LiftPD

DEPTH_BINS = (1.0, 60.0, 0.5)                         # DualBEV grid_config['depth']: 118 bins, left edges
IMG_MEAN = (123.675, 116.28, 103.53)                  # RGB, 0..255
IMG_STD = (58.395, 57.12, 57.375)


def depth_centres(d0=DEPTH_BINS[0], d1=DEPTH_BINS[1], step=DEPTH_BINS[2]):
    n = int(round((d1 - d0) / step))
    return d0 + step * (torch.arange(n, dtype=torch.float32) + 0.5)


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


def camera_vector(K, post_rot, post_tran, sensor2ego, bda=None):
    """DualBEV mlp_input (27 values per camera): fx fy cx cy, post_rot 00 01, post_tran x, post_rot 10 11,
    post_tran y, bda 00 01 10 11 22, sensor2ego 3 x 4 row-wise. Inputs: tensors (B, 3, 3), (B, 2, 2) or (B, 3, 3),
    (B, 2) or (B, 3), (B, 4, 4) camera -> vehicle; bda (B, 3, 3) or identity."""
    B = K.shape[0]
    if bda is None:
        bda = torch.eye(3, dtype=K.dtype, device=K.device).expand(B, 3, 3)
    v = [K[:, 0, 0], K[:, 1, 1], K[:, 0, 2], K[:, 1, 2],
         post_rot[:, 0, 0], post_rot[:, 0, 1], post_tran[:, 0], post_rot[:, 1, 0], post_rot[:, 1, 1], post_tran[:, 1],
         bda[:, 0, 0], bda[:, 0, 1], bda[:, 1, 0], bda[:, 1, 1], bda[:, 2, 2]]
    return torch.cat([torch.stack(v, 1), sensor2ego[:, :3, :4].reshape(B, 12)], 1)


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


def nuscenes_front_ida(src_hw=(900, 1600), dst_hw=(256, 704)):
    """DualBEV test-time image transform: resize by dst_w / src_w, crop the top rows -> 3x3 matrix (pixels)."""
    s = dst_hw[1] / src_hw[1]
    crop_top = int(round(src_hw[0] * s)) - dst_hw[0]
    return torch.tensor([[s, 0, 0], [0, s, -crop_top], [0, 0, 1]], dtype=torch.float32)


# ----------------------------------------------------------------------------------------------- self-test
def _selftest():
    torch.manual_seed(0)
    enc = FrontCameraEncoder().eval()
    n_backbone = sum(p.numel() for p in enc.img_backbone.parameters())
    n_heads = sum(p.numel() for m in (enc.img_neck, enc.depth_net) for p in m.parameters())
    # key names the DualBEV checkpoint should provide (derived from its code)
    keys = enc.state_dict().keys()
    for k in ("img_backbone.layer3.0.downsample.0.weight", "img_neck.lateral_convs.1.conv.weight",
              "img_neck.fpn_convs.0.conv.weight", "depth_net.bn.running_mean", "depth_net.depth_mlp.fc1.weight",
              "depth_net.depth_se.conv_reduce.weight", "depth_net.depth_conv.3.aspp2.atrous_conv.weight",
              "depth_net.depth_conv.4.weight", "depth_net.context_conv.weight"):
        assert k in keys, k
    # image transform: 1600 x 900 -> 704 x 396, 140 rows cropped at the top
    ida = nuscenes_front_ida()
    assert abs(ida[0, 0] - 0.44) < 1e-6 and ida[1, 2] == -140
    # front camera of a synthetic rig (camera 1.5 m high, looking along +x)
    K = torch.tensor([[1266.0, 0, 799.5], [0, 1266.0, 449.5], [0, 0, 1]])
    R = torch.tensor([[0.0, -1, 0], [0, 0, -1], [1, 0, 0]])
    T = torch.eye(4)
    T[:3, :3], T[:3, 3] = R, -R @ torch.tensor([0.0, 0, 1.5])
    s2e = torch.linalg.inv(T)
    B = 2
    cam = camera_vector(K.expand(B, 3, 3), ida[:2, :2].expand(B, 2, 2), ida[:2, 2].expand(B, 2), s2e.expand(B, 4, 4))
    assert cam.shape == (B, 27) and cam[0, 0] == 1266.0 and cam[0, 15 + 3] == s2e[0, 3]
    img = torch.randn(B, 3, 256, 704)
    with torch.no_grad():
        prob, ctx = enc.image_features(img, cam)
        assert prob.shape == (B, 118, 16, 44) and ctx.shape == (B, 80, 16, 44)
        assert torch.allclose(prob.sum(1), torch.ones(B, 16, 44), atol=1e-5)
        # one camera alone gives the same result as inside a batch (eval mode: no cross-sample statistics)
        p1, c1 = enc.image_features(img[:1], cam[:1])
        assert torch.allclose(p1, prob[:1], atol=1e-5) and torch.allclose(c1, ctx[:1], atol=1e-4)
        bev = enc(img, K, T, ida, cam)
    assert bev.shape == (B, 128, 120, 104)
    # frozen heads: only the BEV side trains
    enc.freeze_camera_heads()
    enc.train()
    trainable = sum(p.numel() for p in enc.parameters() if p.requires_grad)
    out = enc(img, K, T, ida, cam)
    out.mean().backward()
    assert enc.out_conv[0].weight.grad is not None and enc.depth_net.depth_conv[4].weight.grad is None
    assert not enc.depth_net.training
    print(f"camera_encoder selftest ok: backbone {n_backbone / 1e6:.1f} M, neck + DepthNet {n_heads / 1e6:.1f} M "
          f"parameters (frozen), trainable {trainable / 1e3:.0f} k; BEV {tuple(bev.shape)}")


if __name__ == "__main__":
    _selftest()
