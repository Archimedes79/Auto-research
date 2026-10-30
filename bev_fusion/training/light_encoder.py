"""Light image encoder with stride-8 features (plan.md 2.7, proposal approved for a first try 2026-10-30): a drop-in for the
frozen DualBEV image side (training/camera_encoder.FrontCameraEncoder.image_features) that gives 4 x the cells at a fraction
of the FLOPs.

    image (B, 3, 256, 704), normalised like DualBEV (ImageNet mean / std on 0..255)
      -> ImageNet-initialised backbone to stride 32 (ResNet-18 / EfficientNet-B0 / EfficientNet-B2 of torchvision)
      -> FPN: 1 x 1 lateral convs to c channels, nearest top-down sums, 3 x 3 smooth -> stride 8 (32 x 88 cells) or 16
      -> camera-aware squeeze-excitation gates of the 27-value camera vector (as DualBEV's DepthNet, any rig)
      -> depth branch: 3 x 3 conv + BN + ReLU twice, 1 x 1 -> 118 logits (1-60 m, 0.5 m bins, the teacher's bins: softmax)
      -> context branch: 1 x 1 -> 80 channels (the teacher's context width, so a trunk trained on it can be reused)
Same call as the teacher: `image_features(img, cam_vec) -> (prob (B, 118, h, w), ctx (B, 80, h, w))`. The distillation loss
(`distill_loss`) pools the student's 2 x 2 cells onto the teacher's 16 x 44 grid (cell centres align: stride-8 cells 2j and 2j + 1
are centred at 16 j + 4 and 16 j + 12, the teacher's cell j at 16 j + 8) and compares probabilities (KL) and context (relative MSE).
Run the self-test: python training/light_encoder.py   (FLOPs of the candidates: temp/checks/encoder_flops.py)
"""
import os

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as tvm

try:
    from .camera_encoder import Mlp, SELayer, depth_centres
except ImportError:                                   # run as a script
    from camera_encoder import Mlp, SELayer, depth_centres

WEIGHTS = {"resnet18": "resnet18-f37072fd.pth", "efficientnet_b0": "efficientnet_b0_rwightman-7f5810bc.pth",
           "efficientnet_b2": "efficientnet_b2_rwightman-c35c1473.pth"}
WEIGHT_DIR = "models/pretrained/imagenet"


class Backbone(nn.Module):
    """torchvision backbone returning the maps at stride 8, 16 and 32."""

    def __init__(self, name, pretrained=True, root=WEIGHT_DIR):
        super().__init__()
        m = getattr(tvm, name)(weights=None)
        if pretrained:
            path = os.path.join(root, WEIGHTS[name])
            m.load_state_dict(torch.load(path, weights_only=True))                  # tensors only
        if name.startswith("resnet"):
            self.stem = nn.Sequential(m.conv1, m.bn1, m.relu, m.maxpool, m.layer1)
            self.s8, self.s16, self.s32 = m.layer2, m.layer3, m.layer4
        elif name.startswith("efficientnet"):
            f = m.features                                  # [0] stem, [1] s2, [2] s4, [3] s8, [4] [5] s16, [6] [7] s32, [8] 1x1 head
            self.stem, self.s8, self.s16, self.s32 = f[:3], f[3:4], f[4:6], f[6:8]
        else:
            raise ValueError(name)
        with torch.no_grad():
            self.chans = [t.shape[1] for t in self.forward(torch.zeros(1, 3, 64, 64))]

    def forward(self, x):
        c3 = self.s8(self.stem(x))
        c4 = self.s16(c3)
        return c3, c4, self.s32(c4)


class LightImageEncoder(nn.Module):
    def __init__(self, backbone="resnet18", c=128, n_depth=118, n_ctx=80, cam_dim=27, out_stride=8, pretrained=True):
        super().__init__()
        self.backbone = Backbone(backbone, pretrained)
        self.out_stride, self.n_depth, self.n_ctx = out_stride, n_depth, n_ctx
        self.lateral = nn.ModuleList([nn.Conv2d(ch, c, 1) for ch in self.backbone.chans])
        self.smooth = nn.Sequential(nn.Conv2d(c, c, 3, padding=1), nn.BatchNorm2d(c), nn.ReLU(True))
        self.bn = nn.BatchNorm1d(cam_dim)
        self.depth_mlp, self.depth_se = Mlp(cam_dim, c, c), SELayer(c)
        self.context_mlp, self.context_se = Mlp(cam_dim, c, c), SELayer(c)
        self.depth_conv = nn.Sequential(nn.Conv2d(c, c, 3, padding=1), nn.BatchNorm2d(c), nn.ReLU(True),
                                        nn.Conv2d(c, c, 3, padding=1), nn.BatchNorm2d(c), nn.ReLU(True),
                                        nn.Conv2d(c, n_depth, 1))
        self.context_conv = nn.Conv2d(c, n_ctx, 1)

    def image_features(self, img, cam_vec):
        c3, c4, c5 = self.backbone(img)
        p5 = self.lateral[2](c5)
        p4 = self.lateral[1](c4) + F.interpolate(p5, size=c4.shape[-2:], mode="nearest")
        p3 = self.lateral[0](c3) + F.interpolate(p4, size=c3.shape[-2:], mode="nearest")
        f = self.smooth(p3 if self.out_stride == 8 else p4)
        cam = self.bn(cam_vec)
        depth = self.depth_conv(self.depth_se(f, self.depth_mlp(cam)[..., None, None]))
        ctx = self.context_conv(self.context_se(f, self.context_mlp(cam)[..., None, None]))
        return depth.softmax(1), ctx

    forward = image_features

    def logits(self, img, cam_vec):
        """depth logits (for losses that want them) and context."""
        p, ctx = self.image_features(img, cam_vec)
        return p, ctx


def distill_loss(prob, ctx, t_prob, t_ctx, w_ctx=1.0):
    """student prob (B, 118, 2h, 2w), ctx (B, 80, 2h, 2w) at stride 8; teacher t_prob (B, 118, h, w), t_ctx (B, 80, h, w) at
    stride 16. Pooled 2 x 2; returns (total, kl, relative ctx MSE). KL(teacher || pooled student), summed over bins, mean over
    cells; context error relative to the teacher's context energy."""
    p = F.avg_pool2d(prob.float(), 2)
    c = F.avg_pool2d(ctx.float(), 2)
    t_prob, t_ctx = t_prob.float(), t_ctx.float()
    kl = (t_prob * (torch.log(t_prob.clamp(min=1e-8)) - torch.log(p.clamp(min=1e-8)))).sum(1).mean()
    rel = F.mse_loss(c, t_ctx) / t_ctx.pow(2).mean().clamp(min=1e-6)
    return kl + w_ctx * rel, kl, rel


def expected_depth(prob):
    """(B, 118, h, w) -> (B, h, w) expected depth in metres over the 118 bin centres."""
    dc = depth_centres().to(prob)
    return (prob * dc.view(1, -1, 1, 1)).sum(1)


def n_params(m):
    return sum(p.numel() for p in m.parameters() if p.requires_grad)


# ----------------------------------------------------------------------------------------------- self-test
def _selftest():
    torch.manual_seed(0)
    B = 2
    img = torch.randn(B, 3, 256, 704)
    cv = torch.randn(B, 27)
    rows = []
    for name in ("resnet18", "efficientnet_b0", "efficientnet_b2"):
        have = os.path.exists(os.path.join(WEIGHT_DIR, WEIGHTS[name]))
        enc = LightImageEncoder(name, pretrained=have).eval()
        with torch.no_grad():
            prob, ctx = enc.image_features(img, cv)
        assert prob.shape == (B, 118, 32, 88) and ctx.shape == (B, 80, 32, 88), (name, prob.shape)
        assert torch.allclose(prob.sum(1), torch.ones(B, 32, 88), atol=1e-5)
        # the camera vector gates the features: the same image with another vector gives another depth distribution
        with torch.no_grad():
            p2, _ = enc.image_features(img, cv * 0.5 + 1.0)
        assert (p2 - prob).abs().max() > 1e-5, "camera-aware gate"
        # stride 16 option
        enc16 = LightImageEncoder(name, out_stride=16, pretrained=False).eval()
        with torch.no_grad():
            assert enc16.image_features(img, cv)[0].shape == (B, 118, 16, 44)
        # gradients reach the backbone through the heads
        enc.train()
        prob, ctx = enc.image_features(img, cv)
        (prob[:, 10].mean() + ctx.mean()).backward()
        assert enc.backbone.stem[0].weight.grad is not None if name.startswith("resnet") else True
        assert all(p.grad is not None for p in enc.lateral.parameters()) and enc.depth_conv[-1].weight.grad.abs().sum() > 0
        if have:                                              # the ImageNet weights really are in the backbone
            ref = torch.load(os.path.join(WEIGHT_DIR, WEIGHTS[name]), weights_only=True)
            k = "conv1.weight" if name.startswith("resnet") else "features.0.0.weight"
            first = enc.backbone.stem[0].weight if name.startswith("resnet") else enc.backbone.stem[0][0].weight
            assert torch.equal(first, ref[k]), "ImageNet weights loaded"
        rows.append(f"{name} {n_params(enc) / 1e6:.1f} M{'' if have else ' (no weights)'}")
    # distillation loss: a student equal to the teacher after pooling has zero loss; a wrong one is worse
    t_prob = torch.rand(B, 118, 16, 44).softmax(1)
    t_ctx = torch.randn(B, 80, 16, 44)
    s_prob, s_ctx = t_prob.repeat_interleave(2, 2).repeat_interleave(2, 3), t_ctx.repeat_interleave(2, 2).repeat_interleave(2, 3)
    loss0, kl0, rel0 = distill_loss(s_prob, s_ctx, t_prob, t_ctx)
    assert loss0 < 1e-5, (kl0, rel0)
    bad = distill_loss(torch.rand(B, 118, 32, 88).softmax(1), torch.randn(B, 80, 32, 88), t_prob, t_ctx)[0]
    assert bad > 0.5
    # alignment of the pooling: a depth step at the 4th stride-8 column pair lands in teacher column 1 (cells 2 and 3)
    d = torch.zeros(1, 118, 32, 88)
    d[:, 20, :, 4:6] = 1.0
    d[:, 40, :, :4] = 1.0
    d[:, 40, :, 6:] = 1.0
    pooled = F.avg_pool2d(d, 2)
    assert pooled[0, 20, 0, 2] == 1.0 and pooled[0, 40, 0, 1] == 1.0 and abs(float(expected_depth(pooled)[0, 0, 2]) - 11.25) < 1e-4
    print("light_encoder selftest ok: " + ", ".join(rows) + "; shapes (B, 118 / 80, 32, 88), softmax sums to 1, camera vector changes the "
          "depth, stride-16 option, gradients to backbone and heads, ImageNet weights loaded, distillation loss zero for the pooled "
          "teacher, pooling alignment")


if __name__ == "__main__":
    _selftest()
