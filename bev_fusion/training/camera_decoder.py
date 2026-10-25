"""Camera-only BEV decoder (Mirko, 2026-10-25): the mirror of training/radar_decoder.py. Kept as an auxiliary branch of
the fusion network (an extra loss per sensor, and a probe for sensor-specific failures) and testable on its own.

Same techniques as the fusion decoder (plan.md sections 2, 6-8), only the radar branch is missing:
- image encoder (swappable): SmallImageEncoder (ResNet-18 stages to stride 8, from scratch, for the synthetic world)
  or the pre-trained FrontCameraEncoder of training/camera_encoder.py (nuScenes); both give a depth distribution over
  the depth bins and context channels per feature cell
- lift PD (training/lift_pd.py) with a height slice at z = 0 and vertical oversampling (the oracle-lift gate's fix)
- time: the past frames' BEVs are moved to the current time by the ego motion with ONE splat each
  (training/splat.py splat_window, mass channel kept); the trunk learns object motion from the displacement
- trunk: training/bev_trunk.py; heads: a semantic head (lane lines, vehicles, pedestrians) and the object head of
  training/object_head.py for ALL objects (static and moving: a camera sees both)
Run the self-test: python training/camera_decoder.py
"""
import torch
import torch.nn as nn
import torchvision

try:
    from .bev_trunk import BEVTrunk
    from .lift_pd import Grid, LiftPD
    from .object_head import GROUPS, ObjectHead
    from .splat import splat_window
except ImportError:                                   # run as a script
    from bev_trunk import BEVTrunk
    from lift_pd import Grid, LiftPD
    from object_head import GROUPS, ObjectHead
    from splat import splat_window

DEPTH_CENTRES = 1.25 + 0.5 * torch.arange(118, dtype=torch.float32)    # the camera head's bins (1-60 m, 0.5 m)
Z_LEVELS = (-0.5, 0.0, 0.5, 1.0, 1.5, 2.0, 2.5, 3.0)                   # with a slice at the road plane z = 0
SEM_CLASSES = ("lanes", "vehicles", "pedestrians")


class SmallImageEncoder(nn.Module):
    """ResNet-18 stem + layer1 + layer2 (stride 8), random init -> depth logits (D) and context (C) per feature cell."""

    def __init__(self, n_depth=118, context=32):
        super().__init__()
        r = torchvision.models.resnet18(weights=None)
        self.body = nn.Sequential(r.conv1, r.bn1, r.relu, r.maxpool, r.layer1, r.layer2)
        self.neck = nn.Sequential(nn.Conv2d(128, 128, 3, padding=1), nn.BatchNorm2d(128), nn.ReLU())
        self.depth, self.context = nn.Conv2d(128, n_depth, 1), nn.Conv2d(128, context, 1)
        self.stride, self.out_context = 8, context

    def forward(self, img):
        f = self.neck(self.body(img))
        return self.depth(f), self.context(f)


class CameraOnlyDecoder(nn.Module):
    """images (B, F, 3, H, W) of one camera, frame 0 = current -> dict: 'sem' (B, 3, X, Y) logits, the object head's
    maps, 'depth_logits' (B, F, D, h, w)."""

    def __init__(self, encoder=None, grid=None, bev_channels=32, width=0.5, oversample=4, groups=GROUPS):
        super().__init__()
        self.grid = grid or Grid(z_levels=Z_LEVELS)
        self.encoder = encoder or SmallImageEncoder()
        self.lift = LiftPD(self.grid, DEPTH_CENTRES, oversample=oversample)
        nz = len(self.grid.z_levels)
        self.reduce = nn.Sequential(nn.Conv2d(nz * self.encoder.out_context, bev_channels, 1), nn.BatchNorm2d(bev_channels), nn.ReLU())
        self.bev_channels = bev_channels
        self.trunk = BEVTrunk(cin=bev_channels * 2 + 1, width=width)       # [current, past moved to now, past mass]
        c = self.trunk.img_bev_encoder_neck.out_channels
        self.sem_head = nn.Sequential(nn.Conv2d(c, 64, 3, padding=1), nn.ReLU(), nn.Conv2d(64, len(SEM_CLASSES), 1))
        nn.init.constant_(self.sem_head[2].bias, -2.19)
        self.objects = ObjectHead(c, groups)

    def forward(self, img, K, T_cam_car, ida, steps):
        """K, T_cam_car, ida: the rig (one per batch); steps: per frame the ego steps from that frame to now
        (frame 0: []), e.g. [[], [(1/12, v, w)] * 6] for the current frame and one 0.5 s old."""
        B, Fn = img.shape[:2]
        dlog, ctx = self.encoder(img.flatten(0, 1))
        prob = dlog.softmax(1)
        hw = img.shape[-2:]
        bev = self.reduce(self.lift(ctx, prob, K, T_cam_car, hw, ida))           # (B*F, C, X, Y)
        bev = bev.view(B, Fn, *bev.shape[1:])
        cur = bev[:, 0]
        if Fn > 1:
            out_w, mass_w = splat_window([(bev[:, k], steps[k]) for k in range(1, Fn)], self.grid)
            past, mass = out_w.mean(1), mass_w.mean(1).clamp(max=1.0)            # several past frames: averaged
        else:
            past, mass = torch.zeros_like(cur), torch.zeros_like(cur[:, :1])
        f = self.trunk.features(torch.cat([cur, past, mass], 1))
        out = self.objects(f)
        out["sem"] = self.sem_head(f)
        out["depth_logits"] = dlog.view(B, Fn, *dlog.shape[1:])
        return out


def _selftest():
    torch.manual_seed(0)
    model = CameraOnlyDecoder()
    K = torch.tensor([[280.0, 0, 175.5], [0, 280.0, 63.5], [0, 0, 1]])
    R = torch.tensor([[0.0, -1, 0], [0, 0, -1], [1, 0, 0]])
    T = torch.eye(4)
    T[:3, :3], T[:3, 3] = R, -R @ torch.tensor([1.5, 0.0, 1.5])
    img = torch.randn(2, 2, 3, 128, 352)
    out = model(img, K, T, None, [[], [(1 / 12, 10.0, 0.1)] * 6])
    assert out["sem"].shape == (2, 3, 120, 104) and out["hm"].shape == (2, 3, 120, 104)
    assert out["depth_logits"].shape == (2, 2, 118, 16, 44)
    (out["sem"].mean() + out["hm"].mean() + out["depth_logits"].mean()).backward()
    assert model.encoder.depth.weight.grad is not None and model.reduce[0].weight.grad.abs().sum() > 0
    print("camera_decoder selftest ok: shapes of the semantic map, the object maps and the depth logits; gradients "
          "reach the image encoder through the lift and the splat")


if __name__ == "__main__":
    _selftest()
