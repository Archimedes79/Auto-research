"""S1 camera-only BEV segmentation module (plan.md section 11 stage S1; experiments/s1a_lift_pd_vs_p): cached image-head outputs
-> lift -> BEV trunk -> semantic head, with ONE config switch between our design and the published designs it must not
drift away from (Mirko, 2026-10-26: "make sure that we do not go away totally from SOA").

    prob (B, D, h, w) depth distribution + ctx (B, C, h, w) context per feature cell   <- the frozen DualBEV image heads
    (real: training/feature_cache.py; synthetic: studies/early/synthetic_s1_data.py - the SAME dict layout)
      -> lift (per sample calibration): PD pull with depth weight / P pull without depth / S splat (LSS)
      -> height: kept as channels (feat[z_car], Simple-BEV / ours) or summed (DualBEV's bev_pool over its 13 heights, LSS)
      -> 1x1 reduce (skipped when the summed volume already has the trunk's input width, as DualBEV)
      -> BEVTrunk (ResNet pyramid + FPN_LSS, DualBEV's BEV encoder; width 1.0 = its 26.7 M parameters)
      -> semantic head (3x3 conv + 1x1, multi-label logits)

PRESETS (everything not listed is the same, so a preset differs from another in exactly the named choices):
  ours        lift PD, 8 slices incl. z = 0 with 4 sub-heights each (oracle-lift gate fix), heights as channels, 1x1 to 64
              channels, trunk width 0.5 (T2 of plan 6)
  dualbev     DualBEV's HeightTrans as far as the code note says (literature/code/dualbev.md 2.2): 13 fixed heights (-2 .. 2 m in
              steps of 1 and -4.5 .. 2.5 m in steps of 1 in the lidar frame, +1.84 m here: the lidar is about 1.84 m above the
              ground, from memory), depth-weighted gather, heights SUMMED to 80 channels (no reduce), trunk width 1.0.
              Deviations: our gather interpolates (DualBEV rounds to the nearest bin), no depth-threshold filter, no
              LSS branch / fusion / BEV probability (those are its detection design)
  simplebev   lift P (pull, no depth weight) on Simple-BEV's choice: the 8 default slices as channels, no oversampling
  lss         lift S (forward splat of every pixel x bin, BEVPoolV2's sum), heights summed, trunk width 0.5
`CameraBEVSeg(S1Config.preset("dualbev"))` etc.; any field can be overridden. Learned parts: reduce, trunk, head (the image
heads are cached and frozen). Run the self-test: python training/legacy/s1_camera.py
"""
import os
import sys
from dataclasses import dataclass, field, replace

import torch
import torch.nn as nn

_TRAINING = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))     # moved to training/legacy/ (2026-10-31): the live blocks
if _TRAINING not in sys.path:
    sys.path.insert(0, _TRAINING)
from bev_trunk import BEVTrunk, SemanticHead  # noqa: E402
from camera_encoder import depth_centres  # noqa: E402
from fallbacks import LiftS  # noqa: E402
from lift_pd import Grid, LiftPD  # noqa: E402

Z_OURS = (-0.5, 0.0, 0.5, 1.0, 1.5, 2.0, 2.5, 3.0)                       # a slice at the road plane z = 0
Z_GRID = Grid().z_levels                                                   # (-0.75 ... 2.75): Simple-BEV-like 8 heights
LIDAR_HEIGHT = 1.84                                                        # m above the ground (from memory)
Z_DUALBEV = tuple(sorted(z + LIDAR_HEIGHT for z in (-4.5, -3.5, -2.5, -1.5, -0.5, 0.5, 1.5, 2.5, -2.0, -1.0, 0.0, 1.0, 2.0)))


@dataclass
class S1Config:
    lift: str = "pd"                    # pd | p | s
    z_levels: tuple = Z_OURS
    oversample: int = 4                 # sub-heights per slice (pd / p); 1 = the published point sample
    height: str = "channels"            # channels (feat[z_car]) | sum (heights collapsed by summation)
    n_ctx: int = 80
    c_bev: int = 64                     # trunk input width (with height = sum and c_bev == n_ctx: no reduce)
    trunk_width: float = 0.5
    detail: int = 0                     # stride-1 detail path of the trunk (plan 6)
    n_classes: int = 3
    head_bias: float = -3.0             # rare classes: start at p = 0.05
    grid_kwargs: dict = field(default_factory=dict)

    @staticmethod
    def preset(name, **kw):
        base = {"ours": dict(),
                "dualbev": dict(lift="pd", z_levels=Z_DUALBEV, oversample=1, height="sum", c_bev=80, trunk_width=1.0),
                "simplebev": dict(lift="p", z_levels=Z_GRID, oversample=1),
                "lss": dict(lift="s", z_levels=Z_GRID, oversample=1, height="sum", c_bev=80)}[name]
        return replace(S1Config(), **{**base, **kw})


class CameraBEVSeg(nn.Module):
    def __init__(self, cfg=None):
        super().__init__()
        self.cfg = cfg = cfg or S1Config()
        self.grid = Grid(z_levels=cfg.z_levels, **cfg.grid_kwargs)
        if cfg.lift in ("pd", "p"):
            self.lift = LiftPD(self.grid, depth_centres(), use_depth=(cfg.lift == "pd"), oversample=cfg.oversample)
        elif cfg.lift == "s":
            self.lift = LiftS(self.grid, depth_centres())
        else:
            raise ValueError(cfg.lift)
        nz = len(cfg.z_levels)
        c_in = nz * cfg.n_ctx if cfg.height == "channels" else cfg.n_ctx
        if cfg.height not in ("channels", "sum"):
            raise ValueError(cfg.height)
        if c_in == cfg.c_bev:
            self.reduce = nn.Identity()
        else:
            self.reduce = nn.Sequential(nn.Conv2d(c_in, cfg.c_bev, 1, bias=False), nn.BatchNorm2d(cfg.c_bev), nn.ReLU(True))
        self.trunk = BEVTrunk(cin=cfg.c_bev, width=cfg.trunk_width, detail=cfg.detail)
        c = self.trunk.img_bev_encoder_neck.out_channels + cfg.detail
        del self.trunk.head
        self.head = SemanticHead(c, cfg.n_classes)
        nn.init.constant_(self.head.net[-1].bias, cfg.head_bias)

    def lifted(self, prob, ctx, K, T_cam_car, hw, ida=None):
        """(B, c_in, X, Y): the lift per sample (one calibration each), heights as channels or summed."""
        out = []
        for b in range(ctx.shape[0]):
            v = self.lift(ctx[b:b + 1], prob[b:b + 1], K[b], T_cam_car[b], hw, ida)           # (1, Z * C, X, Y)
            if self.cfg.height == "sum":
                v = v.view(1, len(self.cfg.z_levels), self.cfg.n_ctx, *v.shape[-2:]).sum(1)
            out.append(v)
        return torch.cat(out)

    def forward(self, batch):
        """batch: prob (B, D, h, w), ctx (B, C, h, w), K (B, 3, 3), T_cam_car (B, 4, 4), hw, ida -> logits (B, n_classes, X, Y)."""
        v = self.lifted(batch["prob"], batch["ctx"], batch["K"], batch["T_cam_car"], batch["hw"], batch.get("ida"))
        return self.head(self.trunk.features(self.reduce(v)))


def n_params(m):
    return sum(p.numel() for p in m.parameters() if p.requires_grad)


# ----------------------------------------------------------------------------------------------- self-test
def _rig():
    K = torch.tensor([[560.0, 0, 351.5], [0, 560.0, 127.5], [0, 0, 1]])
    R = torch.tensor([[0.0, -1, 0], [0, 0, -1], [1, 0, 0]])
    T = torch.eye(4)
    T[:3, :3], T[:3, 3] = R, -R @ torch.tensor([1.5, 0.0, 1.5])
    return K, T


def _selftest():
    torch.manual_seed(0)
    K, T = _rig()
    B = 2
    batch = {"prob": torch.rand(B, 118, 16, 44).softmax(1), "ctx": torch.randn(B, 80, 16, 44),
             "K": K.expand(B, 3, 3), "T_cam_car": T.expand(B, 4, 4), "hw": (256, 704), "ida": None}
    rows = []
    for name in ("ours", "dualbev", "simplebev", "lss"):
        cfg = S1Config.preset(name)
        m = CameraBEVSeg(cfg)
        assert len(cfg.z_levels) in (8, 13)
        out = m(batch)
        assert out.shape == (B, 3, 120, 104) and torch.isfinite(out).all(), (name, out.shape)
        out.mean().backward()
        assert any(p.grad is not None and p.grad.abs().sum() > 0 for p in m.trunk.parameters()), name
        assert m.head.net[-1].weight.grad.abs().sum() > 0
        rows.append(f"{name} {n_params(m) / 1e6:.1f} M")
    # presets differ in exactly the named choices
    o, d = S1Config.preset("ours"), S1Config.preset("dualbev")
    assert len(d.z_levels) == 13 and abs(min(d.z_levels) - (-4.5 + LIDAR_HEIGHT)) < 1e-9 and d.height == "sum"
    assert o.height == "channels" and o.oversample == 4 and d.oversample == 1 and S1Config.preset("lss").lift == "s"
    # dualbev: the summed volume feeds the trunk directly (80 channels, no reduce, DualBEV's BEV encoder input)
    md = CameraBEVSeg(d)
    assert isinstance(md.reduce, nn.Identity) and md.trunk.img_bev_encoder_backbone.chans[0] == 160
    # PD with a one-hot depth: only voxels at that depth receive features (the summed variant too)
    prob = torch.zeros(1, 118, 16, 44)
    prob[:, 38] = 1.0                                                            # bin centre 1.25 + 0.5 * 38 = 20.25 m
    ctx = torch.ones(1, 80, 16, 44)
    v = md.lifted(prob, ctx, K[None], T[None], (256, 704))
    xs = (torch.arange(120) + 0.5) * 0.5                                         # depth along the optical axis = x - 1.5
    off = ((xs - 1.5 - 20.25).abs() > 0.55).view(1, 1, -1, 1)                    # farther than one bin from the peak
    assert v.shape == (1, 80, 120, 104) and v.amax() > 0.5 and (v.abs() * off).amax() < 1e-6, "one-hot depth support"
    # learned parts only: no image-head parameters exist in the module
    assert not any("depth_net" in n or n.startswith("camera") for n, _ in m.named_parameters())
    print("s1_camera selftest ok: presets run with shapes (B, 3, 120, 104) and gradients to trunk and head ("
          + ", ".join(rows) + "); presets differ in the named choices only; DualBEV preset: 13 heights summed to 80 "
          "channels without reduce; one-hot depth reaches only its own depth")


if __name__ == "__main__":
    _selftest()
