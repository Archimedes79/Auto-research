"""Lane branch (experiment t7): stride-4 image features of the CURRENT camera frame lifted into the BEV next to the stride-16
context of the main camera branch.

Why: the t5 probe showed that the stride-16 context (what lift PD receives) draws painted lines as blurred bands while the
stride-4 stage of the same frozen ResNet-50 and an RGB CNN draw them sharply; one stride-16 cell is 0.57 m x 7.9 m on the ground
at 20 m, a stride-4 cell 0.14 m x 2 m.

    image (B, 3, 256, 704) -> features at stride 4 (64 x 176): frozen DualBEV ResNet-50 stem + layer 1 ("res4", 256 ch)
                                                               or an RGB CNN from scratch ("rgb", 64 ch)
      -> 1 x 1 conv to 32, BN, ReLU, 3 x 3 conv 32
      -> lift PD with the main branch's depth probabilities of that frame (stride 16: the lift samples both maps in
         normalised image coordinates, so the features are read at stride 4 and the depth at stride 16), same heights;
         `use_depth=False` (ModelConfig.lane_use_depth, t24) = lift P: the same table, the pull without the depth weight
      -> 1 x 1 conv over heights x channels to 16, BN, ReLU -> 1 x 1 conv to c_out, ZERO-initialised (the chain starts exactly
         as without the branch; ControlNet-style zero start)
The model adds the output to the camera BEV after the temporal block (model.py, ModelConfig.lane_branch). With camera_features 'dual' (t20)
the depth probabilities, K and ida it reads are those of the FIRST feature source (DualBEV, the same crop as the cached image) and the output
is added to the merged BEV of both sources; model.check_config refuses a first source that is not the DualBEV family.

Image source for cached training: a key-frame image cache (`build-cache`): JPEG (quality 95) of the 256 x 704 image after the
dataset's own resize + crop, one file per CAM_FRONT sample_data token; `CachedFrontRadarDataset(lane_images=...)` reads it and
`collate_cached` stacks it as uint8 `batch["lane_img"]`, normalised here on the device.

Self-test: python net/model/lane_branch.py            (CPU, synthetic; no data needed)
Image cache (net/prep/image_cache.py): python training/build_image_cache.py build-cache <config.json> [--workers 1]   (from the bev_fusion folder; > 1 forks copies of the devkit: memory)
Older frames (t26, model.student_frames 'all': the student reads EVERY frame of the window): python training/build_image_cache.py build-cache-older <config.json> fills <lane_images>/sweeps
with the JPEG of each older frame (0 is the key frame; n_cam 3, cam_stride 2: -2 and -4 camera sweeps). Once every sample is through, each builder writes its BUILT.json (scenes covered, the
builder's quality / crop; <root>/BUILT.json for the key frames, <root>/sweeps/BUILT.json per window); a crash half-way leaves none. check-cache / check-cache-older compare a few val frames
with the dataset's own image and the marker with the config's scenes. A config's "split" is resolved first (splits/<name>.json); the builders read plain datasets (no feature cache, no student).
"""
import os as _os
import sys as _sys
_sys.path.insert(1, _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__))))  # net/ (a flat run snapshot holds netpath.py next to this file)
import netpath  # noqa: E402,F401
import os
import sys

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision

from camera_encoder import IMG_MEAN, IMG_STD
from lift_pd import LiftPD

DUALBEV_CKPT = "models/pretrained/dualbev-r50-cbgs.pth"


def _cbr(cin, cout, k, stride=1):
    return [nn.Conv2d(cin, cout, k, stride, k // 2, bias=False), nn.BatchNorm2d(cout), nn.ReLU(True)]


class Res4Features(nn.Module):
    """Frozen ResNet-50 stem + layer 1 of the DualBEV checkpoint (img_backbone.conv1 / bn1 / layer1): (B, 256, H/4, W/4)."""
    out_channels = 256

    def __init__(self, path=DUALBEV_CKPT):
        super().__init__()
        r = torchvision.models.resnet50(weights=None)
        self.conv1, self.bn1, self.relu, self.maxpool, self.layer1 = r.conv1, r.bn1, r.relu, r.maxpool, r.layer1
        if path:
            try:                                   # third-party file: tensors only; mmap keeps RAM low
                ck = torch.load(path, map_location="cpu", weights_only=True, mmap=True)
            except RuntimeError:
                ck = torch.load(path, map_location="cpu", weights_only=True)
            sd = ck.get("state_dict", ck)
            pre = "img_backbone."
            take = {k[len(pre):]: v for k, v in sd.items()
                    if k.startswith(pre) and k[len(pre):].split(".")[0] in ("conv1", "bn1", "layer1")}
            missing, unexpected = self.load_state_dict(take, strict=False)
            if missing or unexpected:
                raise RuntimeError(f"res4 weights do not fit: missing {missing[:3]} unexpected {unexpected[:3]}")
        for p in self.parameters():
            p.requires_grad_(False)
        self.eval()

    def train(self, mode=True):                    # frozen: BatchNorm always in eval mode
        return super().train(False)

    def forward(self, x):
        with torch.no_grad():
            return self.layer1(self.maxpool(self.relu(self.bn1(self.conv1(x)))))


class RGBFeatures(nn.Module):
    """RGB CNN from scratch to stride 4 (the t5 probe's control features): (B, 64, H/4, W/4)."""
    out_channels = 64

    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(*_cbr(3, 32, 3, 2), *_cbr(32, 32, 3), *_cbr(32, 64, 3, 2), *_cbr(64, 64, 3))

    def forward(self, x):
        return self.net(x)


class LaneBranch(nn.Module):
    def __init__(self, grid, depth_centres, kind="res4", c_mid=32, c_lane=16, c_out=64, oversample=4, pretrained=DUALBEV_CKPT, use_depth=True):
        """use_depth: True = lift PD (the stride-4 features weighted by the main branch's depth probabilities, t7 / t20); False = lift P, the same projection table
        without the weight (t24): a ground cell takes the feature of the pixel it projects to, whatever the predicted depth there. No tensor is added or changed."""
        super().__init__()
        if kind == "res4":
            self.features = Res4Features(pretrained)
        elif kind == "rgb":
            self.features = RGBFeatures()
        else:
            raise ValueError(f"lane branch kind {kind!r}")
        self.kind = kind
        self.reduce = nn.Sequential(*_cbr(self.features.out_channels, c_mid, 1), nn.Conv2d(c_mid, c_mid, 3, padding=1))
        self.lift = LiftPD(grid, depth_centres, use_depth=use_depth, oversample=oversample)   # use_depth False = lift P: pull without the depth weight (t24); the tensors are the same either way
        nz = len(grid.z_levels)
        self.collapse = nn.Sequential(*_cbr(nz * c_mid, c_lane, 1))
        self.mix = nn.Conv2d(c_lane, c_out, 1)
        nn.init.zeros_(self.mix.weight)
        nn.init.zeros_(self.mix.bias)
        self.register_buffer("mean", torch.tensor(IMG_MEAN).view(1, 3, 1, 1), persistent=False)
        self.register_buffer("std", torch.tensor(IMG_STD).view(1, 3, 1, 1), persistent=False)

    def image_map(self, img):
        """img (B, 3, H, W): uint8 RGB (the image cache) or already normalised float (the dataset's own image)."""
        if img.dtype == torch.uint8:
            img = (img.float() - self.mean) / self.std
        return self.reduce(self.features(img))

    def forward(self, img, prob, K, T_cam_car, image_hw, ida=None, keys=None):
        """img (B, 3, H, W) current frame; prob (B, D, h, w) its depth probabilities (any stride); K (B, 3, 3), T_cam_car
        (B, 4, 4) per sample -> (B, c_out, X, Y). keys: the projection-table cache key of each sample (lift_pd.lift_key of its K, T_cam_car, image_hw, ida, made from the host
        copies), or None: made from the GPU tensors, as before."""
        f = self.image_map(img)
        out = []
        for b in range(f.shape[0]):                              # one calibration per sample (scenes differ)
            vol = self.lift(f[b:b + 1], prob[b:b + 1].to(f.dtype), K[b], T_cam_car[b], image_hw, ida, key=None if keys is None else keys[b])
            out.append(self.collapse(vol))
        return self.mix(torch.cat(out))


# ----------------------------------------------------------------------------------------------- image cache


                                                          # <root>/sweeps/BUILT.json for the older frames, key = the window "n_cam=3,cam_stride=2"; <root>/BUILT.json for the key frames, key KEY. Content:
                                                          # {"built": {key: {"scenes": [names], "when": ..., "settings": build_settings()}}}, the scenes merged over builds


# ----------------------------------------------------------------------------------------------- self-test
def _selftest():
    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
    from lift_pd import Grid, _front_camera
    torch.manual_seed(0)
    grid = Grid(z_levels=(-0.5, 0.0, 0.5, 1.0))
    depths = torch.arange(1.0, 60.0, 0.5)
    K, T, hw = _front_camera(torch.float32)
    H, W = hw

    # 1. shapes, zero start, gradients after one step (rgb: no checkpoint needed)
    br = LaneBranch(grid, depths, kind="rgb", c_out=8, oversample=2)
    img = torch.randint(0, 256, (2, 3, H, W), dtype=torch.uint8)
    prob = torch.rand(2, len(depths), H // 16, W // 16).softmax(1)
    Kb, Tb = K.expand(2, 3, 3), T.expand(2, 4, 4)
    out = br(img, prob, Kb, Tb, hw)
    assert out.shape == (2, 8, grid.nx, grid.ny), out.shape
    assert out.abs().max().item() == 0.0, "zero start: the branch must add nothing at step 0"
    opt = torch.optim.SGD(br.parameters(), lr=0.1)
    target = torch.randn_like(out)
    for _ in range(2):
        opt.zero_grad()
        F.mse_loss(br(img, prob, Kb, Tb, hw), target).backward()
        opt.step()
    g = [p.grad for p in br.features.parameters()]
    assert all(x is not None and x.abs().sum() > 0 for x in g), "after one step the gradient must reach the image CNN"
    print("shapes, zero start, gradient reaches the image CNN after one step: ok")

    # 2. geometry at stride 4 with the depth at stride 16: a painted ground line at y = +2 m, x = 6..40 m lands in the BEV
    #    cells of that line in the ground slice (z = 0), not at y = -2 m
    f = torch.zeros(1, 1, H // 4, W // 4)
    for x in torch.arange(6.0, 40.0, 0.05):
        pc = T @ torch.tensor([x, 2.0, 0.0, 1.0])
        u, v = (K @ (pc[:3] / pc[2]))[:2]
        j, i = int((u + 0.5) / 4), int((v + 0.5) / 4)
        if 0 <= i < f.shape[2] and 0 <= j < f.shape[3]:
            f[0, 0, i, j] = 1.0
    uni = torch.full((1, len(depths), H // 16, W // 16), 1.0 / len(depths))
    lift = LiftPD(grid, depths)
    vol = lift(f, uni, K, T, hw)                               # (1, Z * 1, X, Y), height-major channels
    ground = vol[0, 1]                                         # z = 0.0
    xs, ys = grid.xy_centres()
    on = ground[(xs > 8) & (xs < 30)][:, (ys - 2.0).abs() < 0.3].mean()
    off = ground[(xs > 8) & (xs < 30)][:, (ys + 2.0).abs() < 0.3].mean()
    assert on > 20 * max(off.item(), 1e-6) and on > 0, (on, off)
    print(f"stride-4 ground line lands in its BEV cells: mean on the line {on:.4f}, mirrored side {off:.6f}: ok")

    # 3. res4 kind: frozen (no trainable parameter in the features, BatchNorm stays in eval mode under .train())
    br4 = LaneBranch(grid, depths, kind="res4", c_out=8, oversample=1, pretrained="")
    br4.train()
    assert not any(p.requires_grad for p in br4.features.parameters()) and not br4.features.bn1.training
    assert br4.image_map(img[:1]).shape == (1, 32, H // 4, W // 4)
    print("res4 features frozen, BatchNorm in eval mode under train(): ok")
    if os.path.exists(DUALBEV_CKPT):
        Res4Features(DUALBEV_CKPT)
        print("res4 weights load from the DualBEV checkpoint (conv1, bn1, layer1): ok")
    else:
        print(f"res4 weights NOT checked ({DUALBEV_CKPT} not found from {os.getcwd()})")


if __name__ == "__main__":
    if len(sys.argv) > 1:
        sys.exit("lane_branch.py runs its self-test only; the image cache: python training/build_image_cache.py build-cache <config>")
    _selftest()
