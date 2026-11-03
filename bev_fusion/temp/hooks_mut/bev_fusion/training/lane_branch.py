"""Lane branch (experiment t7): stride-4 image features of the CURRENT camera frame lifted into the BEV next to the stride-16
context of the main camera branch.

Why: the t5 probe showed that the stride-16 context (what lift PD receives) draws painted lines as blurred bands while the
stride-4 stage of the same frozen ResNet-50 and an RGB CNN draw them sharply; one stride-16 cell is 0.57 m x 7.9 m on the ground
at 20 m, a stride-4 cell 0.14 m x 2 m.

    image (B, 3, 256, 704) -> features at stride 4 (64 x 176): frozen DualBEV ResNet-50 stem + layer 1 ("res4", 256 ch)
                                                               or an RGB CNN from scratch ("rgb", 64 ch)
      -> 1 x 1 conv to 32, BN, ReLU, 3 x 3 conv 32
      -> lift PD with the main branch's depth probabilities of that frame (stride 16: the lift samples both maps in
         normalised image coordinates, so the features are read at stride 4 and the depth at stride 16), same heights
      -> 1 x 1 conv over heights x channels to 16, BN, ReLU -> 1 x 1 conv to c_out, ZERO-initialised (the chain starts exactly
         as without the branch; ControlNet-style zero start)
The model adds the output to the camera BEV after the temporal block (model.py, ModelConfig.lane_branch).

Image source for cached training: a key-frame image cache (`build-cache`): JPEG (quality 95) of the 256 x 704 image after the
dataset's own resize + crop, one file per CAM_FRONT sample_data token; `CachedFrontRadarDataset(lane_images=...)` reads it and
`collate_cached` stacks it as uint8 `batch["lane_img"]`, normalised here on the device.

Self-test: python training/lane_branch.py            (CPU, synthetic; no data needed)
Image cache: python training/lane_branch.py build-cache <config.json> [--workers 1]   (from the bev_fusion folder; > 1 forks copies of the devkit: memory)
"""
import os
import sys

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision

try:
    from .camera_encoder import IMG_MEAN, IMG_STD
    from .lift_pd import LiftPD
except ImportError:
    from camera_encoder import IMG_MEAN, IMG_STD
    from lift_pd import LiftPD

DUALBEV_CKPT = "models/pretrained/dualbev-r50-cbgs.pth"
JPEG_QUALITY = 95


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
    def __init__(self, grid, depth_centres, kind="res4", c_mid=32, c_lane=16, c_out=64, oversample=4, pretrained=DUALBEV_CKPT):
        super().__init__()
        if kind == "res4":
            self.features = Res4Features(pretrained)
        elif kind == "rgb":
            self.features = RGBFeatures()
        else:
            raise ValueError(f"lane branch kind {kind!r}")
        self.kind = kind
        self.reduce = nn.Sequential(*_cbr(self.features.out_channels, c_mid, 1), nn.Conv2d(c_mid, c_mid, 3, padding=1))
        self.lift = LiftPD(grid, depth_centres, oversample=oversample)
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

    def forward(self, img, prob, K, T_cam_car, image_hw, ida=None):
        """img (B, 3, H, W) current frame; prob (B, D, h, w) its depth probabilities (any stride); K (B, 3, 3), T_cam_car
        (B, 4, 4) per sample -> (B, c_out, X, Y)."""
        f = self.image_map(img)
        out = []
        for b in range(f.shape[0]):                              # one calibration per sample (scenes differ)
            vol = self.lift(f[b:b + 1], prob[b:b + 1].to(f.dtype), K[b], T_cam_car[b], image_hw, ida)
            out.append(self.collapse(vol))
        return self.mix(torch.cat(out))


# ----------------------------------------------------------------------------------------------- image cache
def image_path(root, sd_token):
    return os.path.join(root, sd_token[:2], sd_token + ".jpg")


def load_image(root, sd_token):
    """uint8 tensor (3, H, W) of the cache (torchvision's JPEG decoder)."""
    from torchvision.io import decode_jpeg, read_file
    p = image_path(root, sd_token)
    if not os.path.exists(p):
        raise FileNotFoundError(f"{p}: lane image cache incomplete (python training/lane_branch.py build-cache <config>)")
    return decode_jpeg(read_file(p))


_DS = None                                                # dataset of build_cache, inherited by the forked workers


def _fill(args):
    from PIL import Image
    root, tokens = args
    for tok in tokens:
        sd = _DS.gt.cam(tok)
        p = image_path(root, sd["token"])
        if os.path.exists(p):
            continue
        os.makedirs(os.path.dirname(p), exist_ok=True)
        arr = _DS._image_uint8(sd)
        tmp = p[:-4] + f".tmp{os.getpid()}.jpg"
        Image.fromarray(arr).save(tmp, quality=JPEG_QUALITY)
        os.replace(tmp, p)                                 # atomic: a reader never sees a partial file
    return len(tokens)


def build_cache(datasets, root, workers=1):
    """Write the current CAM_FRONT frame of every sample of the datasets (resumable)."""
    import multiprocessing as mp
    import time
    global _DS
    t0, n = time.time(), 0
    for ds in datasets:
        _DS = ds
        chunks = [ds.samples[i::max(workers, 1) * 16] for i in range(max(workers, 1) * 16)]
        if workers <= 1:                                   # one process: no forked copies of the devkit (shared box, 2026-10-30:
            done = map(_fill, [(root, c) for c in chunks])  # two forked workers + the devkit ran the box out of memory)
            for k in done:
                n += k
                print(f"lane image cache: {n} samples, {time.time() - t0:.0f} s", flush=True)
            continue
        with mp.get_context("fork").Pool(workers) as pool:
            for k in pool.imap_unordered(_fill, [(root, c) for c in chunks]):
                n += k
                print(f"lane image cache: {n} samples, {time.time() - t0:.0f} s", flush=True)
    return n


def _cli(argv):
    import argparse
    import json
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=["build-cache", "check-cache"])
    ap.add_argument("config")
    ap.add_argument("--workers", type=int, default=1)
    ap.add_argument("--n", type=int, default=20, help="check-cache: samples compared with the dataset's own image")
    a = ap.parse_args(argv)
    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
    import train as T
    cfg = T.TrainConfig(**json.load(open(a.config)))
    cfg.lane_images = ""                                   # the datasets must not try to read the cache they build
    dss = T.make_datasets(cfg)
    if a.cmd == "build-cache":
        print("written / checked", build_cache(dss, cfg_root(a.config), a.workers), "samples")
    else:
        import numpy as np
        ds, errs = dss[1], []
        for i in range(0, len(ds.samples), max(1, len(ds.samples) // a.n))[:a.n]:
            sd = ds.gt.cam(ds.samples[i])
            ref = torch.tensor(ds._image_uint8(sd)).permute(2, 0, 1).float()
            got = load_image(cfg_root(a.config), sd["token"]).float()
            errs.append((got - ref).abs().mean().item())
        print(f"cache vs dataset image, mean abs error per pixel (0..255) over {len(errs)} val samples: "
              f"mean {np.mean(errs):.2f}, max {np.max(errs):.2f}")
        assert np.max(errs) < 3.0, "the cache does not match the dataset's image"


def cfg_root(config):
    import json
    root = json.load(open(config)).get("lane_images", "")
    if not root:
        raise SystemExit(f"{config} has no lane_images folder")
    return root


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
        _cli(sys.argv[1:])
    else:
        _selftest()
