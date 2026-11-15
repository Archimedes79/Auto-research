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

Self-test: python training/lane_branch.py            (CPU, synthetic; no data needed)
Image cache: python training/lane_branch.py build-cache <config.json> [--workers 1]   (from the bev_fusion folder; > 1 forks copies of the devkit: memory)
Older frames (t26, model.student_frames 'all': the student reads EVERY frame of the window): python training/lane_branch.py build-cache-older <config.json> fills <lane_images>/sweeps
with the JPEG of each older frame (0 is the key frame; n_cam 3, cam_stride 2: -2 and -4 camera sweeps). Once every sample is through, each builder writes its BUILT.json (scenes covered, the
builder's quality / crop; <root>/BUILT.json for the key frames, <root>/sweeps/BUILT.json per window); a crash half-way leaves none. check-cache / check-cache-older compare a few val frames
with the dataset's own image and the marker with the config's scenes. A config's "split" is resolved first (splits/<name>.json); the builders read plain datasets (no feature cache, no student).
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
def image_path(root, sd_token):
    return os.path.join(root, sd_token[:2], sd_token + ".jpg")


def load_image(root, sd_token):
    """uint8 tensor (3, H, W) of the cache (torchvision's JPEG decoder)."""
    from torchvision.io import decode_jpeg, read_file
    p = image_path(root, sd_token)
    if not os.path.exists(p):
        builder = "build-cache-older" if os.path.basename(os.path.normpath(root)) == OLDER else "build-cache"
        raise FileNotFoundError(f"{p}: lane image cache incomplete (python training/lane_branch.py {builder} <config>)")
    return decode_jpeg(read_file(p))


OLDER = "sweeps"                                          # sub-folder of the image cache with the OLDER frames of a window (t26, model.student_frames 'all'): same layout, one file per CAM_FRONT sample_data token
BUILT = "BUILT.json"                                      # the finished-build marker of a cache folder, written when the builder has been through EVERY sample of its datasets (a crash half-way leaves none):
                                                          # <root>/sweeps/BUILT.json for the older frames, key = the window "n_cam=3,cam_stride=2"; <root>/BUILT.json for the key frames, key KEY. Content:
                                                          # {"built": {key: {"scenes": [names], "when": ..., "settings": build_settings()}}}, the scenes merged over builds
KEY = "key_frames"


def older_root(root):
    return os.path.join(root, OLDER)


def older_key(n_cam, cam_stride):
    """the frames a build covers depend on the window: the older frames of n_cam = 3, cam_stride = 2 (0, -2, -4 sweeps) are other sweeps than those of n_cam = 2, cam_stride = 4"""
    return f"n_cam={n_cam},cam_stride={cam_stride}"


def _built(folder):
    """{key: {"scenes": [names], "when": ..., "settings": ...}} of the finished builds in <folder>/BUILT.json ({} when there is none)"""
    import json
    try:
        return json.load(open(os.path.join(folder, BUILT)))["built"]
    except (OSError, ValueError, KeyError):
        return {}


def older_built(root):
    return _built(older_root(root))


def key_built(root):
    return _built(root)


def older_missing_scenes(root, n_cam, cam_stride, scenes):
    """the scenes of `scenes` (names) that no finished build of this window covers (all of them when there is none)"""
    return sorted(set(scenes) - set(older_built(root).get(older_key(n_cam, cam_stride), {}).get("scenes", [])))


def key_missing_scenes(root, scenes):
    """the scenes of `scenes` (names) that no finished key-frame build covers (all of them when there is none)"""
    return sorted(set(scenes) - set(key_built(root).get(KEY, {}).get("scenes", [])))


def build_settings():
    """what the builder makes the JPEGs with, recorded in the marker: the quality and the resize + crop of the dataset's own image (nuscenes_dataset._image_uint8); a cache made with other ones is
    not the cache of this code"""
    from camera_encoder import nuscenes_front_ida
    from nuscenes_dataset import DST_HW
    return {"jpeg_quality": JPEG_QUALITY, "hw": list(DST_HW), "ida": [round(float(x), 6) for x in nuscenes_front_ida(dst_hw=DST_HW).flatten().tolist()]}


def older_records(ds, sample_token):
    """the CAM_FRONT sample_data records of the older frames (k = 1 .. n_cam - 1) of one sample: the track of the scene, cam_stride records apart, clamped at its start
    (the rule of nuscenes_dataset._cam_frames and feature_cache.used_frames, so these are the frames the teacher caches hold)"""
    track = ds.gt.camera_track(sample_token)
    i0 = [x["token"] for x in track].index(ds.gt.cam(sample_token)["token"])
    return [track[max(0, i0 - k * ds.cam_stride)] for k in range(1, ds.n_cam)]


def scene_names(ds):
    """the names of the scenes the samples of a dataset belong to"""
    return {ds.nusc.get("scene", ds.nusc.get("sample", t)["scene_token"])["name"] for t in ds.samples}


_DS = None                                                # dataset of build_cache, inherited by the forked workers


def _write(root, records):
    """the JPEG of every CAM_FRONT sample_data record that the cache folder `root` does not hold yet, in the dataset's own crop"""
    from PIL import Image
    for sd in records:
        p = image_path(root, sd["token"])
        if os.path.exists(p):
            continue
        os.makedirs(os.path.dirname(p), exist_ok=True)
        arr = _DS._image_uint8(sd)
        tmp = p[:-4] + f".tmp{os.getpid()}.jpg"
        Image.fromarray(arr).save(tmp, quality=JPEG_QUALITY)
        os.replace(tmp, p)                                 # atomic: a reader never sees a partial file


def _fill(args):
    """the current (key) frame of the samples into <root>"""
    root, tokens = args
    _write(root, [_DS.gt.cam(t) for t in tokens])
    return len(tokens)


def _fill_older(args):
    """the older frames of the samples (older_records) into <root>/sweeps/: every record, also one that is a key frame (a clamped frame at the start of a scene)"""
    root, tokens = args
    _write(older_root(root), [sd for t in tokens for sd in older_records(_DS, t)])
    return len(tokens)


def _mark_built(folder, key, datasets):
    """<folder>/BUILT.json: the scenes of these datasets are now complete in the cache folder `folder` under `key` (merged with the scenes of earlier builds). The read-modify-write is under a lock file
    (flock, released when the process dies), so two builds of one folder do not lose each other's update"""
    import datetime
    import fcntl
    import json
    os.makedirs(folder, exist_ok=True)
    with open(os.path.join(folder, BUILT + ".lock"), "w") as lock:
        try:
            fcntl.flock(lock, fcntl.LOCK_EX)
        except OSError:                                    # a file system without locks (not seen on WSL): one writer at a time is then the caller's business
            pass
        built = _built(folder)
        scenes = set(built.get(key, {}).get("scenes", [])).union(*(scene_names(ds) for ds in datasets))
        built[key] = {"scenes": sorted(scenes), "when": datetime.datetime.now().isoformat(timespec="seconds"), "settings": build_settings()}
        path = os.path.join(folder, BUILT)
        tmp = path + f".tmp{os.getpid()}"
        with open(tmp, "w") as fh:
            json.dump({"built": built}, fh, indent=1)
        os.replace(tmp, path)


def build_cache(datasets, root, workers=1, older=False):
    """Write the current CAM_FRONT frame of every sample of the datasets (resumable); older=True: the older frames of the window into <root>/sweeps/ instead. The finished-build marker
    (BUILT.json of that folder) is written once every sample is through."""
    import multiprocessing as mp
    import time
    global _DS
    fill = _fill_older if older else _fill
    t0, n = time.time(), 0
    for ds in datasets:
        _DS = ds
        chunks = [ds.samples[i::max(workers, 1) * 16] for i in range(max(workers, 1) * 16)]
        if workers <= 1:                                   # one process: no forked copies of the devkit (shared box, 2026-10-30:
            done = map(fill, [(root, c) for c in chunks])   # two forked workers + the devkit ran the box out of memory)
            for k in done:
                n += k
                print(f"lane image cache{' (older frames)' if older else ''}: {n} samples, {time.time() - t0:.0f} s", flush=True)
            continue
        with mp.get_context("fork").Pool(workers) as pool:
            for k in pool.imap_unordered(fill, [(root, c) for c in chunks]):
                n += k
                print(f"lane image cache{' (older frames)' if older else ''}: {n} samples, {time.time() - t0:.0f} s", flush=True)
    if datasets:                                           # startable.py asks for this (student_frames 'all'): a half-built cache is not started on
        _mark_built(older_root(root), older_key(datasets[0].n_cam, datasets[0].cam_stride), datasets) if older else _mark_built(root, KEY, datasets)
    return n


def load_config(path):
    """the TrainConfig of a config file as the trainer reads it: a "split" is replaced by its scene lists (training/splits.py, what `Training check` does), so the builders
    cover the scenes of the run, not the whole official split"""
    import json
    import splits
    import train as T
    return T.TrainConfig(**{k: v for k, v in splits.resolve(json.load(open(path))).items() if not k.startswith("_")})


def _cli(argv):
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=["build-cache", "check-cache", "build-cache-older", "check-cache-older"])
    ap.add_argument("config")
    ap.add_argument("--workers", type=int, default=1)
    ap.add_argument("--n", type=int, default=20, help="check-cache: samples compared with the dataset's own image")
    a = ap.parse_args(argv)
    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
    cfg = load_config(a.config)
    cfg.lane_images = cfg.feature_cache = cfg.feature_cache2 = ""     # the builders need the plain datasets: no cache to read (the one they build, the teachers' features), no student to feed
    import train as T
    dss = T.make_datasets(cfg)
    root, older = cfg_root(a.config), a.cmd.endswith("older")
    if a.cmd.startswith("build"):
        print("written / checked" + (" the older frames of" if older else ""), build_cache(dss, root, a.workers, older=older), "samples")
        return
    import numpy as np
    ds, errs = dss[1], []
    for i in range(0, len(ds.samples), max(1, len(ds.samples) // a.n))[:a.n]:
        for sd in (older_records(ds, ds.samples[i]) if older else [ds.gt.cam(ds.samples[i])]):
            ref = torch.tensor(ds._image_uint8(sd)).permute(2, 0, 1).float()
            got = load_image(older_root(root) if older else root, sd["token"]).float()
            errs.append((got - ref).abs().mean().item())
    print(f"{'older-frame ' if older else ''}cache vs dataset image, mean abs error per pixel (0..255) over {len(errs)} val frames: mean {np.mean(errs):.2f}, max {np.max(errs):.2f}")
    assert np.max(errs) < 3.0, "the cache does not match the dataset's image"
    scenes = set(cfg.train_scenes) | set(cfg.val_scenes)
    missing = older_missing_scenes(root, cfg.n_cam, cfg.cam_stride, scenes) if older else key_missing_scenes(root, scenes)
    assert not missing, f"BUILT.json has no finished build of {older_key(cfg.n_cam, cfg.cam_stride) if older else KEY} for {len(missing)} scene(s) of the config, e.g. {missing[:3]}"
    print(f"BUILT.json covers all {len(scenes)} scenes of the config")


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
