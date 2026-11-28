"""The cached features of module 1 (DESIGN.md): the frozen GaussianCaR (camera + radar BEV map segmentation) run on every frame, its decoder output
(256 channels) projected to 32 by a fixed PCA (dec32) plus its 7 map logits (logit7) = the 39 channels at 200 x 200 cells of 0.5 m, float16, that
`<data>/lane_center/b123/feats/{train,val}_{dec32,logit7}.npy` hold. The one command (GPU, about 0.4 s per frame):

    python -P -m lane_center.net.prep.foundation.gcar_features --out <new folder> [--parts val train] [--clone DIR] [--weights FILE] [--workers 2]

Outside the repository (README.md here): the clone of the foundation's code at COMMIT with its rasteriser built (`--clone` or GCAR_REPO), its
weights WEIGHTS (`--weights` or GCAR_WEIGHTS), raw nuScenes trainval (`bflc_paths.NUSC`), the environment gcar. In the repository, next to this
file: the PCA basis (`pca_dec32.json`) and the sample token of every row (`b123_samples.json`).

Frame: the network's BEV is laid on the rear axle at the time of CAM_FRONT, flat (yaw only): camera extrinsics and radar points go into that
frame, no lidar file is read. The output is turned into the lane layout (row = x to the right, column = y forward): `to_lane_layout`.
An array that exists is never overwritten: a part is written into a new folder, an interrupted part goes on from its mark `<part>.extracting`.
The rasteriser is not deterministic, so two extractions differ slightly (`check_features.py` measures it).
"""
import argparse
import hashlib
import json
import os
import subprocess
import sys
import time

import numpy as np
import torch

from lane_center.training import bflc_paths as BP  # where code, data and runs are

HERE = os.path.dirname(os.path.abspath(__file__))

REMOTE = "https://github.com/santimontiel/gaussiancar"
COMMIT = "c0e748b9e95a6a6062579ddebdcd1c1e79500c2e"
RASTERISER = os.path.join("gaussiancar", "ops", "diff-gaussian-rasterization")           # inside the clone; built in place by build_rasteriser.sh
WEIGHTS = {"name": "nuscenes_map.ckpt", "bytes": 1130353854, "sha256": "a184b022ef6d73fcf3abd075176273912a0e1fbbdf589e284151f1e388fb9baa",
           "url": "https://huggingface.co/santimontieleu/gaussiancar/resolve/main/nuscenes_map.ckpt"}
VERSION = "v1.0-trainval"
CAMERAS = ["CAM_FRONT_LEFT", "CAM_FRONT", "CAM_FRONT_RIGHT", "CAM_BACK_LEFT", "CAM_BACK", "CAM_BACK_RIGHT"]
LAYERS = ["lane", "road_segment", "road_divider", "lane_divider", "ped_crossing", "walkway", "carpark_area"]      # the order of logit7
GROUPS = {"dec32": 32, "logit7": 7}                                                      # stored channel group -> channels
PARTS = ("val", "train")
SAMPLES_SHA256 = {"train": "9ee6d3b0c2a536fa925eb7ca684ab5c7ed37c8de85ff9a2956a812e661265470",      # of json.dumps(list): the stored token files
                  "val": "33e629787b4e9895303ee874f79b32735dcf52d4aac56f43919025b1a3e83e50"}


class Missing(Exception):
    """a piece outside the repository is not there; the text says which, where it is expected and how to get it"""


# ------------------------------------------------------------------ the recipe in the repository


def basis():
    """(mean [256], basis [256, 32]) float32 of the PCA that makes dec32; checked against the hash the file carries"""
    with open(os.path.join(HERE, "pca_dec32.json"), encoding="utf-8") as f:
        z = json.load(f)
    mean, base = np.asarray(z["mean"], np.float32), np.asarray(z["basis"], np.float32)
    assert mean.shape == (256,) and base.shape == (256, GROUPS["dec32"]), (mean.shape, base.shape)
    assert hashlib.sha256(mean.tobytes() + base.tobytes()).hexdigest() == z["sha256"], "pca_dec32.json: the numbers do not match its sha256"
    return mean, base


def sample_tokens():
    """{"train": [...], "val": [...]}: the nuScenes sample token of every row of the stored arrays"""
    with open(os.path.join(HERE, "b123_samples.json"), encoding="utf-8") as f:
        return json.load(f)


# ------------------------------------------------------------------ the pieces outside the repository


def need_clone(path):
    """the clone of the foundation's code at COMMIT -> its folder; puts it and its rasteriser on the module path"""
    if not path:
        raise Missing(f"the clone of the GaussianCaR code is not named: pass --clone DIR or set GCAR_REPO (git clone {REMOTE}, commit {COMMIT[:7]})")
    path = os.path.abspath(os.path.expanduser(path))
    if not os.path.isdir(os.path.join(path, "configs")) or not os.path.isdir(os.path.join(path, RASTERISER)):
        raise Missing(f"no GaussianCaR clone at {path} (configs/ and {RASTERISER}/ expected): git clone {REMOTE} {path} && git -C {path} checkout {COMMIT}")
    head = subprocess.run(["git", "-C", path, "rev-parse", "HEAD"], capture_output=True, text=True).stdout.strip()
    if head != COMMIT:
        raise Missing(f"the clone at {path} is at commit {head[:12] or '(no git commit)'}, the features were made at {COMMIT[:12]}: "
                      f"git -C {path} checkout {COMMIT}")
    for p in (os.path.join(path, RASTERISER), path):
        if p not in sys.path:  # path work: the foundation's clone is a foreign checkout with no install (README.md here)
            sys.path.insert(0, p)  # path work: the same
    try:
        import diff_gaussian_rasterization  # noqa: F401
    except ImportError as e:
        raise Missing(f"the Gaussian rasteriser of the clone is not built ({e}): bash {os.path.join(HERE, 'build_rasteriser.sh')} {path}") from None
    return path


def need_weights(path, expect=WEIGHTS):
    """the checkpoint by size and sha256 -> its path"""
    if not path:
        raise Missing(f"the GaussianCaR weights are not named: pass --weights FILE or set GCAR_WEIGHTS ({expect['name']}, {expect['bytes']} bytes, "
                      f"sha256 {expect['sha256']}, from {expect['url']})")
    path = os.path.abspath(os.path.expanduser(path))
    if not os.path.isfile(path):
        raise Missing(f"no weights at {path}: {expect['name']} ({expect['bytes']} bytes, sha256 {expect['sha256']}) comes from {expect['url']}")
    if os.path.getsize(path) != expect["bytes"]:
        raise Missing(f"{path} has {os.path.getsize(path)} bytes, {expect['name']} has {expect['bytes']} (sha256 {expect['sha256']}, from {expect['url']})")
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(1 << 24), b""):
            h.update(block)
    if h.hexdigest() != expect["sha256"]:
        raise Missing(f"{path} has sha256 {h.hexdigest()}, {expect['name']} has {expect['sha256']} (from {expect['url']})")
    return path


def need_raw(root):
    """raw nuScenes trainval: the tables, six cameras, radar key frames and sweeps -> its folder"""
    want = [os.path.join(VERSION, "sample_data.json"), os.path.join("samples", "CAM_BACK_RIGHT"), os.path.join("samples", "RADAR_FRONT"),
            os.path.join("sweeps", "RADAR_FRONT")]
    gone = [w for w in want if not os.path.exists(os.path.join(root, w))]
    if gone:
        raise Missing(f"raw nuScenes {VERSION} is not complete at {root} (missing: {', '.join(gone)}); how to get and link it: datasets/nuscenes/README.md")
    return root


# ------------------------------------------------------------------ the frozen network


def load(clone, weights, device="cuda"):
    """-> (the frozen GaussianCaR map model in eval mode, its config). PTv3 runs without flash attention (the same weights, attention in fp32)."""
    clone, weights = need_clone(clone), need_weights(weights)
    from hydra import compose, initialize_config_dir
    from hydra.utils import instantiate

    import gaussiancar.ops.ptv3.model as ptv3
    from gaussiancar.utils.config import register_new_resolvers

    register_new_resolvers()
    with initialize_config_dir(config_dir=os.path.join(clone, "configs"), version_base="1.3"):
        cfg = compose(config_name="inference", overrides=["task=map", "module.model.image_encoder.pretrained=false", "debug_mode=false"])
    init = ptv3.PointTransformerV3.__init__
    ptv3.PointTransformerV3.__init__ = lambda self, *a, **k: init(self, *a, **dict(k, enable_flash=False))
    module = instantiate(cfg.module, cfg=cfg)
    state = torch.load(weights, map_location="cpu", weights_only=True)["state_dict"]
    res = module.load_state_dict(state, strict=False)
    assert not res.missing_keys, f"the checkpoint lacks {len(res.missing_keys)} tensors of the model, e.g. {res.missing_keys[:4]}"
    print(f"GaussianCaR loaded: {len(state)} tensors, {len(res.unexpected_keys)} not used by the model", flush=True)
    module.to(device).eval()
    for p in module.parameters():
        p.requires_grad_(False)
    return module, cfg


@torch.no_grad()
def forward(model, batch):
    """-> dec [B, 256, 200, 200] (the input of the map head) and logits [B, 7, 200, 200] (LAYERS), in the network's layout"""
    cam = model.forward_features_camera(batch)
    rad = model.forward_features_radar(batch)
    cam_bev, _ = model.gs_render_image(cam["features"], cam["centers"], cam["covariances"], cam["opacities"])
    rad_bev, _ = model.gs_render_radar(rad["features"], rad["centers"], rad["covariances"], rad["opacities"])
    dec = model.decoder(model.fuser(cam_bev, rad_bev))
    out = model.head(dec)
    return dec, torch.cat([out[k] for k in LAYERS], 1)


def project(x, mean, base):
    """x [C, H, W] -> its PCA channels [K, H, W]: (x - mean) @ base per cell"""
    return torch.einsum("chw,ck->khw", x - mean[:, None, None], base)


def to_lane_layout(x):
    """network output [..., H, W] -> the lane layout (row = x to the right, column = y forward): transpose, then mirror the columns"""
    return x.transpose(-1, -2).flip(-1)


# ------------------------------------------------------------------ the frames


class GcarData(torch.utils.data.Dataset):
    """a sample = 6 camera pictures (448 x 800), their projection matrices and 7 radar sweeps, in the frame of the rear axle at the time of CAM_FRONT"""

    def __init__(self, nusc, tokens, img_params, bev_conf):
        from gaussiancar.data.common import get_pose
        from gaussiancar.data.transforms.augmentations import RandomTransformBev, RandomTransformImage
        from gaussiancar.data.transforms.nuscenes.loading import ImageDataLoader, RadarDataLoader

        self.nusc, self.tokens, self.pose = nusc, tokens, get_pose
        self.steps = [
            RandomTransformImage(img_params, training=False, orig_img_size=(img_params["W"], img_params["H"])),
            RandomTransformBev(bev_conf, training=False),
            ImageDataLoader(img_params, nusc.dataroot),
            RadarDataLoader(nusc, num_sweeps=7, lidar_name="CAM_FRONT"),
        ]

    def __len__(self):
        return len(self.tokens)

    def sample(self, token):
        n = self.nusc
        rec = n.get("sample", token)
        ref = n.get("ego_pose", n.get("sample_data", rec["data"]["CAM_FRONT"])["ego_pose_token"])
        world_from_ego = self.pose(ref["rotation"], ref["translation"], flat=True)
        images, intr, extr = [], [], []
        for ch in CAMERAS:
            sd = n.get("sample_data", rec["data"][ch])
            ego = n.get("ego_pose", sd["ego_pose_token"])
            cs = n.get("calibrated_sensor", sd["calibrated_sensor_token"])
            cam_from_ego = self.pose(cs["rotation"], cs["translation"], inv=True)
            ego_from_world = self.pose(ego["rotation"], ego["translation"], inv=True)
            extr.append((cam_from_ego @ ego_from_world @ world_from_ego).tolist())
            intr.append(cs["camera_intrinsic"])
            images.append(os.path.relpath(n.get_sample_data_path(sd["token"]), n.dataroot))
        return dict(token=token, images=images, intrinsics=intr, extrinsics=extr)

    def __getitem__(self, i):
        d = self.sample(self.tokens[i])
        for step in self.steps:
            d = step(d)
        return dict(image=d["image"], lidar2img=d["lidar2img"], radar_points=d["radar_points"])


def collate(batch):
    return dict(image=torch.stack([b["image"] for b in batch]), lidar2img=torch.stack([b["lidar2img"] for b in batch]),
                radar_points=[b["radar_points"] for b in batch])


def scene_names(tokens, root):
    """the nuScenes scenes the sample tokens are in (the light table reader keeps only their records)"""
    tab = os.path.join(root, VERSION)
    with open(os.path.join(tab, "scene.json")) as f:
        name = {s["token"]: s["name"] for s in json.load(f)}
    with open(os.path.join(tab, "sample.json")) as f:
        scene = {s["token"]: s["scene_token"] for s in json.load(f)}
    return sorted({name[scene[t]] for t in tokens})


def frames(module, cfg, tokens, workers=2, device="cuda"):
    """the stored channels of every sample token, in order: {"dec32": [32, 200, 200], "logit7": [7, 200, 200]} float16, batch 1"""
    from omegaconf import OmegaConf

    import GtGenerator

    root = need_raw(BP.NUSC)
    mean, base = (torch.from_numpy(x).to(device) for x in basis())
    nusc = GtGenerator.light_nuscenes(root, VERSION, scenes=scene_names(tokens, root), cameras=CAMERAS)
    data = GcarData(nusc, tokens, OmegaConf.to_container(cfg.data.img_params, resolve=True),
                    OmegaConf.to_container(cfg.data.data_config.bev_aug_conf, resolve=True))
    for b in torch.utils.data.DataLoader(data, batch_size=1, shuffle=False, num_workers=workers, collate_fn=collate):
        batch = dict(image=b["image"].to(device), lidar2img=b["lidar2img"].to(device), radar_points=[r.to(device) for r in b["radar_points"]])
        dec, logits = forward(module.model, batch)
        out = {"dec32": project(dec[0].float(), mean, base), "logit7": logits[0].float()}
        yield {g: to_lane_layout(v).half().cpu().numpy() for g, v in out.items()}


# ------------------------------------------------------------------ writing a part


def open_part(out, part, n):
    """-> (the arrays of a part in `out`, the row to go on from). New arrays start a part; arrays with a mark are an interrupted part; arrays
    without a mark are finished or someone else's and are refused."""
    paths = {g: os.path.join(out, f"{part}_{g}.npy") for g in GROUPS}
    mark = os.path.join(out, f"{part}.extracting")
    have = [p for p in paths.values() if os.path.exists(p)]
    if have and not os.path.exists(mark):
        raise Missing(f"{have[0]} exists and no extraction of it is under way: stored features are never overwritten "
                      f"(a new folder with --out, or leave the finished part out with --parts)")
    if not have:
        os.makedirs(out, exist_ok=True)
        _set_mark(mark, 0)
        return {g: np.lib.format.open_memmap(paths[g], mode="w+", dtype=np.float16, shape=(n, c, 200, 200)) for g, c in GROUPS.items()}, 0
    arrays = {g: np.load(p, mmap_mode="r+") for g, p in paths.items()}
    assert all(arrays[g].shape == (n, c, 200, 200) and arrays[g].dtype == np.float16 for g, c in GROUPS.items()), f"{out}: {part} has another shape"
    with open(mark) as f:
        return arrays, int(f.read())


def _set_mark(mark, row):
    with open(mark + ".tmp", "w") as f:
        f.write(str(row))
    os.replace(mark + ".tmp", mark)


def write_part(out, part, tokens, source, every=100):
    """the rows of a part from `source(tokens) -> the frames in order` into `out`; the mark names the rows that are safely on disk"""
    arrays, start = open_part(out, part, len(tokens))
    mark, t0 = os.path.join(out, f"{part}.extracting"), time.time()
    for row, frame in enumerate(source(tokens[start:]), start):
        for g in GROUPS:
            arrays[g][row] = frame[g]
        if (row + 1) % every == 0 or row + 1 == len(tokens):
            for a in arrays.values():
                a.flush()
            _set_mark(mark, row + 1)
            print(f"{part} {row + 1}/{len(tokens)} ({(time.time() - t0) / (row + 1 - start):.2f} s per frame)", flush=True)
    with open(os.path.join(out, f"{part}_tokens.json"), "w") as f:
        json.dump(tokens, f)
    os.remove(mark)


def parse(argv=None):
    ap = argparse.ArgumentParser(description="the GaussianCaR features of the lane problem, made again into a new folder")
    ap.add_argument("--out", required=True, help="the folder of the new arrays, an absolute path (30 GB); an existing array is never overwritten")
    ap.add_argument("--parts", nargs="+", default=list(PARTS), choices=PARTS)
    ap.add_argument("--clone", default=os.environ.get("GCAR_REPO"), help="the GaussianCaR clone (default: GCAR_REPO)")
    ap.add_argument("--weights", default=os.environ.get("GCAR_WEIGHTS"), help="nuscenes_map.ckpt (default: GCAR_WEIGHTS)")
    ap.add_argument("--workers", type=int, default=2, help="processes that read pictures and radar")
    a = ap.parse_args(argv)
    if not os.path.isabs(a.out):
        ap.error(f"--out {a.out}: an absolute path (a relative one would land below the working folder, which is not the problem folder)")
    return a


def main():
    a = parse()
    tokens = sample_tokens()
    try:
        need_raw(BP.NUSC)
        module, cfg = load(a.clone, a.weights)
        for part in a.parts:
            write_part(a.out, part, tokens[part], lambda toks: frames(module, cfg, toks, a.workers))
    except Missing as e:
        raise SystemExit(f"gcar_features: {e}")
    print(f"done: {', '.join(a.parts)} in {a.out}")


if __name__ == "__main__":
    main()
