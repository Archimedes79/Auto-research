"""Builder of the feature caches (an offline step of the preprocessing that runs the frozen camera heads, hence it imports from net/model):
    python -P -m bev_fusion.net.prep.build_feature_cache build <dataroot> <version> <root> [--split train|val] [--device cpu]    the frozen-head feature cache
    python -P -m bev_fusion.net.prep.build_feature_cache radar <dataroot> <version> <root> [--workers 4]                         the radar cache
Tests: tests/test_build_feature_cache.py, tests/test_feature_cache.py, tests/test_cached_model.py
Reading the caches is net/prep/feature_cache.py."""
import os
import sys
import time

import numpy as np
import torch

from bev_fusion.net.model.camera_encoder import FrontCameraEncoder
from bev_fusion.net.prep.feature_cache import (CHECKPOINT, FEAT_HW, N_CTX, N_PROB, CachedFrontRadarDataset, FeatureStore, build_radar_cache,
                           shard_complete, used_frames, write_shard)
from bev_fusion.net.prep.images import camera_vector
from bev_fusion.net.prep.nuscenes_dataset import FrontRadarDataset, available_scenes, pose_matrix


def frame_inputs(ds, sd_token):
    """The network input of one camera frame, exactly as the dataset builds it: normalised image and 27-value camera vector."""
    sd = ds.nusc.get("sample_data", sd_token)
    cs = ds.nusc.get("calibrated_sensor", sd["calibrated_sensor_token"])
    K = torch.tensor(cs["camera_intrinsic"], dtype=torch.float32)
    T_c2e = torch.tensor(pose_matrix(cs), dtype=torch.float32)
    cv = camera_vector(K[None], ds.ida[None, :2, :2], ds.ida[None, :2, 2], T_c2e[None])[0]
    return ds._image(sd), cv


def build(ds, root, sample_tokens=None, device="cpu", batch=16, checkpoint=CHECKPOINT, verbose=True):
    """Compute and store the frozen heads' outputs for every camera frame the dataset uses (resumable per scene)."""
    os.makedirs(root, exist_ok=True)
    camera = FrontCameraEncoder()
    rep = camera.load_pretrained(checkpoint)
    if rep["missing"] or rep["shape_mismatch"]:
        raise RuntimeError(f"camera weights do not fit: {rep}")
    camera.freeze_camera_heads()
    camera = camera.to(device).eval()
    todo = used_frames(ds, sample_tokens)
    n_done, t0 = 0, time.time()
    for scene, tokens in sorted(todo.items()):
        if shard_complete(root, scene, tokens):
            continue
        toks = sorted(tokens)
        arr = np.zeros((len(toks), N_PROB + N_CTX) + FEAT_HW, np.float16)
        for i in range(0, len(toks), batch):
            chunk = toks[i:i + batch]
            ins = [frame_inputs(ds, t) for t in chunk]
            img, cv = torch.stack([a for a, _ in ins]).to(device), torch.stack([b for _, b in ins]).to(device)
            with torch.no_grad():
                prob, ctx = camera.image_features(img, cv)
            arr[i:i + len(chunk)] = torch.cat([prob, ctx], 1).cpu().numpy().astype(np.float16)
        write_shard(root, scene, arr, toks)
        n_done += len(toks)
        if verbose:
            print(f"{scene}: {len(toks)} frames ({n_done} done, {time.time() - t0:.0f} s)", flush=True)
    return FeatureStore(root)


def _cli(argv):
    import argparse
    p = argparse.ArgumentParser(description="build the frozen-head feature cache (build) or the radar cache (radar: root = "
                                            "the radar cache folder)")
    p.add_argument("cmd", choices=["build", "radar"])
    p.add_argument("--workers", type=int, default=4)
    p.add_argument("dataroot")
    p.add_argument("version")
    p.add_argument("root")
    p.add_argument("--split", default=None, choices=[None, "train", "val"])
    p.add_argument("--device", default="cpu")
    p.add_argument("--limit", type=int, default=None, help="only the first N samples (for tests)")
    p.add_argument("--n-cam", type=int, default=3)
    p.add_argument("--cam-stride", type=int, default=2)
    p.add_argument("--batch", type=int, default=16)
    a = p.parse_args(argv)
    from nuscenes.nuscenes import NuScenes
    nusc = NuScenes(a.version, dataroot=a.dataroot, verbose=False)
    scenes = available_scenes(nusc, a.split) if a.split else None
    if a.cmd == "radar":
        ds = CachedFrontRadarDataset(a.dataroot, a.version, scenes=scenes, store=None, radar_cache=a.root, n_cam=a.n_cam,
                                     cam_stride=a.cam_stride, with_gt=False, nusc=nusc)
        if a.limit:
            ds.samples = ds.samples[:a.limit]
        n = build_radar_cache(ds, a.workers)
        print(f"radar cache: {n} samples written to {a.root}; all done")
        return
    ds = FrontRadarDataset(a.dataroot, a.version, scenes=scenes, n_cam=a.n_cam, cam_stride=a.cam_stride, with_gt=False, nusc=nusc)
    tokens = ds.samples[:a.limit] if a.limit else None
    store = build(ds, a.root, tokens, device=a.device, batch=a.batch)
    print(f"cache: {len(store)} frames in {a.root}")


if __name__ == "__main__":
    _cli(sys.argv[1:])
