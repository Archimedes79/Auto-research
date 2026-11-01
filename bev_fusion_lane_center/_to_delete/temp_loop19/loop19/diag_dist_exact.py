"""Diagnosis (E235.chk2 FAIL): why does the CPU exact-distance map differ from the stored b123 one by up to 4.5 px? CPU only.

    wsl bash tools/env/_py14.sh temp/loop19/diag_dist_exact.py

For a few frames of the check set: the stored b123 row (GPU torch.cdist, float32), the CPU k-d tree row, a brute-force float64 distance, and
torch.cdist on the CPU in float32 (the same matrix-product formula as on the GPU, without TF32). Prints where the stored values deviate from the
float64 truth (at which true distances) and which of the candidates reproduces the stored row.
"""
import json
import os
import sys

import numpy as np
import torch

H = os.path.expanduser("~")
os.environ.setdefault("GT_FRAME", "ego")
sys.argv = ["x"]  # make_dist_gt's helpers are imported, not run
from nuscenes.map_expansion.map_api import NuScenesMap  # noqa: E402
from nuscenes.nuscenes import NuScenes  # noqa: E402

sys.path.insert(0, os.path.join(os.environ["BFLC_ROOT"], "training/data/prep"))
from centerline_gt import build_lane_polylines, lidar_pose  # noqa: E402

nusc = NuScenes("v1.0-trainval", f"{H}/data/nuscenes_full", verbose=False)
toks = json.load(open(f"{H}/exp_gcar_gtchk/feats/train_tokens.json"))
crn = json.load(open(f"{H}/exp_crn_b123/feats/train_tokens.json"))
stored_all = np.load(f"{H}/exp_crn_b123/feats/train_gt_dist_exact.npy", mmap_mode="r")
kd_all = np.load(f"{H}/exp_gcar_gtchk/feats/train_gt_dist_exact.npy")
cen = -50.0 + (np.arange(200) + 0.5) * 0.5
gx, gy = np.meshgrid(cen, cen, indexing="ij")
pix = np.stack([gx.reshape(-1), gy.reshape(-1)], -1)
maps = {}


def densify(p, step):
    d = np.r_[0, np.cumsum(np.linalg.norm(np.diff(p, axis=0), axis=1))]
    if d[-1] < 1e-6:
        return p[:1]
    s = np.arange(0, d[-1] + 1e-6, step)
    return np.stack([np.interp(s, d, p[:, 0]), np.interp(s, d, p[:, 1])], 1)


print("torch", torch.__version__, "matmul tf32 default", torch.backends.cuda.matmul.allow_tf32)
for k in (0, 10, 20, 30):
    sample = nusc.get("sample", toks[k])
    loc = nusc.get("log", nusc.get("scene", sample["scene_token"])["log_token"])["location"]
    if loc not in maps:
        maps[loc] = build_lane_polylines(NuScenesMap(dataroot=f"{H}/data/nuscenes_full", map_name=loc), 0.5)
    polys, bboxes = maps[loc]
    t, R = lidar_pose(nusc, sample)
    near = (bboxes[:, 2] > t[0] - 80) & (bboxes[:, 0] < t[0] + 80) & (bboxes[:, 3] > t[1] - 80) & (bboxes[:, 1] < t[1] + 80)
    fine = np.concatenate([densify((polys[i] - t) @ R, 0.1) for i in np.nonzero(near)[0]])
    fine = fine[(np.abs(fine) < 56).all(1)].astype(np.float32)
    truth = np.full(len(pix), np.inf)
    f64 = fine.astype(np.float64)
    for a in range(0, len(pix), 2000):
        truth[a:a + 2000] = np.sqrt(((pix[a:a + 2000, None, :] - f64[None]) ** 2).sum(-1)).min(1)
    truth = np.minimum(truth / 0.5, 8.0).astype(np.float16).astype(np.float32)
    P, F = torch.from_numpy(pix.astype(np.float32)), torch.from_numpy(fine)
    mm = torch.cat([torch.cdist(P[a:a + 4000], F).min(1).values for a in range(0, len(pix), 4000)])
    mm = (mm / 0.5).clamp(max=8.0).numpy().astype(np.float16).astype(np.float32)
    stored = np.asarray(stored_all[crn.index(toks[k])], np.float32).reshape(-1)
    kd = kd_all[k].astype(np.float32).reshape(-1)
    e_st, e_kd, e_mm = np.abs(stored - truth), np.abs(kd - truth), np.abs(mm - truth)
    big = e_st > 0.5
    print(f"frame {k}: fine points {len(fine)}")
    print(f"  max |error| vs float64 truth: stored {e_st.max():.3f} px, k-d tree {e_kd.max():.3f} px, cpu cdist float32 {e_mm.max():.3f} px")
    print(f"  pixels with stored error > 0.5 px: {big.sum()} ({100 * big.mean():.2f} %); their true distance: median {np.median(truth[big]) if big.any() else 0:.2f} px, "
          f"max {truth[big].max() if big.any() else 0:.2f} px; stored there: median {np.median(stored[big]) if big.any() else 0:.2f} px")
    print(f"  stored == cpu cdist float32 on {100 * np.mean(stored == mm):.2f} % of pixels; stored == truth on {100 * np.mean(stored == truth):.2f} %")
