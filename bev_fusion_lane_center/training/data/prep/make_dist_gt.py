"""Exakte Abstandskarte zur Mittellinie (Soll) pro Sample, ohne das dick gezeichnete Raster.

Fuer jedes Sample: alle Spur-/Verbinder-Mittellinien im Lidar-System, auf 0.1 m verdichtet; fuer jede
Pixelmitte des 200x200-Rasters (0.5 m) der Abstand zum naechsten Punkt (GPU). Ergebnis in Pixeln (1 px = 0.5 m),
bei 8 px gekappt; ausserdem dichte Soll-Punkte (0.5 m Abstand, nur +-48 m) fuer die Auswertung.

Aufruf: python make_dist_gt.py ~/exp/feats
"""
import os
import pickle
import sys

import numpy as np
import torch
from nuscenes.map_expansion.map_api import NuScenesMap
from nuscenes.nuscenes import NuScenes

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from centerline_gt import build_lane_polylines, lidar_pose  # noqa: E402

out = os.path.expanduser(sys.argv[1])
root = os.path.expanduser(os.environ.get("NUSC_ROOT", "~/data/nuscenes"))
nusc = NuScenes(os.environ.get("NUSC_VERSION", "v1.0-mini"), root, verbose=False)
maps = {}
dev = "cuda"
cen = -50.0 + (torch.arange(200, device=dev).float() + 0.5) * 0.5
gx, gy = torch.meshgrid(cen, cen, indexing="ij")
pix = torch.stack([gx.reshape(-1), gy.reshape(-1)], -1)  # [40000,2] (x,y) m


def densify(p, step):
    d = np.r_[0, np.cumsum(np.linalg.norm(np.diff(p, axis=0), axis=1))]
    if d[-1] < 1e-6:
        return p[:1]
    s = np.arange(0, d[-1] + 1e-6, step)
    return np.stack([np.interp(s, d, p[:, 0]), np.interp(s, d, p[:, 1])], 1)


for split in ("train", "val"):
    infos = pickle.load(open(os.path.join(root, f"nuscenes_infos_{split}.pkl"), "rb"))["infos"]
    infos = sorted(infos, key=lambda e: e["timestamp"])  # gleiche Reihenfolge wie NuScenesDataset (und die Features)
    S = len(infos)
    dist = np.zeros((S, 200, 200), np.float16)
    gts = np.empty(S, dtype=object)
    for si, info in enumerate(infos):
        sample = nusc.get("sample", info["token"])
        loc = nusc.get("log", nusc.get("scene", sample["scene_token"])["log_token"])["location"]
        if loc not in maps:
            m = NuScenesMap(dataroot=root, map_name=loc)
            maps[loc] = (m, *build_lane_polylines(m, 0.5))
        _, polys, bboxes = maps[loc]
        t, R = lidar_pose(nusc, sample)
        near = (bboxes[:, 2] > t[0] - 80) & (bboxes[:, 0] < t[0] + 80) & (bboxes[:, 3] > t[1] - 80) & (bboxes[:, 1] < t[1] + 80)
        fine, coarse = [], []
        for k in np.nonzero(near)[0]:
            q = (polys[k] - t) @ R
            fine.append(densify(q, 0.1))
            coarse.append(densify(q, 0.5))
        fine = np.concatenate(fine)
        coarse = np.concatenate(coarse)
        fine = fine[(np.abs(fine) < 56).all(1)]  # etwas ueber den Raster-Rand hinaus
        coarse = coarse[(np.abs(coarse) < 48).all(1)]
        F_ = torch.from_numpy(fine.astype(np.float32)).to(dev)
        d = torch.empty(pix.shape[0], device=dev)
        for a in range(0, pix.shape[0], 4000):
            d[a : a + 4000] = torch.cdist(pix[a : a + 4000], F_).min(1).values
        dist[si] = (d / 0.5).clamp(max=8.0).view(200, 200).cpu().numpy().astype(np.float16)
        gts[si] = coarse.astype(np.float32)
    np.save(f"{out}/{split}_gt_dist_exact.npy", dist)
    np.save(f"{out}/{split}_gt_pts_exact.npy", gts, allow_pickle=True)
    print(f"{split}: {S} Samples, mittlere Punktzahl Soll {np.mean([len(g) for g in gts]):.0f}", flush=True)
