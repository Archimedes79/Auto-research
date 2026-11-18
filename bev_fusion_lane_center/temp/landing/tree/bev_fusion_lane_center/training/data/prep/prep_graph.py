"""Karten-Knoten (Referenzpunkte auf den Mittellinien) pro Sample im LIDAR-System erzeugen.

Pro Sample: alle Spur- und Spurverbinder-Mittellinien, Knoten alle `--spacing` m (Standard 3 m), nur Knoten
innerhalb +-`--range` m. Gespeichert werden (Nmax Knoten, aufgefuellt):
  nodes   [S, Nmax, 2]  (x_lidar, y_lidar) in m
  tangent [S, Nmax, 2]  Einheitsvektor entlang der Spur
  poly    [S, Nmax]     Index der Mittellinie (-1 = Fuellung)
  mask    [S, Nmax]     gueltige Knoten
Reihenfolge = Reihenfolge der infos-Datei (= Reihenfolge der gespeicherten Features).

Aufruf: python prep_graph.py ~/exp/graph
"""
import os
import pickle
import sys

import numpy as np
from nuscenes.map_expansion.map_api import NuScenesMap
from nuscenes.nuscenes import NuScenes

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from centerline_gt import build_lane_polylines, lidar_pose  # noqa: E402

out_dir = os.path.expanduser(sys.argv[1])
os.makedirs(out_dir, exist_ok=True)
root = os.path.expanduser(os.environ.get("NUSC_ROOT", "~/data/nuscenes"))
spacing, rng_m, nmax = 3.0, 48.0, 512

nusc = NuScenes(os.environ.get("NUSC_VERSION", "v1.0-mini"), root, verbose=False)
maps = {}


def resample(p, step):
    d = np.r_[0, np.cumsum(np.linalg.norm(np.diff(p, axis=0), axis=1))]
    if d[-1] < 1e-6:
        return p[:1]
    s = np.arange(0, d[-1] + 1e-6, step)
    return np.stack([np.interp(s, d, p[:, 0]), np.interp(s, d, p[:, 1])], axis=1)


for split in ("train", "val"):
    infos = pickle.load(open(os.path.join(root, f"nuscenes_infos_{split}.pkl"), "rb"))["infos"]
    # WICHTIG: NuScenesDataset.load_annotations sortiert die infos nach Zeitstempel; Features und tokens.json folgen dieser
    # Reihenfolge. Die Knoten muessen dieselbe haben (frueher Roh-Reihenfolge -> falsch zugeordnet).
    infos = sorted(infos, key=lambda e: e["timestamp"])
    S = len(infos)
    nodes = np.zeros((S, nmax, 2), np.float32)
    tang = np.zeros((S, nmax, 2), np.float32)
    poly = np.full((S, nmax), -1, np.int32)
    mask = np.zeros((S, nmax), bool)
    counts = []
    for si, info in enumerate(infos):
        sample = nusc.get("sample", info["token"])
        loc = nusc.get("log", nusc.get("scene", sample["scene_token"])["log_token"])["location"]
        if loc not in maps:
            m = NuScenesMap(dataroot=root, map_name=loc)
            maps[loc] = (m, *build_lane_polylines(m, 0.5))
        _, polys, bboxes = maps[loc]
        t, R = lidar_pose(nusc, sample)
        near = (bboxes[:, 2] > t[0] - 70) & (bboxes[:, 0] < t[0] + 70) & (bboxes[:, 3] > t[1] - 70) & (bboxes[:, 1] < t[1] + 70)
        P, T, I = [], [], []
        for pi, k in enumerate(np.nonzero(near)[0]):
            q = resample((polys[k] - t) @ R, spacing)
            if len(q) < 2:
                continue
            tg = np.gradient(q, axis=0)
            tg /= np.linalg.norm(tg, axis=1, keepdims=True) + 1e-9
            keep = (np.abs(q) < rng_m).all(axis=1)
            P.append(q[keep]); T.append(tg[keep]); I.append(np.full(keep.sum(), pi))
        P, T, I = np.concatenate(P), np.concatenate(T), np.concatenate(I)
        counts.append(len(P))
        if len(P) > nmax:
            sel = np.random.RandomState(si).choice(len(P), nmax, replace=False)
            sel.sort()
            P, T, I = P[sel], T[sel], I[sel]
        n = len(P)
        nodes[si, :n], tang[si, :n], poly[si, :n], mask[si, :n] = P, T, I, True
    np.savez(os.path.join(out_dir, f"{split}_graph.npz"), nodes=nodes, tangent=tang, poly=poly, mask=mask,
             tokens=np.array([i["token"] for i in infos]))
    print(f"{split}: {S} Samples, Knoten pro Sample: min {min(counts)}, Median {int(np.median(counts))}, max {max(counts)} (Nmax {nmax})")
