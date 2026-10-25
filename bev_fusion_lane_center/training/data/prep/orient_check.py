"""Findet Orientierung und Versatz des Mittellinien-Rasters gegen die ORIGINAL-Labels von BEVFusion.

Referenz: mmdet3d.datasets.pipelines.loading.LoadBEVSegmentation (genau der Code, der die Trainings-Labels
baut: Raster um die LIDAR-Position, Ausrichtung nach der Lidar-x-Achse, danach transponiert).
Gemessen wird, welcher Anteil der Mittellinien-Pixel in der Referenz-Klasse 'drivable_area' liegt, fuer die 8
Dreh-/Spiegel-Varianten (D4) und danach fuer Verschiebungen der besten Variante.

Die Mittellinien werden hier im LIDAR-System gerastert (nicht im Ego-System): Lidar sitzt ca. 0.94 m vor der
Hinterachse und ist um -90 Grad gegen das Auto gedreht.
"""
import os
import sys

import numpy as np
from nuscenes.map_expansion.map_api import NuScenesMap
from nuscenes.nuscenes import NuScenes
from pyquaternion import Quaternion

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from centerline_gt import build_lane_polylines, lidar_pose, rasterize  # noqa: E402
from mmdet3d.datasets.pipelines.loading import LoadBEVSegmentation  # noqa: E402

root = os.path.expanduser("~/data/nuscenes")
nusc = NuScenes("v1.0-mini", root, verbose=False)
half, res = 50.0, 0.5
loader = LoadBEVSegmentation(root, [-50.0, 50.0, 0.5], [-50.0, 50.0, 0.5], ["drivable_area"])


def mat(translation, rotation):
    m = np.eye(4, dtype=np.float32)
    m[:3, :3] = Quaternion(rotation).rotation_matrix
    m[:3, 3] = translation
    return m


variants = {
    "identisch": lambda a: a,
    "transponiert": lambda a: a.T,
    "Zeilen gespiegelt": lambda a: a[::-1, :],
    "Spalten gespiegelt": lambda a: a[:, ::-1],
    "180 Grad": lambda a: a[::-1, ::-1],
    "transponiert + Zeilen gespiegelt": lambda a: a.T[::-1, :],
    "transponiert + Spalten gespiegelt": lambda a: a.T[:, ::-1],
    "transponiert + beide gespiegelt": lambda a: a.T[::-1, ::-1],
}
shifts = [(0, 0), (2, 0), (-2, 0), (0, 2), (0, -2), (4, 0), (-4, 0), (0, 4), (0, -4)]
score = {k: [] for k in variants}
shift_score = {s: [] for s in shifts}
maps = {}
best_name = None
cases = []
step = max(1, len(nusc.sample) // 40)
for sample in nusc.sample[::step]:
    scene = nusc.get("scene", sample["scene_token"])
    loc = nusc.get("log", scene["log_token"])["location"]
    if loc not in maps:
        m = NuScenesMap(dataroot=root, map_name=loc)
        maps[loc] = (m, *build_lane_polylines(m, 0.5))
    nmap, polys, bboxes = maps[loc]

    sd = nusc.get("sample_data", sample["data"]["LIDAR_TOP"])
    ep = nusc.get("ego_pose", sd["ego_pose_token"])
    cs = nusc.get("calibrated_sensor", sd["calibrated_sensor_token"])
    ref = loader(
        dict(
            lidar_aug_matrix=np.eye(4, dtype=np.float32),
            lidar2ego=mat(cs["translation"], cs["rotation"]),
            ego2global=mat(ep["translation"], ep["rotation"]),
            location=loc,
        )
    )["gt_masks_bev"][0].astype(bool)

    t, R = lidar_pose(nusc, sample)
    near = (bboxes[:, 2] > t[0] - 75) & (bboxes[:, 0] < t[0] + 75) & (bboxes[:, 3] > t[1] - 75) & (bboxes[:, 1] < t[1] + 75)
    cl = rasterize([(polys[k] - t) @ R for k in np.nonzero(near)[0]], half, res, 1) > 0
    if cl.sum() < 50:
        continue
    cases.append((cl, ref))
    for k, f in variants.items():
        c = f(cl)
        score[k].append((c & ref).sum() / c.sum())

print(f"{len(cases)} Samples. Anteil der Mittellinien-Pixel in 'drivable_area' (BEVFusion-Original-Label):")
ranked = sorted(score.items(), key=lambda kv: -np.mean(kv[1]))
for k, v in ranked:
    print(f"  {np.mean(v):.3f}  {k}")
best = variants[ranked[0][0]]
print(f"\nBeste Variante: {ranked[0][0]}. Empfindlichkeit gegen Verschiebung (Zellen, 1 Zelle = 0.5 m):")
for dr, dc in shifts:
    vals = []
    for cl, ref in cases:
        c = np.roll(best(cl), (dr, dc), axis=(0, 1))
        vals.append((c & ref).sum() / max(c.sum(), 1))
    print(f"  Zeile {dr:+d}, Spalte {dc:+d}: {np.mean(vals):.3f}")
