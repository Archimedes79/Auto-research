"""scratch: the live generator (v7 in progress) against the stored v5 GT on nuScenes mini keyframes: masks, shares, time per keyframe.
Run: wsl bash autogt/run_nice.sh <this file> [scene ...]"""
import os
import sys
import time

import numpy as np

sys.path.insert(0, ".")
from autogt import AutoGT, scene_tokens
from autogt.cli import load_keyframe

scenes = sys.argv[1:] or ["scene-0061", "scene-1077", "scene-0655"]
root = os.path.expanduser("~/data/nuscenes")
gen = AutoGT(root, "v1.0-mini", light=True, scenes=scenes)
store = "data/autogt/15efbde2d1"
t_all, n = 0.0, 0
same_cam = same_rad = same_und = sub_occ = 0
fewer = 0
shares = []
for s in scenes:
    for tok in scene_tokens(s, "v1.0-mini", root)[::3]:
        t0 = time.time()
        new = gen.keyframe(tok)
        t_all += time.time() - t0
        n += 1
        old = load_keyframe(store, tok)
        same_cam += np.array_equal(new["camera"] > 0, old["camera"] > 0)
        same_rad += np.array_equal(new["radar"] > 0, old["radar"] > 0)
        same_und += np.array_equal(new["undefined"] > 0, old["undefined"] > 0)
        sub_occ += bool(((new["occluded"] > 0) <= (old["occluded"] > 0)).all())
        fewer += int((old["occluded"] > 0).sum() - (new["occluded"] > 0).sum())
        for b in new["meta"]["boxes"]:
            shares.append((b["group"], b["share_camera"], b["share_radar"], b["share_fusion"], b["num_lidar_pts"], b["num_radar_pts"]))
print(f"{n} keyframes, {t_all / n * 1000:.0f} ms per keyframe (all generator work)")
print(f"camera identical to v5: {same_cam}/{n}, radar: {same_rad}/{n}, undefined: {same_und}/{n}, occluded subset of v5's: {sub_occ}/{n}; "
      f"hidden cells removed by the soft rule: {fewer} ({fewer / n:.0f} per keyframe)")
f = np.array([s[3] for s in shares])
print(f"boxes {len(f)}: share_fusion = 0: {(f == 0).sum()}, 0 < f < 0.1: {((f > 0) & (f < 0.1)).sum()}, 0.1 <= f < 0.5: {((f >= .1) & (f < .5)).sum()}, "
      f">= 0.5: {(f >= .5).sum()}, = 1: {(f == 1).sum()}")
