"""One-off (2026-10-25): after link_views.sh, check that code still reads nuScenes through the WSL views.

Loads v1.0-mini with the devkit, reads one keyframe image and one radar sweep of the full set, the map expansion of
one city, and one CRN info file of the lane project's additions. Run: wsl bash ... _py14.sh-free: plain python of
the gcar env (`~/miniconda3/envs/gcar/bin/python datasets/NuScenes/tools/history/smoke_test_views.py`).
"""
import json
import os
import pickle
import time

full = os.path.expanduser("~/data/nuscenes_full")
mini = os.path.expanduser("~/data/nuscenes")
t = time.time()
from nuscenes.nuscenes import NuScenes  # noqa: E402

nusc = NuScenes(version="v1.0-mini", dataroot=mini, verbose=False)
s = nusc.sample[0]
cam = nusc.get("sample_data", s["data"]["CAM_FRONT"])
assert os.path.getsize(os.path.join(mini, cam["filename"])) > 0
print(f"mini devkit ok: {len(nusc.scene)} scenes, one CAM_FRONT image read ({time.time() - t:.1f} s)")

sd = json.load(open(os.path.join(full, "v1.0-trainval", "sample_data.json")))
img = next(r for r in sd if r["filename"].startswith("samples/CAM_FRONT/"))
rad = next(r for r in sd if r["filename"].startswith("sweeps/RADAR_FRONT/"))
for r in (img, rad):
    p = os.path.join(full, r["filename"])
    with open(p, "rb") as f:
        n = len(f.read())
    print(f"full read ok: {r['filename']} ({n} bytes)")
exp = json.load(open(os.path.join(full, "maps", "expansion", "boston-seaport.json")))
print(f"map expansion ok: {len(exp['lane'])} lanes in boston-seaport")
info = pickle.load(open(os.path.join(full, "nuscenes_infos_val.pkl"), "rb"))
print(f"CRN info ok: {type(info).__name__} with {len(info)} entries")
print(f"SMOKE TEST PASS ({time.time() - t:.1f} s)")
