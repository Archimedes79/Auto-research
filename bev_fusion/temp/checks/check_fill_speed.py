"""Speed and exactness of Raster.fill (clipped version) against the previous implementation (gt_bev_old.py from git HEAD):
all layers of several mini keyframes must be bit-identical, and the time per keyframe is reported.
Usage: python temp/checks/check_fill_speed.py <dir with gt_bev_old.py>"""
import os, sys, time
import numpy as np
sys.path.insert(0, "training")
sys.path.insert(0, sys.argv[1])
import gt_bev, gt_bev_old
from lift_pd import Grid
root = os.path.expanduser("~/data/nuscenes")
new, old = gt_bev.NuScenesGT(root, "v1.0-mini"), gt_bev_old.NuScenesGT(root, "v1.0-mini")
G = Grid()
tn = to = 0.0
bad = []
for sc in new.nusc.scene:
    tok = sc["first_sample_token"]
    for _ in range(12):
        tok = new.nusc.get("sample", tok)["next"]
    t = time.time(); a = new.keyframe_gt(tok, G); tn += time.time() - t
    t = time.time(); b = old.keyframe_gt(tok, G); to += time.time() - t
    for k, v in a.items():
        if isinstance(v, np.ndarray) and not np.array_equal(v, b[k]):
            bad.append((sc["name"], k, float(np.abs(v - b[k]).max())))
print(f"keyframe_gt per frame: new {tn / 10:.2f} s, old {to / 10:.2f} s; layers differing: {bad}")
