"""one-off: how many connector points inside intersections lie next to painted lines (lane / road dividers) in the lane GT l3? (stored graphs, every 10th file)"""
import gzip, pickle, glob, os
import numpy as np
from scipy.spatial import cKDTree
fs = sorted(glob.glob("/mnt/c/Auto-research/data/autogt/lane/3576fa1f7e/*.pkl.gz"))[::10]
tot = {1.0: 0, 2.0: 0, 3.0: 0}
n_pts = n_seg = 0
n_seg_any = {1.0: 0, 2.0: 0}
frames_with_div = 0
for f in fs:
    g = pickle.load(gzip.open(f, "rb"))
    div = [l["pts"] for k in ("lane_divider", "road_divider") for l in g["lines"][k] if len(l["pts"])]
    div = np.concatenate(div) if div else np.zeros((0, 2))
    if len(div):
        frames_with_div += 1
    tr = cKDTree(div) if len(div) else None
    for p in g["pieces"]:
        for s in p["segments"]:
            if s["kind"] != "connector" or not s["in_intersection"]:
                continue
            q = p["pts"][s["i0"]: s["i1"] + 1]
            n_seg += 1
            n_pts += len(q)
            if tr is None:
                continue
            d = tr.query(q)[0]
            for r in tot:
                tot[r] += int((d <= r).sum())
            for r in n_seg_any:
                n_seg_any[r] += int((d <= r).any())
print(f"{len(fs)} frames, {frames_with_div} with any divider line; intersection connector segments {n_seg}, points {n_pts}")
for r, v in tot.items():
    print(f"points with a painted divider within {r} m: {v / n_pts:.3f}")
for r, v in n_seg_any.items():
    print(f"segments with at least one point within {r} m of a divider: {v / n_seg:.3f}")
