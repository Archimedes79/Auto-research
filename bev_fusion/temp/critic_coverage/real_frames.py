"""Real stored v5 frames: old rule vs soft rule (infinite height), pre-filter vs none, soft with real heights vs old rule, stored occluded."""
import os, sys, time
import numpy as np
from autogt import coverage as C
from autogt.cli import load_keyframe
from autogt.grid import GridSpec
from autogt.layers import cell_centres, occlusion

d = "data/autogt/15efbde2d1"
files = sorted(f for f in os.listdir(d) if f.endswith(".npz") and ".tmp" not in f)
rng = np.random.default_rng(int(sys.argv[1]) if len(sys.argv) > 1 else 0)
N = int(sys.argv[2]) if len(sys.argv) > 2 else 300
pick = rng.choice(len(files), N, replace=False)
grid = GridSpec(); gx, gy = cell_centres(grid)
agg = dict(n=0, stored_vs_old=0, old_vs_inf=0, wedge_diff_cells=0, wedge_diff_cam=0, wedge_frames=0, soft_not_subset=0,
           old_cells=0, soft_cells=0, old_cam_cells=0, soft_cam_cells=0, frames_many=0, tmax=0.0)
orig_wedge = C._wedge
diffs_inf = []
for i in pick:
    tok = files[i][:-4]
    g = load_keyframe(d, tok)
    m = g["meta"]
    boxes = m["boxes"]
    T = m["T_c2e"]; cam = T[:3, 3]
    hiding = [b for b in boxes if b["observed"]]
    fps = [b["corners"] for b in hiding]
    old = occlusion(gx, gy, T[:2, 3], fps)
    stored = g["occluded"] > 0
    agg["stored_vs_old"] += int((old != stored).sum())
    big = [(fp, -500.0, 500.0) for fp in fps]
    inf = C.hidden_cells(gx, gy, cam, big)
    agg["old_vs_inf"] += int((old != inf).sum())
    if (old != inf).any():
        diffs_inf.append((tok, int((old != inf).sum())))
    prisms = [C.prism_of(b) for b in hiding]
    t0 = time.time(); soft = C.hidden_cells(gx, gy, cam, prisms); dt = time.time() - t0
    agg["tmax"] = max(agg["tmax"], dt)
    C._wedge = lambda c, p, f: np.ones(len(p), bool)
    soft_nf = C.hidden_cells(gx, gy, cam, prisms)
    C._wedge = orig_wedge
    camm = g["camera"] > 0
    agg["n"] += 1
    nd = int((soft != soft_nf).sum())
    agg["wedge_diff_cells"] += nd; agg["wedge_diff_cam"] += int(((soft != soft_nf) & camm).sum()); agg["wedge_frames"] += nd > 0
    agg["soft_not_subset"] += int((soft & ~old).sum())
    agg["old_cells"] += int(old.sum()); agg["soft_cells"] += int(soft.sum())
    agg["old_cam_cells"] += int((old & camm).sum()); agg["soft_cam_cells"] += int((soft & camm).sum())
print(agg)
print("frames where old != soft(inf height):", diffs_inf[:10])
