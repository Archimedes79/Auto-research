"""Scan many stored frames for boxes near the camera where the wedge pre-filter changes hidden_cells; count cells in the camera view."""
import os, sys, numpy as np
from autogt import coverage as C
from autogt.cli import load_keyframe
from autogt.grid import GridSpec
from autogt.layers import cell_centres
grid = GridSpec(); gx, gy = cell_centres(grid)
d = "data/autogt/15efbde2d1"
files = sorted(f for f in os.listdir(d) if f.endswith(".npz") and ".tmp" not in f)
rng = np.random.default_rng(int(sys.argv[1])); N = int(sys.argv[2])
orig = C._wedge; allp = lambda c, p, f: np.ones(len(p), bool)
frames = cells = incam = 0; scanned = 0
for i in rng.choice(len(files), N, replace=False):
    g = load_keyframe(d, files[i][:-4]); m = g["meta"]; cam = m["T_c2e"][:3, 3]
    hid = [b for b in m["boxes"] if b["observed"]]
    # candidate boxes: wedge-relevant ones only (near the camera, so that a prefilter can differ); everything else is cheap to skip
    near = [b for b in hid if abs(b["corners"].mean(0)[1]) < 8 and -10 < b["corners"].mean(0)[0] < 15]
    if not near: continue
    scanned += 1
    pr = [C.prism_of(b) for b in hid]
    a = C.hidden_cells(gx, gy, cam, pr)
    C._wedge = allp; b_ = C.hidden_cells(gx, gy, cam, pr); C._wedge = orig
    dif = a != b_
    if dif.any():
        frames += 1; cells += int(dif.sum()); incam += int((dif & (g["camera"] > 0)).sum())
print(f"scanned frames with a box near the camera: {scanned} of {N}; frames where the pre-filter changes hidden_cells: {frames}; cells {cells}; of them in the camera view {incam}")
