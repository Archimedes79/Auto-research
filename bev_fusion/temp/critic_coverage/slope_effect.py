"""Effect of the flat-plane assumption: boxes as annotated (z_lo, z_hi in the ego frame) vs boxes standing on the plane z = 0 (same height)."""
import os, numpy as np
from autogt import coverage as C
from autogt.cli import load_keyframe
from autogt.grid import GridSpec
from autogt.layers import cell_centres
d = "data/autogt/15efbde2d1"
files = sorted(f for f in os.listdir(d) if f.endswith(".npz") and ".tmp" not in f)
rng = np.random.default_rng(4)
grid = GridSpec(); gx, gy = cell_centres(grid)
tot = dict(ann=0, flat=0, flip_in_cam=0, frames=0, frames_flip10=0)
for i in rng.choice(len(files), 300, replace=False):
    g = load_keyframe(d, files[i][:-4]); m = g["meta"]; cam = m["T_c2e"][:3, 3]
    hid = [b for b in m["boxes"] if b["observed"]]
    p_ann = [C.prism_of(b) for b in hid]
    p_flat = [(fp, 0.0, hi - lo) for fp, lo, hi in p_ann]
    a = C.hidden_cells(gx, gy, cam, p_ann); f = C.hidden_cells(gx, gy, cam, p_flat)
    camm = g["camera"] > 0
    tot["ann"] += int((a & camm).sum()); tot["flat"] += int((f & camm).sum()); n = int(((a != f) & camm).sum()); tot["flip_in_cam"] += n
    tot["frames"] += 1; tot["frames_flip10"] += n > 10
print(tot)
