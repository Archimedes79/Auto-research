import os, time, numpy as np
from autogt import coverage as C
from autogt.cli import load_keyframe
from autogt.grid import GridSpec
from autogt.layers import cell_centres, occlusion
grid = GridSpec(); gx, gy = cell_centres(grid)
d = "data/autogt/15efbde2d1"
files = sorted(f for f in os.listdir(d) if f.endswith(".npz") and ".tmp" not in f)
rng = np.random.default_rng(8)
to = tn = 0.0; n = 0
for i in rng.choice(len(files), 200, replace=False):
    g = load_keyframe(d, files[i][:-4]); m = g["meta"]; T = m["T_c2e"]
    hid = [b for b in m["boxes"] if b["observed"]]
    fps = [b["corners"] for b in hid]; pr = [C.prism_of(b) for b in hid]
    t = time.time(); occlusion(gx, gy, T[:2, 3], fps); to += time.time() - t
    t = time.time(); C.hidden_cells(gx, gy, T[:3, 3], pr); tn += time.time() - t; n += 1
print("per frame: old occlusion %.3f s, soft hidden_cells %.3f s" % (to / n, tn / n))
