import os, numpy as np
from autogt import coverage as C
from autogt.cli import load_keyframe
from autogt.grid import GridSpec
from autogt.layers import cell_centres
grid = GridSpec(); gx, gy = cell_centres(grid)
d = "data/autogt/15efbde2d1"
files = sorted(f for f in os.listdir(d) if f.endswith(".npz") and ".tmp" not in f)
rng = np.random.default_rng(31)
bad = 0; n = 0; zs = set()
for i in rng.choice(len(files), 600, replace=False):
    g = load_keyframe(d, files[i][:-4]); m = g["meta"]
    new = C.camera_cells(gx, gy, m["K"], m["T_c2e"], (1600, 900))
    old_inline = None
    P = np.stack([gx.ravel(), gy.ravel(), np.zeros(gx.size), np.ones(gx.size)], 1) @ np.linalg.inv(m["T_c2e"]).T
    z = P[:, 2]; u = m["K"][0, 0] * P[:, 0] / np.maximum(z, 1e-6) + m["K"][0, 2]; vv = m["K"][1, 1] * P[:, 1] / np.maximum(z, 1e-6) + m["K"][1, 2]
    old_inline = ((z > 0.5) & (u >= 0) & (u < 1600) & (vv >= 0) & (vv < 900)).reshape(gx.shape)
    n += 1; bad += int(not np.array_equal(new, old_inline)) + int(not np.array_equal(new, g["camera"] > 0))
    # z as an array of the same shape
    za = np.zeros_like(gx)
    bad += int(not np.array_equal(C.camera_cells(gx, gy, m["K"], m["T_c2e"], (1600, 900), z=za), new))
    zs.add((m["K"][0, 0].round(3), m["K"][0, 2].round(2)))
print("frames", n, "mismatches (new vs inline code, new vs stored v5 camera, array z vs scalar z):", bad, "; distinct intrinsics", len(zs))
