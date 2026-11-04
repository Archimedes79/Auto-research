import numpy as np, sys
from autogt import coverage as C
from autogt.cli import load_keyframe
from autogt.grid import GridSpec
from autogt.layers import cell_centres, occlusion
d = "data/autogt/15efbde2d1"; tok = sys.argv[1]
g = load_keyframe(d, tok); m = g["meta"]; T = m["T_c2e"]; cam = T[:3, 3]
grid = GridSpec(); gx, gy = cell_centres(grid)
print("cam", cam)
hiding = [b for b in m["boxes"] if b["observed"]]
for i, b in enumerate(hiding):
    fp = b["corners"]
    big = [(fp, -500.0, 500.0)]
    old = occlusion(gx, gy, T[:2, 3], [fp])
    inf = C.hidden_cells(gx, gy, cam, big)
    if (old != inf).any():
        print(i, b["group"], b["size"], "centre", b["center"], "corners", fp.tolist(), "diff cells", int((old != inf).sum()))
        ii, jj = np.nonzero(old != inf)
        print("  cells x", gx[ii, jj].min(), gx[ii, jj].max(), "y", gy[ii, jj].min(), gy[ii, jj].max(), "cam mask", g["camera"][ii, jj].sum())
        v = fp - cam[:2]; vc = fp.mean(0) - cam[:2]
        print("  angles of corners from cam (deg):", np.degrees(np.arctan2(v[:, 1], v[:, 0])).round(1), "centroid", np.degrees(np.arctan2(vc[1], vc[0])).round(1))
        print("  diff cell angles (deg):", np.degrees(np.arctan2(gy[ii, jj] - cam[1], gx[ii, jj] - cam[0])).round(1)[:10])
