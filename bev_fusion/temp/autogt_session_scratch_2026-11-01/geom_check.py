"""scratch: known-answer checks of autogt/coverage.py (run: wsl bash autogt/run_nice.sh <this file>)"""
import sys

import numpy as np

sys.path.insert(0, ".")
from autogt import coverage as C
from autogt.grid import GridSpec

g = GridSpec(res=0.5)
gx, gy = np.meshgrid(np.arange(g.nx) * 0.5 + 0.25, -26 + np.arange(g.ny) * 0.5 + 0.25, indexing="ij")
cam = np.array([1.5, 0.0, 1.6])
box = np.array([[20.0, -1], [24, -1], [24, 1], [20, 1]])              # x 20..24, y -1..1
# a flat box (top 1.0 m): shadow ends at d_far * H / (H - h) = 24 * 1.6 / 0.6 from the camera x (camera x = 1.5): (24 - 1.5) * 1.6 / 0.6 = 60 m
for top, expect_far in ((1.0, 1.5 + (24 - 1.5) * 1.6 / 0.6), (1.5, 1.5 + (24 - 1.5) * 1.6 / 0.1), (3.0, 1e9)):
    hid = C.hidden_cells(gx, gy, cam, [(box, 0.0, top)])
    xs = gx[hid[:, 52]]
    print(f"top {top}: hidden cells on y=0 from x {xs.min():.2f} to {xs.max():.2f} (expected up to {min(expect_far, 60):.1f}), count {hid.sum()}")
old = C.hidden_cells(gx, gy, cam, [(box, 0.0, 3.0)], soft=False)
inf = C.hidden_cells(gx, gy, cam, [(box, -50.0, 50.0)])
print("tall box == old 2D rule:", (old == inf).all(), old.sum(), inf.sum())
# a box on the ground that is lower than the camera does not hide cells beyond its shadow; a pedestrian (1.7 m) behind a car (1.5 m top) is seen
print("cells inside the box footprint are not hidden by it:", not C.hidden_cells(gx, gy, cam, [(box, 0.0, 1.0)])[(gx > 20) & (gx < 24) & (np.abs(gy) < 1)].any())
K = np.array([[1260.0, 0, 800], [0, 1260.0, 450], [0, 0, 1]])
T = np.eye(4)
T[:3, :3] = np.array([[0, 0, 1], [-1, 0, 0], [0, -1, 0]], float)      # camera looks along +x, image x to the right = -y, down = -z
T[:3, 3] = cam
car = (np.array([[10.0, -1], [14, -1], [14, 1], [10, 1]]), 0.0, 1.5)
ped_xy = np.array([[17.0, -0.3], [17.6, -0.3], [17.6, 0.3], [17.0, 0.3]])
ped = {"corners": ped_xy, "center": np.array([17.3, 0.0, 0.85]), "size": [0.6, 0.6, 1.7]}
sh = C.object_shares(ped, None, [car], cam, K, T, (1600, 900), [], g)
print("pedestrian 1.7 m behind a 1.5 m car: camera share", sh["camera"], "(expect 1: the head is visible)")
ped_low = dict(ped, center=np.array([17.3, 0.0, 0.5]), size=[0.6, 0.6, 1.0])
sh = C.object_shares(ped_low, None, [car], cam, K, T, (1600, 900), [], g)
print("object 1.0 m high behind the car: camera share", sh["camera"], "(expect 0: hidden)")
print("footprint share on a half mask:", C.footprint_share(np.array([[10.0, -1], [12, -1], [12, 1], [10, 1]]), gy > 0, g), "(expect 0.5)")
print("outside the grid counts as not observed:", C.footprint_share(np.array([[58.0, -1], [62, -1], [62, 1], [58, 1]]), np.ones((g.nx, g.ny), bool), g), "(expect 0.5)")
