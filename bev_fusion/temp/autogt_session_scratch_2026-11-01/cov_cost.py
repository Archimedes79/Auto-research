"""scratch: cost per frame of the masks that are pure functions of calibration + grid (numpy, one thread)"""
import glob, os, sys, time, math
import numpy as np
sys.path.insert(0, ".")
import autogt
from autogt.layers import cell_centres, occlusion
from autogt.grid import GridSpec

files = sorted(glob.glob("data/autogt/15efbde2d1/*.npz"))[:1]
g = autogt.load(os.path.basename(files[0])[:-4])
K, T = g["meta"]["K"], g["meta"]["T_c2e"]
grid = GridSpec()
gx, gy = cell_centres(grid)
foot = [b["corners"] for b in g["meta"]["boxes"] if b["observed"]]
H, W = 900, 1600

def cam():
    P = np.stack([gx.ravel(), gy.ravel(), np.zeros(gx.size), np.ones(gx.size)], 1) @ np.linalg.inv(T).T
    z = P[:, 2]; u = K[0, 0] * P[:, 0] / np.maximum(z, 1e-6) + K[0, 2]; v = K[1, 1] * P[:, 1] / np.maximum(z, 1e-6) + K[1, 2]
    return ((z > 0.5) & (u >= 0) & (u < W) & (v >= 0) & (v < H)).reshape(gx.shape)

def fans():
    half, rmax = math.radians(60.0), 70.0
    cov = np.zeros(gx.shape, bool)
    for yaw, (tx, ty) in ((0, (3.4, 0)), (0.9, (2.4, 0.9)), (-0.9, (2.4, -0.9)), (2.4, (-2.4, 0.9)), (-2.4, (-2.4, -0.9))):
        dx, dy = gx - tx, gy - ty
        ang = (np.arctan2(dy, dx) - yaw + np.pi) % (2 * np.pi) - np.pi
        cov |= (np.abs(ang) <= half) & (np.hypot(dx, dy) <= rmax)
    return cov

def occ():
    return occlusion(gx, gy, T[:2, 3], foot)

for name, f in (("camera cells (ground plane)", cam), ("five radar fans", fans), (f"hidden behind {len(foot)} boxes", occ)):
    f()
    t0 = time.perf_counter()
    for _ in range(50):
        f()
    print(f"{name:32s} {(time.perf_counter() - t0) / 50 * 1e3:6.2f} ms per frame")
