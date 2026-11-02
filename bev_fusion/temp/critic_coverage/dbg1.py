import numpy as np
from autogt import coverage as C
cam = np.array([1.5, 0.0, 1.6]); p = np.array([0.6188517685962003, -24.002733221329027, 0.0])
fp = np.array([[-5.383784071643015, -22.806215647153596], [-5.730792909829329, -22.319366756744536], [0.8865553458625803, -17.602752679320016], [1.2335641840488942, -18.089601569729076]])
z_lo, z_hi = -0.7774575858072699, 0.7607009416754591
t0, t1 = C._clip(cam[:2], p[None, :2], fp)
print("t0,t1", t0, t1)
za, zb = cam[2] + t0 * (p[2] - cam[2]), cam[2] + t1 * (p[2] - cam[2])
print("z at t0,t1", za, zb, z_lo, z_hi)
print(C.blocked(p[None], cam, [(fp, z_lo, z_hi)]))
# direct sampling
t = np.linspace(0, 1, 200001)
q = cam[None] + t[:, None] * (p - cam)[None]
from shapely.geometry import Polygon, Point
import shapely
poly = Polygon(fp)
ins = shapely.contains_xy(poly, q[:, 0], q[:, 1]) & (q[:, 2] >= z_lo) & (q[:, 2] <= z_hi)
print("dense hit:", ins.any(), "t range", t[ins].min() if ins.any() else None, t[ins].max() if ins.any() else None)
