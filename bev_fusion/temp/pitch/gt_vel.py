import sys
sys.path.insert(0, ".")
sys.argv = ["x"]
import numpy as np
import evaluation.bev_gtcheck as G
from evaluation.bev_views import masks
tot = dict(veh05=0, veh0=0, vel_nz=0, moving=0, fovcells=0)
for name, s in G.pick(24, "15efbde2d1", 0):
    g = s[0.0]; m = masks(g)
    veh = np.asarray(g["vehicle"], np.float32); v = np.asarray(g["velocity"], np.float32)
    sp = np.hypot(v[0], v[1])
    tot["veh05"] += int(((veh > 0.5) & m["obj_fov"]).sum()); tot["veh0"] += int(((veh > 0) & m["obj_fov"]).sum())
    tot["vel_nz"] += int(((sp > 0) & m["obj_fov"]).sum()); tot["moving"] += int(((sp > 0.5) & m["obj_fov"]).sum()); tot["fovcells"] += int(m["obj_fov"].sum())
print(tot)
