import sys, math
sys.path.insert(0, ".")
import numpy as np
import evaluation.bev_gtcheck as G
rows = {"next: cur->next": [], "prev: prev->cur": [], "prev: cur->prev": []}
for _, s in G.pick(24, "15efbde2d1", 0):
    for dt in (0.5, -0.5):
        st = s[dt]["meta"].get("steps")
        if not st:
            continue
        sx, sy, syaw = G.steps_displacement(st)
        Tc, Tn = s[0.0]["meta"]["T_e2g"], s[dt]["meta"]["T_e2g"]
        if dt > 0:
            px, py, pyaw = G.rel_pose(Tc, Tn)
            rows["next: cur->next"].append((np.hypot(px - sx, py - sy), abs(math.remainder(pyaw - syaw, 2 * math.pi)), s[dt]["meta"]["dt"]))
        else:
            px, py, pyaw = G.rel_pose(Tn, Tc)
            rows["prev: prev->cur"].append((np.hypot(px - sx, py - sy), abs(math.remainder(pyaw - syaw, 2 * math.pi)), s[dt]["meta"]["dt"]))
            px, py, pyaw = G.rel_pose(Tc, Tn)
            rows["prev: cur->prev"].append((np.hypot(px - sx, py - sy), abs(math.remainder(pyaw - syaw, 2 * math.pi)), s[dt]["meta"]["dt"]))
for k, v in rows.items():
    a = np.array(v)
    print(f"{k:18s} n {len(a)}  xy mean {a[:,0].mean():.3f} max {a[:,0].max():.3f}  yaw max {a[:,1].max():.4f}  | true gap dt: min {a[:,2].min():.2f} max {a[:,2].max():.2f}")
