import os, sys
import numpy as np
from scipy.spatial import cKDTree
R = os.environ["BFLC_ROOT"]
for sub in ("training/data", "training/grid", ".."):
    sys.path.insert(0, os.path.normpath(os.path.join(R, sub)))
import lane_gt_data as LD
import lane_tokcache as LC
import lane_map_data as L
c = LC.TokCache(os.path.join(R, "runs/e365_cache/be292_a_0_K512_9ad1d4ce30_v1/val"))
lane_gt, _ = LD.lane_store()

def runs(p):
    pts = np.asarray(p["pts"], float)
    keep = np.ones(len(pts), bool)
    for s in p["segments"]:
        if s["kind"] == "connector" and s["in_intersection"]:
            keep[s["i0"]:s["i1"] + 1] = False
    out, cur = [], []
    for k, q in enumerate(pts):
        if keep[k]:
            cur.append(q)
        else:
            if len(cur) >= 2: out.append(np.array(cur))
            cur = []
    if len(cur) >= 2: out.append(np.array(cur))
    return out

stats = {"raw": [], "cut": []}
spac = []
for i in np.linspace(0, 872, 40).astype(int):
    g = lane_gt.load(c.tokens[i])
    rem = []
    for p in g["pieces"]:
        pts = np.asarray(p["pts"], float)
        for s in p["segments"]:
            if s["kind"] == "connector" and s["in_intersection"]:
                rem.append(pts[s["i0"]:s["i1"] + 1])
                spac.append(np.linalg.norm(np.diff(pts, axis=0), axis=1).mean())
    rem = np.concatenate(rem) if rem else np.zeros((0, 2))
    raw_pts = [np.asarray(p["pts"], float) for p in g["pieces"]]
    cut_pts = [r for p in g["pieces"] for r in runs(p)]
    for name, pts in (("raw", raw_pts), ("cut", cut_pts)):
        P = L.exact_map(pts)[0]
        if len(rem) and len(P):
            d = cKDTree(rem).query(P)[0]
            stats[name].append(d)
for k, v in stats.items():
    d = np.concatenate(v)
    print(k, len(d), "within 0.5 m: %.4f, 0.25 m: %.4f, 0.1 m: %.4f" % ((d <= 0.5).mean(), (d <= 0.25).mean(), (d <= 0.1).mean()), "min %.3f" % d.min(), "p1 %.3f" % np.percentile(d, 1))
print("mean point spacing of pieces with a connector range", np.mean(spac), np.min(spac), np.max(spac))

print("--- interior of the removed ranges (arc distance to both range ends >= K m)")
for K in (1.0, 2.0, 2.5, 3.0):
    res = {"raw": [], "cut": []}
    for i in np.linspace(0, 872, 40).astype(int):
        g = lane_gt.load(c.tokens[i])
        rem = []
        for p in g["pieces"]:
            pts = np.asarray(p["pts"], float)
            for s in p["segments"]:
                if s["kind"] == "connector" and s["in_intersection"]:
                    seg = pts[s["i0"]:s["i1"] + 1]
                    arc = np.concatenate([[0], np.cumsum(np.linalg.norm(np.diff(seg, axis=0), axis=1))])
                    inner = (arc >= K) & (arc[-1] - arc >= K)
                    rem.append(seg[inner])
        rem = np.concatenate(rem) if rem else np.zeros((0, 2))
        for name, pts in (("raw", [np.asarray(p["pts"], float) for p in g["pieces"]]), ("cut", [r for p in g["pieces"] for r in runs(p)])):
            P = L.exact_map(pts)[0]
            if len(rem) and len(P):
                res[name].append(cKDTree(rem).query(P)[0])
    print(K, {k: "n %d within 0.5 m: %.4f (%d samples)" % (len(np.concatenate(v)), (np.concatenate(v) <= 0.5).mean(), (np.concatenate(v) <= 0.5).sum()) for k, v in res.items()})
