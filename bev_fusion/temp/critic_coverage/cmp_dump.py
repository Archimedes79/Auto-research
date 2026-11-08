import pickle, numpy as np, sys
from autogt import coverage as C
from autogt.grid import GridSpec
from autogt.layers import cell_centres
old = pickle.load(open("bev_fusion/temp/critic_coverage/dump_old.pkl", "rb"))
new = pickle.load(open("bev_fusion/temp/critic_coverage/dump_new.pkl", "rb"))
g = GridSpec(); gx, gy = cell_centres(g)
nlay = 0; diff_layers = {}
for tok in old:
    o, n = old[tok], new[tok]
    for k, v in o.items():
        if k in ("meta", "_radar_poses"): continue
        nlay += 1
        if not np.array_equal(v, n[k]):
            diff_layers.setdefault(k, 0); diff_layers[k] += 1
    # box meta
    for bo, bn in zip(o["meta"]["boxes"], n["meta"]["boxes"]):
        for kk, vv in bo.items():
            a, b = vv, bn[kk]
            same = np.array_equal(a, b) if hasattr(a, "shape") else (a == b or (isinstance(a, float) and np.isnan(a) and np.isnan(b)))
            if not same: diff_layers.setdefault("box." + kk, 0); diff_layers["box." + kk] += 1
        extra = set(bn) - set(bo)
    assert len(o["meta"]["boxes"]) == len(n["meta"]["boxes"])
print("layers compared", nlay, "differing layers (count of keyframes):", diff_layers)
print("new box fields:", sorted(extra))
# more
cam_cnt = []; sh = []
for tok, n in new.items():
    for b in n["meta"]["boxes"]:
        sh.append((b["share_camera"], b["share_radar"], b["share_fusion"], b["observed"], b["group"], b["num_lidar_pts"], b["num_radar_pts"]))
sh = np.array([(s[0], s[1], s[2], s[3]) for s in sh], float)
print("boxes", len(sh), "share_fusion in (0,1):", int(((sh[:, 2] > 0) & (sh[:, 2] < 1)).sum()), "fusion>camera violations:", int((sh[:, 2] > sh[:, 0] + 1e-9).sum()), int((sh[:, 2] > sh[:, 1] + 1e-9).sum()))
# occlusion old vs new
tot_old = tot_new = notsub = 0
for tok in old:
    a, b = old[tok]["occluded"] > 0, new[tok]["occluded"] > 0
    tot_old += a.sum(); tot_new += b.sum(); notsub += (b & ~a).sum()
print("occluded cells old", int(tot_old), "new", int(tot_new), "new not in old", int(notsub))
