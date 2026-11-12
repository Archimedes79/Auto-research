"""how well does the connector drawn from the KEYPOINTS of a pair-head run follow the GT connector of its pair? For every GT pair whose two ends are strictly matched to keypoints: share of the GT connector points within 1.0 / 1.5 m of the curve, for the chord-tangent curve and the curve with the handle scales fitted to the GT (qjunction.fit_pair_scales on the true ends), once from the keypoints and once from the TRUE ends. usage: curve_vs_gt.py <run>"""
import os
import pickle
import sys

import numpy as np
from scipy.spatial import cKDTree

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
for sub in ("evaluation", "training", "training/grid", "training/data", "tools/viz", ".."):
    sys.path.insert(0, os.path.join(ROOT, sub))
import jgt as J
import jrule_control as JR
import qends as QENDS
import qjunction as QJ
import qgraph_view as QV

run = sys.argv[1]
frames = pickle.load(open(os.path.join(ROOT, "runs", run, "ends_val.pkl"), "rb"))
acc = {k: [] for k in ("kp_chord", "kp_fit", "true_chord", "true_fit")}
dpos, dang = [], []
n = 0
for e in frames:
    jg = J.junction_gt(QV.load_gt(e["token"]))
    if jg is None:
        continue
    K = e["logit"].shape[0]
    ve = e["valid"][0] & (e["kp_p"][:K] >= 0.5)
    vx = e["valid"][1] & (e["kp_p"][K:] >= 0.5)
    ma, mb = JR._match(e["kp_pos"][:K], e["kp_dir"][:K], ve, jg["ent"]), JR._match(e["kp_pos"][K:], e["kp_dir"][K:], vx, jg["ext"])
    sc = QJ.fit_pair_scales(jg)
    for ip, (r, pts) in enumerate(zip(jg["pairs"], jg["pair_pts"])):
        a, b = int(r[0]), int(r[1])
        tgt = np.asarray(pts, float)
        if len(tgt) < 3 or ma[a] < 0 or mb[b] < 0 or not np.isfinite(sc[ip]).all():
            continue
        tree_t = tgt
        pa, ta = e["kp_pos"][ma[a]], e["kp_dir"][ma[a]]
        pb, tb = e["kp_pos"][K + mb[b]], e["kp_dir"][K + mb[b]]
        dpos.append(np.linalg.norm(pa - jg["ent"][a, :2]))
        dpos.append(np.linalg.norm(pb - jg["ext"][b, :2]))
        for name, (p0, t0, p1, t1) in (("kp", (pa, ta, pb, tb)), ("true", (jg["ent"][a, :2], jg["ent"][a, 2:], jg["ext"][b, :2], jg["ext"][b, 2:]))):
            for cn, c in (("chord", QENDS.curve(p0, t0, p1, t1)), ("fit", QENDS.curve_s(np.asarray(p0, float), np.asarray(t0, float), np.asarray(p1, float), np.asarray(t1, float), float(sc[ip][0]), float(sc[ip][1])))):
                d = cKDTree(c).query(tgt)[0]
                acc[f"{name}_{cn}"].append((float((d <= 1.0).mean()), float((d <= 1.5).mean()), float(d.max())))
        n += 1
print(f"{run}: {n} GT pairs with both ends matched and a connector; keypoint position error mean {np.mean(dpos):.2f} m")
for k, v in acc.items():
    a = np.array(v)
    print(f"{k:12s} GT connector points within 1.0 m: {a[:, 0].mean():.3f}, within 1.5 m: {a[:, 1].mean():.3f}; pairs with >= 90 % within 1.5 m: {(a[:, 1] >= 0.9).mean():.3f}; median max deviation {np.median(a[:, 2]):.2f} m")
