"""position / heading error of the matched keypoints, per system, on the GT ends that all systems found (strict one-to-one match). usage: kp_err.py <run> <run> ..."""
import os
import pickle
import sys

import numpy as np

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
for sub in ("evaluation", "training", "training/grid", "training/data", "tools/viz", ".."):
    sys.path.insert(0, os.path.join(ROOT, sub))
import jgt as J
import jrule_control as JR
import qgraph_view as QV

runs = sys.argv[1:]
frames = {r: {f["token"]: f for f in pickle.load(open(os.path.join(ROOT, "runs", r, "ends_val.pkl"), "rb"))} for r in runs}
E = {r: dict(pos=[], ang=[]) for r in runs}
n = 0
for sid, (name, toks) in QV.val_scenes().items():
    for t in toks:
        if not all(t in frames[r] for r in runs):
            continue
        jg = J.junction_gt(QV.load_gt(t))
        if jg is None:
            continue
        inv = {}
        for r in runs:
            f = frames[r][t]
            K = f["logit"].shape[0]
            ve = f["valid"][0] & (f["kp_p"][:K] >= 0.5)
            vx = f["valid"][1] & (f["kp_p"][K:] >= 0.5)
            kp, kd = f["kp_pos"], f["kp_dir"]
            ma, mb = JR._match(kp[:K], kd[:K], ve, jg["ent"]), JR._match(kp[K:], kd[K:], vx, jg["ext"])
            inv[r] = ({g: int(k) for g, k in enumerate(ma) if k >= 0}, {g: int(k) for g, k in enumerate(mb) if k >= 0}, K)
        ents = set.intersection(*[set(inv[r][0]) for r in runs])
        exts = set.intersection(*[set(inv[r][1]) for r in runs])
        n += len(ents) + len(exts)
        for r in runs:
            f = frames[r][t]
            ia, ib, K = inv[r]
            for kind, gs, m, gt in (("ent", ents, ia, jg["ent"]), ("ext", exts, ib, jg["ext"])):
                for g in gs:
                    k = m[g] + (0 if kind == "ent" else K)
                    d = np.linalg.norm(f["kp_pos"][k] - gt[g][:2])
                    c = float(np.clip(np.dot(f["kp_dir"][k], gt[g][2:4]) / (np.linalg.norm(f["kp_dir"][k]) * np.linalg.norm(gt[g][2:4]) + 1e-9), -1, 1))
                    E[r]["pos"].append(d)
                    E[r]["ang"].append(np.degrees(np.arccos(c)))
print("common GT ends:", n)
for r in runs:
    p, a = np.array(E[r]["pos"]), np.array(E[r]["ang"])
    print(f"{r:12s} position error: mean {p.mean():.2f} m, median {np.median(p):.2f}, 90% {np.percentile(p, 90):.2f}; heading error: mean {a.mean():.1f} deg, median {np.median(a):.1f}, 90% {np.percentile(a, 90):.1f}")
