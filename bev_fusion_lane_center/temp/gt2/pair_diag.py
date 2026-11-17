"""where does the pair head lose against E284? candidates split by whether the two keypoints are matched to GT ends (strict one-to-one) and the AUC of the logit inside the group 'both ends matched' (true pair vs another pair of true ends)"""
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


def auc(s, y):
    s, y = np.asarray(s), np.asarray(y, bool)
    if y.sum() == 0 or (~y).sum() == 0:
        return float("nan")
    r = np.argsort(np.argsort(s)) + 1
    return (r[y].sum() - y.sum() * (y.sum() + 1) / 2) / (y.sum() * (~y).sum())


for run in sys.argv[1:]:
    frames = {f["token"]: f for f in pickle.load(open(os.path.join(ROOT, "runs", run, "ends_val.pkl"), "rb"))}
    S = {"both": [], "one": [], "none": []}
    L = {"both": [], "one": [], "none": []}
    n_gt = 0
    for sid, (name, toks) in QV.val_scenes().items():
        for t in toks:
            if t not in frames:
                continue
            jg = J.junction_gt(QV.load_gt(t))
            f = frames[t]
            K = f["logit"].shape[0]
            gt_pairs = set() if jg is None else {(int(p[0]), int(p[1])) for p in jg["pairs"]}
            n_gt += len(gt_pairs)
            ve = f["valid"][0] & (f["kp_p"][:K] >= 0.5)
            vx = f["valid"][1] & (f["kp_p"][K:] >= 0.5)
            kp, kd = f["kp_pos"], f["kp_dir"]
            if jg is not None:
                ma, mb = JR._match(kp[:K], kd[:K], ve, jg["ent"]), JR._match(kp[K:], kd[K:], vx, jg["ext"])
                inv_a, inv_b = {int(k): g for g, k in enumerate(ma) if k >= 0}, {int(k): g for g, k in enumerate(mb) if k >= 0}
            else:
                inv_a = inv_b = {}
            ii, jj = np.where(ve[:, None] & vx[None, :] & f["allowed"].astype(bool))
            for i, j in zip(ii, jj):
                a, b = inv_a.get(int(i), -1), inv_b.get(int(j), -1)
                g = "both" if (a >= 0 and b >= 0) else ("one" if (a >= 0 or b >= 0) else "none")
                S[g].append(float(f["logit"][i, j]))
                L[g].append((a, b) in gt_pairs)
    print(run, "GT pairs", n_gt)
    for g in S:
        s, y = np.array(S[g]), np.array(L[g], bool)
        print(f"  candidates with {g:4s} ends matched: n {len(s):6d}, true {int(y.sum()):5d}, mean logit {s.mean() if len(s) else float('nan'):.2f}, delivered (logit>=0) {int((s >= 0).sum()):5d}, AUC {auc(s, y):.3f}")
