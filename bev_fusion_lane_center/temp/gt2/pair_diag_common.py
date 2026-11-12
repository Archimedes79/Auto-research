"""are the pair heads comparable on the same ends? Candidates restricted to the GT ends that BOTH systems found (strict one-to-one match of the keypoints to GT ends), so the head with the harder end set (E287: crowded / far ends) is not judged on ends the other never saw.
usage: pair_diag_common.py <run A> <run B> ...   (all runs: runs/<run>/ends_val.pkl); prints AUC, recall / precision at logit >= 0 and at the recall of the first run, inside the common set."""
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


runs = sys.argv[1:]
frames = {r: {f["token"]: f for f in pickle.load(open(os.path.join(ROOT, "runs", r, "ends_val.pkl"), "rb"))} for r in runs}
S = {r: [] for r in runs}
Y = {r: [] for r in runs}
n_common_pairs = 0
n_gt = 0
for sid, (name, toks) in QV.val_scenes().items():
    for t in toks:
        if not all(t in frames[r] for r in runs):
            continue
        jg = J.junction_gt(QV.load_gt(t))
        if jg is None:
            continue
        gt_pairs = {(int(p[0]), int(p[1])) for p in jg["pairs"]}
        n_gt += len(gt_pairs)
        inv = {}
        for r in runs:
            f = frames[r][t]
            K = f["logit"].shape[0]
            ve = f["valid"][0] & (f["kp_p"][:K] >= 0.5)
            vx = f["valid"][1] & (f["kp_p"][K:] >= 0.5)
            kp, kd = f["kp_pos"], f["kp_dir"]
            ma, mb = JR._match(kp[:K], kd[:K], ve, jg["ent"]), JR._match(kp[K:], kd[K:], vx, jg["ext"])
            inv[r] = ({int(k): g for g, k in enumerate(ma) if k >= 0}, {int(k): g for g, k in enumerate(mb) if k >= 0}, ve, vx)
        ents = set.intersection(*[set(inv[r][0].values()) for r in runs])
        exts = set.intersection(*[set(inv[r][1].values()) for r in runs])
        n_common_pairs += sum(1 for a, b in gt_pairs if a in ents and b in exts)
        for r in runs:
            f = frames[r][t]
            ia, ib, ve, vx = inv[r]
            ii, jj = np.where(ve[:, None] & vx[None, :] & f["allowed"].astype(bool))
            for i, j in zip(ii, jj):
                a, b = ia.get(int(i), -1), ib.get(int(j), -1)
                if a in ents and b in exts:
                    S[r].append(float(f["logit"][i, j]))
                    Y[r].append((a, b) in gt_pairs)
print(f"GT pairs {n_gt}, with both ends found by all {len(runs)} systems: {n_common_pairs}")
ref = None
for r in runs:
    s, y = np.array(S[r]), np.array(Y[r], bool)
    rec0 = ((s >= 0) & y).sum() / n_common_pairs
    prec0 = ((s >= 0) & y).sum() / max(1, (s >= 0).sum())
    if ref is None:
        ref = rec0
    o = np.argsort(-s)
    tp = np.cumsum(y[o])
    k = int(np.searchsorted(tp, ref * n_common_pairs)) + 1
    k = min(k, len(s))
    print(f"{r:12s} candidates {len(s):6d}, true {int(y.sum()):5d}, AUC {auc(s, y):.3f}, at logit >= 0: recall {rec0:.3f} precision {prec0:.3f}; precision at the recall of {runs[0]} ({ref:.3f}): {tp[k - 1] / k:.3f}")
