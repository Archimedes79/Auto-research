"""Loop 18 / WP4 test 1b (one-off): which property separates the true predicted forks (split vertex within 6 m of a ground-truth split) from the false ones?
Features of every predicted split vertex of the champion graphs (873 VAL frames): junction-head heat at the vertex (3 x 3 maximum), fork angle, length of the shorter branch, distance to the ego, number of successors.
Prints AUROC per feature and the precision / recall obtained by thresholding the best one. Merge vertices are treated the same way with the merge heat.
Usage (gcar): p18_fork_features.py"""
import os
import pickle
import sys

import numpy as np
from scipy.spatial import cKDTree

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import graph_eval as GE  # noqa: E402
import p16_infer as PI  # noqa: E402
import p16_timing as TM  # noqa: E402

raw, gts = pickle.load(open(os.path.expanduser("~/exp_gcar_b123/p18/raw_val873.pkl"), "rb"))
net, cfg = TM.load_builder(288, 144)


def auroc(pos, neg):
    allv = np.concatenate([pos, neg])
    rk = allv.argsort().argsort() + 1.0
    return float((rk[: len(pos)].sum() - len(pos) * (len(pos) + 1) / 2) / (len(pos) * len(neg)))


def forks_of(g, kind):
    """list of (position, [successor piece indices]) for split (kind 0) or merge (kind 1) vertices"""
    n = len(g["pts"])
    out, inn = {}, {}
    for i, j in g["edges"]:
        out.setdefault(i, []).append(j)
        inn.setdefault(j, []).append(i)
    res = []
    if kind == 0:
        for i, js in out.items():
            if len(js) >= 2 and len(g["pts"][i]):
                res.append((np.asarray(g["pts"][i][-1], float), js))
    else:
        for j, is_ in inn.items():
            if len(is_) >= 2 and len(g["pts"][j]):
                res.append((np.asarray(g["pts"][j][0], float), is_))
    return res


def plen(p):
    p = np.asarray(p, float)
    return float(np.linalg.norm(np.diff(p, axis=0), axis=1).sum()) if len(p) > 1 else 0.0


rows = {0: [], 1: []}
for r, gt in zip(raw, gts):
    g = PI.assemble(r, None, 0.7, cfg, fast=2, **TM.ASM)
    gs, gm = GE.junctions(gt)
    jm = r["junc"]
    for kind, gtj in ((0, gs), (1, gm)):
        F = forks_of(g, kind)
        if not F:
            continue
        tree = cKDTree(gtj) if len(gtj) else None
        hm = jm[kind].astype(np.float32)
        for pos, js in F:
            i = int(np.clip(round((pos[0] + 50) / 0.5 - 0.5), 1, 198))
            j = int(np.clip(round((pos[1] + 50) / 0.5 - 0.5), 1, 198))
            heat = float(hm[i - 1 : i + 2, j - 1 : j + 2].max())
            dirs = []
            lens = []
            for k in js:
                p = np.asarray(g["pts"][k], float)
                lens.append(plen(p))
                q = p[: min(len(p), 3)] if kind == 0 else p[-min(len(p), 3):][::-1]
                d = (q[-1] - q[0]) if kind == 0 else (q[0] - q[-1])
                dirs.append(d / max(np.linalg.norm(d), 1e-9))
            ang = float(np.degrees(np.arccos(np.clip(dirs[0] @ dirs[1], -1, 1)))) if len(dirs) >= 2 else 0.0
            tp = tree is not None and tree.query(pos)[0] <= 6.0
            rows[kind].append((heat, ang, min(lens), float(np.linalg.norm(pos)), len(js), float(tp)))
for kind, nm in ((0, "split"), (1, "merge")):
    A = np.array(rows[kind])
    tp, fp = A[A[:, 5] == 1], A[A[:, 5] == 0]
    print(f"{nm}: {len(A)} predicted forks, {len(tp)} true ({len(tp) / len(A):.2f})")
    for k, fn in enumerate(("junction-head heat", "fork angle (deg)", "shorter branch (m)", "distance to ego (m)", "successors")):
        print(f"   {fn:22s} AUROC {auroc(tp[:, k], fp[:, k]):.3f}  true mean {tp[:, k].mean():.2f}  false mean {fp[:, k].mean():.2f}")
    for k, fn, ths in ((0, "heat", (0.1, 0.2, 0.3, 0.5, 0.7)), (2, "shorter branch", (3, 5, 8, 12, 20))):
        out = []
        for th in ths:
            keep = A[:, k] >= th
            out.append(f"{fn}>={th}: keep {keep.mean():.2f} precision {A[keep, 5].mean():.2f} recall-share {A[keep, 5].sum() / A[:, 5].sum():.2f}")
        print("   " + " | ".join(out))
od = "/mnt/c/ML-Experiment/bev_fusion_lane_center/runs/e18/fork_prune_val"
os.makedirs(od, exist_ok=True)
np.savez(f"{od}/fork_features.npz", k0=np.array(rows[0]), k1=np.array(rows[1]), columns=np.array(["heat", "angle", "shorter_branch", "range", "successors", "true"]))
print("saved", f"{od}/fork_features.npz")
