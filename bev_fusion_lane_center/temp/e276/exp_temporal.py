"""E276 pre-experiment P5 (video note 3: "no temporally consistent behaviour, especially on intersections"), CPU: how much do the delivered nodes of B flicker from one keyframe (0.5 s) to the next, compared
with the GT nodes? From the E275 node dump (temp/e275/pred_<run>.pkl, e >= 0.2, exact map), the poses of the VAL file (poses_val.npz: t, R, prev) and the lane GT l4 (store folder, plain pickles).
For every pair (previous keyframe, keyframe) of one scene: the nodes of the previous frame are moved into the frame of the keyframe by the ego motion (the convention of the pose file is found by the
GT: the one that aligns the GT nodes of the two frames), and counted as PERSISTENT when a node of the same kind lies within 2 m in the keyframe: delivered nodes (e >= 0.5) against delivered nodes,
GT nodes against GT nodes (the ceiling: the GT snaps to a 2 m grid, so it is not 1.0). Split by the zone of the node (the tier of the nearest GT lane point: own lane 1, neighbours 2, rest 3), and by
"near an intersection" (within 3 m of a GT connector point inside an intersection). The excess flicker = GT persistence - delivered persistence.
    wsl env CUDA_VISIBLE_DEVICES= bash tools/env/_py14.sh exp_temporal.py <run> <l4 store folder>
"""
import gzip
import json
import os
import pickle
import sys

import numpy as np
from scipy.spatial import cKDTree

ROOT = os.environ["BFLC_ROOT"]
for sub in ("training", "training/grid", "training/data", "evaluation"):
    sys.path.insert(0, os.path.join(ROOT, sub))
import qgraph_eval as QE  # noqa: E402

run, store = sys.argv[1], sys.argv[2]
THR, TOL, BOXM = 0.5, 2.0, 45.0
frames = QE.val_frames()
rows = [f.row for f in frames]
by_row = {f.row: i for i, f in enumerate(frames)}
dump = pickle.load(open(os.path.join(ROOT, "temp", "e275", f"pred_{run}.pkl"), "rb"))
assert [d["row"] for d in dump] == rows
poses = np.load(os.path.expanduser("~/exp_gcar_b123/poses_val.npz"))
T, R, PREV = poses["t"], poses["R"], poses["prev"]
pairs = [(by_row[int(PREV[r])], by_row[r]) for r in rows if int(PREV[r]) >= 0 and int(PREV[r]) in by_row]
print(len(pairs), "keyframe pairs of VAL", flush=True)


def graph(f):
    return pickle.load(gzip.open(os.path.join(store, f.token + ".pkl.gz"), "rb"))


def lane2ego(p):
    return np.c_[p[:, 1], -p[:, 0]]


def ego2lane(e):
    return np.c_[-e[:, 1], e[:, 0]]


CONV = {}
CONV["ego pose, lane frame converted"] = lambda p, ra, ta, rb, tb: ego2lane((lane2ego(p) @ ra.T + ta - tb) @ rb)                  # g = R e + t, e in the ego frame
CONV["pose acts on the lane frame"] = lambda p, ra, ta, rb, tb: (p @ ra.T + ta - tb) @ rb
CONV["ego pose, R transposed"] = lambda p, ra, ta, rb, tb: ego2lane((lane2ego(p) @ ra + ta - tb) @ rb.T)


def warp(conv, p, a, b):
    return CONV[conv](np.asarray(p, float), R[frames[a].row], T[frames[a].row], R[frames[b].row], T[frames[b].row])


# 1. the convention of the pose file, from the GT nodes of 40 pairs
test = pairs[:: max(1, len(pairs) // 40)][:40]
score = {c: [] for c in CONV}
for a, b in test:
    Ga, Gb = QE.LaneRef(graph(frames[a])), QE.LaneRef(graph(frames[b]))
    if not len(Ga.nt["pos"]) or not len(Gb.nt["pos"]):
        continue
    tree = cKDTree(Gb.nt["pos"])
    for c in CONV:
        w = warp(c, Ga.nt["pos"], a, b)
        m = (np.abs(w) < BOXM).all(1)
        if m.any():
            score[c].append(float(np.median(tree.query(w[m])[0])))
best = min(score, key=lambda c: np.median(score[c]))
print("alignment of GT nodes between consecutive frames (median NN distance, m):", {c: round(float(np.median(v)), 2) for c, v in score.items()}, "-> convention:", best, flush=True)
assert np.median(score[best]) < 1.5, "no convention of the pose file aligns the GT nodes: the pairs are not usable"

# 2. persistence
acc = {}


def add(key, hit, n=1):
    a = acc.setdefault(key, np.zeros(2))
    a += (hit, n)


for a, b in pairs:
    fa, fb = frames[a], frames[b]
    ra, rb = QE.LaneRef(graph(fa)), QE.LaneRef(graph(fb))
    da, db = dump[a], dump[b]
    Pa, Pb = da["pos"][da["e"] >= THR], db["pos"][db["e"] >= THR]
    sets = {"delivered": (Pa, Pb), "GT": (ra.nt["pos"], rb.nt["pos"])}
    inter = []
    for ref in (rb,):
        pts = []
        g = graph(fb)
        for p in g["pieces"]:
            q = np.asarray(p["pts"], float)
            for s in p.get("segments", []):
                if s.get("kind") == "connector" and s.get("in_intersection"):
                    pts.append(q[s["i0"]: s["i1"] + 1])
        inter = cKDTree(np.concatenate(pts)) if pts else None
    for kind, (A, B) in sets.items():
        if not len(A) or not len(B):
            continue
        w = warp(best, A, a, b)
        m = (np.abs(w) < BOXM).all(1)
        w = w[m]
        if not len(w):
            continue
        d = cKDTree(B).query(w)[0]
        hit = d <= TOL
        zone = np.zeros(len(w), int)
        if rb.tree is not None and rb.labelled:
            dz, iz = rb.tree.query(w)
            zone = np.where(dz <= 2.5, np.where(rb.ptier[iz] > 0, rb.ptier[iz], 3), 0)
        near = (inter.query(w)[0] <= 3.0) if inter is not None else np.zeros(len(w), bool)
        add((kind, "all"), hit.sum(), len(hit))
        for z in (1, 2, 3):
            add((kind, f"zone {z}"), hit[zone == z].sum(), (zone == z).sum())
        add((kind, "near an intersection"), hit[near].sum(), near.sum())
        add((kind, "elsewhere"), hit[~near].sum(), (~near).sum())
        add((kind, "own lane near an intersection"), hit[near & (zone == 1)].sum(), (near & (zone == 1)).sum())
        add((kind, "own lane elsewhere"), hit[~near & (zone == 1)].sum(), (~near & (zone == 1)).sum())
out = {f"{k[0]} / {k[1]}": dict(persistent=float(v[0] / v[1]) if v[1] else None, n=int(v[1])) for k, v in acc.items()}
json.dump(dict(run=run, convention=best, pairs=len(pairs), result=out), open(os.path.join(ROOT, "temp", "e276", f"temporal_{run}.json"), "w"), indent=1)
print(f"\n{run}: persistence of nodes from the previous keyframe (0.5 s) moved by the ego motion, a node of the same kind within {TOL} m in the keyframe, {len(pairs)} pairs")
print("| group | GT nodes | delivered nodes | excess flicker (GT - delivered) |")
print("|---|---|---|---|")
for g in ("all", "zone 1", "zone 2", "zone 3", "near an intersection", "elsewhere", "own lane near an intersection", "own lane elsewhere"):
    a, b = out.get(f"GT / {g}"), out.get(f"delivered / {g}")
    if a and b and a["persistent"] is not None and b["persistent"] is not None:
        print(f"| {g} | {a['persistent']:.3f} (n {a['n']}) | {b['persistent']:.3f} (n {b['n']}) | {a['persistent'] - b['persistent']:+.3f} |")
