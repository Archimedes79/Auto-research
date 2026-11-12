"""E276 P5b (video note 3, the LINKS), CPU: how much do the delivered LINKS of B flicker from one keyframe (0.5 s) to the next, compared with the GT links? From the FULL dump (temp/e276/predfull_<run>.pkl,
nodes e >= 0.2, links p >= 0.05, exact map), the poses of the VAL file (poses_val.npz: t, R, prev) and the lane GT l4 (store folder).
For every pair (previous keyframe, keyframe) of one scene: both ends of a link of the previous frame are moved into the frame of the keyframe by the ego motion (the convention of the pose file is found by the
GT: the one that aligns the GT nodes of the two frames) and the link is PERSISTENT when the keyframe has a link of the same graph with the first end within 2 m and the second end within 2 m (delivered: p >= 0.5
and e >= 0.5 at both ends; GT: the GT node links, the ceiling: the GT snaps to a 2 m grid, so it is not 1.0). Split by short / long link, by the zone of the link's midpoint (the tier of the nearest GT lane point:
own lane 1, neighbours 2, rest 3) and by "near an intersection" (midpoint within 3 m of a GT connector point inside an intersection). The excess flicker = GT persistence - delivered persistence.
    wsl env CUDA_VISIBLE_DEVICES= bash tools/env/_py14.sh exp_temporal_links.py <run, e.g. e273_p_s0> <l4 store folder>
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
dump = pickle.load(open(os.path.join(ROOT, "temp", "e276", f"predfull_{run}.pkl"), "rb"))
assert [d["row"] for d in dump] == rows
poses = np.load(os.path.expanduser("~/exp_gcar_b123/poses_val.npz"))
T, R, PREV = poses["t"], poses["R"], poses["prev"]
pairs = [(by_row[int(PREV[r])], by_row[r]) for r in rows if int(PREV[r]) >= 0 and int(PREV[r]) in by_row]
if "--pairs" in sys.argv:                                              # a smoke on the first N pairs
    pairs = pairs[: int(sys.argv[sys.argv.index("--pairs") + 1])]
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

# 2. persistence of links
acc = {}


def add(key, hit, n=1):
    a = acc.setdefault(key, np.zeros(2))
    a += (hit, n)


def links_of_dump(d):
    keep = d["e"] >= THR
    s, t, p = d["link_src"], d["link_dst"], d["link_p"]
    m = (p >= THR) & keep[s] & keep[t] & (s != t)
    return d["pos"][s[m]], d["pos"][t[m]], np.asarray(d["link_long"], bool)[m]


def links_of_gt(ref):
    nt = ref.nt
    s, t = np.asarray(nt["src"]).astype(int), np.asarray(nt["dst"]).astype(int)
    return nt["pos"][s], nt["pos"][t], np.asarray(nt["long"]).astype(bool)


def persistent(A, B, C, D):
    """A', B' (the moved ends of the old links) against the links C, D of the keyframe -> bool per old link"""
    if not len(C):
        return np.zeros(len(A), bool)
    tc = cKDTree(C)
    hit = np.zeros(len(A), bool)
    for i, cand in enumerate(tc.query_ball_point(A, TOL)):
        if cand and (np.linalg.norm(D[cand] - B[i], axis=1) <= TOL).any():
            hit[i] = True
    return hit


for a, b in pairs:
    fa, fb = frames[a], frames[b]
    ra, rb = QE.LaneRef(graph(fa)), QE.LaneRef(graph(fb))
    gb = graph(fb)
    pts = []
    for p in gb["pieces"]:
        q = np.asarray(p["pts"], float)
        for s in p.get("segments", []):
            if s.get("kind") == "connector" and s.get("in_intersection"):
                pts.append(q[s["i0"]: s["i1"] + 1])
    inter = cKDTree(np.concatenate(pts)) if pts else None
    sets = {"delivered": (links_of_dump(dump[a]), links_of_dump(dump[b])), "GT": (links_of_gt(ra), links_of_gt(rb))}
    for kind, ((A, B, La), (C, D, Lc)) in sets.items():
        if not len(A):
            continue
        wa, wb = warp(best, A, a, b), warp(best, B, a, b)
        m = (np.abs(wa) < BOXM).all(1) & (np.abs(wb) < BOXM).all(1)
        wa, wb, La = wa[m], wb[m], La[m]
        if not len(wa):
            continue
        hit = persistent(wa, wb, C, D)
        mid = 0.5 * (wa + wb)
        zone = np.zeros(len(mid), int)
        if rb.tree is not None and rb.labelled:
            dz, iz = rb.tree.query(mid)
            zone = np.where(dz <= 2.5, np.where(rb.ptier[iz] > 0, rb.ptier[iz], 3), 0)
        near = (inter.query(mid)[0] <= 3.0) if inter is not None else np.zeros(len(mid), bool)
        add((kind, "all"), hit.sum(), len(hit))
        add((kind, "short links"), hit[~La].sum(), (~La).sum())
        add((kind, "long links"), hit[La].sum(), La.sum())
        for z in (1, 2, 3):
            add((kind, f"zone {z}"), hit[zone == z].sum(), (zone == z).sum())
        add((kind, "near an intersection"), hit[near].sum(), near.sum())
        add((kind, "elsewhere"), hit[~near].sum(), (~near).sum())
        add((kind, "own lane near an intersection"), hit[near & (zone == 1)].sum(), (near & (zone == 1)).sum())
        add((kind, "own lane elsewhere"), hit[~near & (zone == 1)].sum(), (~near & (zone == 1)).sum())
out = {f"{k[0]} / {k[1]}": dict(persistent=float(v[0] / v[1]) if v[1] else None, n=int(v[1])) for k, v in acc.items()}
json.dump(dict(run=run, convention=best, pairs=len(pairs), result=out), open(os.path.join(ROOT, "temp", "e276", f"temporal_links_{run}.json"), "w"), indent=1)
print(f"\n{run}: persistence of LINKS from the previous keyframe (0.5 s) moved by the ego motion, a link with both ends within {TOL} m in the keyframe, {len(pairs)} pairs")
print("| group | GT links | delivered links | excess flicker (GT - delivered) |")
print("|---|---|---|---|")
for g in ("all", "short links", "long links", "zone 1", "zone 2", "zone 3", "near an intersection", "elsewhere", "own lane near an intersection", "own lane elsewhere"):
    a, b = out.get(f"GT / {g}"), out.get(f"delivered / {g}")
    if a and b and a["persistent"] is not None and b["persistent"] is not None:
        print(f"| {g} | {a['persistent']:.3f} (n {a['n']}) | {b['persistent']:.3f} (n {b['n']}) | {a['persistent'] - b['persistent']:+.3f} |")
