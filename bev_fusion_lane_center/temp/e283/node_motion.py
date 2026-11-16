"""scratch (answer to Mirko's questions of 2026-11-08): how much do the delivered nodes of baseline B move from one keyframe (0.5 s) to the next after the ego-motion warp? Matched 1:1 within 2 m; the distribution of the
residual displacement by zone (near a junction = within 8 m of a connector point of the lane GT, else away), and the share of nodes without a partner. The warp is the convention of E276 ("the pose acts on the lane frame",
median alignment of GT nodes 0.49 m): the residual contains the true jitter of the head AND the warp error, so it is an UPPER bound of what temporal tracking could smooth.
    wsl env CUDA_VISIBLE_DEVICES= bash tools/env/_py14.sh temp/e283/node_motion.py [--rows N]"""
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
from lane_task import measures as MS  # noqa: E402

N_ROWS = int(sys.argv[sys.argv.index("--rows") + 1]) if "--rows" in sys.argv else 0
BOX = 45.0
frames = QE.val_frames()
if N_ROWS:
    frames = frames[:N_ROWS]
rows = [f.row for f in frames]
by_row = {r: i for i, r in enumerate(rows)}
poses = np.load(os.path.expanduser("~/exp_gcar_b123/poses_val.npz"))
T, R, PREV = poses["t"], poses["R"], poses["prev"]
prev_of = {by_row[r]: by_row[int(PREV[r])] for r in rows if int(PREV[r]) >= 0 and int(PREV[r]) in by_row}


def warp(p, a, b):
    return (np.asarray(p, float) @ R[rows[a]].T + T[rows[a]] - T[rows[b]]) @ R[rows[b]]


out = {}
for s in (0, 1):
    dump = pickle.load(open(os.path.join(ROOT, "temp", "e276", f"predfull_e273_p_s{s}.pkl"), "rb"))[: len(frames)]
    disp = {"near a junction": [], "away": []}
    n_nodes = {"near a junction": 0, "away": 0}
    n_match = {"near a junction": 0, "away": 0}
    for b, a in prev_of.items():
        da, db = dump[a], dump[b]
        ka, kb = da["e"] >= 0.5, db["e"] >= 0.5
        Pa, Pb = warp(da["pos"][ka], a, b), db["pos"][kb]
        inb = (np.abs(Pa) < BOX).all(1)
        Pa = Pa[inb]
        if not len(Pa) or not len(Pb):
            continue
        ref = frames[b].ref()
        sp = ref.nt["pts"][ref.nt["supp"]]
        tree = cKDTree(sp) if len(sp) else None
        zone = np.where(tree.query(Pa)[0] <= 8.0, 0, 1) if tree is not None else np.ones(len(Pa), int)
        pairs = MS.match_1to1(Pa, Pb, 2.0)
        matched = np.zeros(len(Pa), bool)
        d = np.zeros(len(Pa))
        if len(pairs):
            matched[pairs[:, 0]] = True
            d[pairs[:, 0]] = np.linalg.norm(Pa[pairs[:, 0]] - Pb[pairs[:, 1]], axis=1)
        for zi, name in enumerate(("near a junction", "away")):
            m = zone == zi
            n_nodes[name] += int(m.sum())
            n_match[name] += int((m & matched).sum())
            disp[name].append(d[m & matched])
        frames[b]._lane = frames[b]._ref = None
    out[str(s)] = {k: dict(nodes=n_nodes[k], matched_share=n_match[k] / max(n_nodes[k], 1), median=float(np.median(np.concatenate(disp[k]))), p75=float(np.percentile(np.concatenate(disp[k]), 75)),
                           p90=float(np.percentile(np.concatenate(disp[k]), 90)), share_below_0_5=float((np.concatenate(disp[k]) <= 0.5).mean())) for k in disp}
    print("seed", s, json.dumps(out[str(s)]), flush=True)
json.dump(out, open(os.path.join(ROOT, "temp", "e283", "node_motion.json"), "w"), indent=1)
