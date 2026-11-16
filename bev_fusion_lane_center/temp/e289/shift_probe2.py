"""E289 step 1b (scratch): LINE stability of the delivered graph of B under a shift of the input features.
usage: shift_probe2.py <seed> [max frames]    -> runs/e289/shift_probe2_s<seed>.json

As shift_probe.py, but the delivered graphs (nodes + links + sections, the hard step close:1.5@0.5, `qgraph_eval.readout`) of the original and of the shifted-and-un-shifted run are compared as LINES: the lane curves of both are
sampled every 0.5 m (|x|, |y| < 44 m); line_F1 = the harmonic mean of the share of the points of one graph within 0.5 m of a point of the other, both ways; vertex_to_line = the median distance of the delivered vertices (nodes) of the shifted run to the lines of the original."""
import json
import os
import sys
import time

import numpy as np
import torch
from scipy.spatial import cKDTree

R = os.environ["BFLC_ROOT"]
for sub in ("training", "training/grid", "training/data", "training/data/prep", "evaluation"):
    sys.path.insert(0, os.path.join(R, sub))
import lane_gt_data as LD
import qgraph_eval as QE
import qgraph_readout as QR
import qpredict as QP

seed = int(sys.argv[1])
maxn = int(sys.argv[2]) if len(sys.argv) > 2 else 0
dev = "cuda" if torch.cuda.is_available() else "cpu"
m = QP.load_model(os.path.join(R, "runs", f"e273_p_s{seed}"), dev)
m.eval()
ds, dl = LD.loader("val_clean", bs=4, workers=2, shuffle=False, map_mode="exact", pixels=False)
SHIFTS = [(0, 0), (1, 0), (0, 1), (2, 2), (4, 0)]
CUT = 44.0
spec = QR.parse("close:1.5@0.5")


def pts_of(graph):
    out = []
    for ln in graph.lines:
        ln = np.asarray(ln, float)
        if len(ln) < 2:
            continue
        seg = np.linalg.norm(np.diff(ln, axis=0), axis=1)
        s = np.r_[0, np.cumsum(seg)]
        t = np.arange(0, s[-1] + 1e-6, 0.5)
        out.append(np.stack([np.interp(t, s, ln[:, 0]), np.interp(t, s, ln[:, 1])], 1))
    P = np.concatenate(out) if out else np.zeros((0, 2))
    return P[(np.abs(P) < CUT).all(1)]


def graph_of(pred, shift):
    p = dict(pred)
    p["nodes"] = dict(pred["nodes"])
    p["nodes"]["pos"] = np.asarray(pred["nodes"]["pos"], float).reshape(-1, 2) - shift
    q = QR.apply(p, spec)
    g, _ = QE.readout(q, 0.5, 0.5)
    return g


acc = {s: dict(f1=[], d=[]) for s in SHIFTS}
t0, n = time.time(), 0
for b in dl:
    p0 = QP.predict(m, b)
    for s in SHIFTS:
        b2 = dict(b)
        b2["x"] = torch.roll(b["x"], shifts=(s[0], s[1]), dims=(2, 3))
        ps = QP.predict(m, b2)
        sh = np.array([0.5 * s[0], 0.5 * s[1]])
        for i in range(len(p0)):
            g0, gs = graph_of(p0[i], np.zeros(2)), graph_of(ps[i], sh)
            A, B_ = pts_of(g0), pts_of(gs)
            if len(A) < 5 or len(B_) < 5:
                continue
            ta, tb = cKDTree(A), cKDTree(B_)
            prec = float((ta.query(B_)[0] <= 0.5).mean())
            rec = float((tb.query(A)[0] <= 0.5).mean())
            acc[s]["f1"].append(2 * prec * rec / max(prec + rec, 1e-9))
            nodes = np.asarray(gs.nodes if hasattr(gs, "nodes") else [], dtype=object)
            vp = np.asarray(ps[i]["nodes"]["pos"], float).reshape(-1, 2) - sh
            ev = np.asarray(ps[i]["nodes"]["e"], float).reshape(-1)
            vp = vp[(ev >= 0.5) & (np.abs(vp) < CUT).all(1)]
            if len(vp):
                acc[s]["d"] += ta.query(vp)[0].tolist()
    n += len(p0)
    if maxn and n >= maxn:
        break
res = dict(seed=seed, frames=n, seconds=round(time.time() - t0))
for s, a in acc.items():
    res[f"shift_{s[0] * 0.5:g}x_{s[1] * 0.5:g}y"] = dict(line_f1_mean=float(np.mean(a["f1"])) if a["f1"] else None, line_f1_p10=float(np.percentile(a["f1"], 10)) if a["f1"] else None,
                                                         vertex_to_line_median_m=float(np.median(a["d"])) if a["d"] else None, vertex_to_line_p90_m=float(np.percentile(a["d"], 90)) if a["d"] else None, frames=len(a["f1"]))
print(json.dumps(res, indent=1))
os.makedirs(os.path.join(R, "runs", "e289"), exist_ok=True)
json.dump(res, open(os.path.join(R, "runs", "e289", f"shift_probe2_s{seed}.json"), "w"), indent=1)
