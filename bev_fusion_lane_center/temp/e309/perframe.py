"""E309 scratch: the delivered graphs of B, B + E284 curves, B + E287 p2 curves per VAL frame (seed 0), the metrics per frame and the lenient coverage numbers.
usage: wsl bash tools/env/_py14.sh perframe.py [seed]   -> temp/e309/perframe_s<seed>.pkl"""
import os
import pickle
import sys
import time

import numpy as np
from scipy.spatial import cKDTree

ROOT = os.environ["BFLC_ROOT"]
for sub in ("evaluation", "training/grid", "training", "training/data", "evaluation/sparse_lanes", ".."):
    sys.path.insert(0, os.path.normpath(os.path.join(ROOT, sub)))
import qends_eval as QEE  # noqa: E402
import qgraph_eval as QE  # noqa: E402
import qgraph_readout as QR  # noqa: E402

seed = int(sys.argv[1]) if len(sys.argv) > 1 else 0
RUNS = {"e284": f"e284_h2_s{seed}", "e287p2": f"e287_p2_s{seed}", "e312": f"e312_p_s{seed}"}
THR = {"e312": 0.1}
SNAP = {"e312": True}
dump = pickle.load(open(os.path.join(ROOT, "temp", "e276", f"predfull_e273_p_s{seed}.pkl"), "rb"))
ends = {k: {f["row"]: f for f in pickle.load(open(os.path.join(ROOT, "runs", r, "ends_val.pkl"), "rb"))} for k, r in RUNS.items()}
frames = QE.val_frames()
spec = QR.parse("close:1.5@0.5")


def resample(p, ds=0.5):
    p = np.asarray(p, float)
    if len(p) < 2:
        return p
    seg = np.r_[0, np.cumsum(np.linalg.norm(np.diff(p, axis=0), axis=1))]
    if seg[-1] < 1e-6:
        return p[:1]
    s = np.arange(0, seg[-1] + 1e-9, ds)
    return np.c_[np.interp(s, seg, p[:, 0]), np.interp(s, seg, p[:, 1])]


def lenient(lines, gt_pieces, tol=2.0, box=48.0):
    """share of the GT length within `tol` m of a delivered point (coverage) and share of the delivered length within `tol` of GT (precision), inside the metrics' box, no direction, no topology"""
    P = np.vstack([resample(l) for l in lines if len(l) > 1] or [np.zeros((0, 2))])
    G = np.vstack([resample(p) for p in gt_pieces if len(p) > 1] or [np.zeros((0, 2))])
    inb = lambda A: A[(np.abs(A[:, 0]) < box) & (np.abs(A[:, 1]) < box)] if len(A) else A  # noqa: E731
    P, G = inb(P), inb(G)
    if len(G) == 0 or len(P) == 0:
        return dict(cov=float("nan") if len(G) == 0 else 0.0, prec=float("nan") if len(P) == 0 else 0.0, n_gt=len(G) * 0.5, n_pred=len(P) * 0.5)
    tg, tp = cKDTree(G), cKDTree(P)
    return dict(cov=float((tp.query(G)[0] <= tol).mean()), prec=float((tg.query(P)[0] <= tol).mean()), n_gt=len(G) * 0.5, n_pred=len(P) * 0.5)


out = []
t0 = time.time()
for b, (f, d) in enumerate(zip(frames, dump)):
    pred = QR.apply(dict(nodes=dict(pos=d["pos"], dirv=d["dirv"], e=d["e"], flags=d["flags"], section=d["section"], s=d.get("s")),
                         links=dict(src=d["link_src"], dst=d["link_dst"], p=d["link_p"], long=d["link_long"], a=d["link_a"], b=d["link_b"])), spec)
    graph, bk = QE.readout(pred, 0.5, 0.5)
    g = f.lane()
    gt_pieces = [np.asarray(pc["pts"]) for pc in g["pieces"]]
    rec = dict(token=f.token, row=f.row, scene=f.scene, gt_pieces=gt_pieces, n_gt_lines=len(gt_pieces))
    variants = {"B": (graph.lines, graph.edges)}
    for k in ends:
        e = ends[k][f.row]
        kk = e["kp_pos"].shape[0] // 2
        p = 1.0 / (1.0 + np.exp(-e["logit"])) * e["allowed"]
        ia, ib = np.nonzero(p >= THR.get(k, 0.5))
        pairs = [(e["kp_pos"][i], e["kp_dir"][i], e["kp_pos"][kk + j], e["kp_dir"][kk + j]) + (tuple(e["scale"][i, j]) if e.get("scale") is not None else ()) for i, j in zip(ia, ib)]
        raw, _, _ = QEE.add_connectors(graph.lines, graph.edges, graph.nodes, bk["kind"], bk["piece_links"], np.asarray(pred["links"]["long"], bool), pairs, snap=SNAP.get(k, False))
        variants[k] = (raw["pts"] if isinstance(raw, dict) else raw.lines, None)
        variants[k + "_graph"] = raw
    for k in ("B",) + tuple(ends):
        if k == "B":
            c = QE.graph_counts(graph, f)
            lines = graph.lines
        else:
            raw = variants[k + "_graph"]
            c = QE.graph_counts(raw, f)
            lines = variants[k][0]
        num = QE.graph_numbers([c])
        rec[k] = dict(num={x: float(num[x]) for x in ("topo_F1", "topo_P", "topo_R", "F1@1.0", "Q", "lanes_found", "junction_F1", "dup_frac") if x in num}, len=lenient(lines, gt_pieces),
                      lines=[np.asarray(l, np.float32) for l in lines])
    f._lane = f._ref = None
    out.append(rec)
    if (b + 1) % 100 == 0:
        print(f"{b + 1}/{len(frames)} {time.time() - t0:.0f} s", flush=True)
pickle.dump(out, open(os.path.join(ROOT, "temp", "e309", f"perframe_s{seed}.pkl"), "wb"))
print("done")
