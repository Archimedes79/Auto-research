"""one-off (E275 step 3): what the close-node suppression drops, from the dumped nodes (temp/e275/pred_<run>.pkl, exact map, candidates e >= 0.2) and the lane GT of the 873 VAL frames. CPU only.

    wsl env CUDA_VISIBLE_DEVICES= bash tools/env/_py14.sh analyse.py <run> [<run> ...]       e.g. e273_p_s0 e273_p_s1 e274_r_s0 e274_r_s1
writes temp/e275/analysis_<run>.json and the pictures results/E275/figures/<run>_dropped_<k>.png (20 frames with the most dropped nodes; not in git)
numbers (predictions of the card): (2) share of the dropped nodes inside intersections (proxy: within 3 m of a GT connector point inside an intersection) against the base rate of all delivered nodes;
(3) share of GT nodes that have another GT node within 1.5 m (the recall the step cannot keep); node recall / precision before and after (the scorer's one-to-one matching).
"""
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
import qgraph_readout as RO  # noqa: E402
from lane_task import measures as MS  # noqa: E402

R, THR, NEAR_INTER = 1.5, 0.5, 3.0
runs = sys.argv[1:]
DIR = os.environ.get("E275_DIR", "e275")                       # e275smoke for the smoke test on the first frames
all_frames = QE.val_frames()
os.makedirs(os.path.join(ROOT, "results", DIR.upper(), "figures"), exist_ok=True)


def inter_tree(g):
    """KD-tree of the GT points that lie in an intersection connector segment (None when the frame has none)"""
    pts = []
    for p in g["pieces"]:
        q = np.asarray(p["pts"], float)
        for s in p.get("segments", []):
            if s.get("kind") == "connector" and s.get("in_intersection"):
                pts.append(q[s["i0"]: s["i1"] + 1])
    return cKDTree(np.concatenate(pts)) if pts else None


for run in runs:
    dump = pickle.load(open(os.path.join(ROOT, "temp", DIR, f"pred_{run}.pkl"), "rb"))
    frames = all_frames[: len(dump)]
    assert DIR != "e275" or len(dump) == 873, len(dump)
    acc = dict(delivered=0, dropped=0, dropped_matched=0, delivered_matched=0, dropped_inter=0, delivered_inter=0, dropped_event=0, delivered_event=0, gt_nodes=0, gt_close_pair=0, gt_close_pair_event=0,
               gt_event=0, tp_before=0, tp_after=0, lost_true=0, lost_true_gt_pair=0)
    dist_hist, e_drop, e_sup, ratio, nd_frame = [], [], [], [], []
    for k, (f, d) in enumerate(zip(frames, dump)):
        assert d["row"] == f.row
        ref = f.ref()
        g = f.lane()
        G = ref.nt["pos"]
        Gev = np.asarray(ref.nt["flags"]).any(1)
        acc["gt_nodes"] += len(G)
        acc["gt_event"] += int(Gev.sum())
        if len(G) > 1:
            dd = cKDTree(G).query(G, k=2)[0][:, 1]
            near = dd < R
            acc["gt_close_pair"] += int(near.sum())
            acc["gt_close_pair_event"] += int((near & Gev).sum())
        sel = np.flatnonzero(d["e"] >= THR)
        pos, e, fl = d["pos"][sel].astype(float), d["e"][sel].astype(float), d["flags"][sel]
        keep = RO.suppress_close_nodes(pos, e, R, THR)
        acc["delivered"] += len(sel)
        acc["dropped"] += int((~keep).sum())
        nd_frame.append(int((~keep).sum()))
        pairs = MS.match_1to1(pos, G, QE.NODE_TOL) if len(pos) and len(G) else np.zeros((0, 2), int)
        matched = np.zeros(len(pos), bool)
        matched[pairs[:, 0]] = True
        acc["delivered_matched"] += int(matched.sum())
        acc["dropped_matched"] += int((matched & ~keep).sum())
        pk = pos[keep]
        pairs2 = MS.match_1to1(pk, G, QE.NODE_TOL) if len(pk) and len(G) else np.zeros((0, 2), int)
        acc["tp_before"] += len(pairs)
        acc["tp_after"] += len(pairs2)
        lost = np.setdiff1d(pairs[:, 1], pairs2[:, 1]) if len(pairs) else np.zeros(0, int)       # GT nodes matched before the step, not after
        acc["lost_true"] += len(lost)
        if len(lost) and len(G) > 1:
            dd = cKDTree(G).query(G[lost], k=2)[0][:, 1]
            acc["lost_true_gt_pair"] += int((dd < R).sum())                                      # the GT node has another GT node within R: the ceiling explains the loss
        tr = inter_tree(g)
        near_i = (tr.query(pos)[0] <= NEAR_INTER) if (tr is not None and len(pos)) else np.zeros(len(pos), bool)
        acc["delivered_inter"] += int(near_i.sum())
        acc["dropped_inter"] += int((near_i & ~keep).sum())
        ev = fl.max(1) >= 0.5 if len(fl) else np.zeros(0, bool)
        acc["delivered_event"] += int(ev.sum())
        acc["dropped_event"] += int((ev & ~keep).sum())
        if (~keep).any() and keep.any():                                                            # distance to the nearest kept node, e of the dropped node and of that kept node
            tk = cKDTree(pk)
            dist, j = tk.query(pos[~keep])
            dist_hist += dist.tolist()
            e_drop += e[~keep].tolist()
            e_sup += e[keep][j].tolist()
    A = acc
    out = dict(run=run, frames=len(frames), step=f"close {R} m at e >= {THR}", delivered=A["delivered"], dropped=A["dropped"], dropped_share=A["dropped"] / A["delivered"],
               dropped_counted_true_share=A["dropped_matched"] / max(A["dropped"], 1), delivered_counted_true_share=A["delivered_matched"] / A["delivered"],
               node_recall_before=A["tp_before"] / A["gt_nodes"], node_recall_after=A["tp_after"] / A["gt_nodes"], node_precision_before=A["tp_before"] / A["delivered"],
               node_precision_after=A["tp_after"] / (A["delivered"] - A["dropped"]),
               dropped_in_intersection_share=A["dropped_inter"] / max(A["dropped"], 1), delivered_in_intersection_share=A["delivered_inter"] / A["delivered"],
               dropped_event_flag_share=A["dropped_event"] / max(A["dropped"], 1), delivered_event_flag_share=A["delivered_event"] / A["delivered"],
               gt_nodes_with_gt_neighbour_within_R_share=A["gt_close_pair"] / A["gt_nodes"], gt_event_nodes_with_gt_neighbour_within_R=A["gt_close_pair_event"], gt_event_nodes=A["gt_event"],
               true_nodes_lost=A["lost_true"], true_nodes_lost_explained_by_gt_pair_share=A["lost_true_gt_pair"] / max(A["lost_true"], 1),
               distance_to_kept_node_hist={f"{a}-{b}": int(((np.array(dist_hist) >= a) & (np.array(dist_hist) < b)).sum()) for a, b in ((0, 0.5), (0.5, 1.0), (1.0, 1.5))},
               mean_e_dropped=float(np.mean(e_drop)), mean_e_kept_neighbour=float(np.mean(e_sup)), frames_with_dropped=int(sum(1 for x in nd_frame if x > 0)),
               dropped_per_frame_median=float(np.median(nd_frame)), dropped_per_frame_max=int(max(nd_frame)))
    json.dump(out, open(os.path.join(ROOT, "temp", DIR, f"analysis_{run}.json"), "w"), indent=1)
    print(json.dumps(out, indent=1), flush=True)
    # pictures: the 20 frames with the most dropped nodes
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    top = np.argsort(-np.array(nd_frame))[:20]
    for rank, k in enumerate(top):
        f, d = frames[k], dump[k]
        ref, g = f.ref(), f.lane()
        sel = np.flatnonzero(d["e"] >= THR)
        pos, e = d["pos"][sel].astype(float), d["e"][sel].astype(float)
        keep = RO.suppress_close_nodes(pos, e, R, THR)
        fig, ax = plt.subplots(figsize=(7, 7))
        for p in g["pieces"]:
            q = np.asarray(p["pts"], float)
            ax.plot(q[:, 0], q[:, 1], color="0.8", lw=1.5, zorder=1)
        G = ref.nt["pos"]
        ax.scatter(G[:, 0], G[:, 1], s=60, facecolors="none", edgecolors="green", lw=1.2, zorder=2, label="GT nodes")
        ax.scatter(pos[keep, 0], pos[keep, 1], s=14, c="tab:blue", zorder=3, label="kept")
        ax.scatter(pos[~keep, 0], pos[~keep, 1], s=40, c="tab:red", marker="x", zorder=4, label="dropped by the step")
        ax.set_xlim(-30, 30)
        ax.set_ylim(-30, 30)
        ax.set_aspect("equal")
        ax.set_title(f"{run}  frame {k}  dropped {int((~keep).sum())} of {len(keep)}", fontsize=9)
        ax.legend(loc="upper right", fontsize=7)
        fig.savefig(os.path.join(ROOT, "results", DIR.upper(), "figures", f"{run}_dropped_{rank:02d}.png"), dpi=90, bbox_inches="tight")
        plt.close(fig)
