"""one-off: sensitivity of the delivered graph to the read-out thresholds (150 VAL frames, diagnostic only): link threshold, long-link threshold, node threshold"""
import sys, os
for p in ("training/train", "training/grid", "training/data", "evaluation", "training"):
    sys.path.insert(0, os.path.join("/mnt/c/Auto-research/bev_fusion_lane_center", p))
import numpy as np, torch
import qgates as G, qpredict as QP, qgraph_eval as QE
run = sys.argv[1]
m = QP.load_model("/mnt/c/Auto-research/bev_fusion_lane_center/runs/" + run, "cuda")
frames = QE.val_frames()
idx = np.linspace(0, len(frames) - 1, 150).astype(int)
fr = [frames[i] for i in idx]
preds = []
for b in G._batches("exact", 150):
    preds += QP.predict(m, b)
preds = [G.component_sections(p) for p in preds]
def run_thr(te, tl, tlong):
    import copy
    ps = []
    for p in preds:
        q = dict(nodes=p["nodes"], links=dict(p["links"]), assign=p["assign"], maps=None)
        L = q["links"]
        keep = np.where(L["long"], L["p"] >= tlong, L["p"] >= tl)
        for k in ("src", "dst", "p", "long", "a", "b"):
            L[k] = np.asarray(L[k])[keep]
        ps.append(q)
    out = QE.score_frames(ps, fr, thr_e=te, thr_link=0.0)
    g = out["graph"]
    return {k: round(float(g.get(k, float("nan"))), 3) for k in ("F1@1.0", "lanes_found", "topo_P", "topo_R", "topo_F1", "junction_P", "junction_R", "junction_F1", "dup_frac", "vert100_dp")}
for te, tl, tlong in ((0.5, 0.2, 0.2), (0.3, 0.2, 0.2), (0.2, 0.2, 0.2), (0.1, 0.2, 0.2), (0.2, 0.1, 0.2), (0.1, 0.1, 0.1)):
    print((te, tl, tlong), run_thr(te, tl, tlong), flush=True)
