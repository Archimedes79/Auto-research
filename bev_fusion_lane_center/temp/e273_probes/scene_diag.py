"""one-off: delivered graph numbers by scene group (junction-rich scenes vs the rest), 150 VAL frames"""
import sys, os
for p in ("training/train", "training/grid", "training/data", "evaluation", "training"):
    sys.path.insert(0, os.path.join("/mnt/c/Auto-research/bev_fusion_lane_center", p))
import numpy as np, torch
torch.set_num_threads(2)
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
sc = np.array([f.scene for f in fr])
for name, sel in (("all", np.ones(len(fr), bool)), ("junction scenes 11, 39", np.isin(sc, [11, 39])), ("scene 23 (straight)", sc == 23), ("rest", ~np.isin(sc, [11, 39, 23]))):
    ii = np.where(sel)[0]
    if not len(ii):
        continue
    g = QE.score_frames([preds[i] for i in ii], [fr[i] for i in ii])["graph"]
    print(f"{name:26s} n={len(ii):3d}", {k: round(float(g.get(k, float('nan'))), 3) for k in ("F1@1.0", "lanes_found", "topo_P", "topo_R", "topo_F1", "junction_F1", "dup_frac")}, flush=True)
