"""one-off diagnostic: what would a RULE step at the read-out that bridges gaps (a kept node without a kept successor links to the nearest kept node ahead within R m) do to the delivered graph? (150 VAL frames)"""
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
def bridge(p, R, cone=0.5, dcos=0.7):
    nd, lk = p["nodes"], p["links"]
    keep = np.where(nd["e"] >= 0.5)[0]
    pos, d = nd["pos"], nd["dirv"]
    out = set(np.asarray(lk["src"])[np.asarray(lk["p"]) >= 0.5].tolist())
    src, dst = [], []
    for a in keep:
        if a in out:
            continue
        v = pos[keep] - pos[a]
        dist = np.linalg.norm(v, axis=1)
        ok = (dist > 0.5) & (dist <= R) & ((v @ d[a]) / np.maximum(dist, 1e-6) > cone) & ((d[keep] @ d[a]) > dcos)
        if ok.any():
            c = keep[np.where(ok)[0][np.argmin(dist[ok])]]
            src.append(a); dst.append(c)
    q = dict(nodes=nd, assign=p["assign"], maps=None, links={k: np.concatenate([np.asarray(lk[k]), np.asarray(v)]) for k, v in (("src", src), ("dst", dst), ("p", np.ones(len(src))), ("long", np.zeros(len(src), bool)), ("a", np.full(len(src), 1 / 3)), ("b", np.full(len(src), 1 / 3)))}, )
    q["links"]["curves"] = None
    return q
for R in (0, 4, 6, 8):
    ps = preds if R == 0 else [bridge(p, R) for p in preds]
    g = QE.score_frames(ps, fr)["graph"]
    print("bridge R", R, {k: round(float(g.get(k, float('nan'))), 3) for k in ("F1@1.0", "lanes_found", "topo_P", "topo_R", "topo_F1", "junction_F1", "dup_frac", "pieces_per_gt_piece")}, flush=True)
