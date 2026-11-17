import time, numpy as np
import qgraph_eval as QE, qformat
fr = QE.val_frames()[::29]
print(len(fr))
kinds = {}
cs = {a: [] for a in ("g0", "g1", "l2")}
t = time.time()
tl = 0
for f in fr:
    for k in f.gt["kind"]:
        kinds[k] = kinds.get(k, 0) + 1
    g = f.lane()
    cs["g0"].append(QE.graph_counts(f.gt, f.gt))
    idx = {p["id"]: i for i, p in enumerate(g["pieces"])}
    raw = dict(pts=[np.asarray(p["pts"], float) for p in g["pieces"]], edges=[(i, idx[q]) for i, p in enumerate(g["pieces"]) for q in p["succ"] if q in idx])
    cs["g1"].append(QE.graph_counts(raw, f.gt))
    t1 = time.time()
    pred = qformat.oracle_prediction(g)
    graph, bk = QE.readout(pred)
    tl += time.time() - t1
    cs["l2"].append(QE.graph_counts(graph, f.gt))
print(kinds, "time", time.time() - t, "oracle+readout", tl)
for a, c in cs.items():
    n = QE.graph_numbers(c)
    print(a, {k: round(n[k], 4) for k in ("topo_F1", "F1@1.0", "lanes_found", "junction_F1", "vert100", "vert100_dp")})
