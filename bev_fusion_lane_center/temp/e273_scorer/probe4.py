import time, sys, json, numpy as np
import qgraph_eval as QE, qformat
t0 = time.time()
fr = QE.val_frames()
print("frames", len(fr), "stored share", QE.stored_share(fr), time.time() - t0)
sel = fr[::12]
preds = []
t0 = time.time()
for f in sel:
    preds.append(qformat.oracle_prediction(f.lane()))
print("oracle preds", len(preds), time.time() - t0)
t0 = time.time()
out, rec = QE.score_frames(preds, sel, records=True)
print("score time", time.time() - t0)
for k in ("graph", "nodes", "assign"):
    print(k, {a: (round(b, 4) if isinstance(b, float) else b) for a, b in out[k].items()})
# worst frames by F1
f1 = [QE.graph_numbers([c])["F1@1.0"] for c in rec["graph"]]
o = np.argsort(f1)[:5]
print("worst F1@1.0", [(int(sel[i].k), round(f1[i], 4)) for i in o])
