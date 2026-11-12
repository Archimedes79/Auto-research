import time, sys, json, numpy as np
import qgraph_eval as QE, qformat
fr = QE.val_frames()[::12]
preds = [qformat.oracle_prediction(f.lane()) for f in fr]
for tie in (0.05, 0.2, 0.3, 0.5, 1.0):
    QE.TIE = tie
    out = QE.score_frames(preds, fr)
    print(tie, round(out["assign"]["lane_acc"], 5), round(out["assign"]["purity"], 5), out["graph"]["pieces_per_gt_piece"], out["graph"]["id_switches_per_100m"])
QE.TIE = 0.05
# which nodes fail
bad = []
for f, p in zip(fr, preds):
    ref = f.ref(); nd = p["nodes"]; d, tied, pj = ref.near(nd["pos"])
    for n in range(len(d)):
        own = int(p["table"]["piece"][n])
        if d[n] <= 1.0 and own not in tied[n]:
            bad.append((f.k, n, float(d[n]), own, tied[n].tolist(), [k for k in range(8) if nd["flags"][n][k]]))
print(len(bad)); print(bad[:12])
