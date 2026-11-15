"""one-off: VAL scenes (ids of lane_prototype.py) ranked by GT splits + merges per frame; the even ids of poses_val are TEST scenes and are skipped"""
import pickle, os, collections
import numpy as np
H = os.path.expanduser("~")
VAL = [1, 3, 5, 7, 9, 11, 13, 15, 17, 19, 21, 23, 25, 27, 29, 31, 33, 35, 37, 39, 41, 42]
G = pickle.load(open(f"{H}/exp_gcar_b123/graph_gt_val.pkl", "rb"))
sc = np.load(f"{H}/exp_gcar_b123/poses_val.npz")["scene"]
cnt, n = collections.Counter(), collections.Counter()
for k, g in enumerate(G):
    s = int(sc[k])
    if s not in VAL:
        continue
    succ, pred = collections.Counter(), collections.Counter()
    for i, j in g["edges"]:
        succ[i] += 1; pred[j] += 1
    cnt[s] += sum(v >= 2 for v in succ.values()) + sum(v >= 2 for v in pred.values())
    n[s] += 1
for s, c in sorted(cnt.items(), key=lambda x: -x[1] / n[x[0]])[:6]:
    print(s, n[s], round(c / n[s], 2))
