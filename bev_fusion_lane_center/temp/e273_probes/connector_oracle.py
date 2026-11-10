"""one-off: can every GT connector be written as a relation between the END of a lane piece and the START of a lane piece in the window? (VAL scenes only)"""
import pickle, os, collections
import numpy as np
H = os.path.expanduser("~")
VAL = {1, 3, 5, 7, 9, 11, 13, 15, 17, 19, 21, 23, 25, 27, 29, 31, 33, 35, 37, 39, 41, 42}
G = pickle.load(open(f"{H}/exp_gcar_b123/graph_gt_val.pkl", "rb"))
sc = np.load(f"{H}/exp_gcar_b123/poses_val.npz")["scene"]
c = collections.Counter()
for k, g in enumerate(G):
    if int(sc[k]) not in VAL:
        continue
    pred, succ = collections.defaultdict(list), collections.defaultdict(list)
    for i, j in g["edges"]:
        succ[i].append(j); pred[j].append(i)
    for i, kd in enumerate(g["kind"]):
        if kd != "connector" or len(g["pts"][i]) < 2:
            continue
        L = float(np.linalg.norm(np.diff(np.asarray(g["pts"][i], float), axis=0), axis=1).sum())
        a = any(g["kind"][p] == "lane" for p in pred[i])
        b = any(g["kind"][s] == "lane" for s in succ[i])
        so, eo = bool(g["start_open"][i]), bool(g["end_open"][i])
        key = "both ends on lane pieces" if a and b else ("one end on a lane piece, other end open at the window edge" if (a and eo) or (b and so) else
              ("chained to another connector" if (pred[i] or succ[i]) else ("both ends open (crosses the window edge twice)" if so and eo else "isolated")))
        c[key] += 1; c["length:" + key] += L
n = sum(v for k, v in c.items() if not k.startswith("length:"))
L = sum(v for k, v in c.items() if k.startswith("length:"))
for k, v in c.items():
    if not k.startswith("length:"):
        print(f"{k}: {v / n:.3f} of connectors, {c['length:' + k] / L:.3f} of connector length")
