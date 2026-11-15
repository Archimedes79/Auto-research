import os, pickle, sys, time
import numpy as np
R = os.environ["BFLC_ROOT"]
for sub in ("evaluation", "training", "training/grid", "training/data", ".."):
    sys.path.insert(0, os.path.join(R, sub))
import torch
torch.set_num_threads(2)
import qgraph_eval as QE
d = pickle.load(open(f"{R}/runs/e324_s_s0/ends_card_snap_thr0.1.counts.pkl", "rb"))
print(type(d), d.keys(), len(d["head"]), type(d["head"][0]), list(d["head"][0].keys()))
t = time.time()
for _ in range(5):
    QE.graph_numbers(d["head"])
print("graph_numbers 873 frames:", (time.time() - t) / 5)
