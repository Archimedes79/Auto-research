import json
import os
R = os.environ.get("BFLC_ROOT") or "."
import pickle
import sys

import numpy as np
import torch

for run in ("runs/e273_p_s0", "runs/e292_a_s0", "runs/e292_c2_s0"):
    sd = torch.load(os.path.join(R, run, "model.pt"), map_location="cpu")
    a = sd.get("args") or {}
    print(run, {k: a[k] for k in sorted(a) if k in ("width", "soft_nms", "ends", "sparse", "node_loss", "seed", "steps", "phases", "init", "freeze", "tier_w", "neg_w")})
    print("   keys", sorted(sd.keys()), "state tensors", len(sd["state"]))
for p in ("temp/e276/predfull_e273_p_s0.pkl", "temp/e292/predfull_e292_a_s0.pkl"):
    d = pickle.load(open(os.path.join(R, p), "rb"))
    n5 = np.array([int((f["e"] >= 0.5).sum()) for f in d])
    n2 = np.array([len(f["e"]) for f in d])
    print(p, "frames", len(d), "e>=0.2 nodes/frame mean %.1f max %d | e>=0.5 mean %.1f max %d | keys" % (n2.mean(), n2.max(), n5.mean(), n5.max()), sorted(d[0].keys()))
    print("   dtype e", d[0]["e"].dtype, "pos", d[0]["pos"].dtype, "idx", d[0]["idx"].dtype, "row", d[0]["row"])
