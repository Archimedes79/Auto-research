import time, numpy as np, autogt

t0 = time.time()
big = autogt.AutoGT("~/data/nuscenes_full", "v1.0-trainval", light=True, scenes=["scene-0561"])
t1 = time.time()
toks = big.keyframes("scene-0561")
print("devkit init s", round(t1 - t0, 1), "n keyframes", len(toks), flush=True)
shares, per = [], []
for tok in toks:
    g = autogt.load(tok)
    moving = (np.hypot(*g["velocity"]) > 1.0) & (g["flow_weight"] > 0)
    covered = (g["vehicle"] > 0)
    n_cov = int(covered.sum())
    if n_cov == 0:
        per.append(None)
        continue
    s = float((moving & covered).sum()) / n_cov
    per.append(s)
    shares.append(s)
print("frames with vehicle cells", len(shares), "of", len(toks))
print("mean share moving among vehicle-covered cells (mean over frames)", float(np.mean(shares)))
# pooled over all cells of the sequence
tot_m = tot_c = 0
for tok in toks:
    g = autogt.load(tok)
    moving = (np.hypot(*g["velocity"]) > 1.0) & (g["flow_weight"] > 0)
    covered = g["vehicle"] > 0
    tot_m += int((moving & covered).sum()); tot_c += int(covered.sum())
print("pooled share", tot_m / tot_c, "cells", tot_c)
print("total s", round(time.time() - t0, 1))
