import os, sys, time, cProfile, pstats
import numpy as np
import torch
torch.set_num_threads(2)
ROOT = os.environ["BFLC_ROOT"]
for s in ("training/data", "training/grid", "evaluation"):
    sys.path.insert(0, os.path.join(ROOT, s))
import lane_ptok_data as LP, lane_tokcache as LC, lane_tok_data as LT
ds = LP.PTokData("val", 0, "e292_a")
td = LT.TokData("val", res=2.0, feats=False)
assert td.tokens == ds.tokens
f = ds.c.f; K = ds.K
idx = [3, 50, 200, 400, 600, 800]
tabs = [td.table(i) for i in idx]
tks = [{k: np.array(f[k][i]) for k in ("pos", "dirv", "valid", "e_p")} for i in idx]
tks = [dict(t, pos=t["pos"].astype(np.float32)) for t in tks]
def run():
    for t, tb in zip(tks, tabs):
        LC.label_frame(t, tb, K=K, allowed=None)
run()
t0 = time.perf_counter(); 
for _ in range(3): run()
print("label_frame ms/frame", (time.perf_counter() - t0) / 3 / len(idx) * 1000, "M", [len(t["pos"]) for t in tabs], "valid", [int(t["valid"].sum()) for t in tks])
pr = cProfile.Profile(); pr.enable(); run(); pr.disable()
pstats.Stats(pr).sort_stats("cumtime").print_stats(8)
# assign_greedy loop alone
near, d = LC.near_matrix(tks[2]["pos"], tks[2]["dirv"], tks[2]["valid"], tabs[2]["pos"], tabs[2]["dirv"])
t0 = time.perf_counter()
for _ in range(20): LC.assign_greedy(near, d, K)
print("assign_greedy ms", (time.perf_counter() - t0) / 20 * 1000, "near rows with any:", int(near.any(1).sum()))
