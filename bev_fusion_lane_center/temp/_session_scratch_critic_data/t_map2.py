import os, sys, time, cProfile, pstats
import numpy as np
import torch
torch.set_num_threads(2)
ROOT = os.environ["BFLC_ROOT"]
for s in ("training/data", "training/data/prep", "training/grid", "evaluation", ".."):
    sys.path.insert(0, os.path.normpath(os.path.join(ROOT, s)))
import lane_map_data as LM
ds = LM.MapData("val", 0, "e292_a", None, "error")
for i in range(3): ds.table(i)
def T(fn, n=30):
    t0 = time.perf_counter()
    for i in range(n): fn(i * 20)
    return (time.perf_counter() - t0) / n * 1000
print("pieces (GT load) ms", T(lambda i: ds.pieces(i)))
print("frame_pts ms", T(lambda i: ds.pts(i)))
P = [ds.pts(i * 20) for i in range(30)]
print("roads_of ms", T(lambda i: LM.roads_of(P[i // 20])))
R = [LM.roads_of(p) for p in P]
print("sample_roads ms", T(lambda i: LM.sample_roads(R[i // 20])))
print("error_map total ms", T(lambda i: ds.table(i, "error")))
pr = cProfile.Profile(); pr.enable()
for i in range(0, 300, 10): ds.table(i, "error")
pr.disable(); pstats.Stats(pr).sort_stats("tottime").print_stats(8)
